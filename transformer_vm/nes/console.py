"""NES console: the frame loop over a 6502-in-the-graph CPU.

Wires the frame-level PPU, OAM DMA, and controllers onto the MMIO bus and
paces frames by CPU cycle budget:

    run to scanline 241  ->  render frame  ->  vblank + NMI  ->  run to
    frame end  ->  vblank clear  ->  repeat

NMI is delivered through the graph's in-model interrupt sequence (the
console only flags the next STATE token); vblank timing uses the base
cycle counts shared by every backend, so a kernel-driven console and a
reference-driven console execute identical instruction streams.

NTSC timing, rounded to integer CPU cycles per frame:
    scanline      = 341 PPU dots / 3 = 113.667 CPU cycles
    vblank start  = scanline 241 -> 27394 cycles into the frame
    frame         = 262 scanlines   -> 29781 cycles
"""

from __future__ import annotations

import argparse
import logging
import time

from transformer_vm.nes.bus import Bus, GraphCore, KernelBus, ReferenceCore
from transformer_vm.nes.cartridge import Cartridge
from transformer_vm.nes.ppu import PPU

logger = logging.getLogger(__name__)

CYCLES_PER_SCANLINE = 341 / 3
VBLANK_START_CYCLES = 27394  # scanline 241
FRAME_CYCLES = 29781         # 262 scanlines
OAM_DMA_STALL = 513

BUTTONS = ("a", "b", "select", "start", "up", "down", "left", "right")


class Controller:
    """Standard controller serial protocol at $4016/$4017 (port 1 only)."""

    def __init__(self):
        self.buttons = dict.fromkeys(BUTTONS, False)
        self._strobe = False
        self._index = 0

    def set_buttons(self, **pressed: bool) -> None:
        for name, value in pressed.items():
            if name not in self.buttons:
                raise KeyError(f"unknown button {name!r}")
            self.buttons[name] = value

    def read(self, addr: int) -> int:
        if addr != 0x4016:
            return 0x40  # port 2 not connected; open-bus bit
        if self._strobe:
            return 0x40 | (1 if self.buttons["a"] else 0)
        if self._index >= 8:
            return 0x41  # official controllers return 1 after the report
        bit = 1 if self.buttons[BUTTONS[self._index]] else 0
        self._index += 1
        return 0x40 | bit

    def write(self, addr: int, value: int) -> None:
        if addr == 0x4016:
            strobe = bool(value & 1)
            if self._strobe and not strobe:
                self._index = 0
            self._strobe = strobe
            if strobe:
                self._index = 0


class OamDma:
    """$4014: copy CPU page -> PPU OAM, stalling the CPU for 513 cycles."""

    def __init__(self, ppu: PPU, read_cpu):
        self.ppu = ppu
        self._read_cpu = read_cpu
        self._stall = 0

    def read(self, addr: int) -> int:
        return 0

    def write(self, addr: int, value: int) -> None:
        base = (value & 0xFF) << 8
        page = bytes(self._read_cpu(base + i) for i in range(256))
        self.ppu.oam_dma(page)
        self._stall += OAM_DMA_STALL

    def take_stall(self) -> int:
        stall, self._stall = self._stall, 0
        return stall


class Console:
    """The NES: a 6502 machine backend + PPU + input over the MMIO bus."""

    def __init__(self, cart: Cartridge | str, *, backend: str = "kernel",
                 checkpoint_tokens: int = 2_000_000):
        if isinstance(cart, str):
            cart = Cartridge.load(cart)
        self.cart = cart
        self.ppu = PPU(cart)
        self.controller = Controller()
        image = cart.cpu_image()

        self._boot(backend, image, cart.reset_vector, state=None)
        self.dma = OamDma(self.ppu, self._read_cpu)
        self._attach_devices()

        self.cycles = 0        # console-side base-cycle clock
        self.frame_index = 0
        self.halted = False
        self._frame_base = 0
        self._sprite0_scanline: int | None = None
        # Snapshot-and-rebuild: once the append-only token log passes this
        # many tokens, it is reseeded from the RAM mirror at the next frame
        # boundary (a quiet point: no pending NMI or read pause).  Keeps
        # hull attention memory and query depth bounded over long runs.
        # 0 disables; the reference backend has no log and never rebuilds.
        self.checkpoint_tokens = checkpoint_tokens
        self.checkpoint_count = 0

    # ── Backend construction / handoff ──────────────────────────────

    def _boot(self, backend: str, image: bytes, start_pc: int, state: dict | None):
        """Create the CPU machine + bus for a backend, optionally resuming
        from a snapshot (RAM image + architectural state)."""
        self.backend = backend
        self._uses_kernel_bus = False
        if backend == "kernel":
            from transformer_vm.cpu6502.kernel import Cpu6502CppRuntime

            self._rt = Cpu6502CppRuntime(bcd=False)
            self.bus = KernelBus(self._rt, image, start_pc=start_pc)
            if state is not None:
                self._rt.set_state(state)
            self._read_cpu = self._rt.read_ram
            self._uses_kernel_bus = True
            return
        if backend == "weights":
            # The real transformer through its generated C++ run loop:
            # KernelBus drives it natively (MMIO pauses back to Python), the
            # same protocol as the kernel.
            from transformer_vm.cpu6502.weights_runner import Cpu6502TransformerMachine

            machine = Cpu6502TransformerMachine(bcd=False, checkpoint_tokens=None)
            if not machine.supports_run_loop:
                raise RuntimeError("the weights backend requires the codegen run loop")
            self._rt = machine
            self.bus = KernelBus(machine, image, start_pc=start_pc)
            if state is not None:
                machine.set_state(state)
            self._read_cpu = machine.read_ram
            self._uses_kernel_bus = True
            return
        if backend == "reference":
            from transformer_vm.cpu6502.reference import Cpu6502

            cpu = Cpu6502(memory=bytearray(image), bcd_enabled=False)
            cpu.pc = start_pc
            if state is not None:
                cpu.pc, cpu.a, cpu.x, cpu.y, cpu.s, cpu.p = (
                    int(state[k]) for k in ("pc", "a", "x", "y", "s", "p")
                )
            core = ReferenceCore(cpu)
        elif backend == "graph":
            from transformer_vm.cpu6502.evaluator import Cpu6502Runtime

            rt = Cpu6502Runtime(bcd=False)
            rt.boot_image(list(image), start_pc=start_pc, state=state)
            core = GraphCore(rt)
        else:
            raise ValueError(f"unknown backend {backend!r}")
        self.bus = Bus(core, image)
        self._read_cpu = lambda addr: self.bus.ram[addr & 0xFFFF]

    def _attach_devices(self):
        self.bus.attach(self.ppu, 0x2000, 0x3FFF)     # registers + mirrors
        self.bus.attach(self.dma, 0x4014, 0x4014)
        self.bus.attach(self.controller, 0x4016, 0x4017)

    def _snapshot(self) -> tuple[bytes, dict]:
        if self._uses_kernel_bus:
            return self._rt.dump_ram(), self._rt.get_state()
        ram = bytes(self.bus.ram)
        core = self.bus.core
        if isinstance(core, ReferenceCore):
            cpu = core.cpu
            state = {"pc": cpu.pc, "a": cpu.a, "x": cpu.x, "y": cpu.y,
                     "s": cpu.s, "p": cpu.p, "halted": cpu.halted}
        else:
            state = core.rt.machine_state
        return ram, state

    def handoff(self, backend: str) -> None:
        """Transfer the running console to another CPU backend mid-game.

        Must happen at a frame boundary (no pending NMI or read pause);
        run_frame() always returns at one.  The PPU, controllers, and the
        console clock carry over untouched -- only the CPU machine and its
        token log are rebuilt from the snapshot.
        """
        ram, state = self._snapshot()
        self._boot(backend, ram, int(state["pc"]), state)
        self.dma._read_cpu = self._read_cpu
        self._attach_devices()
        if isinstance(self.bus, Bus):
            self.bus.cycles = self.cycles  # console clock continues

    # ── CPU driving ─────────────────────────────────────────────────

    def _set_nmi(self) -> None:
        if self._uses_kernel_bus:
            self._rt.set_nmi()
        else:
            self.bus.set_nmi()

    def _check_nmi_edge(self) -> None:
        if self.ppu.take_nmi_edge():
            self._set_nmi()

    def _run_until(self, target: int) -> None:
        """Run the CPU until the console clock reaches target cycles."""
        if self.halted:
            return
        if self._uses_kernel_bus:
            if target <= self.cycles:
                return
            out = self.bus.run(
                1 << 60,
                max_cycles=target - self.cycles,
                on_pause=lambda _out: self._check_nmi_edge(),
            )
            self.cycles += int(out["cycles"])
            if out["reason"] in ("halt", "trap"):
                self.halted = True
        else:
            bus = self.bus
            while bus.cycles < target and not bus.halted:
                bus.step()
                self._check_nmi_edge()
            self.cycles = bus.cycles
            if bus.halted:
                self.halted = True

    # ── Frame loop ──────────────────────────────────────────────────

    def run_frame(self):
        """Run one NTSC frame; returns the rendered 240x256 index frame."""
        base = self._frame_base

        # Visible portion, split by the sprite-0 hit event when pending.
        # Each PPU status transition invalidates any $2002 read lease.
        if self._sprite0_scanline is not None and self.ppu.rendering_enabled:
            hit_cycle = base + int((self._sprite0_scanline + 1) * CYCLES_PER_SCANLINE)
            self._run_until(min(hit_cycle, base + VBLANK_START_CYCLES))
            self.bus.clear_read_lease()
            self.ppu.sprite0_hit = True
        self._run_until(base + VBLANK_START_CYCLES)
        self.bus.clear_read_lease()

        frame = self.ppu.render_frame()
        self._sprite0_scanline = self.ppu.sprite0_scanline

        self.ppu.begin_vblank()
        if self.ppu.nmi_enabled:
            self._set_nmi()
        self._run_until(base + FRAME_CYCLES)
        self.bus.clear_read_lease()
        self.ppu.end_vblank()

        self._frame_base += FRAME_CYCLES
        self.frame_index += 1

        if (
            self.checkpoint_tokens
            and not self.halted
            and self.bus.log_tokens >= self.checkpoint_tokens
            and self.bus.checkpoint()
        ):
            self.checkpoint_count += 1
        return frame

    def run(self, frames: int):
        """Run several frames; returns the last rendered frame."""
        frame = None
        for _ in range(frames):
            if self.halted:
                break
            frame = self.run_frame()
        return frame


def write_ppm(path: str, frame) -> None:
    """Write a rendered frame as a binary PPM (no dependencies)."""
    rgb = PPU.to_rgb(frame)
    with open(path, "wb") as f:
        f.write(b"P6\n256 240\n255\n")
        f.write(rgb.tobytes())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run an NES ROM (NROM) on the 6502 graph machine.")
    parser.add_argument("rom", help="path to an iNES .nes file")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument(
        "--backend", choices=["kernel", "reference", "graph", "weights"], default="kernel"
    )
    parser.add_argument(
        "--warmup-frames", type=int, default=0,
        help="run this many frames on the kernel first, then hand the running "
             "game off to --backend for the remaining frames",
    )
    parser.add_argument("--out", help="write the final frame as a PPM image")
    args = parser.parse_args()

    build_start = time.perf_counter()
    first_backend = "kernel" if args.warmup_frames else args.backend
    console = Console(args.rom, backend=first_backend)
    build_elapsed = time.perf_counter() - build_start

    run_start = time.perf_counter()
    if args.warmup_frames:
        console.run(args.warmup_frames)
        logger.info(
            "handoff to %s backend after %d frames (PC=%04X)",
            args.backend, console.frame_index, console._snapshot()[1]["pc"],
        )
        console.handoff(args.backend)
    frame = console.run(args.frames - console.frame_index)
    run_elapsed = time.perf_counter() - run_start

    fps = console.frame_index / run_elapsed if run_elapsed > 0 else 0.0
    logger.info(
        "%d frames in %.2fs (%.1f fps), %d CPU cycles, machine build %.2fs%s",
        console.frame_index, run_elapsed, fps, console.cycles, build_elapsed,
        " [HALTED]" if console.halted else "",
    )
    if args.out and frame is not None:
        write_ppm(args.out, frame)
        logger.info("final frame written to %s", args.out)


if __name__ == "__main__":
    main()
