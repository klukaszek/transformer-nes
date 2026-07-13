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
import hashlib
import json
import logging
import os
import time

from transformer_nes.nes.apu import make_apu
from transformer_nes.nes.bus import Bus, GraphCore, KernelBus, ReferenceCore
from transformer_nes.nes.cartridge import Cartridge
from transformer_nes.nes.mappers import make_mapper
from transformer_nes.nes.ppu import make_ppu, to_rgb

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
        self.strobe = False
        self.index = 0

    def set_buttons(self, **pressed: bool) -> None:
        for name, value in pressed.items():
            if name not in self.buttons:
                raise KeyError(f"unknown button {name!r}")
            self.buttons[name] = value

    def read(self, addr: int) -> int:
        if addr != 0x4016:
            return 0x40  # port 2 not connected; open-bus bit
        if self.strobe:
            return 0x40 | (1 if self.buttons["a"] else 0)
        if self.index >= 8:
            return 0x41  # official controllers return 1 after the report
        bit = 1 if self.buttons[BUTTONS[self.index]] else 0
        self.index += 1
        return 0x40 | bit

    def write(self, addr: int, value: int) -> None:
        if addr == 0x4016:
            strobe = bool(value & 1)
            if self.strobe and not strobe:
                self.index = 0
            self.strobe = strobe
            if strobe:
                self.index = 0

    def read_lease(self, addr: int):
        """Serial reads are fully determined between strobes: lease the
        remaining report bits as a queue (int = repeat one value)."""
        if addr != 0x4016:
            return None
        if self.strobe:
            return 0x40 | (1 if self.buttons["a"] else 0)
        if self.index >= 8:
            return 0x41
        return bytes(
            0x40 | (1 if self.buttons[BUTTONS[j]] else 0)
            for j in range(self.index, 8)
        )

    def lease_consumed(self, addr: int, n: int) -> None:
        if addr == 0x4016 and not self.strobe:
            self.index = min(8, self.index + n)


class ApuFrameCounter:
    """$4017 read shim: writes go to the APU, reads are open bus and
    leased (the native APU object cannot carry a read_lease attribute)."""

    def __init__(self, apu):
        self.apu = apu

    def read(self, addr: int) -> int:
        return self.apu.read(addr)

    def write(self, addr: int, value: int) -> None:
        self.apu.write(addr, value)

    def read_lease(self, addr: int):
        return 0x40


class OamDma:
    """$4014: copy CPU page -> PPU OAM, stalling the CPU for 513 cycles."""

    def __init__(self, ppu, read_cpu):
        self.ppu = ppu
        self.read_cpu = read_cpu
        self.stall = 0

    def read(self, addr: int) -> int:
        return 0

    def write(self, addr: int, value: int) -> None:
        base = (value & 0xFF) << 8
        page = bytes(self.read_cpu(base + i) for i in range(256))
        self.ppu.oam_dma(page)
        self.stall += OAM_DMA_STALL

    def take_stall(self) -> int:
        stall, self.stall = self.stall, 0
        return stall


# Register-journal entry kinds (debugging: dumped by app.py's F1).
J_PPU_WRITE = 0
J_PPU_STATUS_READ = 1
J_MAPPER_WRITE = 2
J_MAPPER_STATUS_READ = 3
J_EVENT = 4  # value: 0 sprite0, 1 irq delivery, 2 nmi

# run_frame phases for journal entries.
PHASE_VISIBLE = 0      # frame start through the last mid-frame event
PHASE_POST_EVENT = 1   # after a sprite-0 / IRQ event, still visible
PHASE_VBLANK = 2


class RegJournal:
    """Rolling log of device register traffic, the ground truth for
    debugging mid-frame raster state (order is exact: every entry is a
    serviced bus pause)."""

    def __init__(self, cap: int = 200_000):
        self.entries: list[tuple[int, int, int, int, int, int, int]] = []
        self.cap = cap

    def add(self, frame: int, phase: int, kind: int, addr: int, value: int,
            v_addr: int = -1, latch: int = -1) -> None:
        if len(self.entries) >= self.cap:
            del self.entries[:self.cap // 4]
        self.entries.append((frame, phase, kind, addr, value, v_addr, latch))


class JournaledPpu:
    """Bus shim over the PPU recording $2000-$2007 traffic with the
    write-time loopy state."""

    def __init__(self, ppu, journal: RegJournal, console: Console):
        self.ppu = ppu
        self.journal = journal
        self.console = console

    def read(self, addr: int) -> int:
        value = self.ppu.read(addr)
        if ((addr - 0x2000) & 7) == 2:
            self.journal.add(self.console.frame_index, self.console.journal_phase,
                             J_PPU_STATUS_READ, 0x2002, value,
                             self.ppu.vram_addr, int(self.ppu.write_latch))
        return value

    def read_lease(self, addr: int):
        return self.ppu.read_lease(addr)

    def write(self, addr: int, value: int) -> None:
        reg = (addr - 0x2000) & 7
        if reg != 3 and reg != 4:  # OAM traffic is bulky and never raster state
            self.journal.add(self.console.frame_index, self.console.journal_phase,
                             J_PPU_WRITE, 0x2000 + reg, value,
                             self.ppu.vram_addr, int(self.ppu.write_latch))
        self.ppu.write(addr, value)


class JournaledMapper:
    """Bus shim over a mapper recording configuration-register traffic."""

    def __init__(self, mapper, journal: RegJournal, console: Console):
        self.mapper = mapper
        self.journal = journal
        self.console = console

    def read(self, addr: int) -> int:
        value = self.mapper.read(addr)
        if addr == 0x5204:
            self.journal.add(self.console.frame_index, self.console.journal_phase,
                             J_MAPPER_STATUS_READ, addr, value)
        return value

    def write(self, addr: int, value: int) -> None:
        if 0x5100 <= addr <= 0x52FF:
            self.journal.add(self.console.frame_index, self.console.journal_phase,
                             J_MAPPER_WRITE, addr, value)
        self.mapper.write(addr, value)

    def apply_native(self, addr: int, value: int) -> None:
        self.mapper.apply_native(addr, value)


def profile_fallbacks(console, min_count: int) -> None:
    """Dump hot per-instruction-tier pcs with disassembly: the evidence
    that justifies a SHAPE_LIBRARY addition (ROADMAP START HERE item 4)."""
    from transformer_nes.cpu6502.spec import OPCODES

    engine = getattr(console.rt, "engine", None)
    if engine is None or not hasattr(engine, "jit_hot"):
        logger.info("--profile requires the weights backend")
        return
    hot = sorted(int(pc) for pc in engine.jit_hot(min_count))
    logger.info("%d fallback pcs >= %d steps:", len(hot), min_count)
    for pc in hot:
        text, off = [], 0
        for _ in range(4):
            byte = console.rt.read_ram((pc + off) & 0xFFFF)
            op = OPCODES.get(byte)
            if op is None:
                text.append(f"${byte:02X}?")
                break
            operand = [console.rt.read_ram((pc + off + 1 + j) & 0xFFFF)
                       for j in range(op.length - 1)]
            arg = "".join(f"{b:02X}" for b in reversed(operand))
            text.append(f"{op.mnemonic} {arg}".strip())
            off += op.length
        logger.info("  %04X  %s", pc, "; ".join(text))


def jit_sidecar_path(cart: Cartridge) -> str:
    """Per-ROM store for JIT window proposals, next to the engine cache."""
    from transformer_nes.cpu6502.backends.weights.windows import MAX_INSTRUCTIONS
    from transformer_nes.model.engine_gen import CACHE_DIR

    policy = (
        f"sequence-v4:safe-prefix:shape-promotion:"
        f"max-instructions={MAX_INSTRUCTIONS}"
    ).encode()
    digest = hashlib.sha256(cart.prg + policy).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"jitwin_{digest}.json")


def load_jit_windows(path: str):
    try:
        with open(path) as f:
            return tuple(
                (int(h), bytes.fromhex(c), int(r)) for h, c, r in json.load(f)
            )
    except (OSError, ValueError):
        return ()


def collect_sequence_windows(mapper, image, *, proposals=(), cap: int = 128):
    """Micro-loop windows from the boot image, last runs' hot JIT window
    proposals (proven profitable), then every runtime-reachable PRG bank
    mapping (mapper.prg_scan_regions), so banked-in code gets the pinned
    tier too.  The module's dispatch byte-verifies live RAM, so a window
    for a bank that is not currently mapped -- or RAM code not yet copied
    in -- is inert.  Deduped on (head, code), in that priority order under
    the cap."""
    from transformer_nes.cpu6502.backends.weights.windows import scan_sequence_windows

    windows = list(scan_sequence_windows(image))
    seen = {(head, bytes(code)) for head, code, _ in windows}
    for head, code, repeat in tuple(proposals)[:64]:  # sub-budget: banks
        key = (head, bytes(code))                     # still deserve slots
        if key not in seen:
            seen.add(key)
            windows.append((head, code, repeat))
    for base, data in mapper.prg_scan_regions():
        if len(windows) >= cap:
            break
        for head, code, repeat in scan_sequence_windows(data, base=base):
            key = (head, bytes(code))
            if key not in seen:
                seen.add(key)
                windows.append((head, code, repeat))
    return tuple(windows[:cap])


class Console:
    """The NES: a 6502 machine backend + PPU + input over the MMIO bus."""

    def __init__(self, cart: Cartridge | str, *, backend: str = "kernel",
                 checkpoint_tokens: int | None = None,
                 sample_rate: float = 44100.0):
        if isinstance(cart, str):
            cart = Cartridge.load(cart)
        self.cart = cart
        self.ppu = make_ppu(cart)
        self.apu = make_apu(sample_rate)
        self.controller = Controller()
        self.mapper = make_mapper(cart)
        self.journal = RegJournal()
        self.journal_phase = PHASE_VISIBLE
        image = self.mapper.cpu_image()
        reset_vector = image[0xFFFC] | (image[0xFFFD] << 8)

        self.boot(backend, image, reset_vector, state=None)
        self.dma = OamDma(self.ppu, self.read_cpu)
        self.attach_devices()

        self.cycles = 0        # console-side base-cycle clock
        self.frame_index = 0
        self.halted = False
        self.frame_base = 0
        self.sprite0_scanline: int | None = None
        self.last_audio = None      # this frame's APU samples (float32 mono)
        self.irq_sources: set[str] = set()
        # Snapshot-and-rebuild for the GRAPH evaluator only, whose
        # Python-side token log genuinely grows: past this many tokens it
        # reseeds from the RAM mirror at the next frame boundary (a quiet
        # point: no pending NMI or read pause).  The native engines'
        # attention state is O(1) in the token log and never rebuilds.
        if checkpoint_tokens is None:
            checkpoint_tokens = 2_000_000
        self.checkpoint_tokens = checkpoint_tokens
        self.checkpoint_count = 0

    # ── Backend construction / handoff ──────────────────────────────

    def boot(self, backend: str, image: bytes, start_pc: int, state: dict | None):
        """Create the CPU machine + bus for a backend, optionally resuming
        from a snapshot (RAM image + architectural state)."""
        self.backend = backend
        self.uses_kernel_bus = False
        if backend == "kernel":
            from transformer_nes.cpu6502.backends.kernel import Cpu6502CppRuntime

            self.rt = Cpu6502CppRuntime(bcd=False)
            self.bus = KernelBus(self.rt, image, start_pc=start_pc)
            if state is not None:
                self.rt.set_state(state)
            self.read_cpu = self.rt.read_ram
            self.poke_cpu = self.rt.inject_write
            self.poke_region = self.rt.inject_region
            self.map_region = self.rt.map_region
            self.uses_kernel_bus = True
            return
        if backend == "weights":
            # The real transformer through its generated C++ run loop:
            # KernelBus drives it natively (MMIO pauses back to Python), the
            # same protocol as the kernel.
            from transformer_nes.cpu6502.backends.weights.runner import Cpu6502TransformerMachine

            self.jit_sidecar = jit_sidecar_path(self.cart)
            machine = Cpu6502TransformerMachine(
                bcd=False,
                sequence_windows=collect_sequence_windows(
                    self.mapper, image,
                    proposals=load_jit_windows(self.jit_sidecar),
                ),
            )
            if not machine.supports_run_loop:
                raise RuntimeError("the weights backend requires the codegen run loop")
            self.rt = machine
            self.bus = KernelBus(machine, image, start_pc=start_pc)
            if state is not None:
                machine.set_state(state)
            self.read_cpu = machine.read_ram
            self.poke_cpu = machine.inject_write
            self.poke_region = machine.inject_region
            self.map_region = machine.map_region
            self.uses_kernel_bus = True
            return
        if backend == "reference":
            from transformer_nes.cpu6502.backends.reference import Cpu6502

            cpu = Cpu6502(memory=bytearray(image), bcd_enabled=False)
            cpu.pc = start_pc
            if state is not None:
                cpu.pc, cpu.a, cpu.x, cpu.y, cpu.s, cpu.p = (
                    int(state[k]) for k in ("pc", "a", "x", "y", "s", "p")
                )
            core = ReferenceCore(cpu)
        elif backend == "graph":
            from transformer_nes.cpu6502.backends.evaluator import Cpu6502Runtime

            rt = Cpu6502Runtime(bcd=False)
            rt.boot_image(list(image), start_pc=start_pc, state=state)
            core = GraphCore(rt)
        else:
            raise ValueError(f"unknown backend {backend!r}")
        self.bus = Bus(core, image)
        self.read_cpu = lambda addr: self.bus.ram[addr & 0xFFFF]

        def poke(addr: int, value: int) -> None:
            core.poke(addr, value)
            self.bus.ram[addr & 0xFFFF] = value & 0xFF
        self.poke_cpu = poke

        def poke_region(base: int, data: bytes) -> None:
            for i, byte in enumerate(data):
                poke(base + i, byte)
        self.poke_region = poke_region
        self.map_region = poke_region

    def attach_devices(self):
        ppu_dev = JournaledPpu(self.ppu, self.journal, self)
        # Journaled writes never pause: PPU scroll/address/VRAM/OAM traffic
        # and pulse/triangle/noise registers are order-only state consumed
        # at drain boundaries.  $2000 stays pausing (its NMI edge must reach
        # check_interrupt_lines at the write), as do $4010-$4017 (DMC/frame
        # IRQ acks, DMA stall, controller strobe vs. serial-read leases).
        self.bus.attach(ppu_dev, 0x2000, 0x3FFF,      # registers + mirrors
                        journal_writes=lambda a: (a & 7) != 0)
        self.bus.attach(self.dma, 0x4014, 0x4014)     # OAM DMA
        self.bus.attach(self.apu, 0x4000, 0x4013,     # channel registers
                        journal_writes=lambda a: a <= 0x400F)
        self.bus.attach(self.apu, 0x4015, 0x4015)     # status / enables
        self.bus.attach(self.controller, 0x4016, 0x4016)  # controller 1 (port 1)
        self.bus.attach(ApuFrameCounter(self.apu), 0x4017, 0x4017)  # read: open bus
        if self.mapper.banked:
            mapper_dev = JournaledMapper(self.mapper, self.journal, self)
            for lo, hi in getattr(self.mapper, "register_ranges", ()):
                self.bus.attach(mapper_dev, lo, hi)
            # Registers live in ROM space; reads there must stay native.
            self.bus.attach(mapper_dev, 0x8000, 0xFFFF, writes_only=True)
            self.mapper.poke = self.poke_cpu
            self.mapper.poke_region = self.map_region
            self.mapper.on_mirroring = self.set_mirroring
            if hasattr(self.mapper, "mmc2_banks"):
                self.mapper.set_mmc2_banks = self.ppu.set_mmc2_banks
                self.ppu.set_mmc2_banks(self.mapper.mmc2_banks())
            elif hasattr(self.mapper, "mmc4_banks"):
                self.mapper.set_mmc4_banks = self.ppu.set_mmc4_banks
                self.ppu.set_mmc4_banks(self.mapper.mmc4_banks())
            elif hasattr(self.mapper, "set_mmc5_state"):
                self.mapper.set_mmc5_state = self.ppu.set_mmc5_state
                self.mapper.set_mmc5_exram = self.ppu.set_mmc5_exram
                self.mapper.set_mmc5_chr = self.ppu.set_mmc5_chr
                self.ppu.set_mmc5_state(
                    self.mapper.exram_mode, self.mapper.nametable_map,
                    self.mapper.fill_tile, self.mapper.fill_color,
                    self.mapper.chr_upper,
                )
                for i, value in enumerate(self.mapper.exram):
                    if value:
                        self.ppu.set_mmc5_exram(i, value)
                self.mapper.sync_mmc5_chr()
            elif getattr(self.mapper, "chr_rom", False):
                self.mapper.set_chr_banks = self.ppu.set_chr_banks
                self.ppu.set_chr_banks(self.mapper.chr_banks())
            if hasattr(self.mapper, "on_irq_ack"):
                self.mapper.on_irq_ack = self.clear_irq
            # MMC1/MMC3-class PRG banking runs as native console silicon
            # when the backend has the run loop; register writes reach the
            # Python mapper as journal drains (apply_native).
            if self.uses_kernel_bus and hasattr(self.mapper, "native_attach"):
                self.mapper.native_attach(self.rt)

    def set_mirroring(self, mode: int) -> None:
        self.ppu.mirroring = mode

    def snapshot(self) -> tuple[bytes, dict]:
        if self.uses_kernel_bus:
            return self.rt.dump_ram(), self.rt.get_state()
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
        ram, state = self.snapshot()
        self.boot(backend, ram, int(state["pc"]), state)
        self.dma.read_cpu = self.read_cpu
        self.attach_devices()
        self._sync_irq()
        if isinstance(self.bus, Bus):
            self.bus.cycles = self.cycles  # console clock continues

    # ── CPU driving ─────────────────────────────────────────────────

    def set_nmi(self) -> None:
        if self.uses_kernel_bus:
            self.rt.set_nmi()
        else:
            self.bus.set_nmi()

    def _sync_irq(self) -> None:
        action = "set_irq" if self.irq_sources else "clear_irq"
        if self.uses_kernel_bus:
            getattr(self.rt, action)()
        else:
            getattr(self.bus, action)()

    def set_irq(self, source: str = "mapper") -> None:
        self.irq_sources.add(source)
        self._sync_irq()

    def clear_irq(self, source: str = "mapper") -> None:
        self.irq_sources.discard(source)
        self._sync_irq()

    def check_nmi_edge(self) -> None:
        if self.ppu.take_nmi_edge():
            self.set_nmi()

    def check_interrupt_lines(self) -> None:
        """Refresh device-owned interrupt levels after an MMIO pause."""
        self.check_nmi_edge()
        if "apu" in self.irq_sources and not self.apu.irq_pending:
            self.clear_irq("apu")

    def run_until(self, target: int) -> None:
        """Run the CPU until the console clock reaches target cycles."""
        if self.halted:
            return
        if self.uses_kernel_bus:
            if target <= self.cycles:
                return
            out = self.bus.run(
                1 << 60,
                max_cycles=target - self.cycles,
                on_pause=lambda out: self.check_interrupt_lines(),
                trace=False,
            )
            self.cycles += int(out["cycles"])
            if out["reason"] in ("halt", "trap"):
                self.halted = True
        else:
            bus = self.bus
            while bus.cycles < target and not bus.halted:
                bus.step()
                self.check_interrupt_lines()
            self.cycles = bus.cycles
            if bus.halted:
                self.halted = True

    # ── Frame loop ──────────────────────────────────────────────────

    def run_frame(self):
        """Run one NTSC frame; returns the rendered 240x256 index frame."""
        base = self.frame_base

        # Visible portion, split by mid-frame events: the sprite-0 hit and
        # the mapper's scanline IRQ (which also marks the render split).
        # Each PPU status transition invalidates any $2002 read lease.
        self.journal_phase = PHASE_VISIBLE
        # The event list is dynamic: an IRQ handler may re-arm its target
        # for a SECOND IRQ later in the same frame (CV3 chains one at the
        # HUD seam and one a line below doing the real raster switch), so
        # re-poll the mapper after every delivery.  IRQ lines must strictly
        # increase within a frame; sprite-0 fires at most once.
        fired_sprite0 = False
        last_irq_line = -1
        for _ in range(8):  # safety cap on mid-frame events
            candidates = []
            if (not fired_sprite0 and self.sprite0_scanline is not None
                    and self.ppu.rendering_enabled):
                candidates.append((self.sprite0_scanline, "sprite0"))
            irq_line = self.mapper.irq_scanline()
            if (irq_line is not None and irq_line > last_irq_line
                    and self.ppu.rendering_enabled):
                candidates.append((irq_line, "irq"))
            if not candidates:
                break
            scanline, kind = sorted(candidates)[0]
            cycle = base + int((scanline + 1) * CYCLES_PER_SCANLINE)
            self.run_until(min(cycle, base + VBLANK_START_CYCLES))
            self.bus.clear_read_lease()
            self.journal_phase = PHASE_POST_EVENT
            if kind == "sprite0":
                fired_sprite0 = True
                self.journal.add(self.frame_index, PHASE_POST_EVENT,
                                 J_EVENT, scanline, 0)
                self.ppu.sprite0_hit = True
                continue
            last_irq_line = scanline
            # Re-validate the IRQ at delivery: the game may have disabled
            # rendering (the counter only clocks while rendering) or
            # disarmed the IRQ ($E000) since the frame started.
            if self.ppu.rendering_enabled and self.mapper.irq_scanline() == scanline:
                self.journal.add(self.frame_index, PHASE_POST_EVENT,
                                 J_EVENT, scanline, 1)
                self.ppu.mark_split(scanline)
                if hasattr(self.mapper, "irq_pending"):
                    self.mapper.irq_pending = True
                self.set_irq("mapper")
                # Run the handler now (a couple of scanlines is ample):
                # a re-armed same-frame target is invisible until the
                # handler's $5203 write has actually executed.
                self.run_until(min(cycle + int(2 * CYCLES_PER_SCANLINE),
                                   base + VBLANK_START_CYCLES))
        self.run_until(base + VBLANK_START_CYCLES)
        self.bus.clear_read_lease()

        frame = self.ppu.render_frame()
        self.sprite0_scanline = self.ppu.sprite0_scanline

        self.journal_phase = PHASE_VBLANK
        self.ppu.begin_vblank()
        if self.ppu.nmi_enabled:
            self.journal.add(self.frame_index, PHASE_VBLANK, J_EVENT, 241, 2)
            self.set_nmi()
        self.run_until(base + FRAME_CYCLES)
        self.bus.clear_read_lease()
        self.ppu.end_vblank()

        self.frame_base += FRAME_CYCLES
        self.frame_index += 1

        # Audio for the frame just executed (includes this frame's NMI
        # music writes). Frame and DMC IRQs become visible at this batch
        # boundary; their device flags retain the hardware read/ack rules.
        self.last_audio = self.apu.end_frame(FRAME_CYCLES, self.read_cpu)
        # DMC DMA happened during the just-synthesized batch. Charge its
        # four-cycle steals against the following CPU run; the frame clock
        # remains fixed, so the next run_until naturally executes fewer
        # CPU cycles by exactly this amount.
        self.cycles += self.apu.take_dma_stall()
        if isinstance(self.bus, Bus):
            self.bus.cycles = self.cycles
        if self.apu.irq_pending:
            self.set_irq("apu")
        else:
            self.clear_irq("apu")

        if (
            self.checkpoint_tokens
            and not self.uses_kernel_bus
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

    def save_jit_proposals(self) -> None:
        """Persist this run's hot-window proposals: the NEXT boot folds
        them into the pinned module (a mid-run fold + compile would stall
        the frame loop).  Callers invoke this once at shutdown."""
        proposals = getattr(self.rt, "jit_proposals", None)
        if not proposals:
            return
        known = {
            head: (head, code.hex(), repeat)
            for head, code, repeat in load_jit_windows(self.jit_sidecar)
        }
        new = 0
        for head, (h, code, repeat) in proposals.items():
            if head not in known and len(known) < 192:
                known[head] = (h, code.hex(), repeat)
                new += 1
        if new:
            with open(self.jit_sidecar, "w") as f:
                json.dump(sorted(known.values()), f)
            logger.info("saved %d new jit window proposals (%d total)",
                        new, len(known))


def write_ppm(path: str, frame) -> None:
    """Write a rendered frame as a binary PPM (no dependencies)."""
    rgb = to_rgb(frame)
    with open(path, "wb") as f:
        f.write(b"P6\n256 240\n255\n")
        f.write(rgb.tobytes())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Run an NES ROM on the 6502 graph machine.")
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
    parser.add_argument(
        "--profile", type=int, default=0, metavar="MIN",
        help="after the run, list fallback pcs executed >= MIN times "
             "(weights backend; shape-library growth data)",
    )
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
            args.backend, console.frame_index, console.snapshot()[1]["pc"],
        )
        console.handoff(args.backend)
    frame = console.run(args.frames - console.frame_index)
    run_elapsed = time.perf_counter() - run_start
    console.save_jit_proposals()

    fps = console.frame_index / run_elapsed if run_elapsed > 0 else 0.0
    logger.info(
        "%d frames in %.2fs (%.1f fps), %d CPU cycles, machine build %.2fs%s",
        console.frame_index, run_elapsed, fps, console.cycles, build_elapsed,
        " [HALTED]" if console.halted else "",
    )
    if args.profile:
        profile_fallbacks(console, args.profile)
    if args.out and frame is not None:
        write_ppm(args.out, frame)
        logger.info("final frame written to %s", args.out)


if __name__ == "__main__":
    main()
