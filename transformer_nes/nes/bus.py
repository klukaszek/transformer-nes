"""MMIO bus: memory-mapped devices and NMI for the 6502 machine.

Devices (PPU, APU, controllers) live *outside* the computation graph, but
they talk to the CPU over the model's own bus, exactly like real hardware:

- CPU -> device: every store already surfaces as a WRITE token; the bus
  watches the device's address range and forwards the value.
- Device -> CPU: a read of a device register must see the device's current
  value, not a stale byte in the memory log.  The bus shadow-decodes the
  next instruction from a RAM mirror; if it reads a device address, the
  device's answer is placed on the bus as a WRITE token *before* the
  instruction executes.  The CPU then reads it through the same attention
  lookup as any other byte -- a real 6502 also just reads the bus.
- NMI: the bus sets the nmi flag on the next STATE token; the interrupt
  sequence itself (stack pushes, I flag, $FFFA vector) is graph logic in
  interpreter.py, not harness code.

This module drives the Python-stepped cores (the reference oracle and the
graph evaluator) one instruction at a time.  The C++ kernel implements the
same protocol natively for speed; kernel.py's machine API pauses back into
Python whenever a watched address is touched, and KernelBus services those
pauses with the same Device objects.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from transformer_nes.cpu6502.backends.reference import irq_deliverable
from transformer_nes.cpu6502.spec import (
    MEMORY_MODES,
    MODE_ABSOLUTE,
    MODE_ABSOLUTE_X,
    MODE_ABSOLUTE_Y,
    MODE_INDIRECT_X,
    MODE_INDIRECT_Y,
    MODE_ZERO_PAGE,
    MODE_ZERO_PAGE_X,
    MODE_ZERO_PAGE_Y,
    OPCODES,
    READS_MEMORY,
)
from transformer_nes.cpu6502.trace import TraceRow


class Device(Protocol):
    """A memory-mapped device: answers reads, observes writes.

    A device may additionally define ``take_stall() -> int`` returning CPU
    cycles its last access stalled the CPU for (e.g. OAM DMA's 513); the
    bus drains it into its cycle counter right after forwarding the access.
    """

    def read(self, addr: int) -> int: ...
    def write(self, addr: int, value: int) -> None: ...


def drain_stall(device) -> int:
    take = getattr(device, "take_stall", None)
    return take() if take is not None else 0


def read_target(ram, pc: int, x: int, y: int) -> int | None:
    """Shadow-decode the instruction at pc; return the address its execution
    will read, or None if it does not read memory (or is an unknown byte).

    Pointer bytes for the indirect modes are fetched from the RAM mirror;
    pointers never live in device registers, so this is exact.
    """
    op = OPCODES.get(ram[pc])
    if op is None or op.operation not in READS_MEMORY or op.mode not in MEMORY_MODES:
        return None
    lo = ram[(pc + 1) & 0xFFFF]
    mode = op.mode
    if mode == MODE_ZERO_PAGE:
        return lo
    if mode == MODE_ZERO_PAGE_X:
        return (lo + x) & 0xFF
    if mode == MODE_ZERO_PAGE_Y:
        return (lo + y) & 0xFF
    hi = ram[(pc + 2) & 0xFFFF]
    if mode == MODE_ABSOLUTE:
        return lo | (hi << 8)
    if mode == MODE_ABSOLUTE_X:
        return ((lo | (hi << 8)) + x) & 0xFFFF
    if mode == MODE_ABSOLUTE_Y:
        return ((lo | (hi << 8)) + y) & 0xFFFF
    if mode == MODE_INDIRECT_X:
        zp = (lo + x) & 0xFF
        return ram[zp] | (ram[(zp + 1) & 0xFF] << 8)
    if mode == MODE_INDIRECT_Y:
        base = ram[lo] | (ram[(lo + 1) & 0xFF] << 8)
        return (base + y) & 0xFFFF
    return None


# ── Core adapters ───────────────────────────────────────────────────
# The bus needs five things from a core: the registers that feed shadow
# decode, the pending-NMI flag, a way to request an NMI, a way to place a
# device value on the bus, and a step.


class ReferenceCore:
    """Bus adapter for the plain-Python oracle (reference.Cpu6502)."""

    def __init__(self, cpu):
        self.cpu = cpu

    @property
    def pc(self) -> int:
        return self.cpu.pc

    @property
    def x(self) -> int:
        return self.cpu.x

    @property
    def y(self) -> int:
        return self.cpu.y

    @property
    def halted(self) -> bool:
        return self.cpu.halted

    @property
    def nmi_pending(self) -> bool:
        return self.cpu.nmi_pending

    @property
    def irq_ready(self) -> bool:
        return irq_deliverable(self.cpu.p, self.cpu.irq_pending)

    def set_nmi(self) -> None:
        self.cpu.nmi_pending = True

    def set_irq(self) -> None:
        self.cpu.irq_pending = True

    def clear_irq(self) -> None:
        self.cpu.irq_pending = False

    def poke(self, addr: int, value: int) -> None:
        self.cpu.memory[addr & 0xFFFF] = value & 0xFF

    def step(self) -> TraceRow:
        return self.cpu.step()


class GraphCore:
    """Bus adapter for the graph evaluator (evaluator.Cpu6502Runtime).

    The runtime must already be booted via boot_image().
    """

    def __init__(self, rt):
        self.rt = rt

    @property
    def pc(self) -> int:
        return self.rt.mstate["pc"]

    @property
    def x(self) -> int:
        return self.rt.mstate["x"]

    @property
    def y(self) -> int:
        return self.rt.mstate["y"]

    @property
    def halted(self) -> bool:
        return self.rt.machine_halted

    @property
    def nmi_pending(self) -> bool:
        return self.rt.nmi_pending

    @property
    def irq_ready(self) -> bool:
        return irq_deliverable(self.rt.mstate["p"], self.rt.irq_pending)

    def set_nmi(self) -> None:
        self.rt.set_nmi()

    def set_irq(self) -> None:
        self.rt.set_irq()

    def clear_irq(self) -> None:
        self.rt.clear_irq()

    def poke(self, addr: int, value: int) -> None:
        self.rt.inject_write(addr, value)

    def step(self) -> TraceRow:
        return self.rt.machine_step()

    @property
    def log_tokens(self) -> int:
        return self.rt.pos

    def checkpoint(self, ram: bytearray) -> None:
        """Reseed the token log from a RAM snapshot; resume mid-program."""
        rebuild = getattr(self.rt, "checkpoint", None)
        if rebuild is not None:
            # The runtime's own snapshot-and-rebuild also keeps the trace
            # numbering and any pending NMI (its RAM mirror is kept in sync
            # with ours through inject_write).
            rebuild()
            return
        state = self.rt.machine_state
        self.rt.boot_image(list(ram), start_pc=state["pc"], state=state)


# ── The bus ─────────────────────────────────────────────────────────


class Bus:
    """Instruction-stepped MMIO bus over a Python core adapter."""

    def __init__(self, core, memory_image):
        self.core = core
        # RAM mirror for shadow decode; kept in sync with every CPU write
        # and every device injection.
        self.ram = bytearray(memory_image)
        if len(self.ram) != 0x10000:
            raise ValueError("memory_image must be exactly 64 KiB")
        self.ranges: list[tuple[int, int, Device, bool]] = []
        # Base CPU cycles executed (no penalty cycles; NMI = 7 cycles;
        # device stalls included).  Matches the C++ kernel's accounting.
        self.cycles = 0

    def attach(self, device: Device, lo: int, hi: int, *,
               writes_only: bool = False) -> None:
        """Map a device over [lo, hi] (inclusive).  writes_only devices
        (mappers: registers in ROM space) never intercept reads."""
        self.ranges.append((lo, hi, device, writes_only))

    def set_nmi(self) -> None:
        self.core.set_nmi()

    def set_irq(self) -> None:
        self.core.set_irq()

    def clear_irq(self) -> None:
        self.core.clear_irq()

    @property
    def halted(self) -> bool:
        return self.core.halted

    @property
    def log_tokens(self) -> int:
        """Token-log length of the underlying core (0 = no append-only log)."""
        return getattr(self.core, "log_tokens", 0)

    def checkpoint(self) -> bool:
        """Snapshot-and-rebuild: reseed the core's append-only token log
        from the RAM mirror, keeping the architectural state.

        Semantically invisible: unwritten and zero bytes both read as 0
        through the exact-address mask, so dropping the history and
        reseeding only the nonzero RAM bytes preserves every future read.
        No-op for cores without a token log (the reference oracle).
        """
        rebuild = getattr(self.core, "checkpoint", None)
        if rebuild is None:
            return False
        rebuild(self.ram)
        return True

    def device_at(self, addr: int, *, for_read: bool = False) -> Device | None:
        for lo, hi, device, writes_only in self.ranges:
            if lo <= addr <= hi and not (for_read and writes_only):
                return device
        return None

    def step(self) -> TraceRow:
        """Execute one instruction (or pending interrupt) with device servicing."""
        # Device -> CPU: pre-inject the register value before a device read.
        # A pending interrupt preempts the instruction, so no read happens.
        if self.core.nmi_pending or self.core.irq_ready:
            self.cycles += 7
        else:
            op = OPCODES.get(self.ram[self.core.pc])
            if op is not None:
                self.cycles += op.cycles
            target = read_target(self.ram, self.core.pc, self.core.x, self.core.y)
            if target is not None:
                device = self.device_at(target, for_read=True)
                if device is not None:
                    value = device.read(target) & 0xFF
                    self.core.poke(target, value)
                    self.ram[target] = value
                    self.cycles += drain_stall(device)

        row = self.core.step()

        # CPU -> device: forward watched writes, keep the mirror in sync.
        for w in row.writes:
            self.ram[w.addr] = w.value
            device = self.device_at(w.addr)
            if device is not None:
                device.write(w.addr, w.value)
                self.cycles += drain_stall(device)
        return row

    def run(self, max_steps: int, *, nmi_at: set[int] | None = None,
            irq_at: set[int] | None = None) -> list[TraceRow]:
        """Step up to max_steps, requesting an NMI/IRQ before each step index
        in nmi_at/irq_at (absolute instruction counts from this call)."""
        rows: list[TraceRow] = []
        for i in range(max_steps):
            if nmi_at and i in nmi_at:
                self.set_nmi()
            if irq_at and i in irq_at:
                self.set_irq()
            if self.halted:
                break
            rows.append(self.step())
        return rows

    def clear_read_lease(self) -> None:
        pass  # per-step Python bus reads devices directly; nothing leased


class KernelBus:
    """MMIO bus over the C++ kernel machine (kernel.Cpu6502CppRuntime).

    The kernel runs the instruction stream natively and pauses back into
    Python only when a watched address is touched; this class services
    those pauses with the same Device objects Bus uses, so a KernelBus run
    is directly comparable to a Bus run.
    """

    def __init__(self, rt, memory_image, *, start_pc: int):
        self.rt = rt
        # Queue-leased devices by address (controller serial reads):
        # flushed with the engine's served-read counts before any device
        # is next consulted.
        self.lease_devices: dict[int, Device] = {}
        rt.load_image(memory_image, start_pc=start_pc)
        self.ranges: list[tuple[int, int, Device, bool]] = []

    def attach(self, device: Device, lo: int, hi: int, *,
               writes_only: bool = False) -> None:
        """Map a device over [lo, hi].  writes_only devices (mappers:
        registers in ROM space) never watch reads -- ROM fetches and
        table reads stay native."""
        self.ranges.append((lo, hi, device, writes_only))
        if not writes_only:
            self.rt.watch_read(lo, hi)
        self.rt.watch_write(lo, hi)

    def device_at(self, addr: int) -> Device | None:
        for lo, hi, device, _writes_only in self.ranges:
            if lo <= addr <= hi:
                return device
        return None

    def flush_lease(self) -> None:
        """Advance queue-leased devices by the reads the engine served
        (tracking survives the flush; entries drop on clear/eviction)."""
        for addr, device in self.lease_devices.items():
            n = self.rt.take_lease_served(addr)
            if n:
                device.lease_consumed(addr, n)

    def clear_read_lease(self) -> None:
        self.flush_lease()
        self.lease_devices.clear()
        self.rt.clear_read_lease()

    @property
    def log_tokens(self) -> int:
        return int(self.rt.get_state()["pos"])

    def checkpoint(self) -> bool:
        """Snapshot-and-rebuild the kernel's token log from its RAM mirror.

        Captures the architectural state, reloads the RAM snapshot as a
        fresh (short) WRITE-token prefix, restores the state, and
        re-registers the watch ranges that load_image cleared.
        """
        state = self.rt.get_state()
        if state["halted"]:
            return False
        ram = self.rt.dump_ram()
        self.rt.load_image(ram, start_pc=state["pc"])
        self.rt.set_state(state)
        for lo, hi, _device, writes_only in self.ranges:
            if not writes_only:
                self.rt.watch_read(lo, hi)
            self.rt.watch_write(lo, hi)
        return True

    def run(self, max_steps: int, *, nmi_at: set[int] | None = None,
            irq_at: set[int] | None = None,
            stop_on_trap: bool = False, max_cycles: int = 0,
            on_pause=None, trace: bool = True) -> dict:
        """Run up to max_steps instructions, servicing device pauses.

        nmi_at/irq_at use the same convention as Bus.run: the interrupt is
        requested before executing step index i (absolute count from this
        call).  max_cycles > 0 also stops once that many base CPU cycles
        (device stalls included) have elapsed in this call.  on_pause, if
        given, is called after every serviced pause -- the hook the console
        uses to check device-raised NMI edges before execution resumes.
        Returns the merged kernel result arrays plus the final "reason"
        and "cycles" (elapsed this call, stalls included).  trace=False
        skips the per-step arrays entirely (the console's gameplay path
        reads only "reason"/"cycles").
        """
        nmi_list = sorted(nmi_at) if nmi_at else []
        irq_list = sorted(irq_at) if irq_at else []
        outs: list[dict] = []
        executed = 0
        cycles_done = 0
        reason = "budget"
        while executed < max_steps and (max_cycles <= 0 or cycles_done < max_cycles):
            while nmi_list and nmi_list[0] <= executed:
                self.rt.set_nmi()
                nmi_list.pop(0)
            while irq_list and irq_list[0] <= executed:
                self.rt.set_irq()
                irq_list.pop(0)
            limit = max_steps - executed
            if nmi_list:
                limit = min(limit, nmi_list[0] - executed)
            if irq_list:
                limit = min(limit, irq_list[0] - executed)
            cycle_limit = max_cycles - cycles_done if max_cycles > 0 else 0
            out = self.rt.run(limit, stop_on_trap=stop_on_trap, max_cycles=cycle_limit,
                              trace=trace)
            outs.append(out)
            executed += int(out["steps"])
            cycles_done += int(out["cycles"])
            # CPU -> device: watched writes from the final step of this run.
            if len(out["hit_addr"]):
                # A device write may change what a leased register reads.
                self.clear_read_lease()
            for addr, value in zip(out["hit_addr"], out["hit_value"], strict=True):
                device = self.device_at(int(addr))
                if device is not None:
                    device.write(int(addr), int(value))
                    cycles_done += drain_stall(device)
            reason = out["reason"]
            if len(outs) > 100_000:
                st = self.rt.get_state()
                raise RuntimeError(
                    f"bus runaway: {len(outs)} pauses in one run "
                    f"(executed={executed}, cycles={cycles_done}/{max_cycles}, "
                    f"reason={reason!r}, read_addr={int(out['read_addr'])}, "
                    f"pc={st['pc']:04X}, p={st['p']:02X})"
                )
            if reason == "watch_read":
                # Device -> CPU: place the register value on the bus, resume.
                # Any queue-leased device must catch up with the engine's
                # served reads before its read() runs.
                self.flush_lease()
                addr = int(out["read_addr"])
                device = self.device_at(addr)
                if device is not None:
                    value = device.read(addr) & 0xFF
                    cycles_done += drain_stall(device)
                    # Registers whose value is frozen for the rest of the
                    # span (int) or whose next values are already determined
                    # (bytes: serial reports) are served natively until the
                    # lease is cleared or the queue drains.
                    lease = getattr(device, "read_lease", None)
                    if lease is not None:
                        leased = lease(addr)
                        if isinstance(leased, (bytes, bytearray)):
                            evicted = self.rt.set_read_lease_seq(addr, bytes(leased))
                            self.lease_devices.pop(int(evicted), None)
                            self.lease_devices[addr] = device
                        elif leased is not None:
                            evicted = self.rt.set_read_lease(addr, leased)
                            self.lease_devices.pop(int(evicted), None)
                else:
                    value = self.rt.read_ram(addr)
                self.rt.inject_write(addr, value)
            if on_pause is not None:
                on_pause(out)
            if reason in ("halt", "trap"):
                break

        merged = {"reason": reason, "cycles": cycles_done, "steps": executed}
        if trace:
            for key in ("pc", "a", "x", "y", "s", "p", "ir", "oplo", "ophi",
                        "ea", "data", "halted", "w_row", "w_addr", "w_value"):
                merged[key] = np.concatenate([out[key] for out in outs]) if outs else np.array([])
        return merged
