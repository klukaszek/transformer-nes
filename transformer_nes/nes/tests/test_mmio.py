"""Verify the MMIO bus and the in-graph NMI across all three backends.

One scenario, three runs: a program that reads and writes a memory-mapped
device (absolute and (zp),Y addressing) while NMIs fire at fixed
instruction indices (including one that lands inside the handler).  The
reference-core bus, the graph-evaluator bus, and the C++ kernel bus must
produce identical traces, identical memory writes, and an identical
device event log.
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from transformer_nes.cpu6502.backends.evaluator import Cpu6502Runtime
from transformer_nes.cpu6502.backends.reference import Cpu6502
from transformer_nes.nes.bus import Bus, GraphCore, KernelBus, ReferenceCore

DEVICE_LO, DEVICE_HI = 0xD000, 0xD00F
START_PC = 0x8000
STEPS = 120
NMI_AT = {5, 17, 18, 40}  # 18 fires on the first handler instruction (nested)


class CounterDevice:
    """Deterministic test device: reads count upward, all events are logged."""

    def __init__(self):
        self.counter = 0
        self.log: list[tuple[str, int, int]] = []

    def read(self, addr: int) -> int:
        self.counter = (self.counter + 7) & 0xFF
        self.log.append(("r", addr, self.counter))
        return self.counter

    def write(self, addr: int, value: int) -> None:
        self.log.append(("w", addr, value))


def make_image() -> bytes:
    image = bytearray(0x10000)
    main = bytes([
        0xA2, 0x00,              # LDX #$00
        # loop ($8002):
        0xAD, 0x00, 0xD0,        # LDA $D000     ; device read, absolute
        0x9D, 0x00, 0x02,        # STA $0200,X   ; plain RAM write
        0x8D, 0x01, 0xD0,        # STA $D001     ; device write
        0xA9, 0x04, 0x85, 0x10,  # LDA #$04 ; STA $10   ; pointer -> $D004
        0xA9, 0xD0, 0x85, 0x11,  # LDA #$D0 ; STA $11
        0xA0, 0x00,              # LDY #$00
        0xB1, 0x10,              # LDA ($10),Y   ; device read, (zp),Y
        0x8D, 0x01, 0xD0,        # STA $D001     ; device write
        0xE8,                    # INX
        0x4C, 0x02, 0x80,        # JMP loop
    ])
    image[START_PC:START_PC + len(main)] = main
    handler = bytes([
        0xAD, 0x02, 0xD0,        # LDA $D002     ; device read inside handler
        0x8D, 0x03, 0xD0,        # STA $D003     ; device write inside handler
        0xE6, 0x20,              # INC $20       ; NMI counter in RAM
        0x40,                    # RTI
    ])
    image[0x9000:0x9000 + len(handler)] = handler
    image[0xFFFA] = 0x00
    image[0xFFFB] = 0x90
    return bytes(image)


def row_key(row):
    s = row.state
    writes = tuple(sorted((w.addr, w.value) for w in row.writes))
    return (s.pc, s.a, s.x, s.y, s.s, s.p, s.halted, writes)


def assert_registers_match(out, ref_rows):
    """A native run's per-step registers match the reference trace."""
    assert len(out["pc"]) == len(ref_rows)
    for i, ref_row in enumerate(ref_rows):
        s = ref_row.state
        got = (int(out["pc"][i]), int(out["a"][i]), int(out["x"][i]),
               int(out["y"][i]), int(out["s"][i]), int(out["p"][i]))
        assert (s.pc, s.a, s.x, s.y, s.s, s.p) == got, f"step {i}"


def run_reference(image: bytes):
    cpu = Cpu6502(memory=bytearray(image))
    cpu.pc = START_PC
    device = CounterDevice()
    bus = Bus(ReferenceCore(cpu), image)
    bus.attach(device, DEVICE_LO, DEVICE_HI)
    rows = bus.run(STEPS, nmi_at=set(NMI_AT))
    return rows, device, bus


def test_graph_bus_matches_reference_bus():
    image = make_image()
    ref_rows, ref_dev, ref_bus = run_reference(image)

    rt = Cpu6502Runtime()
    rt.boot_image(list(image), start_pc=START_PC)
    device = CounterDevice()
    bus = Bus(GraphCore(rt), image)
    bus.attach(device, DEVICE_LO, DEVICE_HI)
    rows = bus.run(STEPS, nmi_at=set(NMI_AT))

    assert len(ref_rows) == len(rows) == STEPS
    for i, (a, b) in enumerate(zip(ref_rows, rows, strict=True)):
        assert row_key(a) == row_key(b), f"step {i}: ref={a.state} graph={b.state}"
    assert ref_dev.log == device.log
    # Each scheduled NMI ran the handler exactly once.
    assert ref_bus.ram[0x20] == bus.ram[0x20] == len(NMI_AT)


IRQ_START = 0x8000
IRQ_STEPS = 80
# 1 fires inside the SEI-protected prologue (delivery deferred until the
# CLI lands); 30 and 55 fire in the unprotected loop.
IRQ_AT = {1, 30, 55}


def make_irq_image() -> bytes:
    image = bytearray(0x10000)
    main = bytes([
        0x78,                    # SEI
        0xA9, 0x00, 0x85, 0x30,  # LDA #$00 ; STA $30  ; IRQ counter
        0xEA, 0xEA,              # NOP ; NOP           ; deferral window
        0x58,                    # CLI
        # loop ($8008):
        0xE6, 0x31,              # INC $31
        0x4C, 0x08, 0x80,        # JMP loop
    ])
    image[IRQ_START:IRQ_START + len(main)] = main
    handler = bytes([
        0xE6, 0x30,              # INC $30
        0x40,                    # RTI
    ])
    image[0xA000:0xA000 + len(handler)] = handler
    image[0xFFFE] = 0x00
    image[0xFFFF] = 0xA0
    return bytes(image)


def run_irq_reference(image: bytes):
    cpu = Cpu6502(memory=bytearray(image))
    cpu.pc = IRQ_START
    bus = Bus(ReferenceCore(cpu), image)
    rows = bus.run(IRQ_STEPS, irq_at=set(IRQ_AT))
    return rows, bus


def test_irq_graph_bus_matches_reference_bus():
    image = make_irq_image()
    ref_rows, ref_bus = run_irq_reference(image)
    # The SEI-window IRQ was deferred, not lost: all three delivered.
    assert ref_bus.ram[0x30] == len(IRQ_AT)

    rt = Cpu6502Runtime()
    rt.boot_image(list(image), start_pc=IRQ_START)
    bus = Bus(GraphCore(rt), image)
    rows = bus.run(IRQ_STEPS, irq_at=set(IRQ_AT))

    assert len(ref_rows) == len(rows) == IRQ_STEPS
    for i, (a, b) in enumerate(zip(ref_rows, rows, strict=True)):
        assert row_key(a) == row_key(b), f"step {i}: ref={a.state} graph={b.state}"
    assert bus.ram[0x30] == len(IRQ_AT)


@pytest.mark.slow
def test_irq_kernel_bus_matches_reference_bus():
    from transformer_nes.cpu6502.backends.kernel import Cpu6502CppRuntime

    image = make_irq_image()
    ref_rows, _ref_bus = run_irq_reference(image)

    rt = Cpu6502CppRuntime(bcd=True)
    bus = KernelBus(rt, image, start_pc=IRQ_START)
    out = bus.run(IRQ_STEPS, irq_at=set(IRQ_AT))

    assert_registers_match(out, ref_rows)
    assert rt.read_ram(0x30) == len(IRQ_AT)


@pytest.mark.slow
def test_irq_weights_bus_matches_reference_bus():
    from transformer_nes.cpu6502.backends.weights.runner import (
        Cpu6502TransformerMachine,
    )

    image = make_irq_image()
    ref_rows, _ref_bus = run_irq_reference(image)

    machine = Cpu6502TransformerMachine(bcd=False)
    bus = KernelBus(machine, image, start_pc=IRQ_START)
    out = bus.run(IRQ_STEPS, irq_at=set(IRQ_AT))

    assert_registers_match(out, ref_rows)
    assert machine.read_ram(0x30) == len(IRQ_AT)


@pytest.mark.slow
def test_kernel_bus_matches_reference_bus():
    from transformer_nes.cpu6502.backends.kernel import Cpu6502CppRuntime

    image = make_image()
    ref_rows, ref_dev, ref_bus = run_reference(image)

    rt = Cpu6502CppRuntime(bcd=True)
    device = CounterDevice()
    bus = KernelBus(rt, image, start_pc=START_PC)
    bus.attach(device, DEVICE_LO, DEVICE_HI)
    out = bus.run(STEPS, nmi_at=set(NMI_AT))

    assert len(out["pc"]) == STEPS
    writes_by_row = defaultdict(set)
    for row, addr, value in zip(out["w_row"], out["w_addr"], out["w_value"], strict=True):
        writes_by_row[int(row)].add((int(addr), int(value)))
    for i, ref_row in enumerate(ref_rows):
        s = ref_row.state
        got = (int(out["pc"][i]), int(out["a"][i]), int(out["x"][i]),
               int(out["y"][i]), int(out["s"][i]), int(out["p"][i]))
        assert (s.pc, s.a, s.x, s.y, s.s, s.p) == got, f"step {i}"
        assert {(w.addr, w.value) for w in ref_row.writes} == writes_by_row[i], f"step {i}"
    assert ref_dev.log == device.log
    assert rt.read_ram(0x20) == len(NMI_AT)
