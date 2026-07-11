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


def _image() -> bytes:
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


def _row_key(row):
    s = row.state
    writes = tuple(sorted((w.addr, w.value) for w in row.writes))
    return (s.pc, s.a, s.x, s.y, s.s, s.p, s.halted, writes)


def _run_reference(image: bytes):
    cpu = Cpu6502(memory=bytearray(image))
    cpu.pc = START_PC
    device = CounterDevice()
    bus = Bus(ReferenceCore(cpu), image)
    bus.attach(device, DEVICE_LO, DEVICE_HI)
    rows = bus.run(STEPS, nmi_at=set(NMI_AT))
    return rows, device, bus


def test_graph_bus_matches_reference_bus():
    image = _image()
    ref_rows, ref_dev, ref_bus = _run_reference(image)

    rt = Cpu6502Runtime()
    rt.boot_image(list(image), start_pc=START_PC)
    device = CounterDevice()
    bus = Bus(GraphCore(rt), image)
    bus.attach(device, DEVICE_LO, DEVICE_HI)
    rows = bus.run(STEPS, nmi_at=set(NMI_AT))

    assert len(ref_rows) == len(rows) == STEPS
    for i, (a, b) in enumerate(zip(ref_rows, rows, strict=True)):
        assert _row_key(a) == _row_key(b), f"step {i}: ref={a.state} graph={b.state}"
    assert ref_dev.log == device.log
    # Each scheduled NMI ran the handler exactly once.
    assert ref_bus.ram[0x20] == bus.ram[0x20] == len(NMI_AT)


@pytest.mark.slow
def test_kernel_bus_matches_reference_bus():
    from transformer_nes.cpu6502.backends.kernel import Cpu6502CppRuntime

    image = _image()
    ref_rows, ref_dev, _ref_bus = _run_reference(image)

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
