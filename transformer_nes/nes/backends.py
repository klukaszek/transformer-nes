"""Adapt the CPU runtime backends to the native console's interface.

The console drives one core protocol (pc/x/y, interrupt lines, poke,
step); each backend gets a thin adapter, and backend_factory returns
the (runtime, is_engine) pair the Console constructor expects.
OPCODES_NATIVE is the (cycles, read-mode, dynamic-kind) table the native run loop
uses for the stepped backends.
"""

from __future__ import annotations

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
    dynamic_cycle_table,
)


class ReferenceCore:
    def __init__(self, cpu):
        self.cpu = cpu

    pc = property(lambda s: s.cpu.pc)
    x = property(lambda s: s.cpu.x)
    y = property(lambda s: s.cpu.y)
    p = property(lambda s: s.cpu.p)
    halted = property(lambda s: s.cpu.halted)
    nmi_pending = property(lambda s: s.cpu.nmi_pending)
    irq_ready = property(lambda s: irq_deliverable(s.cpu.p, s.cpu.irq_pending))

    def set_nmi(self):
        self.cpu.nmi_pending = True

    def set_irq(self):
        self.cpu.irq_pending = True

    def clear_irq(self):
        self.cpu.irq_pending = False

    def poke(self, addr, value):
        self.cpu.write_memory(addr, value)

    def step(self):
        return self.cpu.step()


def backend_factory(name: str):
    def build(image: bytes, start_pc: int):
        if name == "weights":
            from transformer_nes.cpu6502.backends.weights.runner import Cpu6502TransformerMachine

            return Cpu6502TransformerMachine(bcd=False, mirror_cpu_ram=True), True
        if name == "reference":
            from transformer_nes.cpu6502.backends.reference import Cpu6502

            cpu = Cpu6502(
                memory=bytearray(image), bcd_enabled=False, mirror_cpu_ram=True,
            )
            cpu.pc = start_pc
            return ReferenceCore(cpu), False
        raise ValueError(f"unknown backend {name!r}")

    return build


MODE_CODES = {
    MODE_ZERO_PAGE: 0,
    MODE_ZERO_PAGE_X: 1,
    MODE_ZERO_PAGE_Y: 2,
    MODE_ABSOLUTE: 3,
    MODE_ABSOLUTE_X: 4,
    MODE_ABSOLUTE_Y: 5,
    MODE_INDIRECT_X: 6,
    MODE_INDIRECT_Y: 7,
}
_DYNAMIC_CYCLES = dynamic_cycle_table()
OPCODES_NATIVE = [(2, -1, 0)] * 256
for byte, op in OPCODES.items():
    mode = MODE_CODES[op.mode] if op.operation in READS_MEMORY and op.mode in MEMORY_MODES else -1
    OPCODES_NATIVE[byte] = (op.cycles, mode, _DYNAMIC_CYCLES[byte])
