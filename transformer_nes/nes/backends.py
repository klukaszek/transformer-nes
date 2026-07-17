"""Adapt the CPU runtime backends to the native console's interface.

The console drives one core protocol (pc/x/y, interrupt lines, poke,
step); each backend gets a thin adapter, and backend_factory returns
the (runtime, is_engine) pair the Console constructor expects.
OPCODES_NATIVE is the (cycles, read-mode) table the native run loop
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
)


class ReferenceCore:
    def __init__(self, cpu):
        self.cpu = cpu

    pc = property(lambda s: s.cpu.pc)
    x = property(lambda s: s.cpu.x)
    y = property(lambda s: s.cpu.y)
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
        self.cpu.memory[addr & 0xFFFF] = value & 0xFF

    def step(self):
        return self.cpu.step()


class GraphCore:
    def __init__(self, runtime):
        self.runtime = runtime

    pc = property(lambda s: s.runtime.mstate["pc"])
    x = property(lambda s: s.runtime.mstate["x"])
    y = property(lambda s: s.runtime.mstate["y"])
    halted = property(lambda s: s.runtime.machine_halted)
    nmi_pending = property(lambda s: s.runtime.nmi_pending)
    irq_ready = property(lambda s: irq_deliverable(s.runtime.mstate["p"], s.runtime.irq_pending))
    log_tokens = property(lambda s: s.runtime.pos)

    def set_nmi(self):
        self.runtime.set_nmi()

    def set_irq(self):
        self.runtime.set_irq()

    def clear_irq(self):
        self.runtime.clear_irq()

    def poke(self, addr, value):
        self.runtime.inject_write(addr, value)

    def step(self):
        return self.runtime.machine_step()

    def checkpoint(self, _ram):
        self.runtime.checkpoint()


def backend_factory(name: str):
    def build(image: bytes, start_pc: int):
        if name == "weights":
            from transformer_nes.cpu6502.backends.weights.runner import Cpu6502TransformerMachine

            return Cpu6502TransformerMachine(bcd=False), True
        if name == "reference":
            from transformer_nes.cpu6502.backends.reference import Cpu6502

            cpu = Cpu6502(memory=bytearray(image), bcd_enabled=False)
            cpu.pc = start_pc
            return ReferenceCore(cpu), False
        if name == "graph":
            from transformer_nes.cpu6502.backends.graph import Cpu6502Runtime

            runtime = Cpu6502Runtime(bcd=False)
            runtime.boot_image(list(image), start_pc=start_pc)
            return GraphCore(runtime), False
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
OPCODES_NATIVE = [(0, -1)] * 256
for byte, op in OPCODES.items():
    mode = MODE_CODES[op.mode] if op.operation in READS_MEMORY and op.mode in MEMORY_MODES else -1
    OPCODES_NATIVE[byte] = (op.cycles, mode)
