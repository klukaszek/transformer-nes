"""Adapt the CPU runtime backends to the native console's interface.

The console drives one core protocol (interrupt lines, poke, peek,
step); each backend gets a thin adapter, and backend_factory returns the
(runtime, is_engine) pair the Console constructor expects.
"""

from __future__ import annotations

from transformer_nes.cpu6502.spec import MEMORY_MODES, READS_MEMORY, opcode_for


class ReferenceCore:
    def __init__(self, cpu):
        self.cpu = cpu

    halted = property(lambda s: s.cpu.halted)

    def set_nmi(self):
        self.cpu.nmi_pending = True

    def set_irq(self):
        self.cpu.irq_pending = True

    def clear_irq(self):
        self.cpu.irq_pending = False

    def poke(self, addr, value):
        self.cpu.write_memory(addr, value)

    def peek(self) -> tuple[int, int]:
        """The next step's cycles and the device read it issues (-1:
        none): an interrupt sequence reads nothing a device answers."""
        from transformer_nes.cpu6502.backends.reference import instruction_cycles, irq_deliverable

        cpu = self.cpu
        if cpu.nmi_pending or irq_deliverable(cpu.p, cpu.irq_pending):
            return 7, -1
        op = opcode_for(cpu.read_memory(cpu.pc))
        cycles = instruction_cycles(cpu, op, cpu.pc, cpu.x, cpu.y, cpu.p)
        if op.operation not in READS_MEMORY or op.mode not in MEMORY_MODES:
            return cycles, -1
        lo, hi, _ = cpu.read_operand(op)
        return cycles, cpu.resolve_ea(op, lo, hi)

    def step(self):
        return self.cpu.step()


def backend_factory(name: str):
    def build(image: bytes, start_pc: int):
        if name == "weights":
            from transformer_nes.cpu6502.backends.weights.runner import Cpu6502TransformerMachine

            machine = Cpu6502TransformerMachine(
                bcd=False,
                mirror_cpu_ram=True,
            )
            return machine, True
        if name == "reference":
            from transformer_nes.cpu6502.backends.reference import Cpu6502

            cpu = Cpu6502(
                memory=bytearray(image),
                bcd_enabled=False,
                mirror_cpu_ram=True,
            )
            cpu.pc = start_pc
            return ReferenceCore(cpu), False
        raise ValueError(f"unknown backend {name!r}")

    return build
