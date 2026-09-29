"""Boring Python 6502 reference core.

No transformer, no graph -- just a trusted oracle that executes the full
official NMOS 6502 opcode set using the shared spec table.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

from transformer_nes.cpu6502.spec import (
    BRANCH_CONDITIONS,
    FLAG_B,
    FLAG_C,
    FLAG_D,
    FLAG_I,
    FLAG_MASKS,
    FLAG_N,
    FLAG_U,
    FLAG_V,
    FLAG_Z,
    MODE_ABSOLUTE,
    MODE_ABSOLUTE_X,
    MODE_ABSOLUTE_Y,
    MODE_ACCUMULATOR,
    MODE_IMMEDIATE,
    MODE_IMPLIED,
    MODE_INDIRECT,
    MODE_INDIRECT_X,
    MODE_INDIRECT_Y,
    MODE_RELATIVE,
    MODE_ZERO_PAGE,
    MODE_ZERO_PAGE_X,
    MODE_ZERO_PAGE_Y,
    OPCODES,
    PAGE_CROSS_READ_OPERATIONS,
    Opcode,
    opcode_for,
    written_addresses,
)
from transformer_nes.cpu6502.trace import Cpu6502State, MemWrite, Trace, TraceRow

# What an operation handler returns: (the trace row's data value, the next
# PC, the memory writes).
Result = tuple[int, int, tuple[MemWrite, ...]]


# The four shifts as (value, carry in) -> (result, bit shifted out).
def ASL(v: int, c: int) -> tuple[int, int]:
    return (v << 1) & 0xFF, v & 0x80


def LSR(v: int, c: int) -> tuple[int, int]:
    return v >> 1, v & 0x01


def ROL(v: int, c: int) -> tuple[int, int]:
    return ((v << 1) | (1 if c else 0)) & 0xFF, v & 0x80


def ROR(v: int, c: int) -> tuple[int, int]:
    return (v >> 1) | (0x80 if c else 0), v & 0x01


def irq_deliverable(p: int, irq_pending: bool | int,
                    nmi_pending: bool | int = False) -> bool:
    """An IRQ is taken at a step boundary only with the I flag clear and no
    NMI ahead of it (NMI wins).  The canonical Python gate; the generated
    C++ engines carry the identical test as their own irq_ready()."""
    return bool(irq_pending and not nmi_pending and not (p & FLAG_I))


def instruction_cycles(cpu: Cpu6502, opcode: Opcode, pc: int,
                       x: int, y: int, p: int) -> int:
    """One instruction's NMOS cycle count: the opcode's, +1 for a taken
    branch and +1 more when it lands on another page, +1 when an indexed
    read carries into the high address byte."""
    cycles = opcode.cycles
    lo = cpu.read_memory(pc + 1)
    if opcode.operation in BRANCH_CONDITIONS:
        mask, expected = BRANCH_CONDITIONS[opcode.operation]
        if (p & mask) == expected:
            landing = (pc + 2) % 0x100 + lo - (0x100 if lo >= 0x80 else 0)
            cycles += 1 + (not 0 <= landing <= 0xFF)
    elif opcode.operation in PAGE_CROSS_READ_OPERATIONS:
        low = {
            MODE_ABSOLUTE_X: lo + x,
            MODE_ABSOLUTE_Y: lo + y,
            MODE_INDIRECT_Y: cpu.read_memory(lo) + y,
        }.get(opcode.mode, 0)
        cycles += low > 0xFF
    return cycles


@dataclass
class Cpu6502:
    """Reference 6502 CPU.

    Memory is modeled as a dense 64 KiB array.  This is intentionally simple:
    the weights compiler models as an attention-backed append-only memory log,
    but the reference is allowed to be straightforward.
    """

    pc: int = 0x8000
    a: int = 0x00
    x: int = 0x00
    y: int = 0x00
    s: int = 0xFD
    p: int = FLAG_U | FLAG_I  # 0x24 -- interrupt disable, unused bit set
    memory: bytearray = field(default_factory=lambda: bytearray(0x10000))
    halted: bool = False
    step_count: int = 0
    # NMOS 6502 honors the D flag; the NES 2A03 ignores it (always binary).
    bcd_enabled: bool = True
    # Enable the NES's $0000-$1FFF internal-RAM mirrors when needed.
    mirror_cpu_ram: bool = False
    # Level-style NMI request: when set, the next step() executes the NMI
    # sequence instead of the instruction at PC, then clears the flag.
    nmi_pending: bool = False
    # IRQ request: delivered at a step boundary only while the I flag is
    # clear (stays pending until then); NMI wins when both are set.
    irq_pending: bool = False

    # operation dispatch table, populated in __post_init__
    _ops: dict[str, Callable[[Opcode, int, int, int], tuple[int, int, tuple[MemWrite, ...]]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self):
        if self.mirror_cpu_ram:
            for addr in range(0x0800, 0x2000):
                self.memory[addr] = self.memory[addr & 0x07FF]
        self._ops = self.operations()
        # 256-entry dispatch table: (opcode, handler) or None for
        # unsupported bytes.  Avoids two dict lookups per instruction.
        self.dispatch = [None] * 256
        for byte, opcode in OPCODES.items():
            self.dispatch[byte] = (opcode, self._ops[opcode.operation])

    # ── State helpers ───────────────────────────────────────────────

    def state(self, ir: int = 0, operand_lo: int = 0, operand_hi: int = 0,
              ea: int = 0, data: int = 0) -> Cpu6502State:
        return Cpu6502State(
            pc=self.pc,
            a=self.a,
            x=self.x,
            y=self.y,
            s=self.s,
            p=self.p,
            ir=ir,
            operand_lo=operand_lo,
            operand_hi=operand_hi,
            ea=ea,
            data=data,
            halted=self.halted,
        )

    def load_program(self, program: bytes | bytearray, start_pc: int | None = None) -> None:
        """Write program bytes into memory and reset PC."""
        if start_pc is None:
            start_pc = self.pc
        for i, b in enumerate(program):
            self.write_memory(start_pc + i, b)
        self.pc = start_pc & 0xFFFF

    def memory_address(self, addr: int) -> int:
        addr &= 0xFFFF
        return (addr & 0x07FF) if self.mirror_cpu_ram and addr < 0x2000 else addr

    def read_memory(self, addr: int) -> int:
        return self.memory[self.memory_address(addr)]

    def write_memory(self, addr: int, value: int) -> None:
        for target in written_addresses(addr, self.mirror_cpu_ram):
            self.memory[target] = value & 0xFF

    # ── Flag helpers ────────────────────────────────────────────────

    def set_nz(self, value: int) -> None:
        value &= 0xFF
        self.p = (self.p & ~(FLAG_N | FLAG_Z)) | (FLAG_N if value & 0x80 else 0) | (FLAG_Z if value == 0 else 0)

    def set_flag(self, flag: str, value: bool | int) -> None:
        mask = FLAG_MASKS[flag]
        if value:
            self.p |= mask
        else:
            self.p &= ~mask

    # ── Stack helpers ───────────────────────────────────────────────

    def push_byte(self, value: int) -> None:
        self.write_memory(0x0100 + self.s, value)
        self.s = (self.s - 1) & 0xFF

    def pull_byte(self) -> int:
        self.s = (self.s + 1) & 0xFF
        return self.read_memory(0x0100 + self.s)

    def push_word(self, value: int) -> None:
        self.push_byte((value >> 8) & 0xFF)
        self.push_byte(value & 0xFF)

    def pull_word(self) -> int:
        lo = self.pull_byte()
        hi = self.pull_byte()
        return lo | (hi << 8)

    # ── Instruction execution ───────────────────────────────────────

    def fetch_byte(self, addr: int) -> int:
        return self.read_memory(addr)

    def read_operand(self, opcode: Opcode) -> tuple[int, int, int]:
        """Return (operand_lo, operand_hi, next_pc_after_operand_fetch)."""
        lo = self.fetch_byte(self.pc + 1)
        if opcode.length == 1:
            return 0, 0, self.pc + 1
        if opcode.length == 2:
            return lo, 0, self.pc + 2
        hi = self.fetch_byte(self.pc + 2)
        return lo, hi, self.pc + 3

    def resolve_ea(self, opcode: Opcode, lo: int, hi: int) -> int:
        mode = opcode.mode
        if mode in (MODE_IMPLIED, MODE_ACCUMULATOR):
            return 0
        if mode == MODE_IMMEDIATE:
            return lo
        if mode == MODE_ZERO_PAGE:
            return lo
        if mode == MODE_ZERO_PAGE_X:
            return (lo + self.x) & 0xFF
        if mode == MODE_ZERO_PAGE_Y:
            return (lo + self.y) & 0xFF
        if mode == MODE_ABSOLUTE:
            return lo | (hi << 8)
        if mode == MODE_ABSOLUTE_X:
            return ((lo | (hi << 8)) + self.x) & 0xFFFF
        if mode == MODE_ABSOLUTE_Y:
            return ((lo | (hi << 8)) + self.y) & 0xFFFF
        if mode == MODE_RELATIVE:
            offset = lo if lo < 0x80 else lo - 0x100
            return (self.pc + opcode.length + offset) & 0xFFFF
        if mode == MODE_INDIRECT:
            ptr = lo | (hi << 8)
            # NMOS page-wrap bug for JMP ($xxFF)
            lo_addr = ptr
            hi_addr = (ptr & 0xFF00) | ((ptr + 1) & 0xFF)
            return self.fetch_byte(lo_addr) | (self.fetch_byte(hi_addr) << 8)
        if mode == MODE_INDIRECT_X:
            zp = (lo + self.x) & 0xFF
            return self.fetch_byte(zp) | (self.fetch_byte((zp + 1) & 0xFF) << 8)
        if mode == MODE_INDIRECT_Y:
            base = self.fetch_byte(lo) | (self.fetch_byte((lo + 1) & 0xFF) << 8)
            return (base + self.y) & 0xFFFF
        return 0

    def step(self) -> TraceRow:
        """Execute one instruction (or a pending NMI) and return a trace row."""
        if self.halted:
            row = TraceRow(self.step_count, self.state(), ())
            self.step_count += 1
            return row

        if self.nmi_pending:
            self.nmi_pending = False
            return self.interrupt_step(0xFFFA)
        if irq_deliverable(self.p, self.irq_pending):
            self.irq_pending = False
            return self.interrupt_step(0xFFFE)

        ir = self.fetch_byte(self.pc)
        opcode = opcode_for(ir)
        lo, hi, _ = self.read_operand(opcode)
        ea = self.resolve_ea(opcode, lo, hi)
        data, new_pc, writes = self._ops[opcode.operation](opcode, lo, hi, ea)

        row = TraceRow(
            step=self.step_count,
            state=self.state(ir=ir, operand_lo=lo, operand_hi=hi, ea=ea, data=data),
            writes=writes,
        )

        self.pc = new_pc & 0xFFFF
        self.step_count += 1
        return row

    def interrupt_step(self, vector_addr: int) -> TraceRow:
        """Execute an interrupt sequence (NMI or IRQ) as one step.  The
        caller clears the pending flag it acted on.

        Matches the graph's convention: push PCH, PCL, then P with B clear
        and U set; set I; vector through $FFFA (NMI) or $FFFE (IRQ).  The
        trace row reports the interrupted PC, the (not executed)
        instruction bytes at PC, and ea=0 / data=0.
        """
        pc = self.pc
        pushed_p = ((self.p | FLAG_U) & ~FLAG_B) & 0xFF
        writes = (
            MemWrite(self.step_count, 0x0100 + self.s, (pc >> 8) & 0xFF),
            MemWrite(self.step_count, 0x0100 + ((self.s - 1) & 0xFF), pc & 0xFF),
            MemWrite(self.step_count, 0x0100 + ((self.s - 2) & 0xFF), pushed_p),
        )
        self.push_word(pc)
        self.push_byte(pushed_p)
        self.set_flag("I", True)
        vector = self.fetch_byte(vector_addr) | (self.fetch_byte(vector_addr + 1) << 8)

        row = TraceRow(
            step=self.step_count,
            state=self.state(
                ir=self.fetch_byte(pc),
                operand_lo=self.fetch_byte(pc + 1),
                operand_hi=self.fetch_byte(pc + 2),
                ea=0,
                data=0,
            ),
            writes=writes,
        )
        self.pc = vector & 0xFFFF
        self.step_count += 1
        return row

    def run(self, program: bytes | bytearray | None = None,
            *, start_pc: int = 0x8000, max_steps: int = 100,
            initial_memory: bytes | bytearray | None = None) -> Trace:
        """Run a program and return its trace.

        Args:
            program: bytes to write at start_pc.  May be None if memory is
                already initialized (e.g. via initial_memory).
            start_pc: address where execution begins.
            max_steps: safety limit on instruction count.
            initial_memory: optional full 64 KiB image to load before writing
                the program.
        """
        if initial_memory is not None:
            if len(initial_memory) != 0x10000:
                raise ValueError("initial_memory must be exactly 64 KiB")
            self.memory = bytearray(initial_memory)
        if program is not None:
            self.load_program(program, start_pc)
        else:
            self.pc = start_pc & 0xFFFF

        self.halted = False
        self.step_count = 0
        trace = Trace()

        for _ in range(max_steps):
            row = self.step()
            trace.append(row)
            if self.halted:
                break

        return trace

    def run_traps(self, *, start_pc: int | None = None,
                  max_steps: int = 100_000_000) -> int:
        """Execute until the PC traps in a jump-to-self, then return the step count.

        Fast path for long conformance runs (e.g. the Klaus functional test
        suite, ~30M instructions): identical semantics to step(), but no
        trace rows are built and dispatch is a flat table.
        """
        if start_pc is not None:
            self.pc = start_pc & 0xFFFF
        dispatch = self.dispatch
        resolve_ea = self.resolve_ea
        steps = 0
        while steps < max_steps:
            pc = self.pc
            entry = dispatch[self.read_memory(pc)]
            if entry is None:
                raise ValueError(f"unsupported opcode 0x{self.read_memory(pc):02X}")
            opcode, handler = entry
            length = opcode.length
            if length == 1:
                lo = hi = 0
            elif length == 2:
                lo = self.read_memory(pc + 1)
                hi = 0
            else:
                lo = self.read_memory(pc + 1)
                hi = self.read_memory(pc + 2)
            ea = resolve_ea(opcode, lo, hi)
            _, new_pc, _ = handler(opcode, lo, hi, ea)
            new_pc &= 0xFFFF
            self.pc = new_pc
            steps += 1
            if new_pc == pc or self.halted:
                break
        return steps

    # ── Operations ──────────────────────────────────────────────────
    # A handler takes (opcode, operand lo, operand hi, effective address)
    # and returns (the trace row's data value, the next PC, the memory
    # writes).  Most operations are one of a few families, bound to a
    # register, a flag or a shift by operations().

    def operations(self) -> dict[str, Callable[[Opcode, int, int, int], Result]]:
        """The handler for every spec operation name."""
        ops = {
            **{f"load_{r}": partial(self.load, r) for r in "axy"},
            **{f"store_{r}": partial(self.store, r) for r in "axy"},
            "cmp": partial(self.compare, "a"),
            "cpx": partial(self.compare, "x"),
            "cpy": partial(self.compare, "y"),
            "and": partial(self.logic, lambda a, v: a & v),
            "ora": partial(self.logic, lambda a, v: a | v),
            "eor": partial(self.logic, lambda a, v: a ^ v),
            "asl": partial(self.shift, ASL),
            "lsr": partial(self.shift, LSR),
            "rol": partial(self.shift, ROL),
            "ror": partial(self.shift, ROR),
            "inc": partial(self.step_memory, 1),
            "dec": partial(self.step_memory, -1),
            "inx": partial(self.transfer, "x", "x", 1),
            "iny": partial(self.transfer, "y", "y", 1),
            "dex": partial(self.transfer, "x", "x", -1),
            "dey": partial(self.transfer, "y", "y", -1),
            "tax": partial(self.transfer, "x", "a", 0),
            "tay": partial(self.transfer, "y", "a", 0),
            "txa": partial(self.transfer, "a", "x", 0),
            "tya": partial(self.transfer, "a", "y", 0),
            "tsx": partial(self.transfer, "x", "s", 0),
            **{op: partial(self.branch, mask, expected)
               for op, (mask, expected) in BRANCH_CONDITIONS.items()},
            **{op: partial(self.flag, flag, value) for op, (flag, value) in (
                ("clc", ("C", False)), ("sec", ("C", True)), ("cli", ("I", False)),
                ("sei", ("I", True)), ("clv", ("V", False)), ("cld", ("D", False)),
                ("sed", ("D", True)))},
        }
        for name in ("adc", "sbc", "bit", "txs", "pha", "pla", "php", "plp", "jmp", "jsr", "rts",
                     "rti", "brk", "nop", "lax", "sax", "dcp", "isb", "slo", "rla", "sre", "rra"):
            ops[name] = getattr(self, f"op_{name}")
        return ops

    def write(self, ea: int, value: int) -> MemWrite:
        self.write_memory(ea, value)
        return MemWrite(self.step_count, ea, value & 0xFF)

    def push_writes(self, *values: int) -> tuple[MemWrite, ...]:
        """The stack writes of pushing `values` in order (before pushing)."""
        return tuple(
            MemWrite(self.step_count, 0x0100 + ((self.s - i) & 0xFF), v & 0xFF)
            for i, v in enumerate(values)
        )

    def set_register(self, register: str, value: int, opcode: Opcode) -> Result:
        """register = value & 0xFF, setting N and Z."""
        value &= 0xFF
        setattr(self, register, value)
        self.set_nz(value)
        return value, self.pc + opcode.length, ()

    def load(self, register: str, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:
        return self.set_register(register, (lo if opcode.mode == MODE_IMMEDIATE else self.read_memory(ea)), opcode)

    def store(self, register: str, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:
        value = getattr(self, register)
        return value, self.pc + opcode.length, (self.write(ea, value),)

    def transfer(self, register: str, source: str, delta: int, opcode: Opcode, lo: int, hi: int,
                 ea: int) -> Result:
        """register = source + delta: the transfers, INX/INY/DEX/DEY."""
        return self.set_register(register, getattr(self, source) + delta, opcode)

    def compare(self, register: str, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:
        result = getattr(self, register) - (lo if opcode.mode == MODE_IMMEDIATE else self.read_memory(ea))
        self.set_flag("C", result >= 0)
        self.set_nz(result & 0xFF)
        return result & 0xFF, self.pc + opcode.length, ()

    def logic(self, combine, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:
        return self.set_register("a", combine(self.a, (lo if opcode.mode == MODE_IMMEDIATE else self.read_memory(ea))), opcode)

    def shift(self, rule, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:
        """ASL/LSR/ROL/ROR on A or on memory: rule(value, carry) gives the
        result and the bit shifted out into C."""
        on_a = opcode.mode == MODE_ACCUMULATOR
        value, carry = rule(self.a if on_a else self.fetch_byte(ea), self.p & FLAG_C)
        self.set_flag("C", carry)
        self.set_nz(value)
        if on_a:
            self.a = value
            return value, self.pc + opcode.length, ()
        return value, self.pc + opcode.length, (self.write(ea, value),)

    def step_memory(self, delta: int, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:
        value = (self.fetch_byte(ea) + delta) & 0xFF
        self.set_nz(value)
        return value, self.pc + opcode.length, (self.write(ea, value),)

    def branch(self, mask: int, expected: int, opcode: Opcode, lo: int, hi: int, ea: int
               ) -> Result:
        if (self.p & mask) != expected:
            return 0, self.pc + opcode.length, ()
        offset = lo if lo < 0x80 else lo - 0x100
        return 0, (self.pc + opcode.length + offset) & 0xFFFF, ()

    def flag(self, name: str, value: bool, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:
        self.set_flag(name, value)
        return 0, self.pc + opcode.length, ()

    # ADC and SBC: V always reflects the binary result, as on NMOS hardware;
    # with D set (and BCD enabled) A and C take the decimal result.

    def add(self, value: int, *, subtract: bool = False) -> None:
        carry = 1 if self.p & FLAG_C else 0
        addend = (~value) & 0xFF if subtract else value
        raw = self.a + addend + carry
        binary = raw & 0xFF
        if self.p & FLAG_D and self.bcd_enabled:
            result, new_c = (self.sbc_decimal if subtract else self.adc_decimal)(value)
        else:
            result, new_c = binary, raw > 0xFF
        self.set_flag("C", new_c)
        self.set_flag("V", (self.a ^ binary) & (addend ^ binary) & 0x80)
        self.a = result
        self.set_nz(self.a)

    def adc_decimal(self, value: int) -> tuple[int, bool]:
        carry = 1 if self.p & FLAG_C else 0
        al = (self.a & 0x0F) + (value & 0x0F) + carry
        ah = (self.a >> 4) + (value >> 4) + (al >= 10)
        return ((ah - 10 if ah >= 10 else ah) & 0x0F) << 4 | ((al - 10 if al >= 10 else al) & 0x0F), ah >= 10

    def sbc_decimal(self, value: int) -> tuple[int, bool]:
        carry = 1 if self.p & FLAG_C else 0
        al = (self.a & 0x0F) - (value & 0x0F) - (1 - carry)
        ah = (self.a >> 4) - (value >> 4) - (al < 0)
        return ((ah + 10 if ah < 0 else ah) & 0x0F) << 4 | ((al + 10 if al < 0 else al) & 0x0F), ah >= 0

    def op_adc(self, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:
        self.add(lo if opcode.mode == MODE_IMMEDIATE else self.read_memory(ea))
        return self.a, self.pc + opcode.length, ()

    def op_sbc(self, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:
        self.add((lo if opcode.mode == MODE_IMMEDIATE else self.read_memory(ea)), subtract=True)
        return self.a, self.pc + opcode.length, ()

    def op_bit(self, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:
        value = self.fetch_byte(ea)
        self.p = (self.p & ~(FLAG_N | FLAG_V | FLAG_Z)) | (value & (FLAG_N | FLAG_V))
        self.set_flag("Z", (self.a & value) == 0)
        return value, self.pc + opcode.length, ()

    def op_txs(self, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:
        self.s = self.x  # the one transfer that sets no flags
        return self.s, self.pc + opcode.length, ()

    def op_pha(self, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:
        writes = self.push_writes(self.a)
        self.push_byte(self.a)
        return self.a, self.pc + opcode.length, writes

    def op_php(self, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:
        value = (self.p | FLAG_B | FLAG_U) & 0xFF  # pushed with B and U set
        writes = self.push_writes(value)
        self.push_byte(value)
        return self.p, self.pc + opcode.length, writes

    def op_pla(self, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:
        return self.set_register("a", self.pull_byte(), opcode)

    def op_plp(self, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:
        self.p = (self.pull_byte() & ~FLAG_B) | FLAG_U
        return self.p, self.pc + opcode.length, ()

    def op_jmp(self, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:
        return 0, ea, ()

    def op_jsr(self, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:
        return_addr = (self.pc + opcode.length - 1) & 0xFFFF
        writes = self.push_writes(return_addr >> 8, return_addr)
        self.push_word(return_addr)
        return 0, ea, writes

    def op_rts(self, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:
        return 0, (self.pull_word() + 1) & 0xFFFF, ()

    def op_rti(self, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:
        self.p = (self.pull_byte() & ~FLAG_B) | FLAG_U
        return 0, self.pull_word(), ()

    def op_brk(self, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:
        """Push PC+2 and P (B set), set I, jump through $FFFE; a zero vector
        halts (the harnesses' end-of-program convention)."""
        return_addr = (self.pc + 2) & 0xFFFF
        pushed_p = (self.p | FLAG_B | FLAG_U) & 0xFF
        writes = self.push_writes(return_addr >> 8, return_addr, pushed_p)
        self.push_word(return_addr)
        self.push_byte(pushed_p)
        self.set_flag("I", True)
        vector = self.fetch_byte(0xFFFE) | (self.fetch_byte(0xFFFF) << 8)
        self.halted = vector == 0x0000
        return 0, vector, writes

    def op_nop(self, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:
        return 0, self.pc + opcode.length, ()

    # ── Unofficial opcodes: compositions of the official ones ───────

    def op_lax(self, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:
        self.x = (lo if opcode.mode == MODE_IMMEDIATE else self.read_memory(ea)) & 0xFF
        return self.set_register("a", self.x, opcode)

    def op_sax(self, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:
        value = self.a & self.x
        return value, self.pc + opcode.length, (self.write(ea, value),)

    def op_dcp(self, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:  # DEC, CMP
        value, next_pc, writes = self.step_memory(-1, opcode, lo, hi, ea)
        self.compare("a", opcode, value, hi, ea)
        return value, next_pc, writes

    def op_isb(self, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:  # INC, SBC
        value, next_pc, writes = self.step_memory(1, opcode, lo, hi, ea)
        self.add(value, subtract=True)
        return value, next_pc, writes

    def shift_then(self, rule, then, opcode: Opcode, ea: int) -> Result:
        """A memory shift whose result then feeds A (the RMW unofficials)."""
        value, next_pc, writes = self.shift(rule, opcode, 0, 0, ea)
        then(value)
        return value, next_pc, writes

    def op_slo(self, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:  # ASL, ORA
        return self.shift_then(ASL, lambda v: self.set_register("a", self.a | v, opcode), opcode, ea)

    def op_rla(self, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:  # ROL, AND
        return self.shift_then(ROL, lambda v: self.set_register("a", self.a & v, opcode), opcode, ea)

    def op_sre(self, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:  # LSR, EOR
        return self.shift_then(LSR, lambda v: self.set_register("a", self.a ^ v, opcode), opcode, ea)

    def op_rra(self, opcode: Opcode, lo: int, hi: int, ea: int) -> Result:  # ROR, ADC
        return self.shift_then(ROR, self.add, opcode, ea)



# ── Convenience entry point ─────────────────────────────────────────


def run(program: bytes | bytearray | None = None, *, start_pc: int = 0x8000,
        max_steps: int = 100, initial_memory: bytes | bytearray | None = None) -> Trace:
    """Run a program on a fresh CPU and return its trace."""
    cpu = Cpu6502()
    return cpu.run(program, start_pc=start_pc, max_steps=max_steps,
                   initial_memory=initial_memory)


def run_image(image: bytes | bytearray, *, start_pc: int = 0x0400,
              max_steps: int = 100_000) -> Trace:
    """Run a full 64KB memory image on a fresh CPU."""
    return run(None, start_pc=start_pc, max_steps=max_steps, initial_memory=image)


def main() -> None:

    parser = argparse.ArgumentParser(description="Run a 6502 binary through the reference core.")
    parser.add_argument("program", help="Path to a .bin file")
    parser.add_argument("--start-pc", type=lambda x: int(x, 0), default=0x8000)
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()

    with open(args.program, "rb") as f:
        program = f.read()

    trace = run(program, start_pc=args.start_pc, max_steps=args.max_steps)
    print(trace.format())


if __name__ == "__main__":
    main()
