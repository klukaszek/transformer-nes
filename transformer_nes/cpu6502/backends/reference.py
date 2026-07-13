"""Boring Python 6502 reference core.

No transformer, no graph -- just a trusted oracle that executes the full
official NMOS 6502 opcode set using the shared spec table.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from transformer_nes.cpu6502.spec import (
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
    Opcode,
    opcode_for,
)
from transformer_nes.cpu6502.trace import Cpu6502State, MemWrite, Trace, TraceRow


def irq_deliverable(p: int, irq_pending: bool, nmi_pending: bool = False) -> bool:
    """An IRQ is taken at a step boundary only with the I flag clear and no
    NMI ahead of it (NMI wins).  The canonical Python gate; the generated
    C++ engines carry the identical test as their own irq_ready()."""
    return irq_pending and not nmi_pending and not (p & FLAG_I)


@dataclass
class Cpu6502:
    """Reference 6502 CPU.

    Memory is modeled as a dense 64 KiB array.  This is intentionally simple:
    the graph interpreter will use an attention-based append-only memory log,
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
        self._ops = {
            "load_a": self.op_load_a,
            "load_x": self.op_load_x,
            "load_y": self.op_load_y,
            "store_a": self.op_store_a,
            "store_x": self.op_store_x,
            "store_y": self.op_store_y,
            "adc": self.op_adc,
            "sbc": self.op_sbc,
            "cmp": self.op_cmp,
            "cpx": self.op_cpx,
            "cpy": self.op_cpy,
            "and": self.op_and,
            "ora": self.op_ora,
            "eor": self.op_eor,
            "bit": self.op_bit,
            "asl": self.op_asl,
            "lsr": self.op_lsr,
            "rol": self.op_rol,
            "ror": self.op_ror,
            "inc": self.op_inc,
            "dec": self.op_dec,
            "inx": self.op_inx,
            "iny": self.op_iny,
            "dex": self.op_dex,
            "dey": self.op_dey,
            "tax": self.op_tax,
            "tay": self.op_tay,
            "txa": self.op_txa,
            "tya": self.op_tya,
            "tsx": self.op_tsx,
            "txs": self.op_txs,
            "pha": self.op_pha,
            "php": self.op_php,
            "pla": self.op_pla,
            "plp": self.op_plp,
            "jmp": self.op_jmp,
            "jsr": self.op_jsr,
            "rts": self.op_rts,
            "rti": self.op_rti,
            "brk": self.op_brk,
            "bpl": self.op_bpl,
            "bmi": self.op_bmi,
            "bvc": self.op_bvc,
            "bvs": self.op_bvs,
            "bcc": self.op_bcc,
            "bcs": self.op_bcs,
            "bne": self.op_bne,
            "beq": self.op_beq,
            "clc": self.op_clc,
            "sec": self.op_sec,
            "cli": self.op_cli,
            "sei": self.op_sei,
            "clv": self.op_clv,
            "cld": self.op_cld,
            "sed": self.op_sed,
            "nop": self.op_nop,
            "lax": self.op_lax,
            "sax": self.op_sax,
            "dcp": self.op_dcp,
            "isb": self.op_isb,
            "slo": self.op_slo,
            "rla": self.op_rla,
            "sre": self.op_sre,
            "rra": self.op_rra,
        }
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
            self.memory[(start_pc + i) & 0xFFFF] = b
        self.pc = start_pc & 0xFFFF

    # ── Flag helpers ────────────────────────────────────────────────

    def set_nz(self, value: int) -> None:
        value &= 0xFF
        self.p = (self.p & ~(FLAG_N | FLAG_Z)) | (FLAG_N if value & 0x80 else 0) | (FLAG_Z if value == 0 else 0)

    def set_flag(self, flag: str, value: bool) -> None:
        mask = FLAG_MASKS[flag]
        if value:
            self.p |= mask
        else:
            self.p &= ~mask

    # ── Stack helpers ───────────────────────────────────────────────

    def push_byte(self, value: int) -> None:
        self.memory[0x0100 + self.s] = value & 0xFF
        self.s = (self.s - 1) & 0xFF

    def pull_byte(self) -> int:
        self.s = (self.s + 1) & 0xFF
        return self.memory[0x0100 + self.s]

    def push_word(self, value: int) -> None:
        self.push_byte((value >> 8) & 0xFF)
        self.push_byte(value & 0xFF)

    def pull_word(self) -> int:
        lo = self.pull_byte()
        hi = self.pull_byte()
        return lo | (hi << 8)

    # ── Instruction execution ───────────────────────────────────────

    def fetch_byte(self, addr: int) -> int:
        return self.memory[addr & 0xFFFF]

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

    def read_mem(self, opcode: Opcode, lo: int, hi: int, ea: int) -> int:
        mode = opcode.mode
        if mode == MODE_IMMEDIATE:
            return lo
        if mode == MODE_ACCUMULATOR:
            return self.a
        return self.fetch_byte(ea)

    def branch(self, opcode: Opcode, lo: int, taken: bool) -> tuple[int, int, tuple[MemWrite, ...]]:
        if taken:
            offset = lo if lo < 0x80 else lo - 0x100
            target = (self.pc + opcode.length + offset) & 0xFFFF
            return 0, target, ()
        return 0, self.pc + opcode.length, ()

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
        lo, hi, fetch_end = self.read_operand(opcode)
        ea = self.resolve_ea(opcode, lo, hi)
        data = self.read_mem(opcode, lo, hi, ea)

        handler = self._ops[opcode.operation]
        writes: tuple[MemWrite, ...]
        data, new_pc, writes = handler(opcode, lo, hi, ea)

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
        memory = self.memory
        dispatch = self.dispatch
        resolve_ea = self.resolve_ea
        steps = 0
        while steps < max_steps:
            pc = self.pc
            entry = dispatch[memory[pc]]
            if entry is None:
                raise ValueError(f"unsupported opcode 0x{memory[pc]:02X}")
            opcode, handler = entry
            length = opcode.length
            if length == 1:
                lo = hi = 0
            elif length == 2:
                lo = memory[(pc + 1) & 0xFFFF]
                hi = 0
            else:
                lo = memory[(pc + 1) & 0xFFFF]
                hi = memory[(pc + 2) & 0xFFFF]
            ea = resolve_ea(opcode, lo, hi)
            _, new_pc, _ = handler(opcode, lo, hi, ea)
            new_pc &= 0xFFFF
            self.pc = new_pc
            steps += 1
            if new_pc == pc or self.halted:
                break
        return steps

    # ── Operation implementations ───────────────────────────────────

    def op_load_a(self, opcode: Opcode, lo: int, hi: int, ea: int
                    ) -> tuple[int, int, tuple[MemWrite, ...]]:
        value = lo if opcode.mode == MODE_IMMEDIATE else self.fetch_byte(ea)
        self.a = value & 0xFF
        self.set_nz(self.a)
        return self.a, self.pc + opcode.length, ()

    def op_load_x(self, opcode: Opcode, lo: int, hi: int, ea: int
                    ) -> tuple[int, int, tuple[MemWrite, ...]]:
        value = lo if opcode.mode == MODE_IMMEDIATE else self.fetch_byte(ea)
        self.x = value & 0xFF
        self.set_nz(self.x)
        return self.x, self.pc + opcode.length, ()

    def op_load_y(self, opcode: Opcode, lo: int, hi: int, ea: int
                    ) -> tuple[int, int, tuple[MemWrite, ...]]:
        value = lo if opcode.mode == MODE_IMMEDIATE else self.fetch_byte(ea)
        self.y = value & 0xFF
        self.set_nz(self.y)
        return self.y, self.pc + opcode.length, ()

    def op_store_a(self, opcode: Opcode, lo: int, hi: int, ea: int
                     ) -> tuple[int, int, tuple[MemWrite, ...]]:
        self.memory[ea] = self.a & 0xFF
        write = MemWrite(self.step_count, ea, self.a & 0xFF)
        return self.a, self.pc + opcode.length, (write,)

    def op_store_x(self, opcode: Opcode, lo: int, hi: int, ea: int
                     ) -> tuple[int, int, tuple[MemWrite, ...]]:
        self.memory[ea] = self.x & 0xFF
        write = MemWrite(self.step_count, ea, self.x & 0xFF)
        return self.x, self.pc + opcode.length, (write,)

    def op_store_y(self, opcode: Opcode, lo: int, hi: int, ea: int
                     ) -> tuple[int, int, tuple[MemWrite, ...]]:
        self.memory[ea] = self.y & 0xFF
        write = MemWrite(self.step_count, ea, self.y & 0xFF)
        return self.y, self.pc + opcode.length, (write,)

    def adc_apply(self, value: int) -> None:
        carry = 1 if self.p & FLAG_C else 0
        raw = self.a + value + carry
        bin_result = raw & 0xFF
        if self.p & FLAG_D and self.bcd_enabled:
            result, new_c = self.adc_decimal(value)
        else:
            result = bin_result
            new_c = raw > 0xFF
        # V always reflects the binary sum, as on NMOS hardware.
        a_sign = self.a & 0x80
        v_sign = value & 0x80
        r_sign = bin_result & 0x80
        self.set_flag("C", new_c)
        self.set_flag("V", (a_sign == v_sign) and (a_sign != r_sign))
        self.a = result
        self.set_nz(self.a)

    def op_adc(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        value = lo if opcode.mode == MODE_IMMEDIATE else self.fetch_byte(ea)
        self.adc_apply(value)
        return self.a, self.pc + opcode.length, ()

    def adc_decimal(self, value: int) -> tuple[int, bool]:
        carry = 1 if self.p & FLAG_C else 0
        al = (self.a & 0x0F) + (value & 0x0F) + carry
        if al >= 10:
            al = (al - 10) & 0x0F
            ah_carry = 1
        else:
            ah_carry = 0
        ah = (self.a >> 4) + (value >> 4) + ah_carry
        if ah >= 10:
            ah = (ah - 10) & 0x0F
            final_c = True
        else:
            final_c = False
        return (ah << 4) | al, final_c

    def sbc_apply(self, value: int) -> None:
        carry = 1 if self.p & FLAG_C else 0
        raw = self.a + ((~value) & 0xFF) + carry
        bin_result = raw & 0xFF
        if self.p & FLAG_D and self.bcd_enabled:
            result, new_c = self.sbc_decimal(value)
        else:
            result = bin_result
            new_c = raw > 0xFF
        # V always reflects the binary difference, as on NMOS hardware.
        a_sign = self.a & 0x80
        v_sign = (~value) & 0x80
        r_sign = bin_result & 0x80
        self.set_flag("C", new_c)
        self.set_flag("V", (a_sign == v_sign) and (a_sign != r_sign))
        self.a = result
        self.set_nz(self.a)

    def op_sbc(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        value = lo if opcode.mode == MODE_IMMEDIATE else self.fetch_byte(ea)
        self.sbc_apply(value)
        return self.a, self.pc + opcode.length, ()

    def sbc_decimal(self, value: int) -> tuple[int, bool]:
        carry = 1 if self.p & FLAG_C else 0
        # A - M - (1-C) = A + (99-M) + C  in decimal, but adjust result
        # Standard BCD subtraction algorithm
        al = (self.a & 0x0F) - (value & 0x0F) - (1 - carry)
        borrow = 0
        if al < 0:
            al = (al + 10) & 0x0F
            borrow = 1
        ah = (self.a >> 4) - (value >> 4) - borrow
        if ah < 0:
            ah = (ah + 10) & 0x0F
            final_c = False
        else:
            final_c = True
        return (ah << 4) | al, final_c

    def op_cmp(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        value = lo if opcode.mode == MODE_IMMEDIATE else self.fetch_byte(ea)
        result = self.a - value
        self.set_flag("C", result >= 0)
        self.set_nz(result & 0xFF)
        return result & 0xFF, self.pc + opcode.length, ()

    def op_cpx(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        value = lo if opcode.mode == MODE_IMMEDIATE else self.fetch_byte(ea)
        result = self.x - value
        self.set_flag("C", result >= 0)
        self.set_nz(result & 0xFF)
        return result & 0xFF, self.pc + opcode.length, ()

    def op_cpy(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        value = lo if opcode.mode == MODE_IMMEDIATE else self.fetch_byte(ea)
        result = self.y - value
        self.set_flag("C", result >= 0)
        self.set_nz(result & 0xFF)
        return result & 0xFF, self.pc + opcode.length, ()

    def op_and(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        value = lo if opcode.mode == MODE_IMMEDIATE else self.fetch_byte(ea)
        self.a = (self.a & value) & 0xFF
        self.set_nz(self.a)
        return self.a, self.pc + opcode.length, ()

    def op_ora(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        value = lo if opcode.mode == MODE_IMMEDIATE else self.fetch_byte(ea)
        self.a = (self.a | value) & 0xFF
        self.set_nz(self.a)
        return self.a, self.pc + opcode.length, ()

    def op_eor(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        value = lo if opcode.mode == MODE_IMMEDIATE else self.fetch_byte(ea)
        self.a = (self.a ^ value) & 0xFF
        self.set_nz(self.a)
        return self.a, self.pc + opcode.length, ()

    def op_bit(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        value = self.fetch_byte(ea)
        self.p = (self.p & ~(FLAG_N | FLAG_V | FLAG_Z)) | (value & (FLAG_N | FLAG_V))
        self.set_flag("Z", (self.a & value) == 0)
        return value, self.pc + opcode.length, ()

    def op_asl(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        if opcode.mode == MODE_ACCUMULATOR:
            value = self.a
            new_value = (value << 1) & 0xFF
            self.a = new_value
            self.set_flag("C", value & 0x80)
            self.set_nz(new_value)
            return self.a, self.pc + opcode.length, ()
        value = self.fetch_byte(ea)
        new_value = (value << 1) & 0xFF
        self.memory[ea] = new_value
        self.set_flag("C", value & 0x80)
        self.set_nz(new_value)
        write = MemWrite(self.step_count, ea, new_value)
        return new_value, self.pc + opcode.length, (write,)

    def op_lsr(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        if opcode.mode == MODE_ACCUMULATOR:
            value = self.a
            new_value = (value >> 1) & 0xFF
            self.a = new_value
            self.set_flag("C", value & 0x01)
            self.set_nz(new_value)
            return self.a, self.pc + opcode.length, ()
        value = self.fetch_byte(ea)
        new_value = (value >> 1) & 0xFF
        self.memory[ea] = new_value
        self.set_flag("C", value & 0x01)
        self.set_nz(new_value)
        write = MemWrite(self.step_count, ea, new_value)
        return new_value, self.pc + opcode.length, (write,)

    def op_rol(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        carry = 1 if self.p & FLAG_C else 0
        if opcode.mode == MODE_ACCUMULATOR:
            value = self.a
            new_value = ((value << 1) | carry) & 0xFF
            self.a = new_value
            self.set_flag("C", value & 0x80)
            self.set_nz(new_value)
            return self.a, self.pc + opcode.length, ()
        value = self.fetch_byte(ea)
        new_value = ((value << 1) | carry) & 0xFF
        self.memory[ea] = new_value
        self.set_flag("C", value & 0x80)
        self.set_nz(new_value)
        write = MemWrite(self.step_count, ea, new_value)
        return new_value, self.pc + opcode.length, (write,)

    def op_ror(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        carry = 1 if self.p & FLAG_C else 0
        if opcode.mode == MODE_ACCUMULATOR:
            value = self.a
            new_value = ((value >> 1) | (carry << 7)) & 0xFF
            self.a = new_value
            self.set_flag("C", value & 0x01)
            self.set_nz(new_value)
            return self.a, self.pc + opcode.length, ()
        value = self.fetch_byte(ea)
        new_value = ((value >> 1) | (carry << 7)) & 0xFF
        self.memory[ea] = new_value
        self.set_flag("C", value & 0x01)
        self.set_nz(new_value)
        write = MemWrite(self.step_count, ea, new_value)
        return new_value, self.pc + opcode.length, (write,)

    def op_inc(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        new_value = (self.fetch_byte(ea) + 1) & 0xFF
        self.memory[ea] = new_value
        self.set_nz(new_value)
        write = MemWrite(self.step_count, ea, new_value)
        return new_value, self.pc + opcode.length, (write,)

    def op_dec(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        new_value = (self.fetch_byte(ea) - 1) & 0xFF
        self.memory[ea] = new_value
        self.set_nz(new_value)
        write = MemWrite(self.step_count, ea, new_value)
        return new_value, self.pc + opcode.length, (write,)

    def op_inx(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        self.x = (self.x + 1) & 0xFF
        self.set_nz(self.x)
        return self.x, self.pc + opcode.length, ()

    def op_iny(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        self.y = (self.y + 1) & 0xFF
        self.set_nz(self.y)
        return self.y, self.pc + opcode.length, ()

    def op_dex(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        self.x = (self.x - 1) & 0xFF
        self.set_nz(self.x)
        return self.x, self.pc + opcode.length, ()

    def op_dey(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        self.y = (self.y - 1) & 0xFF
        self.set_nz(self.y)
        return self.y, self.pc + opcode.length, ()

    def op_tax(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        self.x = self.a & 0xFF
        self.set_nz(self.x)
        return self.x, self.pc + opcode.length, ()

    def op_tay(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        self.y = self.a & 0xFF
        self.set_nz(self.y)
        return self.y, self.pc + opcode.length, ()

    def op_txa(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        self.a = self.x & 0xFF
        self.set_nz(self.a)
        return self.a, self.pc + opcode.length, ()

    def op_tya(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        self.a = self.y & 0xFF
        self.set_nz(self.a)
        return self.a, self.pc + opcode.length, ()

    def op_tsx(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        self.x = self.s & 0xFF
        self.set_nz(self.x)
        return self.x, self.pc + opcode.length, ()

    def op_txs(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        self.s = self.x & 0xFF
        return self.s, self.pc + opcode.length, ()

    def op_pha(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        write = MemWrite(self.step_count, 0x0100 + self.s, self.a & 0xFF)
        self.push_byte(self.a)
        return self.a, self.pc + opcode.length, (write,)

    def op_php(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        # PHP always pushes with B and U set.
        value = (self.p | FLAG_B | FLAG_U) & 0xFF
        write = MemWrite(self.step_count, 0x0100 + self.s, value)
        self.push_byte(value)
        return self.p, self.pc + opcode.length, (write,)

    def op_pla(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        self.a = self.pull_byte()
        self.set_nz(self.a)
        return self.a, self.pc + opcode.length, ()

    def op_plp(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        self.p = (self.pull_byte() & ~FLAG_B) | FLAG_U
        return self.p, self.pc + opcode.length, ()

    def op_jmp(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        return 0, ea, ()

    def op_jsr(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        return_addr = (self.pc + opcode.length - 1) & 0xFFFF
        writes = (
            MemWrite(self.step_count, 0x0100 + self.s, (return_addr >> 8) & 0xFF),
            MemWrite(self.step_count, 0x0100 + ((self.s - 1) & 0xFF), return_addr & 0xFF),
        )
        self.push_word(return_addr)
        return 0, ea, writes

    def op_rts(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        return_addr = (self.pull_word() + 1) & 0xFFFF
        return 0, return_addr, ()

    def op_rti(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        self.p = (self.pull_byte() & ~FLAG_B) | FLAG_U
        return 0, self.pull_word(), ()

    def op_brk(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        # BRK pushes PC+2, then P with B set, then loads IRQ vector at $FFFE.
        return_addr = (self.pc + 2) & 0xFFFF
        pushed_p = (self.p | FLAG_B | FLAG_U) & 0xFF
        writes = (
            MemWrite(self.step_count, 0x0100 + self.s, (return_addr >> 8) & 0xFF),
            MemWrite(self.step_count, 0x0100 + ((self.s - 1) & 0xFF), return_addr & 0xFF),
            MemWrite(self.step_count, 0x0100 + ((self.s - 2) & 0xFF), pushed_p),
        )
        self.push_word(return_addr)
        self.push_byte(pushed_p)
        self.set_flag("I", True)
        vector = self.fetch_byte(0xFFFE) | (self.fetch_byte(0xFFFF) << 8)
        if vector == 0x0000:
            self.halted = True
        return 0, vector, writes

    def op_bpl(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        return self.branch(opcode, lo, not (self.p & FLAG_N))

    def op_bmi(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        return self.branch(opcode, lo, bool(self.p & FLAG_N))

    def op_bvc(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        return self.branch(opcode, lo, not (self.p & FLAG_V))

    def op_bvs(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        return self.branch(opcode, lo, bool(self.p & FLAG_V))

    def op_bcc(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        return self.branch(opcode, lo, not (self.p & FLAG_C))

    def op_bcs(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        return self.branch(opcode, lo, bool(self.p & FLAG_C))

    def op_bne(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        return self.branch(opcode, lo, not (self.p & FLAG_Z))

    def op_beq(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        return self.branch(opcode, lo, bool(self.p & FLAG_Z))

    def op_clc(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        self.set_flag("C", False)
        return 0, self.pc + opcode.length, ()

    def op_sec(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        self.set_flag("C", True)
        return 0, self.pc + opcode.length, ()

    def op_cli(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        self.set_flag("I", False)
        return 0, self.pc + opcode.length, ()

    def op_sei(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        self.set_flag("I", True)
        return 0, self.pc + opcode.length, ()

    def op_clv(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        self.set_flag("V", False)
        return 0, self.pc + opcode.length, ()

    def op_cld(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        self.set_flag("D", False)
        return 0, self.pc + opcode.length, ()

    def op_sed(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        self.set_flag("D", True)
        return 0, self.pc + opcode.length, ()

    def op_nop(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        return 0, self.pc + opcode.length, ()

    # ── Unofficial opcodes ──────────────────────────────────────────

    def op_lax(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        value = lo if opcode.mode == MODE_IMMEDIATE else self.fetch_byte(ea)
        self.a = self.x = value & 0xFF
        self.set_nz(self.a)
        return self.a, self.pc + opcode.length, ()

    def op_sax(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        value = self.a & self.x & 0xFF
        self.memory[ea] = value
        write = MemWrite(self.step_count, ea, value)
        return value, self.pc + opcode.length, (write,)

    def op_dcp(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        new_value = (self.fetch_byte(ea) - 1) & 0xFF
        self.memory[ea] = new_value
        write = MemWrite(self.step_count, ea, new_value)
        result = self.a - new_value
        self.set_flag("C", result >= 0)
        self.set_nz(result & 0xFF)
        return new_value, self.pc + opcode.length, (write,)

    def op_isb(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        new_value = (self.fetch_byte(ea) + 1) & 0xFF
        self.memory[ea] = new_value
        write = MemWrite(self.step_count, ea, new_value)
        self.sbc_apply(new_value)
        return new_value, self.pc + opcode.length, (write,)

    def op_slo(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        value = self.fetch_byte(ea)
        new_value = (value << 1) & 0xFF
        self.memory[ea] = new_value
        write = MemWrite(self.step_count, ea, new_value)
        self.set_flag("C", value & 0x80)
        self.a = (self.a | new_value) & 0xFF
        self.set_nz(self.a)
        return new_value, self.pc + opcode.length, (write,)

    def op_rla(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        value = self.fetch_byte(ea)
        carry = 1 if self.p & FLAG_C else 0
        new_value = ((value << 1) | carry) & 0xFF
        self.memory[ea] = new_value
        write = MemWrite(self.step_count, ea, new_value)
        self.set_flag("C", value & 0x80)
        self.a = (self.a & new_value) & 0xFF
        self.set_nz(self.a)
        return new_value, self.pc + opcode.length, (write,)

    def op_sre(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        value = self.fetch_byte(ea)
        new_value = (value >> 1) & 0xFF
        self.memory[ea] = new_value
        write = MemWrite(self.step_count, ea, new_value)
        self.set_flag("C", value & 0x01)
        self.a = (self.a ^ new_value) & 0xFF
        self.set_nz(self.a)
        return new_value, self.pc + opcode.length, (write,)

    def op_rra(self, opcode: Opcode, lo: int, hi: int, ea: int
                ) -> tuple[int, int, tuple[MemWrite, ...]]:
        value = self.fetch_byte(ea)
        carry = 1 if self.p & FLAG_C else 0
        new_value = ((value >> 1) | (carry << 7)) & 0xFF
        self.memory[ea] = new_value
        write = MemWrite(self.step_count, ea, new_value)
        # ADC consumes the carry that ROR just produced.
        self.set_flag("C", value & 0x01)
        self.adc_apply(new_value)
        return new_value, self.pc + opcode.length, (write,)


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
    import argparse

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
