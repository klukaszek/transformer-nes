"""6502 opcode table and flag definitions shared by the CPU implementations."""

from __future__ import annotations

import builtins
from dataclasses import dataclass

# ── Processor status flag bits (NV-BDIZC) ─────────────────────────
FLAG_N = 0b1000_0000  # negative
FLAG_V = 0b0100_0000  # overflow
FLAG_U = 0b0010_0000  # unused, always 1 on 6502
FLAG_B = 0b0001_0000  # break
FLAG_D = 0b0000_1000  # decimal
FLAG_I = 0b0000_0100  # interrupt disable
FLAG_Z = 0b0000_0010  # zero
FLAG_C = 0b0000_0001  # carry

FLAG_MASKS = {
    "N": FLAG_N,
    "V": FLAG_V,
    "U": FLAG_U,
    "B": FLAG_B,
    "D": FLAG_D,
    "I": FLAG_I,
    "Z": FLAG_Z,
    "C": FLAG_C,
}


@dataclass(frozen=True)
class Opcode:
    """A single 6502 opcode entry.

    The reference core and graph construction both consume this table so
    behavior is declared in exactly one place.
    """

    byte: int
    mnemonic: str
    mode: str
    cycles: int
    operation: str  # dispatch key used by the reference core and graph

    @property
    def length(self) -> int:
        """Instruction bytes: the opcode and its mode's operand."""
        return MODE_LENGTH[self.mode]


# Addressing mode names used in the table.
MODE_IMMEDIATE = "imm"
MODE_ZERO_PAGE = "zpg"
MODE_ZERO_PAGE_X = "zpx"
MODE_ZERO_PAGE_Y = "zpy"
MODE_ABSOLUTE = "abs"
MODE_ABSOLUTE_X = "abx"
MODE_ABSOLUTE_Y = "aby"
MODE_INDIRECT = "ind"
MODE_INDIRECT_X = "idx"
MODE_INDIRECT_Y = "idy"
MODE_RELATIVE = "rel"
MODE_IMPLIED = "imp"
MODE_ACCUMULATOR = "acc"
MODE_LENGTH = {
    MODE_IMPLIED: 1, MODE_ACCUMULATOR: 1,
    MODE_IMMEDIATE: 2, MODE_ZERO_PAGE: 2, MODE_ZERO_PAGE_X: 2, MODE_ZERO_PAGE_Y: 2,
    MODE_RELATIVE: 2, MODE_INDIRECT_X: 2, MODE_INDIRECT_Y: 2,
    MODE_ABSOLUTE: 3, MODE_ABSOLUTE_X: 3, MODE_ABSOLUTE_Y: 3, MODE_INDIRECT: 3,
}

# Reads that take a cycle more when an indexed address carries into the
# high byte; the unofficial absolute,X NOPs read too (nestest's CYC column).
PAGE_CROSS_READ_OPERATIONS = frozenset({
    "load_a", "load_x", "load_y", "adc", "sbc", "cmp",
    "and", "ora", "eor", "lax", "nop",
})
BRANCH_CONDITIONS = {
    "bpl": (FLAG_N, 0), "bmi": (FLAG_N, FLAG_N),
    "bvc": (FLAG_V, 0), "bvs": (FLAG_V, FLAG_V),
    "bcc": (FLAG_C, 0), "bcs": (FLAG_C, FLAG_C),
    "bne": (FLAG_Z, 0), "beq": (FLAG_Z, FLAG_Z),
}
BRANCH_OPERATIONS = frozenset(BRANCH_CONDITIONS)


# Every opcode the cores execute: the official NMOS set plus the stable
# unofficial ones nestest exercises.  A row is an instruction -- mnemonic
# and operation -- then each encoding as "mode opcode cycles"; the length
# follows from the mode.  ``cycles`` is the base timing; branch and
# indexed-read penalties are computed by the execution backends from the
# live operands and flags.  Order matters: the graph builds its decode
# gates in this order.
_OPCODE_TABLE = """
LDA load_a   imm A9 2  zpg A5 3  zpx B5 4  abs AD 4  abx BD 4  aby B9 4  idx A1 6  idy B1 5
LDX load_x   imm A2 2  zpg A6 3  zpy B6 4  abs AE 4  aby BE 4
LDY load_y   imm A0 2  zpg A4 3  zpx B4 4  abs AC 4  abx BC 4
STA store_a  zpg 85 3  zpx 95 4  abs 8D 4  abx 9D 5  aby 99 5  idx 81 6  idy 91 6
STX store_x  zpg 86 3  zpy 96 4  abs 8E 4
STY store_y  zpg 84 3  zpx 94 4  abs 8C 4
TAX tax      imp AA 2
TAY tay      imp A8 2
TXA txa      imp 8A 2
TYA tya      imp 98 2
TSX tsx      imp BA 2
TXS txs      imp 9A 2
PHA pha      imp 48 3
PHP php      imp 08 3
PLA pla      imp 68 4
PLP plp      imp 28 4
ADC adc      imm 69 2  zpg 65 3  zpx 75 4  abs 6D 4  abx 7D 4  aby 79 4  idx 61 6  idy 71 5
SBC sbc      imm E9 2  zpg E5 3  zpx F5 4  abs ED 4  abx FD 4  aby F9 4  idx E1 6  idy F1 5
CMP cmp      imm C9 2  zpg C5 3  zpx D5 4  abs CD 4  abx DD 4  aby D9 4  idx C1 6  idy D1 5
CPX cpx      imm E0 2  zpg E4 3  abs EC 4
CPY cpy      imm C0 2  zpg C4 3  abs CC 4
AND and      imm 29 2  zpg 25 3  zpx 35 4  abs 2D 4  abx 3D 4  aby 39 4  idx 21 6  idy 31 5
ORA ora      imm 09 2  zpg 05 3  zpx 15 4  abs 0D 4  abx 1D 4  aby 19 4  idx 01 6  idy 11 5
EOR eor      imm 49 2  zpg 45 3  zpx 55 4  abs 4D 4  abx 5D 4  aby 59 4  idx 41 6  idy 51 5
BIT bit      zpg 24 3  abs 2C 4
ASL asl      acc 0A 2  zpg 06 5  zpx 16 6  abs 0E 6  abx 1E 7
LSR lsr      acc 4A 2  zpg 46 5  zpx 56 6  abs 4E 6  abx 5E 7
ROL rol      acc 2A 2  zpg 26 5  zpx 36 6  abs 2E 6  abx 3E 7
ROR ror      acc 6A 2  zpg 66 5  zpx 76 6  abs 6E 6  abx 7E 7
INC inc      zpg E6 5  zpx F6 6  abs EE 6  abx FE 7
DEC dec      zpg C6 5  zpx D6 6  abs CE 6  abx DE 7
INX inx      imp E8 2
INY iny      imp C8 2
DEX dex      imp CA 2
DEY dey      imp 88 2
JMP jmp      abs 4C 3  ind 6C 5
JSR jsr      abs 20 6
RTS rts      imp 60 6
RTI rti      imp 40 6
BRK brk      imp 00 7
BPL bpl      rel 10 2
BMI bmi      rel 30 2
BVC bvc      rel 50 2
BVS bvs      rel 70 2
BCC bcc      rel 90 2
BCS bcs      rel B0 2
BNE bne      rel D0 2
BEQ beq      rel F0 2
CLC clc      imp 18 2
SEC sec      imp 38 2
CLI cli      imp 58 2
SEI sei      imp 78 2
CLV clv      imp B8 2
CLD cld      imp D8 2
SED sed      imp F8 2
NOP nop      imp EA 2
# ── Unofficial (stable) NMOS opcodes, as exercised by nestest ──────
# NOP variants that fetch (and discard) an operand.
NOP nop      imp 1A 2  imp 3A 2  imp 5A 2  imp 7A 2  imp DA 2  imp FA 2  imm 80 2  zpg 04 3
NOP nop      zpg 44 3  zpg 64 3  zpx 14 4  zpx 34 4  zpx 54 4  zpx 74 4  zpx D4 4  zpx F4 4
NOP nop      abs 0C 4  abx 1C 4  abx 3C 4  abx 5C 4  abx 7C 4  abx DC 4  abx FC 4
# LAX: load A and X together.
LAX lax      zpg A7 3  zpy B7 4  abs AF 4  aby BF 4  idx A3 6  idy B3 5
# SAX: store A & X.
SAX sax      zpg 87 3  zpy 97 4  abs 8F 4  idx 83 6
# SBC $EB: identical to official SBC immediate.
SBC sbc      imm EB 2
# DCP: DEC memory then CMP.
DCP dcp      zpg C7 5  zpx D7 6  abs CF 6  abx DF 7  aby DB 7  idx C3 8  idy D3 8
# ISB (aka ISC): INC memory then SBC.
ISB isb      zpg E7 5  zpx F7 6  abs EF 6  abx FF 7  aby FB 7  idx E3 8  idy F3 8
# SLO: ASL memory then ORA.
SLO slo      zpg 07 5  zpx 17 6  abs 0F 6  abx 1F 7  aby 1B 7  idx 03 8  idy 13 8
# RLA: ROL memory then AND.
RLA rla      zpg 27 5  zpx 37 6  abs 2F 6  abx 3F 7  aby 3B 7  idx 23 8  idy 33 8
# SRE: LSR memory then EOR.
SRE sre      zpg 47 5  zpx 57 6  abs 4F 6  abx 5F 7  aby 5B 7  idx 43 8  idy 53 8
# RRA: ROR memory then ADC.
RRA rra      zpg 67 5  zpx 77 6  abs 6F 6  abx 7F 7  aby 7B 7  idx 63 8  idy 73 8
"""


def _parse_opcodes(table: str) -> dict[int, Opcode]:
    opcodes = {}
    for line in table.strip().splitlines():
        if line.startswith("#"):
            continue
        mnemonic, operation, *cells = line.split()
        for mode, byte, cycles in zip(*[iter(cells)] * 3, strict=True):
            opcodes[int(byte, 16)] = Opcode(int(byte, 16), mnemonic, mode, int(cycles), operation)
    return opcodes


OPCODES: dict[int, Opcode] = _parse_opcodes(_OPCODE_TABLE)


def opcode_for(byte: int) -> Opcode:
    """Return the opcode entry for a byte, raising on unsupported bytes."""
    try:
        return OPCODES[byte]
    except KeyError as exc:
        raise ValueError(f"unsupported opcode 0x{byte:02X}") from exc


# ── MMIO classification ────────────────────────────────────────────
# Operations whose execution reads their memory operand, so with a memory
# addressing mode they can trigger a device read.  RMW ops count (they read
# before writing); so does NOP, whose unofficial memory modes issue real
# bus reads on hardware.
READS_MEMORY = frozenset({
    "load_a", "load_x", "load_y", "adc", "sbc", "cmp", "cpx", "cpy",
    "and", "ora", "eor", "bit", "asl", "lsr", "rol", "ror", "inc", "dec",
    "lax", "dcp", "isb", "slo", "rla", "sre", "rra", "nop",
})

# Addressing modes that access a memory operand (immediate, accumulator,
# implied, and relative never touch the bus for their operand).
MEMORY_MODES = frozenset({
    MODE_ZERO_PAGE, MODE_ZERO_PAGE_X, MODE_ZERO_PAGE_Y,
    MODE_ABSOLUTE, MODE_ABSOLUTE_X, MODE_ABSOLUTE_Y,
    MODE_INDIRECT_X, MODE_INDIRECT_Y,
})


# ── Tiny assembler helpers (for tests / examples) ─────────────────

class Asm:
    """Minimal assembler for the initial opcode subset.

    Usage:
        program = (
            Asm()
            .lda_imm(0x42)
            .sta_zpg(0x10)
            .brk()
            .bytes()
        )
    """

    def __init__(self):
        self.buf: bytearray = bytearray()

    def append(self, *values: int) -> Asm:
        for v in values:
            if not 0 <= v <= 0xFF:
                raise ValueError(f"byte out of range: {v}")
            self.buf.append(v)
        return self

    def bytes(self) -> builtins.bytes:
        return bytes(self.buf)

    # Load / store
    def lda_imm(self, value: int) -> Asm:
        return self.append(0xA9, value)

    def ldx_imm(self, value: int) -> Asm:
        return self.append(0xA2, value)

    def ldy_imm(self, value: int) -> Asm:
        return self.append(0xA0, value)

    def sta_zpg(self, addr: int) -> Asm:
        return self.append(0x85, addr)

    def sta_abs(self, addr: int) -> Asm:
        return self.append(0x8D, addr & 0xFF, (addr >> 8) & 0xFF)

    # Arithmetic
    def adc_imm(self, value: int) -> Asm:
        return self.append(0x69, value)

    def sbc_imm(self, value: int) -> Asm:
        return self.append(0xE9, value)

    # Register inc/dec/transfer
    def inx(self) -> Asm:
        return self.append(0xE8)

    def dex(self) -> Asm:
        return self.append(0xCA)

    def tax(self) -> Asm:
        return self.append(0xAA)

    # Flow control
    def jmp_abs(self, addr: int) -> Asm:
        return self.append(0x4C, addr & 0xFF, (addr >> 8) & 0xFF)

    def beq(self, offset: int) -> Asm:
        if not -128 <= offset <= 127:
            raise ValueError(f"branch offset out of range: {offset}")
        return self.append(0xF0, offset & 0xFF)

    def brk(self) -> Asm:
        return self.append(0x00)


def asm() -> Asm:
    """Return a fresh assembler builder."""
    return Asm()


def written_addresses(addr: int, mirror_cpu_ram: bool) -> range:
    """The bytes a CPU write to `addr` lands in: with the NES's RAM
    mirroring, a write below $2000 reaches all four 2 KiB copies."""
    addr &= 0xFFFF
    if mirror_cpu_ram and addr < 0x2000:
        return range(addr & 0x07FF, 0x2000, 0x0800)
    return range(addr, addr + 1)
