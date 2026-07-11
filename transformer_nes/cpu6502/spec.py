"""6502 opcode table and flag definitions shared by reference and graph code."""

from __future__ import annotations

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

    The reference core and the graph interpreter both consume this table so
    behavior is declared in exactly one place.
    """

    byte: int
    mnemonic: str
    mode: str
    length: int
    cycles: int
    operation: str  # dispatch key used by the reference core and graph
    flags: tuple[str, ...]  # flags that the operation may update


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


# Full official NMOS 6502 opcode set (151 opcodes).  Cycle counts are not
# modeled exactly; they are kept as the nominal cycle count for reference only.
OPCODES: dict[int, Opcode] = {
    0xA9: Opcode(0xA9, "LDA", MODE_IMMEDIATE, 2, 2, "load_a", ("N", "Z")),
    0xA5: Opcode(0xA5, "LDA", MODE_ZERO_PAGE, 2, 3, "load_a", ("N", "Z")),
    0xB5: Opcode(0xB5, "LDA", MODE_ZERO_PAGE_X, 2, 4, "load_a", ("N", "Z")),
    0xAD: Opcode(0xAD, "LDA", MODE_ABSOLUTE, 3, 4, "load_a", ("N", "Z")),
    0xBD: Opcode(0xBD, "LDA", MODE_ABSOLUTE_X, 3, 4, "load_a", ("N", "Z")),
    0xB9: Opcode(0xB9, "LDA", MODE_ABSOLUTE_Y, 3, 4, "load_a", ("N", "Z")),
    0xA1: Opcode(0xA1, "LDA", MODE_INDIRECT_X, 2, 6, "load_a", ("N", "Z")),
    0xB1: Opcode(0xB1, "LDA", MODE_INDIRECT_Y, 2, 5, "load_a", ("N", "Z")),
    0xA2: Opcode(0xA2, "LDX", MODE_IMMEDIATE, 2, 2, "load_x", ("N", "Z")),
    0xA6: Opcode(0xA6, "LDX", MODE_ZERO_PAGE, 2, 3, "load_x", ("N", "Z")),
    0xB6: Opcode(0xB6, "LDX", MODE_ZERO_PAGE_Y, 2, 4, "load_x", ("N", "Z")),
    0xAE: Opcode(0xAE, "LDX", MODE_ABSOLUTE, 3, 4, "load_x", ("N", "Z")),
    0xBE: Opcode(0xBE, "LDX", MODE_ABSOLUTE_Y, 3, 4, "load_x", ("N", "Z")),
    0xA0: Opcode(0xA0, "LDY", MODE_IMMEDIATE, 2, 2, "load_y", ("N", "Z")),
    0xA4: Opcode(0xA4, "LDY", MODE_ZERO_PAGE, 2, 3, "load_y", ("N", "Z")),
    0xB4: Opcode(0xB4, "LDY", MODE_ZERO_PAGE_X, 2, 4, "load_y", ("N", "Z")),
    0xAC: Opcode(0xAC, "LDY", MODE_ABSOLUTE, 3, 4, "load_y", ("N", "Z")),
    0xBC: Opcode(0xBC, "LDY", MODE_ABSOLUTE_X, 3, 4, "load_y", ("N", "Z")),
    0x85: Opcode(0x85, "STA", MODE_ZERO_PAGE, 2, 3, "store_a", ()),
    0x95: Opcode(0x95, "STA", MODE_ZERO_PAGE_X, 2, 4, "store_a", ()),
    0x8D: Opcode(0x8D, "STA", MODE_ABSOLUTE, 3, 4, "store_a", ()),
    0x9D: Opcode(0x9D, "STA", MODE_ABSOLUTE_X, 3, 5, "store_a", ()),
    0x99: Opcode(0x99, "STA", MODE_ABSOLUTE_Y, 3, 5, "store_a", ()),
    0x81: Opcode(0x81, "STA", MODE_INDIRECT_X, 2, 6, "store_a", ()),
    0x91: Opcode(0x91, "STA", MODE_INDIRECT_Y, 2, 6, "store_a", ()),
    0x86: Opcode(0x86, "STX", MODE_ZERO_PAGE, 2, 3, "store_x", ()),
    0x96: Opcode(0x96, "STX", MODE_ZERO_PAGE_Y, 2, 4, "store_x", ()),
    0x8E: Opcode(0x8E, "STX", MODE_ABSOLUTE, 3, 4, "store_x", ()),
    0x84: Opcode(0x84, "STY", MODE_ZERO_PAGE, 2, 3, "store_y", ()),
    0x94: Opcode(0x94, "STY", MODE_ZERO_PAGE_X, 2, 4, "store_y", ()),
    0x8C: Opcode(0x8C, "STY", MODE_ABSOLUTE, 3, 4, "store_y", ()),
    0xAA: Opcode(0xAA, "TAX", MODE_IMPLIED, 1, 2, "tax", ("N", "Z")),
    0xA8: Opcode(0xA8, "TAY", MODE_IMPLIED, 1, 2, "tay", ("N", "Z")),
    0x8A: Opcode(0x8A, "TXA", MODE_IMPLIED, 1, 2, "txa", ("N", "Z")),
    0x98: Opcode(0x98, "TYA", MODE_IMPLIED, 1, 2, "tya", ("N", "Z")),
    0xBA: Opcode(0xBA, "TSX", MODE_IMPLIED, 1, 2, "tsx", ("N", "Z")),
    0x9A: Opcode(0x9A, "TXS", MODE_IMPLIED, 1, 2, "txs", ()),
    0x48: Opcode(0x48, "PHA", MODE_IMPLIED, 1, 3, "pha", ()),
    0x08: Opcode(0x08, "PHP", MODE_IMPLIED, 1, 3, "php", ()),
    0x68: Opcode(0x68, "PLA", MODE_IMPLIED, 1, 4, "pla", ("N", "Z")),
    0x28: Opcode(0x28, "PLP", MODE_IMPLIED, 1, 4, "plp", ("N", "V", "B", "D", "I", "Z", "C")),
    0x69: Opcode(0x69, "ADC", MODE_IMMEDIATE, 2, 2, "adc", ("N", "V", "Z", "C")),
    0x65: Opcode(0x65, "ADC", MODE_ZERO_PAGE, 2, 3, "adc", ("N", "V", "Z", "C")),
    0x75: Opcode(0x75, "ADC", MODE_ZERO_PAGE_X, 2, 4, "adc", ("N", "V", "Z", "C")),
    0x6D: Opcode(0x6D, "ADC", MODE_ABSOLUTE, 3, 4, "adc", ("N", "V", "Z", "C")),
    0x7D: Opcode(0x7D, "ADC", MODE_ABSOLUTE_X, 3, 4, "adc", ("N", "V", "Z", "C")),
    0x79: Opcode(0x79, "ADC", MODE_ABSOLUTE_Y, 3, 4, "adc", ("N", "V", "Z", "C")),
    0x61: Opcode(0x61, "ADC", MODE_INDIRECT_X, 2, 6, "adc", ("N", "V", "Z", "C")),
    0x71: Opcode(0x71, "ADC", MODE_INDIRECT_Y, 2, 5, "adc", ("N", "V", "Z", "C")),
    0xE9: Opcode(0xE9, "SBC", MODE_IMMEDIATE, 2, 2, "sbc", ("N", "V", "Z", "C")),
    0xE5: Opcode(0xE5, "SBC", MODE_ZERO_PAGE, 2, 3, "sbc", ("N", "V", "Z", "C")),
    0xF5: Opcode(0xF5, "SBC", MODE_ZERO_PAGE_X, 2, 4, "sbc", ("N", "V", "Z", "C")),
    0xED: Opcode(0xED, "SBC", MODE_ABSOLUTE, 3, 4, "sbc", ("N", "V", "Z", "C")),
    0xFD: Opcode(0xFD, "SBC", MODE_ABSOLUTE_X, 3, 4, "sbc", ("N", "V", "Z", "C")),
    0xF9: Opcode(0xF9, "SBC", MODE_ABSOLUTE_Y, 3, 4, "sbc", ("N", "V", "Z", "C")),
    0xE1: Opcode(0xE1, "SBC", MODE_INDIRECT_X, 2, 6, "sbc", ("N", "V", "Z", "C")),
    0xF1: Opcode(0xF1, "SBC", MODE_INDIRECT_Y, 2, 5, "sbc", ("N", "V", "Z", "C")),
    0xC9: Opcode(0xC9, "CMP", MODE_IMMEDIATE, 2, 2, "cmp", ("N", "Z", "C")),
    0xC5: Opcode(0xC5, "CMP", MODE_ZERO_PAGE, 2, 3, "cmp", ("N", "Z", "C")),
    0xD5: Opcode(0xD5, "CMP", MODE_ZERO_PAGE_X, 2, 4, "cmp", ("N", "Z", "C")),
    0xCD: Opcode(0xCD, "CMP", MODE_ABSOLUTE, 3, 4, "cmp", ("N", "Z", "C")),
    0xDD: Opcode(0xDD, "CMP", MODE_ABSOLUTE_X, 3, 4, "cmp", ("N", "Z", "C")),
    0xD9: Opcode(0xD9, "CMP", MODE_ABSOLUTE_Y, 3, 4, "cmp", ("N", "Z", "C")),
    0xC1: Opcode(0xC1, "CMP", MODE_INDIRECT_X, 2, 6, "cmp", ("N", "Z", "C")),
    0xD1: Opcode(0xD1, "CMP", MODE_INDIRECT_Y, 2, 5, "cmp", ("N", "Z", "C")),
    0xE0: Opcode(0xE0, "CPX", MODE_IMMEDIATE, 2, 2, "cpx", ("N", "Z", "C")),
    0xE4: Opcode(0xE4, "CPX", MODE_ZERO_PAGE, 2, 3, "cpx", ("N", "Z", "C")),
    0xEC: Opcode(0xEC, "CPX", MODE_ABSOLUTE, 3, 4, "cpx", ("N", "Z", "C")),
    0xC0: Opcode(0xC0, "CPY", MODE_IMMEDIATE, 2, 2, "cpy", ("N", "Z", "C")),
    0xC4: Opcode(0xC4, "CPY", MODE_ZERO_PAGE, 2, 3, "cpy", ("N", "Z", "C")),
    0xCC: Opcode(0xCC, "CPY", MODE_ABSOLUTE, 3, 4, "cpy", ("N", "Z", "C")),
    0x29: Opcode(0x29, "AND", MODE_IMMEDIATE, 2, 2, "and", ("N", "Z")),
    0x25: Opcode(0x25, "AND", MODE_ZERO_PAGE, 2, 3, "and", ("N", "Z")),
    0x35: Opcode(0x35, "AND", MODE_ZERO_PAGE_X, 2, 4, "and", ("N", "Z")),
    0x2D: Opcode(0x2D, "AND", MODE_ABSOLUTE, 3, 4, "and", ("N", "Z")),
    0x3D: Opcode(0x3D, "AND", MODE_ABSOLUTE_X, 3, 4, "and", ("N", "Z")),
    0x39: Opcode(0x39, "AND", MODE_ABSOLUTE_Y, 3, 4, "and", ("N", "Z")),
    0x21: Opcode(0x21, "AND", MODE_INDIRECT_X, 2, 6, "and", ("N", "Z")),
    0x31: Opcode(0x31, "AND", MODE_INDIRECT_Y, 2, 5, "and", ("N", "Z")),
    0x09: Opcode(0x09, "ORA", MODE_IMMEDIATE, 2, 2, "ora", ("N", "Z")),
    0x05: Opcode(0x05, "ORA", MODE_ZERO_PAGE, 2, 3, "ora", ("N", "Z")),
    0x15: Opcode(0x15, "ORA", MODE_ZERO_PAGE_X, 2, 4, "ora", ("N", "Z")),
    0x0D: Opcode(0x0D, "ORA", MODE_ABSOLUTE, 3, 4, "ora", ("N", "Z")),
    0x1D: Opcode(0x1D, "ORA", MODE_ABSOLUTE_X, 3, 4, "ora", ("N", "Z")),
    0x19: Opcode(0x19, "ORA", MODE_ABSOLUTE_Y, 3, 4, "ora", ("N", "Z")),
    0x01: Opcode(0x01, "ORA", MODE_INDIRECT_X, 2, 6, "ora", ("N", "Z")),
    0x11: Opcode(0x11, "ORA", MODE_INDIRECT_Y, 2, 5, "ora", ("N", "Z")),
    0x49: Opcode(0x49, "EOR", MODE_IMMEDIATE, 2, 2, "eor", ("N", "Z")),
    0x45: Opcode(0x45, "EOR", MODE_ZERO_PAGE, 2, 3, "eor", ("N", "Z")),
    0x55: Opcode(0x55, "EOR", MODE_ZERO_PAGE_X, 2, 4, "eor", ("N", "Z")),
    0x4D: Opcode(0x4D, "EOR", MODE_ABSOLUTE, 3, 4, "eor", ("N", "Z")),
    0x5D: Opcode(0x5D, "EOR", MODE_ABSOLUTE_X, 3, 4, "eor", ("N", "Z")),
    0x59: Opcode(0x59, "EOR", MODE_ABSOLUTE_Y, 3, 4, "eor", ("N", "Z")),
    0x41: Opcode(0x41, "EOR", MODE_INDIRECT_X, 2, 6, "eor", ("N", "Z")),
    0x51: Opcode(0x51, "EOR", MODE_INDIRECT_Y, 2, 5, "eor", ("N", "Z")),
    0x24: Opcode(0x24, "BIT", MODE_ZERO_PAGE, 2, 3, "bit", ("N", "V", "Z")),
    0x2C: Opcode(0x2C, "BIT", MODE_ABSOLUTE, 3, 4, "bit", ("N", "V", "Z")),
    0x0A: Opcode(0x0A, "ASL", MODE_ACCUMULATOR, 1, 2, "asl", ("N", "Z", "C")),
    0x06: Opcode(0x06, "ASL", MODE_ZERO_PAGE, 2, 5, "asl", ("N", "Z", "C")),
    0x16: Opcode(0x16, "ASL", MODE_ZERO_PAGE_X, 2, 6, "asl", ("N", "Z", "C")),
    0x0E: Opcode(0x0E, "ASL", MODE_ABSOLUTE, 3, 6, "asl", ("N", "Z", "C")),
    0x1E: Opcode(0x1E, "ASL", MODE_ABSOLUTE_X, 3, 7, "asl", ("N", "Z", "C")),
    0x4A: Opcode(0x4A, "LSR", MODE_ACCUMULATOR, 1, 2, "lsr", ("N", "Z", "C")),
    0x46: Opcode(0x46, "LSR", MODE_ZERO_PAGE, 2, 5, "lsr", ("N", "Z", "C")),
    0x56: Opcode(0x56, "LSR", MODE_ZERO_PAGE_X, 2, 6, "lsr", ("N", "Z", "C")),
    0x4E: Opcode(0x4E, "LSR", MODE_ABSOLUTE, 3, 6, "lsr", ("N", "Z", "C")),
    0x5E: Opcode(0x5E, "LSR", MODE_ABSOLUTE_X, 3, 7, "lsr", ("N", "Z", "C")),
    0x2A: Opcode(0x2A, "ROL", MODE_ACCUMULATOR, 1, 2, "rol", ("N", "Z", "C")),
    0x26: Opcode(0x26, "ROL", MODE_ZERO_PAGE, 2, 5, "rol", ("N", "Z", "C")),
    0x36: Opcode(0x36, "ROL", MODE_ZERO_PAGE_X, 2, 6, "rol", ("N", "Z", "C")),
    0x2E: Opcode(0x2E, "ROL", MODE_ABSOLUTE, 3, 6, "rol", ("N", "Z", "C")),
    0x3E: Opcode(0x3E, "ROL", MODE_ABSOLUTE_X, 3, 7, "rol", ("N", "Z", "C")),
    0x6A: Opcode(0x6A, "ROR", MODE_ACCUMULATOR, 1, 2, "ror", ("N", "Z", "C")),
    0x66: Opcode(0x66, "ROR", MODE_ZERO_PAGE, 2, 5, "ror", ("N", "Z", "C")),
    0x76: Opcode(0x76, "ROR", MODE_ZERO_PAGE_X, 2, 6, "ror", ("N", "Z", "C")),
    0x6E: Opcode(0x6E, "ROR", MODE_ABSOLUTE, 3, 6, "ror", ("N", "Z", "C")),
    0x7E: Opcode(0x7E, "ROR", MODE_ABSOLUTE_X, 3, 7, "ror", ("N", "Z", "C")),
    0xE6: Opcode(0xE6, "INC", MODE_ZERO_PAGE, 2, 5, "inc", ("N", "Z")),
    0xF6: Opcode(0xF6, "INC", MODE_ZERO_PAGE_X, 2, 6, "inc", ("N", "Z")),
    0xEE: Opcode(0xEE, "INC", MODE_ABSOLUTE, 3, 6, "inc", ("N", "Z")),
    0xFE: Opcode(0xFE, "INC", MODE_ABSOLUTE_X, 3, 7, "inc", ("N", "Z")),
    0xC6: Opcode(0xC6, "DEC", MODE_ZERO_PAGE, 2, 5, "dec", ("N", "Z")),
    0xD6: Opcode(0xD6, "DEC", MODE_ZERO_PAGE_X, 2, 6, "dec", ("N", "Z")),
    0xCE: Opcode(0xCE, "DEC", MODE_ABSOLUTE, 3, 6, "dec", ("N", "Z")),
    0xDE: Opcode(0xDE, "DEC", MODE_ABSOLUTE_X, 3, 7, "dec", ("N", "Z")),
    0xE8: Opcode(0xE8, "INX", MODE_IMPLIED, 1, 2, "inx", ("N", "Z")),
    0xC8: Opcode(0xC8, "INY", MODE_IMPLIED, 1, 2, "iny", ("N", "Z")),
    0xCA: Opcode(0xCA, "DEX", MODE_IMPLIED, 1, 2, "dex", ("N", "Z")),
    0x88: Opcode(0x88, "DEY", MODE_IMPLIED, 1, 2, "dey", ("N", "Z")),
    0x4C: Opcode(0x4C, "JMP", MODE_ABSOLUTE, 3, 3, "jmp", ()),
    0x6C: Opcode(0x6C, "JMP", MODE_INDIRECT, 3, 5, "jmp", ()),
    0x20: Opcode(0x20, "JSR", MODE_ABSOLUTE, 3, 6, "jsr", ()),
    0x60: Opcode(0x60, "RTS", MODE_IMPLIED, 1, 6, "rts", ()),
    0x40: Opcode(0x40, "RTI", MODE_IMPLIED, 1, 6, "rti", ()),
    0x00: Opcode(0x00, "BRK", MODE_IMPLIED, 1, 7, "brk", ("B",)),
    0x10: Opcode(0x10, "BPL", MODE_RELATIVE, 2, 2, "bpl", ()),
    0x30: Opcode(0x30, "BMI", MODE_RELATIVE, 2, 2, "bmi", ()),
    0x50: Opcode(0x50, "BVC", MODE_RELATIVE, 2, 2, "bvc", ()),
    0x70: Opcode(0x70, "BVS", MODE_RELATIVE, 2, 2, "bvs", ()),
    0x90: Opcode(0x90, "BCC", MODE_RELATIVE, 2, 2, "bcc", ()),
    0xB0: Opcode(0xB0, "BCS", MODE_RELATIVE, 2, 2, "bcs", ()),
    0xD0: Opcode(0xD0, "BNE", MODE_RELATIVE, 2, 2, "bne", ()),
    0xF0: Opcode(0xF0, "BEQ", MODE_RELATIVE, 2, 2, "beq", ()),
    0x18: Opcode(0x18, "CLC", MODE_IMPLIED, 1, 2, "clc", ("C",)),
    0x38: Opcode(0x38, "SEC", MODE_IMPLIED, 1, 2, "sec", ("C",)),
    0x58: Opcode(0x58, "CLI", MODE_IMPLIED, 1, 2, "cli", ("I",)),
    0x78: Opcode(0x78, "SEI", MODE_IMPLIED, 1, 2, "sei", ("I",)),
    0xB8: Opcode(0xB8, "CLV", MODE_IMPLIED, 1, 2, "clv", ("V",)),
    0xD8: Opcode(0xD8, "CLD", MODE_IMPLIED, 1, 2, "cld", ("D",)),
    0xF8: Opcode(0xF8, "SED", MODE_IMPLIED, 1, 2, "sed", ("D",)),
    0xEA: Opcode(0xEA, "NOP", MODE_IMPLIED, 1, 2, "nop", ()),
    # ── Unofficial (stable) NMOS opcodes, as exercised by nestest ──────
    # NOP variants that fetch (and discard) an operand.
    0x1A: Opcode(0x1A, "NOP", MODE_IMPLIED, 1, 2, "nop", ()),
    0x3A: Opcode(0x3A, "NOP", MODE_IMPLIED, 1, 2, "nop", ()),
    0x5A: Opcode(0x5A, "NOP", MODE_IMPLIED, 1, 2, "nop", ()),
    0x7A: Opcode(0x7A, "NOP", MODE_IMPLIED, 1, 2, "nop", ()),
    0xDA: Opcode(0xDA, "NOP", MODE_IMPLIED, 1, 2, "nop", ()),
    0xFA: Opcode(0xFA, "NOP", MODE_IMPLIED, 1, 2, "nop", ()),
    0x80: Opcode(0x80, "NOP", MODE_IMMEDIATE, 2, 2, "nop", ()),
    0x04: Opcode(0x04, "NOP", MODE_ZERO_PAGE, 2, 3, "nop", ()),
    0x44: Opcode(0x44, "NOP", MODE_ZERO_PAGE, 2, 3, "nop", ()),
    0x64: Opcode(0x64, "NOP", MODE_ZERO_PAGE, 2, 3, "nop", ()),
    0x14: Opcode(0x14, "NOP", MODE_ZERO_PAGE_X, 2, 4, "nop", ()),
    0x34: Opcode(0x34, "NOP", MODE_ZERO_PAGE_X, 2, 4, "nop", ()),
    0x54: Opcode(0x54, "NOP", MODE_ZERO_PAGE_X, 2, 4, "nop", ()),
    0x74: Opcode(0x74, "NOP", MODE_ZERO_PAGE_X, 2, 4, "nop", ()),
    0xD4: Opcode(0xD4, "NOP", MODE_ZERO_PAGE_X, 2, 4, "nop", ()),
    0xF4: Opcode(0xF4, "NOP", MODE_ZERO_PAGE_X, 2, 4, "nop", ()),
    0x0C: Opcode(0x0C, "NOP", MODE_ABSOLUTE, 3, 4, "nop", ()),
    0x1C: Opcode(0x1C, "NOP", MODE_ABSOLUTE_X, 3, 4, "nop", ()),
    0x3C: Opcode(0x3C, "NOP", MODE_ABSOLUTE_X, 3, 4, "nop", ()),
    0x5C: Opcode(0x5C, "NOP", MODE_ABSOLUTE_X, 3, 4, "nop", ()),
    0x7C: Opcode(0x7C, "NOP", MODE_ABSOLUTE_X, 3, 4, "nop", ()),
    0xDC: Opcode(0xDC, "NOP", MODE_ABSOLUTE_X, 3, 4, "nop", ()),
    0xFC: Opcode(0xFC, "NOP", MODE_ABSOLUTE_X, 3, 4, "nop", ()),
    # LAX: load A and X together.
    0xA7: Opcode(0xA7, "LAX", MODE_ZERO_PAGE, 2, 3, "lax", ("N", "Z")),
    0xB7: Opcode(0xB7, "LAX", MODE_ZERO_PAGE_Y, 2, 4, "lax", ("N", "Z")),
    0xAF: Opcode(0xAF, "LAX", MODE_ABSOLUTE, 3, 4, "lax", ("N", "Z")),
    0xBF: Opcode(0xBF, "LAX", MODE_ABSOLUTE_Y, 3, 4, "lax", ("N", "Z")),
    0xA3: Opcode(0xA3, "LAX", MODE_INDIRECT_X, 2, 6, "lax", ("N", "Z")),
    0xB3: Opcode(0xB3, "LAX", MODE_INDIRECT_Y, 2, 5, "lax", ("N", "Z")),
    # SAX: store A & X.
    0x87: Opcode(0x87, "SAX", MODE_ZERO_PAGE, 2, 3, "sax", ()),
    0x97: Opcode(0x97, "SAX", MODE_ZERO_PAGE_Y, 2, 4, "sax", ()),
    0x8F: Opcode(0x8F, "SAX", MODE_ABSOLUTE, 3, 4, "sax", ()),
    0x83: Opcode(0x83, "SAX", MODE_INDIRECT_X, 2, 6, "sax", ()),
    # SBC $EB: identical to official SBC immediate.
    0xEB: Opcode(0xEB, "SBC", MODE_IMMEDIATE, 2, 2, "sbc", ("N", "V", "Z", "C")),
    # DCP: DEC memory then CMP.
    0xC7: Opcode(0xC7, "DCP", MODE_ZERO_PAGE, 2, 5, "dcp", ("N", "Z", "C")),
    0xD7: Opcode(0xD7, "DCP", MODE_ZERO_PAGE_X, 2, 6, "dcp", ("N", "Z", "C")),
    0xCF: Opcode(0xCF, "DCP", MODE_ABSOLUTE, 3, 6, "dcp", ("N", "Z", "C")),
    0xDF: Opcode(0xDF, "DCP", MODE_ABSOLUTE_X, 3, 7, "dcp", ("N", "Z", "C")),
    0xDB: Opcode(0xDB, "DCP", MODE_ABSOLUTE_Y, 3, 7, "dcp", ("N", "Z", "C")),
    0xC3: Opcode(0xC3, "DCP", MODE_INDIRECT_X, 2, 8, "dcp", ("N", "Z", "C")),
    0xD3: Opcode(0xD3, "DCP", MODE_INDIRECT_Y, 2, 8, "dcp", ("N", "Z", "C")),
    # ISB (aka ISC): INC memory then SBC.
    0xE7: Opcode(0xE7, "ISB", MODE_ZERO_PAGE, 2, 5, "isb", ("N", "V", "Z", "C")),
    0xF7: Opcode(0xF7, "ISB", MODE_ZERO_PAGE_X, 2, 6, "isb", ("N", "V", "Z", "C")),
    0xEF: Opcode(0xEF, "ISB", MODE_ABSOLUTE, 3, 6, "isb", ("N", "V", "Z", "C")),
    0xFF: Opcode(0xFF, "ISB", MODE_ABSOLUTE_X, 3, 7, "isb", ("N", "V", "Z", "C")),
    0xFB: Opcode(0xFB, "ISB", MODE_ABSOLUTE_Y, 3, 7, "isb", ("N", "V", "Z", "C")),
    0xE3: Opcode(0xE3, "ISB", MODE_INDIRECT_X, 2, 8, "isb", ("N", "V", "Z", "C")),
    0xF3: Opcode(0xF3, "ISB", MODE_INDIRECT_Y, 2, 8, "isb", ("N", "V", "Z", "C")),
    # SLO: ASL memory then ORA.
    0x07: Opcode(0x07, "SLO", MODE_ZERO_PAGE, 2, 5, "slo", ("N", "Z", "C")),
    0x17: Opcode(0x17, "SLO", MODE_ZERO_PAGE_X, 2, 6, "slo", ("N", "Z", "C")),
    0x0F: Opcode(0x0F, "SLO", MODE_ABSOLUTE, 3, 6, "slo", ("N", "Z", "C")),
    0x1F: Opcode(0x1F, "SLO", MODE_ABSOLUTE_X, 3, 7, "slo", ("N", "Z", "C")),
    0x1B: Opcode(0x1B, "SLO", MODE_ABSOLUTE_Y, 3, 7, "slo", ("N", "Z", "C")),
    0x03: Opcode(0x03, "SLO", MODE_INDIRECT_X, 2, 8, "slo", ("N", "Z", "C")),
    0x13: Opcode(0x13, "SLO", MODE_INDIRECT_Y, 2, 8, "slo", ("N", "Z", "C")),
    # RLA: ROL memory then AND.
    0x27: Opcode(0x27, "RLA", MODE_ZERO_PAGE, 2, 5, "rla", ("N", "Z", "C")),
    0x37: Opcode(0x37, "RLA", MODE_ZERO_PAGE_X, 2, 6, "rla", ("N", "Z", "C")),
    0x2F: Opcode(0x2F, "RLA", MODE_ABSOLUTE, 3, 6, "rla", ("N", "Z", "C")),
    0x3F: Opcode(0x3F, "RLA", MODE_ABSOLUTE_X, 3, 7, "rla", ("N", "Z", "C")),
    0x3B: Opcode(0x3B, "RLA", MODE_ABSOLUTE_Y, 3, 7, "rla", ("N", "Z", "C")),
    0x23: Opcode(0x23, "RLA", MODE_INDIRECT_X, 2, 8, "rla", ("N", "Z", "C")),
    0x33: Opcode(0x33, "RLA", MODE_INDIRECT_Y, 2, 8, "rla", ("N", "Z", "C")),
    # SRE: LSR memory then EOR.
    0x47: Opcode(0x47, "SRE", MODE_ZERO_PAGE, 2, 5, "sre", ("N", "Z", "C")),
    0x57: Opcode(0x57, "SRE", MODE_ZERO_PAGE_X, 2, 6, "sre", ("N", "Z", "C")),
    0x4F: Opcode(0x4F, "SRE", MODE_ABSOLUTE, 3, 6, "sre", ("N", "Z", "C")),
    0x5F: Opcode(0x5F, "SRE", MODE_ABSOLUTE_X, 3, 7, "sre", ("N", "Z", "C")),
    0x5B: Opcode(0x5B, "SRE", MODE_ABSOLUTE_Y, 3, 7, "sre", ("N", "Z", "C")),
    0x43: Opcode(0x43, "SRE", MODE_INDIRECT_X, 2, 8, "sre", ("N", "Z", "C")),
    0x53: Opcode(0x53, "SRE", MODE_INDIRECT_Y, 2, 8, "sre", ("N", "Z", "C")),
    # RRA: ROR memory then ADC.
    0x67: Opcode(0x67, "RRA", MODE_ZERO_PAGE, 2, 5, "rra", ("N", "V", "Z", "C")),
    0x77: Opcode(0x77, "RRA", MODE_ZERO_PAGE_X, 2, 6, "rra", ("N", "V", "Z", "C")),
    0x6F: Opcode(0x6F, "RRA", MODE_ABSOLUTE, 3, 6, "rra", ("N", "V", "Z", "C")),
    0x7F: Opcode(0x7F, "RRA", MODE_ABSOLUTE_X, 3, 7, "rra", ("N", "V", "Z", "C")),
    0x7B: Opcode(0x7B, "RRA", MODE_ABSOLUTE_Y, 3, 7, "rra", ("N", "V", "Z", "C")),
    0x63: Opcode(0x63, "RRA", MODE_INDIRECT_X, 2, 8, "rra", ("N", "V", "Z", "C")),
    0x73: Opcode(0x73, "RRA", MODE_INDIRECT_Y, 2, 8, "rra", ("N", "V", "Z", "C")),
}


def opcode_for(byte: int) -> Opcode:
    """Return the opcode entry for a byte, raising on unsupported bytes."""
    try:
        return OPCODES[byte]
    except KeyError as exc:
        raise ValueError(f"unsupported opcode 0x{byte:02X}") from exc


# ── MMIO classification ────────────────────────────────────────────
# Operations whose execution reads their memory operand.  Combined with a
# memory addressing mode, these are the instructions that can trigger a
# device read on the bus.  Stores (STA/STX/STY/SAX), control flow, and
# stack operations are excluded; read-modify-write ops are included (they
# read before writing, and device registers observe that read).  NOP is
# included because the unofficial memory-mode NOPs issue real bus reads on
# hardware (reading e.g. $2007 through them has side effects on a NES).
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

    def bytes(self) -> bytes:
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

    def bne(self, offset: int) -> Asm:
        if not -128 <= offset <= 127:
            raise ValueError(f"branch offset out of range: {offset}")
        return self.append(0xD0, offset & 0xFF)

    def beq(self, offset: int) -> Asm:
        if not -128 <= offset <= 127:
            raise ValueError(f"branch offset out of range: {offset}")
        return self.append(0xF0, offset & 0xFF)

    def brk(self) -> Asm:
        return self.append(0x00)

    def raw(self, *values: int) -> Asm:
        return self.append(*values)


def asm() -> Asm:
    """Return a fresh assembler builder."""
    return Asm()
