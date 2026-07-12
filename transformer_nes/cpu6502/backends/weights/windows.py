"""ROM-independent sequence shapes and window discovery.

This module owns the policy for finding instruction windows worth routing.
It does not build transformer experts or mutate an engine; callers pass the
returned descriptions across that boundary explicitly.
"""

from __future__ import annotations

from transformer_nes.cpu6502.spec import OPCODES, READS_MEMORY

# Ops that end a window: subroutine/interrupt flow and stack PULLS (reads
# past the first instruction).  Plain stores, pushes, and leading RMW
# stay: the fold license (check_writes_licensed) bounds their targets
# below $2000 and rejects what it cannot bound.
SEQUENCE_WRITERS = frozenset({
    "sax", "jsr", "brk", "rti", "rts", "pla", "plp",
})
BRANCH_OPERATIONS = frozenset({"bcc", "bcs", "beq", "bmi", "bne", "bpl", "bvc", "bvs"})
# Long enough to amortize a transformer forward on ordinary straight-line
# 6502 basic blocks, still far below the sequence ABI's 32-row output bound.
# Safety comes from folding/licensing, not this profitability horizon.
MAX_INSTRUCTIONS = 6
MEMORY_WRITE_OPERATIONS = frozenset({
    "store_a", "store_x", "store_y", "sax",
    "asl", "lsr", "rol", "ror", "inc", "dec",
    "dcp", "isb", "slo", "rla", "sre", "rra",
})

# Canonical window shapes compile once into the core engine with pc and every
# operand byte symbolic. Entries are (name, template, spin_repeat); operands
# in the template are placeholders while opcode bytes define the shape.
SHAPE_LIBRARY: tuple[tuple[str, bytes, int], ...] = (
    ("jmp_self", bytes([0x4C, 0, 0]), 4),
    ("dex_bne", bytes([0xCA, 0xD0, 0]), 1),
    ("dey_bne", bytes([0x88, 0xD0, 0]), 1),
    ("inx_bne", bytes([0xE8, 0xD0, 0]), 1),
    ("iny_bne", bytes([0xC8, 0xD0, 0]), 1),
    ("lda_zp_bpl", bytes([0xA5, 0, 0x10, 0]), 1),
    ("lda_zp_bmi", bytes([0xA5, 0, 0x30, 0]), 1),
    ("lda_zp_beq", bytes([0xA5, 0, 0xF0, 0]), 1),
    ("lda_zp_bne", bytes([0xA5, 0, 0xD0, 0]), 1),
    ("lda_abs_bpl", bytes([0xAD, 0, 0, 0x10, 0]), 1),
    ("lda_abs_bmi", bytes([0xAD, 0, 0, 0x30, 0]), 1),
    ("bit_abs_bpl", bytes([0x2C, 0, 0, 0x10, 0]), 1),
    ("bit_abs_bvc", bytes([0x2C, 0, 0, 0x50, 0]), 1),
    ("lda_abs_and_beq", bytes([0xAD, 0, 0, 0x29, 0, 0xF0, 0]), 1),
    ("lda_abs_and_bne", bytes([0xAD, 0, 0, 0x29, 0, 0xD0, 0]), 1),
)


def match_shape_windows(image, shapes_meta, *, base: int = 0, limit: int = 4096):
    """Every anchored shape match over an image region (heads absolute via
    ``base``).  Closure to head is NOT required: dispatch is correct for
    straight-line matches and re-verifies bytes against live RAM, so a
    cold entry is only an unused table slot.  The JMP self-spin keeps its
    operand == head demand (that closure is its dispatch predicate)."""
    data = bytes(image)
    n = len(data)
    out = []
    for meta in shapes_meta:
        signature = [(int(offset), int(byte)) for offset, byte in meta["sig"]]
        length = int(meta["length"])
        first = bytes([signature[0][1]])
        start = 0
        while len(out) < limit:
            head = data.find(first, start, n - length + 1)
            if head < 0:
                break
            start = head + 1
            if base + head + length > 0x10000:  # window would wrap $FFFF
                continue
            if any(data[head + offset] != byte for offset, byte in signature[1:]):
                continue
            if (meta["close"] == "jmp"
                    and data[head + 1] | (data[head + 2] << 8) != base + head):
                continue
            out.append((int(meta["id"]), base + head))
    return out


def match_shape_at(read, pc: int, shapes_meta) -> int | None:
    """Anchored shape match at one pc through a byte-read callable.  Closure
    to head is not required (the caller argues profitability; dispatch is
    correct for straight-line matches too), except the JMP self-spin, whose
    dispatch predicate demands operand == head."""
    for meta in shapes_meta:
        length = int(meta["length"])
        if pc + length > 0x10000:
            continue
        if any(read(pc + off) != byte for off, byte in meta["sig"]):
            continue
        if meta["close"] == "jmp" and read(pc + 1) | (read(pc + 2) << 8) != pc:
            continue
        return int(meta["id"])
    return None


def propose_window(read, pc: int, *, max_len: int = MAX_INSTRUCTIONS):
    """Decode a pinned-window candidate anchored at a hot pc: straight-line
    code obeying the scanner's rules (reads lead, no subroutine/stack-pull
    flow), ended by an optional single branch.  Closure is not required --
    hotness is the profitability argument, dispatch is correct for
    straight-line matches, and the fold license arbitrates stores.
    Returns (head, code, 1) or None."""
    code = bytearray()
    wrote_memory = False
    for index in range(max_len):
        at = pc + len(code)
        if at + 3 > 0x10000:
            break
        op = OPCODES.get(read(at))
        if op is None:
            break
        if op.operation in SEQUENCE_WRITERS:
            break
        reads = op.operation in READS_MEMORY and op.mode not in ("imp", "acc", "imm")
        if wrote_memory and reads:
            break
        if op.operation == "jmp" or op.mode == "ind":
            break
        body = bytes(read(at + j) for j in range(op.length))
        if op.operation in BRANCH_OPERATIONS:
            if index > 0:
                code.extend(body)
            break
        code.extend(body)
        wrote_memory |= op.operation in MEMORY_WRITE_OPERATIONS
    if sum(1 for _ in iter_ops(bytes(code))) < 2:
        return None
    return (pc, bytes(code), 1)


def iter_ops(code: bytes):
    off = 0
    while off < len(code):
        op = OPCODES[code[off]]
        yield op
        off += op.length


def scan_sequence_windows(image, *, base: int = 0, spin_repeat: int = 4,
                          limit: int = 16):
    """Find closed, write-free micro-loops eligible for sequence experts.

    ``base`` scans a bank region at its mapped CPU address: returned heads
    are absolute, and JMP self-targets compare against base + head (branch
    closure is relative, so it needs no adjustment)."""
    data = bytes(image)
    windows = []
    for head in range(len(data)):
        pc = head
        wrote_memory = False
        for index in range(MAX_INSTRUCTIONS):
            op = OPCODES.get(data[pc]) if pc < len(data) - 2 else None
            if op is None:
                break
            if op.operation in SEQUENCE_WRITERS:
                break
            reads = op.operation in READS_MEMORY and op.mode not in ("imp", "acc", "imm")
            if wrote_memory and reads:
                break
            next_pc = pc + op.length
            if op.operation in BRANCH_OPERATIONS:
                offset = data[pc + 1]
                target = (next_pc + offset - (256 if offset & 0x80 else 0)) & 0xFFFF
                if target == head and index > 0:
                    windows.append((base + head, data[head:next_pc], 1))
                break
            if op.operation == "jmp":
                target = data[pc + 1] | (data[pc + 2] << 8)
                if op.mode == "abs" and target == base + head:
                    windows.append((base + head, data[head:next_pc], spin_repeat))
                break
            wrote_memory |= op.operation in MEMORY_WRITE_OPERATIONS
            pc = next_pc
    return tuple(windows[:limit])
