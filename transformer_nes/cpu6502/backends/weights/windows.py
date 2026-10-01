"""ROM-independent sequence shapes and window discovery.

This module owns the policy for finding instruction windows worth routing.
It does not build transformer experts or mutate an engine; callers pass the
returned descriptions across that boundary explicitly.
"""

from __future__ import annotations

from transformer_nes.cpu6502.spec import BRANCH_OPERATIONS, OPCODES, READS_MEMORY

# Ops that end a window: subroutine/interrupt flow and stack PULLS (reads
# past the first instruction).  Plain stores, pushes, and leading RMW
# stay: the fold license (check_writes_licensed) bounds their targets
# below $2000 and rejects what it cannot bound.
SEQUENCE_WRITERS = frozenset({
    "sax", "jsr", "brk", "rti", "rts", "pla", "plp",
})
# Learned windows (block_code) are wider: they hold pushes and pulls (the
# fold licenses a pull only when nothing earlier in the window wrote the
# stack page), run through JSR like through an absolute JMP, and end with a
# branch or a return, whose target is the model's.  What they never hold:
WINDOW_EXCLUDED = frozenset({"brk", "sax"})


def ends_window(op) -> bool:
    """A window's last instruction: a branch, a return."""
    return op.operation in BRANCH_OPERATIONS or op.operation in ("rts", "rti")


def continues_at_target(op) -> bool:
    """An instruction a window runs through to its constant target."""
    return op.operation in ("jmp", "jsr") and op.mode == "abs"


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


def block_code(read, head: int, *, limit: int = 32) -> bytes:
    """The code a window from `head` could run, in execution order, as read
    through `read(address)`: up to `limit` instructions, running through
    absolute JMP and JSR, ending with a branch or a return or before what
    no window holds (WINDOW_EXCLUDED, indirect JMP, undefined bytes).  The
    fold's license checks decide how much of it one forward can take."""
    code = bytearray()
    pc = head
    for _ in range(limit):
        op = OPCODES.get(read(pc))
        if op is None or op.operation in WINDOW_EXCLUDED or (
            op.operation == "jmp" and op.mode != "abs"
        ):
            break
        code += bytes(read((pc + i) & 0xFFFF) for i in range(op.length))
        if ends_window(op):
            break
        if continues_at_target(op):
            target = read((pc + 1) & 0xFFFF) | (read((pc + 2) & 0xFFFF) << 8)
            if target == pc:  # a spin: the shape library's
                code = code[: -op.length]
                break
            pc = target
        else:
            pc = (pc + op.length) & 0xFFFF
    return bytes(code)


def static_window_limit(code: bytes) -> int:
    """How many leading instructions of `code` can possibly share a
    window, judged from constant operand addresses alone (before any fold):
    a store to $2000 or above reaches a device, which no window may; a
    read of $2000-$7FFF may be a device read, which only a window's first
    instruction may make."""
    off = k = 0
    while off < len(code):
        op = OPCODES[code[off]]
        if op.mode in ("abs", "abx", "aby"):
            address = code[off + 1] | (code[off + 2] << 8)
            if op.operation in MEMORY_WRITE_OPERATIONS and address >= 0x2000:
                return k
            reads = op.operation in READS_MEMORY and op.operation not in ("jmp", "jsr")
            if k and reads and 0x2000 <= address < 0x8000:
                return k
        off += op.length
        k += 1
    return k
