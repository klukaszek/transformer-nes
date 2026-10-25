"""Fixed execution shapes: templates, byte matching and recurrence proofs."""

from __future__ import annotations

from transformer_nes.cpu6502.spec import OPCODES
from transformer_nes.graph.core import (
    InputDimension,
    ProgramGraph,
    construction,
    current_graph,
    persist,
    reglu,
    stepglu,
)

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
            if meta["close"] == "jmp" and data[head + 1] | (data[head + 2] << 8) != base + head:
                continue
            out.append((int(meta["id"]), base + head))
    return out


POLL_PREFIXES = {("load_a",), ("bit",), ("load_a", "and")}
COUNTERS = {"inx": (1, 1), "iny": (2, 1), "dex": (1, -1), "dey": (2, -1)}
POLL_BRANCHES = {"bne", "beq", "bpl", "bmi", "bvc", "bvs"}


@construction
def counter_recurrence_graph(register: str, delta: int) -> ProgramGraph:
    """A closed byte-counter loop, advanced within a supplied iteration limit.

    The first instruction shape has already run and closed its branch, so
    the incoming counter is nonzero. Stop before its zero/exit iteration;
    that iteration retains the instruction graph's different branch timing.
    """
    counter = InputDimension(f"in_{register}")
    limit = InputDimension("in_loop_limit")
    cycles = InputDimension("in_loop_cycles")
    result = persist(
        1 + reglu(1, counter - limit - 1)
        if delta < 0 else counter + limit - reglu(1, counter + limit - 255),
        name=f"new_{register}",
    )
    iterations = persist(delta * (result - counter), name="loop_iterations")
    outputs = {
        "counter": result,
        "p_z": persist(1 - stepglu(1, result - 1), name="loop_z"),
        "p_n": persist(stepglu(1, result - 128), name="loop_n"),
        "cycles": persist(reglu(cycles, iterations), name="loop_elapsed"),
        "iterations": iterations,
    }
    return current_graph().capture(
        {"STATE": counter + limit + cycles},
        outputs,
    )


def shape_loop(opcode_checks, length: int) -> list[int]:
    """[kind, closure offset, closure byte, register, delta].

    Operands are symbolic: a poll needs no operand proof because dispatch
    checks its read lease, and a counter has no memory effects. Return kind
    zero for anything outside these fixed, independently checked patterns.
    """
    ops = [OPCODES[byte] for _offset, byte in opcode_checks]
    none = [0, 0, 0, 0, 0]
    if sum(op.length for op in ops) != length:
        return none
    names = tuple(op.operation for op in ops)
    closure = [length - 1, (-length) & 0xFF]
    if names[-1] in POLL_BRANCHES and names[:-1] in POLL_PREFIXES:
        return [1, *closure, 0, 0]
    if len(ops) == 2 and names[-1] == "bne" and names[0] in COUNTERS:
        return [2, *closure, *COUNTERS[names[0]]]
    return none


def with_loops(experts: list[dict]) -> list[dict]:
    for expert in experts:
        if expert["kind"] == "shape":
            single = not expert["spin"] and expert["n_steps"] == len(expert["opcode_checks"])
            expert["loop"] = (
                shape_loop(expert["opcode_checks"], expert["length"]) if single else [0] * 5
            )
    return experts


# Operations whose folded circuits read A's bit state (wide_bits builds):
# logical/bitwise families, ADC/SBC (served-sign overflow + decimal
# nibbles), and any accumulator-mode shift (operand bits mux to A's).
A_BIT_OPERATIONS = frozenset(
    {
        "and",
        "ora",
        "eor",
        "bit",
        "sax",
        "slo",
        "rla",
        "sre",
        "adc",
        "sbc",
    }
)


def last_a_bit_step(ops) -> int:
    """Index of the last step consuming A's bit state; the fold seeds the
    bit feedback chain only strictly before it (wrong pruning fails
    loudly at build, not silently)."""
    last = -1
    for k, op in enumerate(ops):
        if op.operation in A_BIT_OPERATIONS or op.mode == "acc":
            last = k
    return last


def decode_ops(code: bytes) -> list:
    ops, off = [], 0
    while off < len(code):
        op = OPCODES[code[off]]
        ops.append(op)
        off += op.length
    return ops


