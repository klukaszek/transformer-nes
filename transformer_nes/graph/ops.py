"""Arithmetic and memory idioms built from the graph primitives.

Every helper here is exact only on the domain its docstring states
(integer arguments, bounded ranges); the machines that use them (the
6502, the PPU) carry the contracts in their construction notes.
"""

from __future__ import annotations

from collections.abc import Sequence

from transformer_nes.graph import core as graph
from transformer_nes.graph.core import (
    Expression,
    ExprLike,
    fetch,
    make_multiply,
    persist,
    reglu,
    stepglu,
    to_expr,
)


def match_byte(value: ExprLike, byte: int) -> Expression:
    """Return a 0/1 gate: 1 iff value == byte."""
    return match_zero(value - byte)


def match_zero(value: ExprLike) -> Expression:
    """Return a 0/1 gate: 1 iff an integer expression is exactly zero."""
    one = graph.one
    abs_diff = persist(reglu(value, value) + reglu(-value, -value))
    return stepglu(one, -abs_diff)


def multiply(a, b) -> Expression:
    """Multiply two scalar expressions using the graph DSL's internal helper."""
    return make_multiply(to_expr(a), to_expr(b))


def mux(gate: ExprLike, if_one: ExprLike, if_zero: ExprLike) -> Expression:
    """Select between two linear expressions using a 0/1 gate."""
    return if_zero + multiply(if_one - if_zero, gate)


def wrap8(value: ExprLike) -> Expression:
    """Reduce an integer expression to an unsigned 8-bit value."""
    one = graph.one
    return value - 256 * stepglu(one, value - 256) + 256 * stepglu(one, -value - 1)


def wrap16(value: ExprLike) -> Expression:
    """Reduce an integer expression to an unsigned 16-bit value."""
    one = graph.one
    return value - 65536 * stepglu(one, value - 65536) + 65536 * stepglu(one, -value - 1)


def signed_byte(value: ExprLike) -> Expression:
    """Interpret an unsigned byte as a signed 8-bit value."""
    one = graph.one
    return value - 256 * stepglu(one, value - 128)


def extract_bits(value: ExprLike, name: str) -> list[Expression]:
    """Extract bits 7..0 of an 8-bit value expression using binary thresholds.

    Returns a list where bits[0] is bit 7 (MSB) and bits[7] is bit 0 (LSB).
    """
    one = graph.one
    bits = []
    remaining = value
    for i, place in enumerate((128, 64, 32, 16, 8, 4, 2, 1)):
        bit = stepglu(one, remaining - place)
        bit = persist(bit, name=f"{name}bit{7 - i}")
        bits.append(bit)
        remaining = remaining - place * bit
    return bits


def grouped_read(
    address: ExprLike,
    byte_value: ExprLike,
    write_addr: ExprLike,
    is_write: ExprLike,
    *,
    group_count: int | None = None,
    groups: Sequence[int] | None = None,
    dense: bool = False,
    group_size: int = 4096,
) -> Expression:
    """The latest WRITE token's value at `address`, 0 if never written.

    The space splits into 4 KiB lookup groups, each a nearest-address
    attention over its own writes with a 12-bit local-address parabola
    (K-side scores below 2^24, so adjacent addresses stay distinct after
    float32 Q/K rounding).  Equal keys resolve by the LATEST tie-break.

    `groups` (ascending, default all `group_count`) restricts the read to
    groups the address is known to fall in: the gating that picks among
    them skips the bound below the lowest and above the highest, and a
    single group needs none.  `dense` asserts every address the read can ask for has been
    written (the harness seeds the region), so the nearest write is the
    exact one and the exact-address mask is dropped."""
    one = graph.one
    address = to_expr(address)
    write_addr = to_expr(write_addr)
    if groups is None:
        assert group_count is not None
        groups = range(group_count)
    if list(groups) != sorted(set(groups)):
        raise ValueError(f"read groups must ascend: {tuple(groups)}")
    value = Expression()
    for group in groups:
        lo = group * group_size
        hi = lo + group_size - 1
        # Reads: only boundaries between the allowed groups need a test.
        query_checks = []
        if group != groups[0]:
            query_checks.append(stepglu(one, address - lo))
        if group != groups[-1]:
            query_checks.append(stepglu(one, hi - address))
        query_in_group = query_checks[0] if query_checks else None
        for check in query_checks[1:]:
            query_in_group = multiply(query_in_group, check)
        # Writes can land anywhere: both bounds, except the space's ends.
        write_checks = []
        if group > 0:
            write_checks.append(stepglu(one, write_addr - lo))
        if group_count is None or group < group_count - 1:
            write_checks.append(stepglu(one, hi - write_addr))
        write_in_group = write_checks[0]
        for check in write_checks[1:]:
            write_in_group = multiply(write_in_group, check)

        group_write = multiply(is_write, write_in_group)
        group_value, found_addr = fetch(
            [byte_value, write_addr],
            query=address - lo,
            key=write_addr - lo,
            clear_key=1 - group_write,
            tie_break="latest",
        )
        group_value.lookup.memory_group = group
        read = group_value if dense else multiply(group_value, match_zero(address - found_addr))
        if query_in_group is not None:
            read = multiply(query_in_group, read)
        value = value + read
    return value


def memory_site(result: Expression, address, byte_value, write_addr, is_write) -> Expression:
    """Annotate a persisted memory read with its raw ingredients: the
    structural folds (graph/fold.py) rebuild it as one engine-served read
    instead of its attention fabric, and serve bits read_bits annotated
    straight from that read."""
    site = next(iter(result.terms))
    site.memory_read = {
        "address": to_expr(address),
        "byte_value": to_expr(byte_value),
        "write_addr": to_expr(write_addr),
        "is_write": to_expr(is_write),
    }
    return result


def extract_bits_wide(value: ExprLike, name: str) -> list[Expression]:
    """extract_bits, depth-lean: split on the high nibble, then two
    persist-free 15-step ladders ([v >= k] = relu(v-k+1) - relu(v-k),
    exact for integer bytes).  Bits land ~4 layers after the value
    instead of 8, at ~60 ReGLUs of width -- the trade the GPU mapping
    wants (depth is what a single-threadgroup forward pays for) and the
    emitted C++ engine does not (its cost is unit count).  The nibble
    values are wrapped in unit-gated ReGLUs so linear inlining cannot
    smear their ladder sums into every consumer row."""
    v = to_expr(value)
    hi_sum = Expression()
    for k in range(1, 16):
        hi_sum = hi_sum + reglu(1, v - (16 * k - 1)) - reglu(1, v - 16 * k)
    hi = reglu(hi_sum, 1)
    lo = reglu(v - 16 * hi_sum, 1)

    def nibble_bits(nv):
        ge = {k: reglu(1, nv - (k - 1)) - reglu(1, nv - k) for k in range(1, 16)}
        out = []
        for b in range(4):
            r = Expression()
            for start in range(1 << b, 16, 1 << (b + 1)):
                r = r + ge[start]
                if start + (1 << b) < 16:
                    r = r - ge[start + (1 << b)]
            out.append(r)
        return out  # bit 0..3 expressions

    lo_bits = nibble_bits(lo)
    hi_bits = nibble_bits(hi)
    bits = []
    for b in range(7, -1, -1):
        src = hi_bits[b - 4] if b >= 4 else lo_bits[b]
        bits.append(persist(src, name=f"{name}bit{b}"))
    return bits


def read_bits(
    value: Expression, name: str, site_of: Expression | None = None, *, wide: bool = True
) -> list[Expression]:
    """Bits of a memory-read value (MSB first), annotated with the read
    site so structural folds serve them from the fused read itself (the
    kernel computes bits at read time, zero extra depth).  Non-structural
    folds and the union keep the ladder circuit built here.  ``site_of``
    names the raw read when the ladder runs on a later persist of it
    (a pinned persist then const-folds the bits without keeping the raw
    read alive)."""
    site = next(iter((site_of if site_of is not None else value).terms))
    bits = (extract_bits_wide if wide else extract_bits)(value, name)
    dims = []
    for j, b in enumerate(bits):
        d = next(iter(b.terms))
        d.read_bit = (site, j)
        dims.append(d)
    site.read_bit_dims = tuple(dims)
    return bits


# ── 0/1-gated products: one ReGLU each ─────────────────────────────
# relu(b) * a is a * b whenever b is 0 or 1, whatever a's sign, so a
# product with a 0/1 factor needs one unit, where `multiply` spends two
# and a persist to cover any sign of b.


def gate(g: ExprLike, value: ExprLike) -> Expression:
    """value when the 0/1 gate g is 1, else 0."""
    return reglu(value, g)


def both(a: ExprLike, b: ExprLike) -> Expression:
    """a AND b for 0/1 values."""
    return reglu(a, b)


def either(a: ExprLike, b: ExprLike) -> Expression:
    """a OR b for 0/1 values."""
    return to_expr(a) + to_expr(b) - reglu(a, b)


def select(g: ExprLike, if_one: ExprLike, if_zero: ExprLike) -> Expression:
    """if_one when the 0/1 gate g is 1, else if_zero."""
    return to_expr(if_zero) + reglu(to_expr(if_one) - to_expr(if_zero), g)


def equals(value: ExprLike, k: int) -> Expression:
    """1 iff the integer `value` equals k: relu(d+1) - 2 relu(d) + relu(d-1)
    for d = value - k is the unit tent -- three units, one layer, no
    persist (match_byte spends four across three layers)."""
    d = to_expr(value) - k
    return reglu(1, d + 1) - 2 * reglu(1, d) + reglu(1, d - 1)
