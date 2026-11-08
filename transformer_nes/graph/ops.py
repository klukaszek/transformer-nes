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


def match_zero(value: ExprLike) -> Expression:
    """Return a 0/1 gate: 1 iff an integer expression is exactly zero."""
    one = graph.current_graph().one
    abs_diff = persist(reglu(one, value) + reglu(one, -value))
    return stepglu(one, -abs_diff)


def multiply(a, b) -> Expression:
    """Multiply two scalar expressions using the graph DSL's internal helper."""
    return make_multiply(to_expr(a), to_expr(b))


def remainder(value, modulus: int, maximum: int):
    """Reduce integers in [0, maximum] modulo a positive constant.

    Each stage subtracts one power-of-two multiple of the divisor. Circuit
    size is logarithmic in the quotient range; no state/outcome table.
    """
    if modulus < 1 or maximum < 0:
        raise ValueError("modulus must be positive and maximum nonnegative")
    if modulus == 1:
        return 0
    for shift in reversed(range((maximum // modulus).bit_length())):
        multiple = modulus << shift
        value = value - multiple * stepglu(1, value - multiple)
    return value


def remainder_by(value: ExprLike, modulus: ExprLike, maximum: int, *, least: int = 1):
    """Reduce integers in [0, maximum] modulo an integer expression >= `least`.

    Restoring division: each stage compares against one power-of-two multiple
    of the modulus and subtracts it when it fits. The comparison is a 0/1
    gate, so a ReGLU forms the product with the variable multiple. One more
    neuron per stage than `remainder` with a constant modulus.
    """
    if least < 1 or maximum < 0:
        raise ValueError("least modulus must be positive and maximum nonnegative")
    value, modulus = to_expr(value), to_expr(modulus)
    for shift in reversed(range((maximum // least).bit_length())):
        multiple = modulus * (1 << shift)
        value = value - reglu(multiple, stepglu(1, value - multiple))
    return value


def wrap8(value: ExprLike) -> Expression:
    """Reduce an integer expression to an unsigned 8-bit value."""
    one = graph.current_graph().one
    return value - 256 * stepglu(one, value - 256) + 256 * stepglu(one, -value - 1)


def wrap16(value: ExprLike) -> Expression:
    """Reduce an integer expression to an unsigned 16-bit value."""
    one = graph.current_graph().one
    return value - 65536 * stepglu(one, value - 65536) + 65536 * stepglu(one, -value - 1)


def wrapped_nz(value: ExprLike, *, lo: int, hi: int) -> tuple[Expression, Expression]:
    """N/Z of an integer byte result, directly from its pre-wrap value.

    The inclusive bounds must cover ``value``. Finite differences express
    each flag as an affine term plus ReLU hinges, all in one ReGLU stage.
    Shared hinges serve both flags; the construction table is never executed.
    """
    if not -256 <= lo < hi <= 511:
        raise ValueError("byte readout bounds must lie in [-256, 511]")
    value = to_expr(value)
    outputs = []
    for mask in (lambda v: int(v >= 128), lambda v: int(v == 0)):
        samples = [mask(v & 255) for v in range(lo, hi + 1)]
        result = samples[0] + (samples[1] - samples[0]) * (value - lo)
        for i in range(1, len(samples) - 1):
            coefficient = samples[i + 1] - 2 * samples[i] + samples[i - 1]
            if coefficient:
                result = result + coefficient * reglu(1, value - (lo + i))
        outputs.append(to_expr(result))
    return tuple(outputs)


def signed_byte(value: ExprLike) -> Expression:
    """Interpret an unsigned byte as a signed 8-bit value."""
    one = graph.current_graph().one
    return value - 256 * stepglu(one, value - 128)


def extract_bits(value: ExprLike, name: str) -> list[Expression]:
    """Extract bits 7..0 of an 8-bit value expression using binary thresholds.

    Returns a list where bits[0] is bit 7 (MSB) and bits[7] is bit 0 (LSB).
    """
    one = graph.current_graph().one
    bits = []
    remaining = value
    for i, place in enumerate((128, 64, 32, 16, 8, 4, 2, 1)):
        # After seven thresholds the remainder is already exactly 0 or 1.
        bit = remaining if place == 1 else stepglu(one, remaining - place)
        bit = persist(bit, name=f"{name}bit{7 - i}")
        bits.append(bit)
        remaining = remaining - place * bit
    return bits


def high_bit(value: ExprLike, bit: int, width: int = 8) -> Expression:
    """Bit `bit` of an integer in [0, 2**width) in one layer: the alternating
    sum of its 2**(width - bit) - 1 thresholds at multiples of 2**bit. A
    threshold ladder (extract_bits) needs width - bit layers; this is cheap
    for the high bits."""
    one = graph.current_graph().one
    place = 1 << bit
    return sum(
        ((-1) ** (j + 1) * stepglu(one, value - j * place) for j in range(1, 1 << (width - bit))),
        Expression(),
    )


def group_spans(count: int, size: int = 4096) -> list[tuple[int, int]]:
    """`count` consecutive lookup groups of `size` bytes from address 0."""
    return [(g * size, g * size + size - 1) for g in range(count)]


def grouped_read(
    address: ExprLike,
    byte_value: ExprLike,
    write_addr: ExprLike,
    is_write: ExprLike,
    *,
    spans: Sequence[tuple[int, int]],
    space_end: int,
    dense: bool = False,
) -> Expression:
    """The latest WRITE token's value at `address`, 0 if never written.

    The space splits into lookup groups (inclusive address spans), each a
    nearest-address attention over its own writes with a local-address
    parabola. Its square must fit the int32 residual: groups of at most
    32 KiB. Equal keys resolve to the latest write; a read whose nearest
    write is another address is masked to 0.

    `spans` (ascending, disjoint) are the groups the address is known to fall
    in: the gating that picks among them skips the bound below the lowest
    and above the highest, and a single group needs none. Writes can land
    anywhere in [0, space_end). `dense` asserts every address the read can
    ask for has been written (the harness seeds the region), so the nearest
    write is the exact one and the exact-address mask is dropped."""
    one = graph.current_graph().one
    address = to_expr(address)
    write_addr = to_expr(write_addr)
    spans = [tuple(span) for span in spans]
    if any(hi < lo or hi - lo >= 0x8000 for lo, hi in spans) or any(
        a[1] >= b[0] for a, b in zip(spans, spans[1:], strict=False)
    ):
        raise ValueError(f"read spans must ascend, not overlap, and span at most 32 KiB: {spans}")
    value = Expression()
    for lo, hi in spans:
        # Reads: only boundaries between the allowed groups need a test.
        query_checks = []
        if (lo, hi) != spans[0]:
            query_checks.append(stepglu(one, address - lo))
        if (lo, hi) != spans[-1]:
            query_checks.append(stepglu(one, hi - address))
        query_in_group = query_checks[0] if query_checks else None
        for check in query_checks[1:]:
            query_in_group = multiply(query_in_group, check)
        # Writes can land anywhere: both bounds, except the space's ends.
        write_checks = []
        if lo > 0:
            write_checks.append(stepglu(one, write_addr - lo))
        if hi < space_end - 1:
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
        group_value.lookup.memory_group = lo >> 12
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
    """Extract an integer byte's bits with 49 ReGLUs in three stages.

    The thresholds at multiples of 16 supply both the high nibble's value
    and its bits. Only the low nibble needs another threshold ladder.
    Unit-gated projections keep the low nibble and each high bit compact
    when folding inlines linear expressions into their consumers.
    """
    v = to_expr(value)
    high = {k: reglu(1, v - (16 * k - 1)) - reglu(1, v - 16 * k) for k in range(1, 16)}
    lo = reglu(v - 16 * sum(high.values(), Expression()), 1)
    # lo is in [0,15]: relu(lo)=lo and relu(lo-15)=0. The endpoint
    # thresholds share the same interior hinges as the other low bits.
    low = {
        k: (lo if k == 1 else reglu(1, lo - (k - 1)))
        - (0 if k == 15 else reglu(1, lo - k))
        for k in range(1, 16)
    }

    def nibble_bits(ge):
        out = []
        for b in range(4):
            r = Expression()
            for start in range(1 << b, 16, 1 << (b + 1)):
                r = r + ge[start]
                if start + (1 << b) < 16:
                    r = r - ge[start + (1 << b)]
            out.append(r)
        return out  # bit 0..3 expressions

    by_bit = nibble_bits(low) + [reglu(bit, 1) for bit in nibble_bits(high)]
    return [persist(by_bit[b], name=f"{name}bit{b}") for b in range(7, -1, -1)]


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
    g, value = to_expr(g), to_expr(value)
    if set(g.terms) <= {g.one}:
        return max(0, sum(g.terms.values())) * value
    if set(value.terms) <= {value.one}:
        # Selecting between fixed values is affine in the control bit.
        return sum(value.terms.values()) * g
    return reglu(value, g)


def nonzero(value: ExprLike) -> Expression:
    """0/1 for a nonnegative integer, using one ReGLU hinge."""
    value = to_expr(value)
    return value - reglu(1, value - 1)


def gate_nonzero(
    g: ExprLike, value: ExprLike, payload: ExprLike, *, zero_payload: ExprLike | None = None,
) -> Expression:
    """payload when g=1 and value>0, else zero.

    g must be 0/1 and value a nonnegative integer. relu(g-value) is one
    exactly when g=1,value=0. Subtracting that case masks transparency
    with two parallel neurons, without materializing an opacity flag.
    payload may be signed and may depend on value.
    zero_payload may simplify the correction's operand: it must equal
    payload whenever value=0 (e.g. the metadata in metadata+value).
    """
    g = to_expr(g)
    if not g.terms:
        return g
    correction = payload if zero_payload is None else zero_payload
    masked = reglu(payload, g) - reglu(correction, g - value)
    if set(g.terms) <= {g.one}:
        return masked
    out = persist(masked)
    # The domain contract proves zero even when the folder cannot infer
    # that -value is nonpositive. Preserve that proof for specialization.
    next(iter(out.terms)).zero_when = g
    return out


def both(a: ExprLike, b: ExprLike) -> Expression:
    """a AND b for 0/1 values."""
    return gate(b, a)


def either(a: ExprLike, b: ExprLike) -> Expression:
    """a OR b for 0/1 values."""
    return to_expr(a) + to_expr(b) - gate(b, a)


def select(g: ExprLike, if_one: ExprLike, if_zero: ExprLike) -> Expression:
    """if_one when the 0/1 gate g is 1, else if_zero."""
    return to_expr(if_zero) + gate(g, to_expr(if_one) - to_expr(if_zero))


def equals(value: ExprLike, k: int) -> Expression:
    """1 iff the integer `value` equals k: relu(d+1) - 2 relu(d) + relu(d-1)
    for d = value - k is the unit tent: three units in one layer.
    Adjacent comparisons share their hinges through graph interning.
    """
    d = to_expr(value) - k
    return reglu(1, d + 1) - 2 * reglu(1, d) + reglu(1, d - 1)
