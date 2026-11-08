"""Bounded neural timers and a fifteen-bit GF(2) noise recurrence.

The sample/DMC deadlines bound each span. Squared recurrence powers advance
noise in logarithmic work; the clock requests updates only on noise expiries.
"""

from transformer_nes.graph.core import (
    Expression,
    InputDimension,
    construction,
    current_graph,
    persist,
    reglu,
    stepglu,
)
from transformer_nes.graph.ops import both, equals, gate, remainder, select

# The longest span one oscillator forward covers: CPU cycles per sample at
# 44.1 kHz, rounded up (1789773 / 44100 = 40.6). The host splits longer spans,
# so one circuit serves every sample rate; at 44.1 kHz and above each sample
# takes one forward.
SPAN = 41
CLOCK_INPUTS = (
    "cycles",
    "apu_tick",
    *(f"pulse{i}_{f}" for i in range(2) for f in ("period", "counter", "pos")),
    "triangle_period",
    "triangle_counter",
    "triangle_pos",
    "triangle_length",
    "triangle_linear",
    "noise_period",
    "noise_counter",
)
CLOCK_OUTPUTS = (
    "new_apu_tick",
    *(f"new_pulse{i}_{f}" for i in range(2) for f in ("counter", "pos")),
    "new_triangle_counter",
    "new_triangle_pos",
    "new_noise_counter",
    "noise_steps",
)

NOISE_INPUTS = ("steps", "noise_mode", *(f"noise_bit{i}" for i in range(15)))
NOISE_OUTPUTS = tuple(f"new_noise_bit{i}" for i in range(15))


def bits(value, width):
    """Unsigned binary decomposition, least significant lane first."""
    out = []
    for place in reversed(range(width)):
        bit = persist(stepglu(1, value - (1 << place)))
        out.append(bit)
        value = value - (1 << place) * bit
    return out[::-1]


def timer(counter, period, clocks, maximum):
    """Exact down-counter transition for nonnegative counters/periods.

    A counter expires when clocked at zero, including after a period write
    leaves the current counter larger than its new reload value. Restoring
    division only needs enough quotient bits for this span's maximum.
    """
    if maximum <= 1:
        expired = both(equals(counter, 0), clocks)
        return select(expired, period, counter - clocks), expired
    expired = stepglu(1, clocks - counter - 1)
    rest = reglu(1, clocks - counter - 1)
    quotient = Expression()
    for shift in reversed(range((maximum - 1).bit_length())):
        divisor = (period + 1) * (1 << shift)
        take = stepglu(1, rest - divisor)
        rest = rest - gate(take, divisor)
        quotient = quotient + take * (1 << shift)
    return select(expired, period - rest, counter - clocks), gate(expired, quotient + 1)


def xor(a, b):
    return a + b - 2 * both(a, b)


def parity(values):
    """Balanced XOR reduction; every input and intermediate is a bit."""
    while len(values) > 1:
        values = [
            xor(values[i], values[i + 1]) if i + 1 < len(values) else values[i]
            for i in range(0, len(values), 2)
        ]
    return values[0]


def noise_jump(state, count, tap):
    # Each row is a set of input lanes. Squaring composes the linear
    # transform by symmetric difference, entirely at construction time.
    rows = [1 << (i + 1) for i in range(14)] + [1 | (1 << tap)]
    for enabled in count:
        moved = [parity([state[j] for j in range(15) if row & (1 << j)]) for row in rows]
        state = [select(enabled, new, old) for old, new in zip(state, moved, strict=True)]
        squared = []
        for row in rows:
            result = 0
            for j in range(15):
                if row & (1 << j):
                    result ^= rows[j]
            squared.append(result)
        rows = squared
    return state


@construction
def clock_graph(maximum=SPAN):
    if maximum < 1:
        raise ValueError("an oscillator span covers at least one cycle")
    x = {name: InputDimension(name) for name in CLOCK_INPUTS}
    if maximum == 1:
        low, half = x["cycles"], Expression()
    else:
        clock_bits = bits(x["cycles"], maximum.bit_length())
        low = clock_bits[0]
        half = sum((b * (1 << i) for i, b in enumerate(clock_bits[1:])), Expression())
    ticks = half + both(low, 1 - x["apu_tick"])
    tick_max = (maximum + 1) // 2
    values = [xor(x["apu_tick"], low)]
    for i in range(2):
        counter, steps = timer(x[f"pulse{i}_counter"], x[f"pulse{i}_period"], ticks, tick_max)
        position = remainder(x[f"pulse{i}_pos"] + steps, 8, 7 + tick_max)
        values.extend((counter, position))
    counter, steps = timer(x["triangle_counter"], x["triangle_period"], x["cycles"], maximum)
    active = both(stepglu(1, x["triangle_length"] - 1), stepglu(1, x["triangle_linear"] - 1))
    position = remainder(x["triangle_pos"] + gate(active, steps), 32, 31 + maximum)
    values.extend((counter, position))
    counter, steps = timer(x["noise_counter"], x["noise_period"], ticks, tick_max)
    values.extend((counter, steps))
    outputs = {name: persist(value, name=name) for name, value in zip(CLOCK_OUTPUTS, values, strict=True)}
    return current_graph().capture({"SPAN": sum(x.values())}, outputs)


@construction
def noise_graph(maximum=SPAN):
    if maximum < 1:
        raise ValueError("an oscillator span covers at least one cycle")
    x = {name: InputDimension(name) for name in NOISE_INPUTS}
    steps = persist(x["steps"], name="noise_steps_tag")
    mode = persist(x["noise_mode"], name="noise_mode_tag")
    count = bits(steps, ((maximum + 1) // 2).bit_length())
    state = [x[f"noise_bit{i}"] for i in range(15)]
    long = noise_jump(state, count, 1)
    short = noise_jump(state, count, 6)
    values = [select(mode, b, a) for a, b in zip(long, short, strict=True)]
    outputs = {name: persist(value, name=name) for name, value in zip(NOISE_OUTPUTS, values, strict=True)}
    return current_graph().capture({"NOISE": sum(x.values())}, outputs)
