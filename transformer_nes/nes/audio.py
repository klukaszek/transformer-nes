"""2A03 register, frame, oscillator and DMC transitions as weights.

The float32 sample circuit owns digital levels, DAC mixing and filtering.
The host schedules deadlines, transports DMA bytes and plays returned samples.
"""

from dataclasses import replace
from functools import cache

from transformer_nes.graph.core import (
    Expression,
    InputDimension,
    construction,
    current_graph,
    persist,
    reglu,
    stepglu,
)
from transformer_nes.graph.ops import both, either, equals, gate, select
from transformer_nes.model.graph_engine import CompiledGraph, TokenGraph
from transformer_nes.nes.oscillators import MAX_CYCLES, bits

ENVELOPE = ("env_start", "env_loop", "env_constant", "env_period", "env_divider", "env_decay")
PULSE = (
    "timer",
    "length",
    "enabled",
    "sweep_enable",
    "sweep_period",
    "sweep_negate",
    "sweep_shift",
    "sweep_reload",
    "sweep_divider",
    "seq_pos",
    "duty",
    *ENVELOPE,
)
TRIANGLE = (
    "timer",
    "length",
    "length_halt",
    "enabled",
    "linear_reload_value",
    "linear_reload",
    "linear_counter",
    "seq_pos",
)
NOISE = ("length", "enabled", "bit0", *ENVELOPE)
EVENTS = ("quarter", "half", "both")
INPUTS = (
    "event",
    *(f"pulse{i}_{f}" for i in range(2) for f in PULSE),
    *(f"triangle_{f}" for f in TRIANGLE),
    *(f"noise_{f}" for f in NOISE),
)
STATE_OUTPUTS = (
    *(
        f"pulse{i}_{f}"
        for i in range(2)
        for f in (
            "env_start",
            "env_divider",
            "env_decay",
            "length",
            "timer",
            "sweep_reload",
            "sweep_divider",
        )
    ),
    "triangle_length",
    "triangle_linear_reload",
    "triangle_linear_counter",
    "noise_env_start",
    "noise_env_divider",
    "noise_env_decay",
    "noise_length",
)
OUTPUTS = tuple(f"new_{f}" for f in STATE_OUTPUTS)
LEVELS = ("pulse0_level", "pulse1_level", "triangle_level", "noise_level")
FRAME_EVENTS = ("clock", "write", "read", "observe")
FRAME_INPUTS = ("five_step", "irq_inhibit", "frame_irq", "seq_step", "event", "value")
FRAME_OUTPUTS = (
    "new_five_step",
    "new_irq_inhibit",
    "new_frame_irq",
    "new_seq_step",
    "reset_cycle",
    "quarter",
    "half",
    "deadline",
)


# Chip ROM constants become decoder weights, never host-side lookup tables.
LENGTHS = (
    10, 254, 20, 2, 40, 4, 80, 6, 160, 8, 60, 10, 14, 12, 26, 14,
    12, 16, 24, 18, 48, 20, 96, 22, 192, 24, 72, 26, 16, 28, 32, 30,
)
NOISE_PERIODS = (4, 8, 16, 32, 64, 96, 128, 160, 202, 254, 380, 508, 762, 1016, 2034, 4068)
DMC_PERIODS = (428, 380, 340, 320, 286, 254, 226, 214, 190, 160, 142, 128, 106, 85, 72, 54)
PROGRAM_STATE = (*INPUTS[1:], "noise_mode", "noise_period")
PROGRAM_INPUTS = (
    *PROGRAM_STATE,
    "port",
    "value",
    "is_write",
    "dmc_remaining",
    "dmc_irq",
    "frame_irq",
)
PROGRAM_OUTPUTS = (
    *(f"new_{f}" for f in PROGRAM_STATE),
    "read_value",
    "frame_write",
    "frame_read",
    "dmc_write",
)
WRITE_PORTS = (*range(9), 10, 11, 12, 14, 15, 16, 17, 18, 19, 21, 23)
DMC_STATE = (
    "irq_enabled",
    "loop",
    "irq",
    "period",
    "output_level",
    "sample_address",
    "sample_length",
    "current_address",
    "bytes_remaining",
    "sample_buffer",
    "buffer_empty",
    "shift_register",
    "bits_remaining",
    "silence",
)
DMC_INPUTS = (*DMC_STATE, "event", "port", "value")
DMC_OUTPUTS = (
    *(f"new_{f}" for f in DMC_STATE),
    "fetch_due",
    "dma_active",
    "dma_tail",
    "reset_clock",
    "clock_reload",
    "dma_stall",
)
DMC_EVENTS = ("write", "fetch", "clock", "observe")
DMC_WRITE_PORTS = (16, 17, 18, 19, 21)


def decoder(index, values):
    return sum(gate(equals(index, i), value) for i, value in enumerate(values))


@construction
def program_graph():
    x = {name: InputDimension(name) for name in PROGRAM_INPUTS}
    port = persist(x["port"], name="port_tag")
    writing = persist(x["is_write"], name="writing")
    wr = {i: both(writing, equals(port, i)) for i in WRITE_PORTS}
    state = {name: x[name] for name in PROGRAM_STATE}
    b = bits(x["value"], 8)

    def field(lo, width):
        return sum((b[i] * (1 << (i - lo)) for i in range(lo, lo + width)), Expression())

    def assign(name, value, when):
        state[name] = state[name] + gate(when, value - x[name])

    loaded_length = decoder(field(3, 5), LENGTHS)
    for prefix, base in (("pulse0", 0), ("pulse1", 4)):
        for name, value in (
            ("duty", field(6, 2)),
            ("env_loop", b[5]),
            ("env_constant", b[4]),
            ("env_period", field(0, 4)),
        ):
            assign(f"{prefix}_{name}", value, wr[base])
        for name, value in (
            ("sweep_enable", b[7]),
            ("sweep_period", field(4, 3)),
            ("sweep_negate", b[3]),
            ("sweep_shift", field(0, 3)),
            ("sweep_reload", 1),
        ):
            assign(f"{prefix}_{name}", value, wr[base + 1])
        assign(f"{prefix}_seq_pos", 0, wr[base + 3])
        assign(f"{prefix}_env_start", 1, wr[base + 3])
    for prefix, low, high in (("pulse0", 2, 3), ("pulse1", 6, 7), ("triangle", 10, 11)):
        timer_bits = bits(x[f"{prefix}_timer"], 11)
        upper = sum((timer_bits[i] * (1 << i) for i in range(8, 11)), Expression())
        lower = x[f"{prefix}_timer"] - upper
        assign(f"{prefix}_timer", upper + x["value"], wr[low])
        assign(f"{prefix}_timer", lower + 256 * field(0, 3), wr[high])
    for prefix, load in (("pulse0", 3), ("pulse1", 7), ("triangle", 11), ("noise", 15)):
        assign(f"{prefix}_length", loaded_length, both(wr[load], x[f"{prefix}_enabled"]))
    assign("triangle_length_halt", b[7], wr[8])
    assign("triangle_linear_reload_value", field(0, 7), wr[8])
    assign("triangle_linear_reload", 1, wr[11])
    for name, value in (("env_loop", b[5]), ("env_constant", b[4]), ("env_period", field(0, 4))):
        assign(f"noise_{name}", value, wr[12])
    assign("noise_mode", b[7], wr[14])
    assign("noise_period", decoder(field(0, 4), NOISE_PERIODS), wr[14])
    assign("noise_env_start", 1, wr[15])
    for bit, prefix in enumerate(("pulse0", "pulse1", "triangle", "noise")):
        assign(f"{prefix}_enabled", b[bit], wr[21])
        assign(f"{prefix}_length", 0, both(wr[21], 1 - b[bit]))
    status = sum(
        (
            stepglu(1, x[f"{p}_length"] - 1) * (1 << i)
            for i, p in enumerate(("pulse0", "pulse1", "triangle", "noise"))
        ),
        Expression(),
    )
    status = (
        status + 16 * stepglu(1, x["dmc_remaining"] - 1) + 64 * x["frame_irq"] + 128 * x["dmc_irq"]
    )
    effects = (
        select(equals(port, 21), status, 64),
        wr[23],
        both(1 - writing, equals(port, 21)),
        sum(wr[p] for p in DMC_WRITE_PORTS),
    )
    values = (*(state[f] for f in PROGRAM_STATE), *effects)
    outputs = {
        name: persist(value, name=name) for name, value in zip(PROGRAM_OUTPUTS, values, strict=True)
    }
    return current_graph().capture({"ACCESS": sum(x.values())}, outputs)


@construction
def dmc_graph():
    x = {name: InputDimension(name) for name in DMC_INPUTS}
    event = persist(x["event"], name="event_tag")
    port = persist(x["port"], name="port_tag")
    write, fetch, clock, observe = (equals(event, i) for i in range(4))
    wr = {i: both(write, equals(port, i)) for i in DMC_WRITE_PORTS}
    state = {name: x[name] for name in DMC_STATE}
    b = bits(x["value"], 8)
    low4 = sum((b[i] * (1 << i) for i in range(4)), Expression())
    low7 = sum((b[i] * (1 << i) for i in range(7)), Expression())
    state["irq_enabled"] = select(wr[16], b[7], x["irq_enabled"])
    state["loop"] = select(wr[16], b[6], x["loop"])
    state["period"] = select(wr[16], decoder(low4, DMC_PERIODS), x["period"])
    state["output_level"] = select(wr[17], low7, x["output_level"])
    state["sample_address"] = select(wr[18], 0xC000 + 64 * x["value"], x["sample_address"])
    state["sample_length"] = select(wr[19], 16 * x["value"] + 1, x["sample_length"])
    restart = both(wr[21], both(b[4], equals(x["bytes_remaining"], 0)))
    state["bytes_remaining"] = select(
        wr[21],
        select(b[4], select(restart, x["sample_length"], x["bytes_remaining"]), 0),
        x["bytes_remaining"],
    )
    state["current_address"] = select(restart, x["sample_address"], x["current_address"])
    state["irq"] = select(either(wr[21], both(wr[16], 1 - b[7])), 0, x["irq"])

    accepted = both(fetch, both(x["buffer_empty"], stepglu(1, x["bytes_remaining"] - 1)))
    finished = both(accepted, equals(x["bytes_remaining"], 1))
    loop = both(finished, x["loop"])
    next_address = select(equals(x["current_address"], 0xFFFF), 0x8000, x["current_address"] + 1)
    state["sample_buffer"] = select(accepted, x["value"], x["sample_buffer"])
    state["buffer_empty"] = select(accepted, 0, x["buffer_empty"])
    state["current_address"] = select(
        accepted, select(loop, x["sample_address"], next_address), state["current_address"]
    )
    state["bytes_remaining"] = select(
        accepted,
        select(loop, x["sample_length"], x["bytes_remaining"] - 1),
        state["bytes_remaining"],
    )
    state["irq"] = either(state["irq"], both(finished, both(1 - x["loop"], x["irq_enabled"])))

    shift_bits = bits(x["shift_register"], 8)
    change = select(
        shift_bits[0],
        2 * (1 - stepglu(1, x["output_level"] - 126)),
        -2 * stepglu(1, x["output_level"] - 2),
    )
    state["output_level"] = state["output_level"] + gate(both(clock, 1 - x["silence"]), change)
    last = both(clock, equals(x["bits_remaining"], 1))
    load = both(last, 1 - x["buffer_empty"])
    shifted = sum((shift_bits[i] * (1 << (i - 1)) for i in range(1, 8)), Expression())
    state["shift_register"] = select(
        clock, select(load, x["sample_buffer"], shifted), x["shift_register"]
    )
    state["bits_remaining"] = select(
        clock, select(last, 8, x["bits_remaining"] - 1), x["bits_remaining"]
    )
    state["silence"] = select(last, x["buffer_empty"], x["silence"])
    state["buffer_empty"] = either(state["buffer_empty"], last)

    active = stepglu(1, state["bytes_remaining"] - 1)
    due = both(active, state["buffer_empty"])
    count_bits = bits(state["bits_remaining"] - 1, 3)
    tail = 2 + sum(gate(bit, state["period"] * (1 << i)) for i, bit in enumerate(count_bits))
    effects = (
        due,
        active,
        tail,
        either(clock, observe),
        select(clock, state["period"] - 1, state["period"]),
        4 * accepted,
    )
    values = (*(state[f] for f in DMC_STATE), *effects)
    outputs = {
        name: persist(value, name=name) for name, value in zip(DMC_OUTPUTS, values, strict=True)
    }
    return current_graph().capture({"EVENT": sum(x.values())}, outputs)


def sweep_target(p, first):
    timer_bits = bits(p["timer"], 11)
    change = sum(
        gate(
            equals(p["sweep_shift"], shift),
            sum((b * (1 << i) for i, b in enumerate(timer_bits[shift:])), Expression()),
        )
        for shift in range(8)
    )
    target = select(p["sweep_negate"], p["timer"] - change - first, p["timer"] + change)
    muted = either(1 - stepglu(1, p["timer"] - 8), stepglu(1, target - 2048))
    return target, muted


def build_channels():
    x = {name: InputDimension(name) for name in INPUTS}
    event = persist(x["event"], name="event_tag")
    quarter = 1 - equals(event, 1)
    half = 1 - equals(event, 0)
    state, levels = {}, {}

    def channel(prefix, fields):
        return {f: x[f"{prefix}_{f}"] for f in fields}

    def envelope(prefix, p):
        zero = equals(p["env_divider"], 0)
        reload = either(p["env_start"], zero)
        decay = reglu(1, p["env_decay"] - 1) + 15 * both(equals(p["env_decay"], 0), p["env_loop"])
        state[f"{prefix}_env_start"] = select(quarter, 0, p["env_start"])
        state[f"{prefix}_env_divider"] = select(
            quarter, select(reload, p["env_period"], p["env_divider"] - 1), p["env_divider"]
        )
        state[f"{prefix}_env_decay"] = select(
            quarter, select(p["env_start"], 15, select(zero, decay, p["env_decay"])), p["env_decay"]
        )
        return select(p["env_constant"], p["env_period"], p["env_decay"])

    def length(prefix, p, halt):
        state[f"{prefix}_length"] = p["length"] - both(
            half, both(1 - halt, stepglu(1, p["length"] - 1))
        )

    for i in range(2):
        prefix = f"pulse{i}"
        p = channel(prefix, PULSE)
        volume = envelope(prefix, p)
        length(prefix, p, p["env_loop"])
        target, muted = sweep_target(p, i == 0)
        zero = equals(p["sweep_divider"], 0)
        update = both(
            half,
            both(
                zero,
                both(
                    p["sweep_enable"],
                    both(stepglu(1, p["sweep_shift"] - 1), both(1 - muted, stepglu(1, target))),
                ),
            ),
        )
        state[f"{prefix}_timer"] = select(update, target, p["timer"])
        state[f"{prefix}_sweep_reload"] = select(half, 0, p["sweep_reload"])
        state[f"{prefix}_sweep_divider"] = select(
            half,
            select(either(zero, p["sweep_reload"]), p["sweep_period"], p["sweep_divider"] - 1),
            p["sweep_divider"],
        )
        pos = p["seq_pos"]
        duty = (
            gate(equals(p["duty"], 0), equals(pos, 1))
            + gate(equals(p["duty"], 1), both(stepglu(1, pos - 1), 1 - stepglu(1, pos - 3)))
            + gate(equals(p["duty"], 2), both(stepglu(1, pos - 1), 1 - stepglu(1, pos - 5)))
            + gate(equals(p["duty"], 3), either(equals(pos, 0), stepglu(1, pos - 3)))
        )
        active = both(p["enabled"], both(stepglu(1, p["length"] - 1), both(1 - muted, duty)))
        levels[f"{prefix}_level"] = gate(active, volume)

    t = channel("triangle", TRIANGLE)
    length("triangle", t, t["length_halt"])
    state["triangle_linear_reload"] = select(
        quarter, both(t["linear_reload"], t["length_halt"]), t["linear_reload"]
    )
    state["triangle_linear_counter"] = select(
        quarter,
        select(t["linear_reload"], t["linear_reload_value"], reglu(1, t["linear_counter"] - 1)),
        t["linear_counter"],
    )
    active = both(
        t["enabled"],
        both(
            stepglu(1, t["length"] - 1),
            both(stepglu(1, t["linear_counter"] - 1), stepglu(1, t["timer"] - 2)),
        ),
    )
    levels["triangle_level"] = gate(
        active, reglu(1, 15 - t["seq_pos"]) + reglu(1, t["seq_pos"] - 16)
    )
    n = channel("noise", NOISE)
    volume = envelope("noise", n)
    length("noise", n, n["env_loop"])
    low = n["bit0"]
    levels["noise_level"] = gate(
        both(n["enabled"], both(stepglu(1, n["length"] - 1), 1 - low)), volume
    )
    outputs = {
        name: persist(state[f], name=name) for name, f in zip(OUTPUTS, STATE_OUTPUTS, strict=True)
    }
    outputs.update({name: persist(levels[name], name=name) for name in LEVELS})
    return {"STATE": sum(x.values())}, outputs


@construction
def channel_graph():
    return current_graph().capture(*build_channels())


SAMPLE_INPUTS = (*INPUTS, "dmc_output", "previous_mix", "previous_filter")
SAMPLE_OUTPUTS = ("mixed", "filtered", "sample")
DAC_INPUTS = ("pulse", "triangle", "noise", "dmc")


def reciprocal_unit_interval(value):
    """Four Newton steps for 1/x on [1, 2], using only ReGLU products.

    Starting at 2/3 bounds the initial relative error by 1/3. Each step
    squares that error, giving a real-arithmetic bound of 3**-16 before
    float32 rounding. Both factors remain positive over this domain.
    """
    result = 2 / 3
    for _ in range(4):
        result = reglu(result, 2 - reglu(value, result))
    return result


def mixed_level(pulse, triangle, noise, dmc):
    p = (95.88 / 8128) * reglu(pulse, reciprocal_unit_interval(1 + (100 / 8128) * pulse))
    tnd = (1 / 8227) * triangle + (1 / 12241) * noise + (1 / 22638) * dmc
    return persist(p + 159.79 * reglu(tnd, reciprocal_unit_interval(1 + 100 * tnd)), name="mixed")


@construction
def sample_graph():
    tokens, levels = build_channels()
    extra = {name: InputDimension(name) for name in SAMPLE_INPUTS[len(INPUTS):]}
    mixed = mixed_level(
        levels["pulse0_level"] + levels["pulse1_level"],
        levels["triangle_level"], levels["noise_level"], extra["dmc_output"],
    )
    previous = extra["previous_filter"]
    # Small leakage avoids rounding the pole coefficient near one. Stage the
    # signed difference through ReGLUs before adding feedback so residual
    # projections cannot lose it by summing the larger terms first.
    leak = reglu(0.0005, previous) - reglu(0.0005, -previous)
    difference = mixed - extra["previous_mix"]
    delta = reglu(1, difference) - reglu(1, -difference)
    filtered = persist(previous - leak + delta, name="filtered")
    line = 2 * filtered
    sample = persist(line - reglu(1, line - 1) + reglu(1, -1 - line), name="sample")
    tokens["STATE"] += sum(extra.values())
    return current_graph().capture(tokens, {
        "mixed": mixed, "filtered": filtered, "sample": sample,
        **{n: levels[n] for n in LEVELS},
    })


@construction
def dac_graph():
    x = {name: InputDimension(name) for name in DAC_INPUTS}
    mixed = mixed_level(*(x[n] for n in DAC_INPUTS))
    return current_graph().capture({"DAC": sum(x.values())}, {"mixed": mixed})


@cache
def sample_engine():
    def graphs():
        full = TokenGraph(sample_graph(), SAMPLE_INPUTS, (*SAMPLE_OUTPUTS, *LEVELS))
        return {
            "sample": full.project(SAMPLE_OUTPUTS), "levels": full.project(LEVELS),
            "dac": TokenGraph(dac_graph(), DAC_INPUTS, ("mixed",)),
        }

    return CompiledGraph(graphs, tag="apu_sample", sources=(__name__,), memory_size=1, dtype="float32")


@construction
def frame_graph():
    x = {name: InputDimension(name) for name in FRAME_INPUTS}
    event = persist(x["event"], name="event_tag")
    clock, write, read = (equals(event, i) for i in range(3))
    value_bits = bits(x["value"], 8)
    five, inhibit = (
        select(write, value_bits[7], x["five_step"]),
        select(write, value_bits[6], x["irq_inhibit"]),
    )
    last = equals(x["seq_step"], 3)
    reset = either(write, both(clock, last))
    step = select(reset, 0, x["seq_step"] + clock)
    fire = both(clock, both(last, both(1 - five, 1 - inhibit)))
    irq = select(either(read, both(write, inhibit)), 0, either(x["frame_irq"], fire))
    immediate = both(write, five)
    quarter = either(clock, immediate)
    half = either(both(clock, equals(x["seq_step"], 1) + last), immediate)
    deadline = sum(
        gate(equals(step, i), d) for i, d in enumerate((7457, 14913, 22371, 29829))
    ) + gate(both(equals(step, 3), five), 7452)
    values = (five, inhibit, irq, step, reset, quarter, half, deadline)
    outputs = {
        name: persist(value, name=name) for name, value in zip(FRAME_OUTPUTS, values, strict=True)
    }
    return current_graph().capture({"EVENT": sum(x.values())}, outputs)


@cache
def audio_engine(maximum=MAX_CYCLES):
    """Specialize clock widths to the console's immutable sample deadline."""
    if not 1 <= maximum <= MAX_CYCLES:
        raise ValueError("oscillator span bound must be in 1..429")

    def graphs():
        from transformer_nes.nes import oscillators

        token = TokenGraph(channel_graph(), INPUTS, OUTPUTS + LEVELS)
        experts = {
            **{f"mode{i}": {"noise_mode_tag": i} for i in range(2)},
            **{f"tick_mode{i}": {"noise_mode_tag": i, "noise_steps_tag": 1} for i in range(2)},
        }
        return {
            "oscillate": TokenGraph(oscillators.clock_graph(maximum), oscillators.CLOCK_INPUTS,
                                    oscillators.CLOCK_OUTPUTS),
            "tick": TokenGraph(oscillators.clock_graph(1), oscillators.CLOCK_INPUTS,
                               oscillators.CLOCK_OUTPUTS),
            "noise": TokenGraph(oscillators.noise_graph(maximum), oscillators.NOISE_INPUTS,
                                oscillators.NOISE_OUTPUTS, experts),
            "control": replace(
                token.project(OUTPUTS),
                experts={name: {"event_tag": i} for i, name in enumerate(EVENTS)},
            ),
            "frame": TokenGraph(
                frame_graph(),
                FRAME_INPUTS,
                FRAME_OUTPUTS,
                {name: {"event_tag": i} for i, name in enumerate(FRAME_EVENTS)},
            ),
            "program": TokenGraph(
                program_graph(),
                PROGRAM_INPUTS,
                PROGRAM_OUTPUTS,
                {f"write_{p}": {"port_tag": p, "writing": 1} for p in WRITE_PORTS},
            ),
            "dmc": TokenGraph(
                dmc_graph(),
                DMC_INPUTS,
                DMC_OUTPUTS,
                {
                    **{name: {"event_tag": i} for i, name in enumerate(DMC_EVENTS) if i},
                    **{f"write_{p}": {"event_tag": 0, "port_tag": p} for p in DMC_WRITE_PORTS},
                },
            ),
        }

    return CompiledGraph(graphs, tag=f"apu_{maximum}", sources=(__name__,), memory_size=1)
