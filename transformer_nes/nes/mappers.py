"""Neural mapper register transitions, bank layouts and bus effects.

Geometry is specialized at construction; mutable state stays with the caller.
MMC3/MMC5 IRQ circuits also emit qualified-edge deadlines for timing previews.
"""

from functools import cache
from math import gcd

from transformer_nes.graph.core import (
    InputDimension,
    construction,
    current_graph,
    persist,
    reglu,
    stepglu,
)
from transformer_nes.graph.ops import both, either, equals, remainder, select
from transformer_nes.model.graph_engine import CompiledGraph, TokenGraph

STATE = ("counter", "target", "phase", "pending", "enabled")
INPUTS = (*STATE, "event", "value")
OUTPUTS = (
    *(f"new_{name}" for name in STATE),
    "irq",
    "read_value",
    "predictable",
    "next_edge",
    "irq_write",
)
EVENTS = ("clock", "idle", "target", "reload", "enable", "disable", "ack")


@construction
def irq_graph(mmc5: bool):
    x = {name: InputDimension(name) for name in INPUTS}
    counter, target, phase, pending, enabled = (x[name] for name in STATE)
    event = {name: equals(x["event"], i) for i, name in enumerate(EVENTS)}
    clock = event["clock"]
    if mmc5:
        clock_counter = reglu(counter + 1, phase)
        new_counter = counter + reglu(clock_counter - counter, clock)
        new_phase = phase + reglu(1 - phase, clock) - reglu(phase, event["idle"])
        fire = both(both(clock, 1 - equals(target, 0)), equals(clock_counter - target, 0))
        control = stepglu(1, x["value"] - 128)
        read_value = reglu(128 * pending + 64 * phase, event["ack"])
    else:
        clock_counter = select(either(phase, equals(counter, 0)), target, counter - 1)
        new_counter = (
            counter + reglu(clock_counter - counter, clock) - reglu(counter, event["reload"])
        )
        new_phase = phase - reglu(phase, clock) + reglu(1 - phase, event["reload"])
        fire = both(both(clock, enabled), equals(clock_counter, 0))
        control = 1
        read_value = 0
    new_target = target + reglu(x["value"] - target, event["target"])
    new_enabled = (
        enabled + reglu(control - enabled, event["enable"]) - reglu(enabled, event["disable"])
    )
    new_pending = reglu(either(pending, fire), 1 - event["ack"] - event["disable"])
    irq = both(new_enabled, new_pending)
    irq_write = either(both(fire, enabled), event["disable"])
    if mmc5:
        irq_write = either(irq_write, either(event["enable"], event["ack"]))
        distance = select(new_phase, new_target - new_counter, new_target + 1)
        predictable = reglu(1, new_enabled - new_pending - equals(new_target, 0))
        next_edge = reglu(predictable, distance)
    else:
        distance = select(either(new_phase, equals(new_counter, 0)), new_target + 1, new_counter)
        predictable = new_enabled
        next_edge = reglu(new_enabled, distance)
    values = (
        new_counter,
        new_target,
        new_phase,
        new_pending,
        new_enabled,
        irq,
        read_value,
        predictable,
        next_edge,
        irq_write,
    )
    outputs = {name: persist(value, name=name) for name, value in zip(OUTPUTS, values, strict=True)}
    return current_graph().capture({"STATE": sum(x.values())}, outputs)


@cache
def mapper_control_engine():
    def graphs():
        graphs = {
            name: TokenGraph(
                irq_graph(mmc5),
                INPUTS,
                OUTPUTS,
                {event: {"event": i} for i, event in enumerate(EVENTS)},
            )
            for name, mmc5 in (("mmc3", False), ("mmc5", True))
        }
        for chip in BANK_PORTS:
            access = TokenGraph(
                access_graph(chip), ACCESS_INPUTS, ACCESS_OUTPUTS,
                {"read": {"access_writing": 0}, "write": {"access_writing": 1}},
            )
            graphs[f"access_{chip}"] = access
            graphs[f"address_{chip}"] = TokenGraph(
                access.program_graph, ACCESS_INPUTS, ACCESS_OUTPUTS
            ).project(ADDRESS_OUTPUTS)
        return graphs

    return CompiledGraph(graphs, tag="mapper_control", sources=(__name__,), memory_size=1)


BANK_STATE = {
    1: ("control", "prg", "chr0", "chr1", "shift"),
    4: ("select", "prg_mode", "chr_mode", *(f"reg{i}" for i in range(8)), "mirror"),
    5: (
        "prg_mode",
        "chr_mode",
        "protect1",
        "protect2",
        "exram_mode",
        "mirror",
        "fill_tile",
        "fill_color",
        "ram_bank",
        *(f"prg{i}" for i in range(4)),
        *(f"chr{i}" for i in range(12)),
        "upper",
        "mul_a",
        "mul_b",
    ),
    9: ("prg", *(f"chr{i}" for i in range(4)), "mirror"),
    10: ("prg", *(f"chr{i}" for i in range(4)), "mirror"),
}
BANK_PORTS = {
    1: (0x8000, 0xA000, 0xC000, 0xE000),
    4: (0x8000, 0x8001, 0xA000),
    5: (
        *range(0x5100, 0x5108),
        *range(0x5113, 0x5118),
        *range(0x5120, 0x512C),
        0x5130,
        0x5205,
        0x5206,
    ),
    9: (0xA000, 0xB000, 0xC000, 0xD000, 0xE000, 0xF000),
    10: (0xA000, 0xB000, 0xC000, 0xD000, 0xE000, 0xF000),
}

ACCESS_INPUTS = (
    "addr_lo", "addr_hi", "is_write", "original",
    *(f"bank{i}" for i in range(4)), *(f"ram{i}" for i in range(4)),
    "ram_bank", "writable", "mul_lo", "mul_hi",
)
ACCESS_OUTPUTS = (
    "port_index", "bank_event", "irq_event", "schedule_changed",
    "read_source", "read_index", "read_value", "write_source", "repair",
)
ADDRESS_OUTPUTS = ACCESS_OUTPUTS[4:7]
# Passive storage handles consumed by the host. IRQ reads consume the
# acknowledgement event's neural value; literal reads use read_value.
LITERAL, ROM, RAM, EXRAM, IRQ = range(5)


@construction
def access_graph(chip: int):
    """Decode mapper bus effects from split addresses and live bank state.

    The address projection can repair a CPU write after a bank transition.
    """
    x = {name: InputDimension(name) for name in ACCESS_INPUTS}
    lo, hi = x["addr_lo"], x["addr_hi"]
    writing = persist(x["is_write"], name="access_writing")

    def high_range(first, end):
        return stepglu(1, hi - first) - stepglu(1, hi - end)

    def at(address):
        return both(equals(hi, address >> 8), equals(lo, address & 255))

    odd = remainder(lo, 2, 255) if chip == 4 else 0
    matches = []
    for port in BANK_PORTS[chip]:
        if chip == 5:
            match = at(port)
        else:
            stride = 32 if chip in (1, 4) else 16
            match = high_range(port >> 8, (port >> 8) + stride)
            if chip == 4:
                match = both(match, odd if port & 1 else 1 - odd)
        matches.append(both(writing, match))
    port_index = -1 + sum(reglu(i + 1, match) for i, match in enumerate(matches))
    bank_event = 1 + (stepglu(1, x["original"]) if chip == 1 else 0)

    irq_event, schedule = -1, 0
    irq_read = both(1 - writing, at(0x5204)) if chip == 5 else 0
    if chip == 4:
        target = both(writing, high_range(0xC0, 0xE0))
        enable = both(writing, high_range(0xE0, 0x100))
        # event indices in EVENTS: target=2, reload=3, enable=4, disable=5
        irq_event = -1 + reglu(3 + odd, target) + reglu(6 - odd, enable)
        schedule = target + both(enable, odd)
    elif chip == 5:
        target, enable = both(writing, at(0x5203)), both(writing, at(0x5204))
        irq_event = -1 + 3 * target + 5 * enable + 7 * irq_read
        schedule = target + enable

    unit = 0x4000 if chip in (1, 10) else 0x2000
    boundaries = [stepglu(1, hi - (0x80 + i * (unit >> 8)))
                  for i in range(0x8000 // unit)]
    mapped = boundaries[0]
    present = [boundaries[i] - (boundaries[i + 1] if i + 1 < len(boundaries) else 0)
               for i in range(len(boundaries))]
    slot = sum(boundaries[1:])
    bank = sum(reglu(x[f"bank{i}"], match) for i, match in enumerate(present))
    offset = 256 * (hi - 0x80 - (unit >> 8) * slot) + lo
    read_index = reglu(unit * bank + offset, mapped)
    read_source, read_value, write_source = mapped * ROM, 0, 0
    repair = both(writing, mapped)
    if chip == 5:
        ram = sum(reglu(x[f"ram{i}"], match) for i, match in enumerate(present))
        ram_window, exram = high_range(0x60, 0x80), high_range(0x5C, 0x60)
        read_index = (read_index + reglu(0x2000 * x["ram_bank"] + 256 * (hi - 0x60) + lo, ram_window)
                      + reglu(256 * (hi - 0x5C) + lo, exram))
        read_source = mapped + ram + RAM * ram_window + EXRAM * exram + IRQ * irq_read
        read_value = reglu(x["mul_lo"], at(0x5205)) + reglu(x["mul_hi"], at(0x5206))
        writable = both(x["writable"], ram_window + ram)
        write_source = reglu(RAM * writable + EXRAM * exram, writing)
        repair = both(writing, mapped + both(ram_window, 1 - x["writable"]))
    values = (port_index, bank_event, irq_event, schedule, read_source, read_index,
              read_value, write_source, repair)
    outputs = {name: persist(value, name=name)
               for name, value in zip(ACCESS_OUTPUTS, values, strict=True)}
    return current_graph().capture({"ACCESS": sum(x.values())}, outputs)


def high_bit(value):
    return stepglu(1, value - 128)


@construction
def bank_transition(chip: int):
    names = (*BANK_STATE[chip], "event", "port", "value", "original")
    x = {name: InputDimension(name) for name in names}
    reset = equals(x["event"], 0)
    value = x["value"]
    state = {name: x[name] for name in BANK_STATE[chip]}
    port = {p: equals(x["port"], p) for p in BANK_PORTS[chip]}
    if chip == 1:
        first = select(equals(x["event"], 2), x["original"], value)
        first_reset = high_bit(first)
        reset_any = either(first_reset, both(equals(x["event"], 2), high_bit(value)))
        complete = remainder(x["shift"], 2, 31)
        shifted = (x["shift"] - complete) * 0.5 + 16 * remainder(first, 2, 255)
        commit = both(complete, 1 - first_reset)
        for name, p in zip(("control", "chr0", "chr1", "prg"), BANK_PORTS[chip], strict=True):
            state[name] = select(both(commit, port[p]), shifted, x[name])
        # Reset forces PRG mode 3 while preserving the CHR mode and mirror.
        forced = remainder(state["control"], 4, 31) + 12 + 16 * stepglu(1, state["control"] - 16)
        state["control"] = select(reset_any, forced, state["control"])
        state["shift"] = select(either(reset_any, complete), 16, shifted)
        prg = either(reset_any, both(commit, port[0x8000] + port[0xE000]))
        chr_ = both(commit, port[0x8000] + port[0xA000] + port[0xC000])
        mirror = both(commit, port[0x8000])
        initial = (12 + value, 0, 0, 0, 16)
    elif chip == 4:
        mode7 = high_bit(value)
        mode6 = stepglu(1, value - 64) - mode7 + stepglu(1, value - 192)
        state["select"] = select(port[0x8000], remainder(value, 8, 255), x["select"])
        state["prg_mode"] = select(port[0x8000], mode6, x["prg_mode"])
        state["chr_mode"] = select(port[0x8000], mode7, x["chr_mode"])
        for i in range(8):
            state[f"reg{i}"] = select(
                both(port[0x8001], equals(x["select"], i)), value, x[f"reg{i}"]
            )
        state["mirror"] = select(port[0xA000], 2 + remainder(value, 2, 255), x["mirror"])

        def changed(a, b):
            return reglu(1, a - b) + reglu(1, b - a)

        prg = either(
            both(port[0x8000], changed(mode6, x["prg_mode"])),
            both(port[0x8001], stepglu(1, x["select"] - 6)),
        )
        chr_ = either(
            both(port[0x8000], changed(mode7, x["chr_mode"])),
            both(port[0x8001], 1 - stepglu(1, x["select"] - 6)),
        )
        mirror = port[0xA000]
        initial = (0, 0, 0, 0, 2, 4, 5, 6, 7, 0, 1, value)
    elif chip == 5:
        for name, p in zip(BANK_STATE[chip], BANK_PORTS[chip], strict=True):
            modulus = (
                4
                if name
                in (
                    "prg_mode",
                    "chr_mode",
                    "protect1",
                    "protect2",
                    "exram_mode",
                    "fill_color",
                    "upper",
                )
                else 8
                if name == "ram_bank"
                else 256
            )
            state[name] = select(port[p], remainder(value, modulus, 255), x[name])
        prg = sum(port[p] for p in (0x5100, 0x5102, 0x5103, *range(0x5113, 0x5118)))
        chr_ = sum(port[p] for p in (0x5101, *range(0x5120, 0x512C), 0x5130))
        mirror = sum(port[p] for p in (*range(0x5104, 0x5108), 0x5130))
        initial = (3, 3, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 255, *range(8), *range(4), 0, 255, 255)
    else:
        state["prg"] = select(port[0xA000], remainder(value, 16, 255), x["prg"])
        for i in range(4):
            state[f"chr{i}"] = select(
                port[0xB000 + i * 0x1000], remainder(value, 32, 255), x[f"chr{i}"]
            )
        state["mirror"] = select(port[0xF000], 2 + remainder(value, 2, 255), x["mirror"])
        prg, chr_, mirror = port[0xA000], sum(port[p] for p in BANK_PORTS[chip][1:5]), port[0xF000]
        initial = (0, 0, 0, 0, 0, value)
    outputs = {
        f"new_{name}": persist(select(reset, initial[i], state[name]), name=f"new_{name}")
        for i, name in enumerate(BANK_STATE[chip])
    }
    dirty = prg + 2 * chr_ + 4 * mirror
    if chip == 5:
        dirty = dirty + 8 * (port[0x5205] + port[0x5206])
    outputs["dirty"] = persist(select(reset, 15 if chip == 5 else 7, dirty), name="dirty")
    return current_graph().capture({"STATE": sum(x.values())}, outputs)


@construction
def bank_layout(chip: int, n_prg: int, n_chr: int):
    x = {name: InputDimension(name) for name in BANK_STATE[chip]}
    if chip == 1:
        mode = remainder(x["control"], 16, 31)
        pair = 1 - stepglu(1, mode - 8)
        fixed_first = stepglu(1, mode - 8) - stepglu(1, mode - 12)
        bank = remainder(remainder(x["prg"], 16, 31), n_prg, 15)
        even = bank - remainder(bank, 2, 15)
        prg = (
            select(pair, even, select(fixed_first, 0, bank)),
            select(pair, even + 1, select(fixed_first, bank, n_prg - 1)),
        )
        chr_mode = stepglu(1, x["control"] - 16)
        base4 = [4 * remainder(x[f"chr{i}"], n_chr // 4, 31) for i in range(2)]
        raw8 = x["chr0"] - remainder(x["chr0"], 2, 31)
        base8 = 4 * remainder(raw8, n_chr // 4, 30)
        chr_ = []
        for i in range(8):
            address = select(chr_mode, base4[i // 4] + i % 4, base8 + i)
            chr_.append(address - n_chr * stepglu(1, address - n_chr))
        mirror = remainder(x["control"], 4, 31)
    elif chip == 4:
        r6, r7 = (remainder(x[f"reg{i}"], n_prg, 255) for i in (6, 7))
        prg = (
            select(x["prg_mode"], n_prg - 2, r6),
            r7,
            select(x["prg_mode"], r6, n_prg - 2),
            n_prg - 1,
        )
        pairs = []
        for i in range(2):
            even = x[f"reg{i}"] - remainder(x[f"reg{i}"], 2, 255)
            pairs.extend(remainder(even + j, n_chr, 255) for j in range(2))
        singles = [remainder(x[f"reg{i}"], n_chr, 255) for i in range(2, 6)]
        chr_ = [select(x["chr_mode"], singles[i], pairs[i]) for i in range(4)]
        chr_ += [select(x["chr_mode"], pairs[i], singles[i]) for i in range(4)]
        mirror = x["mirror"]
    elif chip == 5:
        prg, chr_, extra = mmc5_layout(x, n_prg, n_chr)
        mirror = x["mirror"]
    else:
        slots = 4 if chip == 9 else 2
        prg = (remainder(x["prg"], n_prg, 15), *(n_prg - (slots - i) for i in range(1, slots)))
        chr_ = [remainder(x[f"chr{i}"], n_chr, 31) for i in range(4)]
        mirror = x["mirror"]
    values = {
        **{f"map_prg{i}": v for i, v in enumerate(prg)},
        **{f"map_chr{i}": v for i, v in enumerate(chr_)},
        "map_mirror": mirror,
    }
    if chip == 5:
        values.update(extra)
    return current_graph().capture(
        {"STATE": sum(x.values())},
        {name: persist(value, name=name) for name, value in values.items()},
    )


def mmc5_layout(x, n_prg: int, n_chr: int):
    mode = [equals(x["prg_mode"], i) for i in range(4)]
    registers = [remainder(x[f"prg{i}"], 128, 255) for i in range(4)]
    sources = ((3, 3, 3, 3), (1, 1, 3, 3), (1, 1, 2, 3), (0, 1, 2, 3))
    spans = ((4, 4, 4, 4), (2, 2, 2, 2), (2, 2, 1, 1), (1, 1, 1, 1))
    prg, ram = [], []
    for slot in range(4):
        bank = is_ram = 0
        for m in range(4):
            source, span = sources[m][slot], spans[m][slot]
            base = registers[source] - remainder(registers[source], span, 127)
            bank = bank + reglu(base + slot % span, mode[m])
            if source != 3:
                is_ram = is_ram + both(mode[m], 1 - high_bit(x[f"prg{source}"]))
        ram.append(is_ram)
        prg.append(select(is_ram, remainder(bank, 8, 127), remainder(bank, n_prg, 127)))
    chr_mode = [equals(x["chr_mode"], i) for i in range(4)]
    chr_ = []
    for background in (False, True):
        for slot in range(8):
            address = 0
            for m, span in enumerate((8, 4, 2, 1)):
                source = slot // span * span + span - 1
                source = 8 + source % 4 if background else source
                # Reduce before expansion. Geometry can discard upper bits
                # and entire registers (e.g. an 8 KiB CHR-RAM map) exactly.
                period = n_chr // gcd(n_chr, span)
                upper = 256 % period
                register = x[f"chr{source}"] + upper * x["upper"]
                base = span * remainder(register, period, 255 + 3 * upper)
                mapped = remainder(base + slot % span, n_chr, period * span - 1)
                address = address + reglu(mapped, chr_mode[m])
            chr_.append(address)
    product = reglu(x["mul_a"], x["mul_b"])
    quotient = 0
    for shift in reversed(range(8)):
        bit = stepglu(1, product - (256 << shift))
        quotient = quotient + (1 << shift) * bit
        product = product - (256 << shift) * bit
    extra = {
        **{f"map_ram{i}": value for i, value in enumerate(ram)},
        "map_ram_bank": x["ram_bank"],
        "map_writable": both(equals(x["protect1"], 2), equals(x["protect2"], 1)),
        **{f"map_{name}": x[name] for name in ("exram_mode", "fill_tile", "fill_color", "upper")},
        "map_mul_lo": product,
        "map_mul_hi": quotient,
    }
    return prg, chr_, extra


@cache
def mapper_bank_engine(chip: int, n_prg: int, n_chr: int):
    if chip not in BANK_STATE or n_prg < 1 or n_chr < 1 or (chip == 1 and n_chr % 4):
        raise ValueError("invalid mapper bank geometry")

    def graphs():
        transition = bank_transition(chip)
        layout = bank_layout(chip, n_prg, n_chr)
        experts = {"reset": {"event": 0}}
        events = ((1, "write"), (2, "rmw")) if chip == 1 else ((1, "write"),)
        for i, name in events:
            experts.update(
                {f"{name}_{port:X}": {"event": i, "port": port} for port in BANK_PORTS[chip]}
            )
        return {
            "transition": TokenGraph(
                transition,
                (*BANK_STATE[chip], "event", "port", "value", "original"),
                (*transition.output_tokens,),
                experts,
            ),
            "layout": TokenGraph(layout, BANK_STATE[chip], (*layout.output_tokens,)),
        }

    return CompiledGraph(
        graphs, tag=f"mapper_banks_{chip}_{n_prg}_{n_chr}", sources=(__name__,), memory_size=1
    )
