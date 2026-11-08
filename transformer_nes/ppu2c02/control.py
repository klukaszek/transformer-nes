"""Neural PPU event, bus, composition and DMA effects.

Inputs use bounded frame phases and flag/pin ages. The host applies returned
state, timestamp and transfer effects on its absolute clock.
"""

from dataclasses import replace
from functools import cache

from transformer_nes.graph.core import InputDimension, construction, current_graph, persist, stepglu
from transformer_nes.graph.ops import both, either, equals, extract_bits, gate, remainder, select
from transformer_nes.model.graph_engine import CompiledGraph, TokenGraph
from transformer_nes.ppu2c02.registers import LOOPY_FIELDS

EVENTS = ("access", "vertical_copy", "sprite_line", "odd_skip", "vblank", "clear")
INPUTS = (
    "event",
    "mask",
    "ctrl",
    "odd",
    "phase",
    "flag_age",
    "vblank",
    "suppressed",
    "nmi_edge",
    "nmi_delayed",
    *(f"t_{f}" for f in LOOPY_FIELDS),
    *(f"v_{f}" for f in LOOPY_FIELDS),
    "oam_addr",
    "clear_vblank",
    "nmi_rise",
    "nmi_fall",
    "status_pending",
)
OUTPUTS = (
    *(f"new_v_{f}" for f in LOOPY_FIELDS),
    "new_oam_addr",
    "new_vblank",
    "new_suppressed",
    "new_nmi_edge",
    "new_nmi_delayed",
    "set_stamp",
    "clear_stamp",
    "clear_status",
    "status_changed",
    "skip_dot",
    "rendering",
)
BUS_EVENTS = ("pin", "background", "sprite")
BUS_INPUTS = ("event", "old_high", "low_age", "level", "ctrl", "extended")
BUS_OUTPUTS = ("new_high", "stamp_low", "rise", "rise_offset")
SPAN_INPUTS = ("old_high", "low_age", "ctrl", "extended", "count")
SPAN_OUTPUTS = (*BUS_OUTPUTS, "last_low_offset", "first_stamp_low")
ACCESS_INPUTS = (
    "reg",
    "is_write",
    "rendering",
    "old_w",
    "old_ctrl",
    "old_mask",
    "new_ctrl",
    "new_mask",
    "new_v_fy",
    "old_v_cx",
    "old_v_cy",
    "old_v_nt",
    "old_v_fy",
    "latch_active",
    "wide_low",
    "latch_lo",
    "latch_hi",
)
ACCESS_OUTPUTS = (
    "drive_a12",
    "a12_high",
    "new_latch_lo",
    "new_latch_hi",
    "chr_changed",
    "schedule_changed",
    "catchup_delay",
)
DELAY_OUTPUTS = ("catchup_delay",)
VIEW_INPUTS = ("exram_mode", "fill_tile", "fill_color", "ctrl")
VIEW_OUTPUTS = (
    "extended",
    "exram_visible",
    "fill_tile_value",
    "fill_attribute",
    "fill_split",
    "background_set",
)


@construction
def bus_graph():
    """Pin or ordered low/high pair; age is saturated to [-1, 11] dots.

    A pair drives low at offset 0 and its pattern pin at offset 4. One
    forward preserves both the low timestamp and the qualified rise dot.
    """
    x = {name: InputDimension(name) for name in BUS_INPUTS}
    event = persist(x["event"], name="bus_event")
    pair = 1 - equals(event, 0)
    bg = equals(event, 1)
    table = extract_bits(x["ctrl"], "bus_ctrl")[3]  # bit 4
    level = select(bg, both(table, 1 - x["extended"]), x["level"])
    stamp = both(x["old_high"], either(pair, 1 - level))
    qualified = select(pair, stepglu(1, x["low_age"] - 7), stepglu(1, x["low_age"] - 11))
    rise = both(level, both(1 - x["old_high"], qualified))
    values = (level, stamp, rise, 4 * pair)
    outputs = {
        name: persist(value, name=name) for name, value in zip(BUS_OUTPUTS, values, strict=True)
    }
    return current_graph().capture({"BUS": sum(x.values())}, outputs)


@construction
def background_span_graph():
    """Compose 1..32 background pairs spaced eight dots apart.

    Only the first pair can qualify a rise. Readouts preserve its stamp for
    early IRQ stops and the final stamp for a completed run.
    """
    x = {name: InputDimension(name) for name in SPAN_INPUTS}
    table = extract_bits(x["ctrl"], "span_ctrl")[3]
    level = both(table, 1 - x["extended"])
    later_stamp = both(stepglu(1, x["count"] - 2), level)
    first_stamp = x["old_high"]
    rise = both(level, both(1 - x["old_high"], stepglu(1, x["low_age"] - 7)))
    values = (level, either(first_stamp, later_stamp), rise, 4,
              gate(later_stamp, 8 * (x["count"] - 1)), first_stamp)
    outputs = {
        name: persist(value, name=name) for name, value in zip(SPAN_OUTPUTS, values, strict=True)
    }
    return current_graph().capture({"SPAN": sum(x.values())}, outputs)


@construction
def access_bus_graph():
    """Register-side bus/latch effects and memory/schedule invalidation."""
    x = {name: InputDimension(name) for name in ACCESS_INPUTS}
    reg = persist(x["reg"], name="access_reg")
    writing = persist(x["is_write"], name="access_writing")
    ctrl = extract_bits(x["new_ctrl"], "new_ctrl")
    old_ctrl = extract_bits(x["old_ctrl"], "old_ctrl")
    mask = extract_bits(x["new_mask"], "new_mask")
    old_mask = extract_bits(x["old_mask"], "old_mask")
    enabled = either(mask[3], mask[4])
    loaded = both(both(equals(reg, 6), writing), both(x["old_w"], 1 - enabled))
    stepped = both(equals(reg, 7), 1 - x["rendering"])
    fy = x["new_v_fy"]
    fy = fy - 4 * stepglu(1, fy - 4)
    high = fy - 2 * stepglu(1, fy - 2)

    old_fy = x["old_v_fy"] - 4 * stepglu(1, x["old_v_fy"] - 4)
    chr_region = 1 - stepglu(1, old_fy - 2)
    half = stepglu(1, old_fy - 1)
    fetch = both(both(equals(reg, 7), 1 - writing), both(x["latch_active"], chr_region))
    cx = x["old_v_cx"]
    fd = both(equals(x["old_v_cy"], 30), stepglu(1, cx - 24))
    fe = both(equals(x["old_v_cy"], 31), both(stepglu(1, cx - 8), 1 - stepglu(1, cx - 16)))
    fd = both(fd, either(either(half, x["wide_low"]), equals(cx, 24)))
    fe = both(fe, either(either(half, x["wide_low"]), equals(cx, 8)))
    fetch = both(fetch, equals(x["old_v_nt"], 3))
    lo, hi = x["latch_lo"], x["latch_hi"]
    for upper in (0, 1):
        active = both(fetch, half if upper else 1 - half)
        old = hi if upper else lo
        new = select(both(active, fd), 0, select(both(active, fe), 1, old))
        if upper:
            hi = new
        else:
            lo = new
    ctrl_changed = 1 - equals(x["old_ctrl"] - x["new_ctrl"], 0)
    changed = either(1 - equals(lo - x["latch_lo"], 0), 1 - equals(hi - x["latch_hi"], 0))
    schedule = both(both(equals(reg, 4), writing), ctrl[2])  # OAM in 8x16 mode
    for old, new in [(old_ctrl[i], ctrl[i]) for i in (2, 3, 4)] + [
        (old_mask[i], mask[i]) for i in (3, 4)
    ]:
        schedule = either(schedule, 1 - equals(old - new, 0))
    values = (
        either(loaded, stepped),
        high,
        lo,
        hi,
        either(changed, ctrl_changed),
        schedule,
        both(equals(reg, 1), writing),
    )
    outputs = {
        name: persist(value, name=name) for name, value in zip(ACCESS_OUTPUTS, values, strict=True)
    }
    return current_graph().capture({"ACCESS": sum(x.values())}, outputs)


@construction
def view_graph():
    """MMC5 storage-view descriptor; payload delivery remains a bulk copy."""
    x = {name: InputDimension(name) for name in VIEW_INPUTS}
    values = (
        equals(x["exram_mode"], 1),
        1 - stepglu(1, x["exram_mode"] - 2),
        x["fill_tile"],
        85 * x["fill_color"],
        960,
        extract_bits(x["ctrl"], "view_ctrl")[2],
    )
    outputs = {
        name: persist(value, name=name) for name, value in zip(VIEW_OUTPUTS, values, strict=True)
    }
    return current_graph().capture({"VIEW": sum(x.values())}, outputs)


@construction
def control_graph():
    x = {name: InputDimension(name) for name in INPUTS}
    event = persist(x["event"], name="event_tag")
    ev = {name: equals(event, i) for i, name in enumerate(EVENTS)}
    mask = extract_bits(x["mask"], "mask")
    rendering = either(mask[3], mask[4])  # bits 4 and 3, MSB first
    nmi_enabled = extract_bits(x["ctrl"], "ctrl")[0]
    horizontal = both(ev["sprite_line"], rendering)
    vertical = both(ev["vertical_copy"], rendering)
    t_nt_hi, v_nt_hi = stepglu(1, x["t_nt"] - 2), stepglu(1, x["v_nt"] - 2)
    nt = (
        x["v_nt"]
        + gate(horizontal, x["t_nt"] - 2 * t_nt_hi - x["v_nt"] + 2 * v_nt_hi)
        + 2 * gate(vertical, t_nt_hi - v_nt_hi)
    )

    clear_flag = both(ev["access"], x["clear_vblank"])
    recent = both(stepglu(1, x["flag_age"]), 1 - stepglu(1, x["flag_age"] - 3))
    suppress = both(clear_flag, both(equals(x["phase"], 241 * 341 + 1), equals(x["flag_age"], -1)))
    set_flag = both(ev["vblank"], 1 - x["suppressed"])
    vblank_after_read = select(clear_flag, 0, x["vblank"])
    delayed = both(
        both(ev["access"], x["nmi_rise"]),
        both(vblank_after_read, 1 - equals(x["phase"], 261 * 341 + 1)),
    )
    cancel = both(ev["access"], both(either(x["clear_vblank"], x["nmi_fall"]), recent))
    clear = ev["clear"]
    values = (
        select(horizontal, x["t_cx"], x["v_cx"]),
        select(vertical, x["t_cy"], x["v_cy"]),
        nt,
        select(vertical, x["t_fy"], x["v_fy"]),
        select(horizontal, 0, x["oam_addr"]),
        select(clear, 0, select(set_flag, 1, vblank_after_read)),
        select(ev["vblank"], 0, either(x["suppressed"], suppress)),
        either(select(cancel, 0, x["nmi_edge"]), both(set_flag, nmi_enabled)),
        either(x["nmi_delayed"], delayed),
        set_flag,
        clear,
        clear,
        set_flag + both(clear, either(x["vblank"], x["status_pending"])),
        both(ev["odd_skip"], both(x["odd"], rendering)),
        rendering,
    )
    outputs = {name: persist(value, name=name) for name, value in zip(OUTPUTS, values, strict=True)}
    return current_graph().capture({"EVENT": sum(x.values())}, outputs)


COMPOSE_INPUTS = ("x", "occupied0", "occupied1", *(f"row{c}" for c in range(8)),
                  *(f"old{c}" for c in range(8)))
COMPOSE_OUTPUTS = (*(f"pixel{c}" for c in range(8)), "new_occupied0", "new_occupied1")

DMA_INPUTS = ("page", "parity", "oam_addr", "ctrl")
DMA_OUTPUTS = (
    "source_base",
    "stall",
    "destination",
    "first_count",
    "second_source",
    "second_count",
    "schedule_changed",
)
DMA_REQUEST = DMA_OUTPUTS[:2]
DMA_TRANSFER = DMA_OUTPUTS[2:]


@construction
def dma_graph():
    """Bulk transfer descriptor, with request and post-catch-up projections.

    Source selection and CPU parity are available at the request. OAMADDR
    and control must be sampled after catching the PPU up to that access:
    rendering can reset OAMADDR in between. The transfer is two spans, so
    wrapping takes no per-byte neural forwards or native address rule.
    """
    x = {name: InputDimension(name) for name in DMA_INPUTS}
    split = 256 - x["oam_addr"]
    values = (
        256 * x["page"],
        513 + x["parity"],
        x["oam_addr"],
        split,
        split,
        x["oam_addr"],
        extract_bits(x["ctrl"], "dma_ctrl")[2],
    )
    outputs = {name: persist(value, name=name)
               for name, value in zip(DMA_OUTPUTS, values, strict=True)}
    return current_graph().capture({"DMA": sum(x.values())}, outputs)


@construction
def composition_graph():
    """Eight lane ordered update, with the existing line window as state.

    Rows arrive in OAM order. A nonzero destination wins even when its
    behind-background bit is set; that bit applies only at final rendering.
    Seven zero padding bytes let the host transport every row identically;
    the circuit clips those lanes and never marks the sentinel tile occupied.
    """
    x = {name: InputDimension(name) for name in COMPOSE_INPUTS}
    fine = remainder(x["x"], 8, 255)
    pixels, occupied = [], [[], []]
    one = current_graph().one
    for c in range(8):
        visible = stepglu(one, 255 - x["x"] - c)
        incoming = gate(visible, x[f"row{c}"])
        pixel = select(stepglu(one, x[f"old{c}"] - 1), x[f"old{c}"], incoming)
        pixels.append(pixel)
        opaque = both(visible, stepglu(one, pixel - 1))
        second = stepglu(one, fine + c - 8)
        occupied[0].append(both(opaque, 1 - second))
        occupied[1].append(both(opaque, second))
    values = (*pixels, *(either(x[f"occupied{i}"], stepglu(one, sum(occupied[i]) - 1))
                         for i in range(2)))
    outputs = {name: persist(value, name=name)
               for name, value in zip(COMPOSE_OUTPUTS, values, strict=True)}
    return current_graph().capture({"COMPOSE": sum(x.values())}, outputs)


@cache
def control_engine():
    def graphs():
        dma = TokenGraph(dma_graph(), DMA_INPUTS, DMA_OUTPUTS)
        access = TokenGraph(
            access_bus_graph(),
            ACCESS_INPUTS,
            ACCESS_OUTPUTS,
            {
                f"{'write' if w else 'read'}{r}": {"access_reg": r, "access_writing": w}
                for w in range(2)
                for r in range(8)
            },
        )
        return {
            "control": TokenGraph(
                control_graph(),
                INPUTS,
                OUTPUTS,
                {name: {"event_tag": i} for i, name in enumerate(EVENTS)},
            ),
            "bus": TokenGraph(
                bus_graph(),
                BUS_INPUTS,
                BUS_OUTPUTS,
                {name: {"bus_event": i} for i, name in enumerate(BUS_EVENTS)},
            ),
            "background_span": TokenGraph(background_span_graph(), SPAN_INPUTS, SPAN_OUTPUTS),
            "access_bus": access,
            "access_delay": replace(access, experts={}).project(DELAY_OUTPUTS),
            "view": TokenGraph(view_graph(), VIEW_INPUTS, VIEW_OUTPUTS),
            "compose": TokenGraph(composition_graph(), COMPOSE_INPUTS, COMPOSE_OUTPUTS),
            "dma": dma,
            "dma_request": dma.project(DMA_REQUEST),
            "dma_transfer": dma.project(DMA_TRANSFER),
            "clock": TokenGraph(clock_graph(), CLOCK_INPUTS, CLOCK_OUTPUTS),
        }

    return CompiledGraph(
        graphs,
        tag="ppu_control",
        sources=(__name__,),
        memory_size=1,
    )


CLOCK_INPUTS = ("line", "index")
CLOCK_OUTPUTS = ("next_line", "wrap", "table", "count", "dot", "event", "slot", "run")
CLOCK_EVENTS = ("slot", "scanline", "sprites", "prefetch", "vertical_copy", "odd_skip",
          "frame_done", "vblank", "clear", "evaluate")


@construction
def clock_graph():
    x = {name: InputDimension(name) for name in CLOCK_INPUTS}
    line, index = x["line"], x["index"]
    visible = 1 - stepglu(1, line - 240)
    post, vblank, pre = (equals(line, n) for n in (240, 241, 261))
    idle = 1 - visible - post - vblank - pre

    def between(lo, hi):
        return both(stepglu(1, index - lo), 1 - stepglu(1, index - hi))

    slot = either(both(visible, either(equals(index, 0), either(between(2, 9), between(10, 34)))),
                  both(pre, between(1, 33)))
    slot_index = select(visible, index - stepglu(1, index - 1) - stepglu(1, index - 9), index - 1)
    ev = {
        "slot": slot,
        "scanline": both(visible, equals(index, 1)),
        "evaluate": both(visible, equals(index, 9)),
        "sprites": either(both(visible, equals(index, 34)), both(pre, equals(index, 33))),
        "prefetch": both(either(visible, pre), equals(index, 35)),
        "vertical_copy": both(pre, equals(index, 34)),
        "odd_skip": both(pre, equals(index, 36)),
        "frame_done": both(post, equals(index, 0)),
        "vblank": both(vblank, equals(index, 0)),
        "clear": both(pre, equals(index, 0)),
    }
    dot = gate(slot, 1 + 8 * slot_index)
    for name, at in (("scanline", 3), ("evaluate", 65), ("sprites", 257),
                     ("prefetch", 321), ("vertical_copy", 280), ("odd_skip", 339),
                     ("vblank", 1), ("clear", 1)):
        dot += at * ev[name]
    end = select(pre, 33, select(equals(index, 0), 1, select(stepglu(1, index - 10), 34, 9)))
    values = (
        line + 1 - 262 * pre,
        pre,
        post + 2 * vblank + 3 * idle + 4 * pre,
        36 * visible + post + vblank + 37 * pre,
        dot,
        sum(i * ev[name] for i, name in enumerate(CLOCK_EVENTS)),
        gate(slot, slot_index),
        1 + gate(slot, end - index - 1),
    )
    outputs = {name: persist(value, name=name) for name, value in zip(CLOCK_OUTPUTS, values, strict=True)}
    return current_graph().capture({"CLOCK": sum(x.values())}, outputs)
