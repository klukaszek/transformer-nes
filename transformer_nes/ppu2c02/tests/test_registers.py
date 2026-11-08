"""The PPU register graph: its compiled engine is the exact transformer
forward, and each register's folded expert computes what the full graph
does.  The console runs with it in test_render.py (whole_ppu)."""

import numpy as np
import pytest

from transformer_nes.graph.core import InputDimension
from transformer_nes.model.graph_engine import CompiledGraph, TokenGraph, construct
from transformer_nes.model.inference import ExactInference
from transformer_nes.ppu2c02 import registers, space
from transformer_nes.ppu2c02.machine import ppu_engine, register_graph


@pytest.mark.parametrize("token", ["bus", "background_span", "access_bus", "view", "dma", "clock"])
def test_bus_effect_tokens_execute_exact_weights(token):
    from transformer_nes.ppu2c02 import control

    engine = control.control_engine()
    graph = {"bus": control.bus_graph, "background_span": control.background_span_graph, "access_bus": control.access_bus_graph,
             "view": control.view_graph, "dma": control.dma_graph, "clock": control.clock_graph}[token]()
    model, _, info = construct(graph)
    eager = ExactInference(model)
    dims = {d.name: d for d in graph.all_dims}
    rng = np.random.default_rng(73)
    io = engine.graphs[token]
    limits = {"event": 3, "reg": 8, "ctrl": 256, "old_ctrl": 256, "new_ctrl": 256,
              "old_mask": 256, "new_mask": 256, "new_v_fy": 8,
              "old_v_cx": 32, "old_v_cy": 32, "old_v_nt": 4, "old_v_fy": 8,
              "exram_mode": 4, "fill_tile": 256, "fill_color": 4,
              "page": 256, "oam_addr": 256, "line": 262, "index": 37}
    for _ in range(256):
        x = {n: int(rng.integers(limits.get(n, 2))) for n in io.inputs}
        if token in ("bus", "background_span"):
            x["low_age"] = int(rng.integers(-1, 12))
        if token == "background_span":
            x["count"] = int(rng.integers(1, 33))
        residual = np.zeros(info.d_model)
        residual[info.slot_of[graph.one]] = 1
        for name, value in x.items():
            residual[info.slot_of[dims[name]]] = value
        actual = eager.forward(residual)
        expected = [round(actual[info.slot_of[dims[n]]]) for n in io.outputs]
        packet = [x[n] for n in io.inputs]
        np.testing.assert_array_equal(engine.step(packet, engine.model_id(token)), expected)
        if token == "access_bus":
            np.testing.assert_array_equal(engine.step(packet, engine.model_id("access_delay")), expected[-1:])
        if token == "dma":
            np.testing.assert_array_equal(engine.step(packet, engine.model_id("dma_request")), expected[:2])
            np.testing.assert_array_equal(engine.step(packet, engine.model_id("dma_transfer")), expected[2:])
        if token in ("bus", "access_bus"):
            name = (control.BUS_EVENTS[x["event"]] if token == "bus"
                    else f"{'write' if x['is_write'] else 'read'}{x['reg']}")
            np.testing.assert_array_equal(engine.step(packet, engine.model_id(token, name)), expected)


def test_clock_covers_every_scanline_and_preserves_tied_events_and_batch_limits():
    from transformer_nes.ppu2c02 import control

    engine = control.control_engine()
    model = engine.model_id("clock")
    for line in range(262):
        expected = []
        if line < 240 or line == 261:
            if line == 261:
                expected.append((1, "clear", 0))
            for slot in range(32):
                if line < 240 and slot == 8:
                    expected.append((65, "evaluate", 0))
                expected.append((1 + 8 * slot, "slot", slot))
                if line < 240 and slot == 0:
                    expected.append((3, "scanline", 0))
            expected.append((257, "sprites", 0))
            if line == 261:
                expected.append((280, "vertical_copy", 0))
            expected.append((321, "prefetch", 0))
            if line == 261:
                expected.append((339, "odd_skip", 0))
        elif line == 240:
            expected.append((0, "frame_done", 0))
        elif line == 241:
            expected.append((1, "vblank", 0))
        for index in range(max(1, len(expected))):
            out = engine.step([line, index], model).tolist()
            assert out[:2] == [(line + 1) % 262, int(line == 261)]
            assert out[2] == {240: 1, 241: 2, 261: 4}.get(line, 0 if line < 240 else 3)
            assert out[3] == len(expected)
            if not expected:
                continue
            dot, event, slot = expected[index]
            run = 1
            if event == "slot":
                while index + run < len(expected) and expected[index + run] == (dot + 8 * run, "slot", slot + run):
                    run += 1
            assert out[4:] == [dot, control.CLOCK_EVENTS.index(event), slot, run]


def test_dma_descriptors_partition_every_destination_and_keep_projections_independent():
    from transformer_nes.ppu2c02 import control

    engine = control.control_engine()
    for address in range(256):
        for parity in range(2):
            for ctrl in (0, 0x20, 0xDF, 0xFF):
                page = (73 * address + parity) & 255
                expected = [page * 256, 513 + parity, address, 256 - address,
                            256 - address, address, bool(ctrl & 0x20)]
                np.testing.assert_array_equal(
                    engine.step([page, parity, address, ctrl], engine.model_id("dma")), expected
                )
                np.testing.assert_array_equal(
                    engine.step([page, parity, -999, -999], engine.model_id("dma_request")), expected[:2]
                )
                np.testing.assert_array_equal(
                    engine.step([-999, -999, address, ctrl], engine.model_id("dma_transfer")), expected[2:]
                )
                source = np.arange(256)
                actual = np.concatenate((source[expected[4]:], source[:expected[3]]))
                np.testing.assert_array_equal(actual, np.roll(source, address))


def test_bus_pin_pairs_keep_timestamp_and_strict_filter_boundary():
    from transformer_nes.ppu2c02 import control

    engine = control.control_engine()
    for event in range(3):
        for age in range(-1, 12):
            for old in range(2):
                for high in range(2):
                    for extended in range(2):
                        packet = [event, old, age, high, 16 * high, extended]
                        # Simulate the two pin changes independently of
                        # the pair circuit, retaining the low timestamp.
                        level, low_since, risen = old, -age, False
                        levels = [(0, high)] if event == 0 else [(0, 0)] + (
                            [] if event == 1 and extended else [(4, high)])
                        stamp = False
                        for dot, next_level in levels:
                            if not next_level and level:
                                low_since, stamp = dot, True
                            if next_level and not level and dot - low_since > 10:
                                risen = True
                            level = next_level
                        expected = [level, stamp, risen, 4 if event else 0]
                        np.testing.assert_array_equal(
                            engine.step(packet, engine.model_id("bus", control.BUS_EVENTS[event])), expected
                        )



def test_background_spans_match_literal_pins_at_every_length_and_filter_age():
    from transformer_nes.ppu2c02 import control

    engine = control.control_engine()
    for old in range(2):
        for age in range(-1, 12):
            for ctrl in (0, 16):
                for extended in range(2):
                    for count in range(1, 33):
                        level, low_since, rises = old, -age, []
                        stamped = False
                        for pair in range(count):
                            for dot, high in ((8*pair, 0), (8*pair+4, bool(ctrl & 16) and not extended)):
                                if not high and level:
                                    low_since, stamped = dot, True
                                if high and not level and dot - low_since > 10:
                                    rises.append(dot)
                                level = high
                        expected = [level, stamped, bool(rises), 4,
                                    low_since if stamped else 0, old]
                        assert rises in ([], [4])
                        packet = [old, age, ctrl, extended, count]
                        np.testing.assert_array_equal(
                            engine.step(packet, engine.model_id("background_span")), expected
                        )


def test_access_bus_latches_include_rows_halves_and_address_aliases():
    from transformer_nes.ppu2c02 import control

    engine = control.control_engine()
    addresses = [base + row for base in (0x0FD0, 0x0FE0, 0x1FD0, 0x1FE0) for row in range(16)]
    for wide in range(2):
        for state in range(4):
            for base in (0, 0x2000, 0x4000, 0x6000):
                for address in addresses:
                    v = base + address
                    for reg, write in ((7, 0), (7, 1), (6, 0)):
                        x = dict.fromkeys(control.ACCESS_INPUTS, 0)
                        x.update(reg=reg, is_write=write, latch_active=1, wide_low=wide,
                                 latch_lo=state&1, latch_hi=state>>1,
                                 old_v_cx=v&31, old_v_cy=(v>>5)&31, old_v_nt=(v>>10)&3, old_v_fy=(v>>12)&7)
                        lo, hi = state&1, state>>1
                        addr = v & 0x3FFF
                        if reg == 7 and not write and addr < 0x2000:
                            if 0x1FD8 <= addr <= 0x1FDF:
                                hi = 0
                            elif 0x1FE8 <= addr <= 0x1FEF:
                                hi = 1
                            elif (wide and 0x0FD8 <= addr <= 0x0FDF) or (not wide and addr == 0x0FD8):
                                lo = 0
                            elif (wide and 0x0FE8 <= addr <= 0x0FEF) or (not wide and addr == 0x0FE8):
                                lo = 1
                        expected = [reg == 7, 0, lo, hi, lo != x["latch_lo"] or hi != x["latch_hi"], 0, 0]
                        name = f"{'write' if write else 'read'}{reg}"
                        np.testing.assert_array_equal(
                            engine.step([x[n] for n in control.ACCESS_INPUTS], engine.model_id("access_bus", name)), expected
                        )


# 1 KiB banks the whole-CHR cases address (and the exact forward seeds).
WHOLE_CHR_BANKS = 8


def random_access(rng, reg=None, is_write=None, whole_chr=0) -> dict[str, int]:
    """An ACCESS token over every register and register-file state, through
    the CHR windows or (whole_chr) MMC5's banks into the whole CHR."""
    ranges = {
        "reg": 8, "value": 256, "is_write": 2, "status": 256, "rendering": 2,
        "chr_writable": 2, "latch_lo": 2, "latch_hi": 2, "fine_x": 8, "w": 2, "ctrl": 256,
        "mask": 256, "oam_addr": 256, "read_buffer": 256,
        **{f"nt_page{k}": 4 for k in range(4)},
        **{f"{r}_{f}": n for r in "tv" for f, n in (("cx", 32), ("cy", 32), ("nt", 4), ("fy", 8))},
    }  # fmt: skip
    access = {"is_access": 1, **{name: int(rng.integers(n)) for name, n in ranges.items()}}
    access.update(
        {f"bank{s}": int(rng.integers(WHOLE_CHR_BANKS)) if whole_chr else 0 for s in range(8)},
        whole_chr=whole_chr,
    )
    if reg is not None:
        access.update(reg=reg, is_write=is_write)
    return access


# Every region an ACCESS token reads below the whole CHR: (base, bytes).
REGIONS = {0: 0x2000, 0x2000: 0x1000, 0x3F00: 0x20, 0x4000: 0x100, 0x5000: 0x2000}


def random_writes(rng, write, count: int) -> None:
    for _ in range(count):
        region = rng.choice(list(REGIONS))
        write(int(region + rng.integers(REGIONS[region])), int(rng.integers(256)))


@pytest.mark.parametrize(
    "whole_chr",
    [0, pytest.param(1, marks=pytest.mark.slow)],  # every seeded byte is a float64 forward
    ids=["chr-windows", "whole-chr"],
)
def test_compiled_engine_is_the_exact_forward(whole_chr):
    """The native engine against model/inference.py's float64 forward of
    the same weights, on random memory writes and accesses."""
    graph = register_graph()
    model, _rows, info = construct(graph)
    reference = ExactInference(model)
    token = TokenGraph(graph, registers.INPUTS, registers.OUTPUTS)
    engine = CompiledGraph(
        lambda: {"access": token},
        tag="ppu_registers",
        sources=("transformer_nes.ppu2c02.machine",),
        memory_size=space.MEMORY_SIZE,
    )
    dims = {d.name: d for d in graph.all_dims}
    inputs = {d.name: d for d in graph.all_dims if isinstance(d, InputDimension)}

    def forward(values: dict[str, int]) -> np.ndarray:
        x = np.zeros(info.d_model)
        x[info.slot_of[graph.one]] = 1.0
        for name, value in values.items():
            if name in inputs:
                x[info.slot_of[inputs[name]]] = value
        return reference.forward(x)

    def write(addr: int, value: int) -> None:
        engine.write(addr, value)
        forward({"is_mem_write": 1, "mem_addr": addr, "mem_value": value})

    rng = np.random.default_rng(3)
    # Every byte a read can reach first, as the console seeds it: reads
    # assume dense memory.
    seeded = [base + i for base, size in REGIONS.items() for i in range(size)]
    if whole_chr:
        seeded += range(space.CHR_FULL_BASE, space.CHR_FULL_BASE + 0x400 * WHOLE_CHR_BANKS)
    for addr in seeded:
        write(addr, int(rng.integers(256)))
    for _ in range(300):
        random_writes(rng, write, 4)
        access = random_access(rng, whole_chr=whole_chr)
        compiled = engine.step([access[name] for name in registers.INPUTS])
        residual = forward(access)
        expected = [round(residual[info.slot_of[dims[name]]]) for name in registers.OUTPUTS]
        assert list(compiled) == expected, access


def test_register_experts_agree_with_the_full_graph():
    engine = ppu_engine()
    rng = np.random.default_rng(11)
    for addr in range(0x7000):
        engine.write(addr, int(rng.integers(256)))
    full = engine.model_id("access")
    for is_write, models in enumerate(engine.register_dispatch):
        for reg, model in enumerate(models):
            for _ in range(200):
                random_writes(rng, engine.write, 2)
                access = random_access(rng, reg, is_write)
                inputs = [access[name] for name in registers.INPUTS]
                assert np.array_equal(engine.step(inputs, full), engine.step(inputs, model)), access


def test_register_experts_report_interrupt_effects():
    engine = ppu_engine()
    rng = np.random.default_rng(29)
    for writing in range(2):
        for reg, model in enumerate(engine.register_dispatch[writing]):
            for ctrl in (0, 0x7F, 0x80, 0xFF):
                for value in (0, 0x7F, 0x80, 0xFF):
                    access = random_access(rng, reg, writing)
                    access.update(ctrl=ctrl, value=value)
                    out = engine.step([access[name] for name in registers.INPUTS], model)
                    expected = (
                        reg == 2 and not writing,
                        reg == 0 and writing and bool(value & 0x80) and not (ctrl & 0x80),
                        reg == 0 and writing and bool(ctrl & 0x80) and not (value & 0x80),
                    )
                    assert tuple(out[-3:]) == expected, access


@pytest.mark.parametrize("event", range(6))
def test_event_control_weights_and_timing_boundaries(event):
    """Event experts and actual matrices against conventional control rules.

    Cross every status/effect bit combination with the suppression/NMI
    boundaries, including the last vblank dot and an already pending edge.
    """
    from transformer_nes.ppu2c02 import control

    engine = control.control_engine()
    graph = control.control_graph()
    model, _, info = construct(graph)
    eager = ExactInference(model)
    dims = {d.name: d for d in graph.all_dims}
    rng = np.random.default_rng(41 + event)
    phases = [0, 241 * 341, 241 * 341 + 1, 241 * 341 + 2, 261 * 341, 261 * 341 + 1, 261 * 341 + 2]
    expert = engine.model_id("control", control.EVENTS[event])
    for phase in phases:
        for age in (-1, 0, 1, 2, 3):
            for flags in range(256):
                x = dict.fromkeys(control.INPUTS, 0)
                x.update(
                    event=event,
                    mask=int(rng.integers(256)),
                    ctrl=int(rng.integers(256)),
                    odd=int(rng.integers(2)),
                    phase=phase,
                    flag_age=age,
                    oam_addr=int(rng.integers(256)),
                )
                for i, name in enumerate(
                    (
                        "vblank",
                        "suppressed",
                        "nmi_edge",
                        "nmi_delayed",
                        "clear_vblank",
                        "nmi_rise",
                        "nmi_fall",
                        "status_pending",
                    )
                ):
                    x[name] = (flags >> i) & 1
                t, v = (int(rng.integers(32768)) for _ in range(2))
                for prefix, value in (("t", t), ("v", v)):
                    x.update(
                        zip(
                            (f"{prefix}_{f}" for f in registers.LOOPY_FIELDS),
                            (value & 31, (value >> 5) & 31, (value >> 10) & 3, value >> 12),
                            strict=True,
                        )
                    )
                vb, suppressed, edge, delayed = (
                    x[n] for n in ("vblank", "suppressed", "nmi_edge", "nmi_delayed")
                )
                rendering = bool(x["mask"] & 0x18)
                oam = x["oam_addr"]
                stamp = clear = changed = skip = 0
                if event == 0:
                    if x["clear_vblank"]:
                        suppressed |= phase == 241 * 341 + 1 and age == -1
                        if 0 <= age <= 2:
                            edge = 0
                        vb = 0
                    if x["nmi_rise"] and vb and phase != 261 * 341 + 1:
                        delayed = 1
                    if x["nmi_fall"] and 0 <= age <= 2:
                        edge = 0
                elif event == 1 and rendering:
                    v = (v & ~0x7BE0) | (t & 0x7BE0)
                elif event == 2 and rendering:
                    v, oam = (v & ~0x041F) | (t & 0x041F), 0
                elif event == 3:
                    skip = bool(x["odd"] and rendering)
                elif event == 4:
                    if not suppressed:
                        vb = stamp = changed = 1
                        if x["ctrl"] & 0x80:
                            edge = 1
                    suppressed = 0
                elif event == 5:
                    changed = bool(vb or x["status_pending"])
                    vb, clear = 0, 1
                expected = [
                    v & 31,
                    (v >> 5) & 31,
                    (v >> 10) & 3,
                    v >> 12,
                    oam,
                    vb,
                    suppressed,
                    edge,
                    delayed,
                    stamp,
                    clear,
                    clear,
                    changed,
                    skip,
                    rendering,
                ]
                inputs = [x[n] for n in control.INPUTS]
                np.testing.assert_array_equal(engine.step(inputs, expert), expected, err_msg=str(x))
                np.testing.assert_array_equal(engine.step(inputs), expected, err_msg=str(x))
                if age == -1 and flags == 0:
                    residual = np.zeros(info.d_model)
                    residual[info.slot_of[graph.one]] = 1
                    for name in control.INPUTS:
                        residual[info.slot_of[dims[name]]] = x[name]
                    actual = eager.forward(residual)
                    assert [
                        round(actual[info.slot_of[dims[n]]]) for n in control.OUTPUTS
                    ] == expected
