"""Sprite selection, fetch and composition against independent hardware rules."""

import numpy as np
import pytest

from transformer_nes.ppu2c02.machine import ppu_engine


@pytest.fixture(scope="module")
def engine():
    return ppu_engine()


def test_composition_executes_exact_weights_and_ordered_clipped_updates():
    from transformer_nes.graph.core import InputDimension
    from transformer_nes.model.graph_engine import construct
    from transformer_nes.model.inference import ExactInference
    from transformer_nes.ppu2c02 import control

    engine = control.control_engine()
    graph = control.composition_graph()
    model, _, info = construct(graph)
    eager = ExactInference(model)
    dims = {d.name: d for d in graph.all_dims}
    inputs = {d.name: d for d in graph.all_dims if isinstance(d, InputDimension)}
    model_id = engine.model_id("compose")
    rng = np.random.default_rng(79)
    for trial in range(64):
        actual, expected = np.zeros(264, dtype=np.int32), np.zeros(264, dtype=np.int32)
        occupied = np.zeros(33, dtype=np.int32)
        # Every x, reversed and repeated, includes overlaps past the hardware
        # eight-sprite cap and all right-edge and tile-boundary cases.
        for x in range(256) if trial % 2 else reversed(range(256)):
            row = rng.integers(256, size=8) * rng.integers(2, size=8)
            packet = [x, *occupied[x >> 3:(x >> 3) + 2], *row, *actual[x:x + 8]]
            out = engine.step(packet, model_id)
            if x == trial:
                residual = np.zeros(info.d_model)
                residual[info.slot_of[graph.one]] = 1
                for name, value in zip(control.COMPOSE_INPUTS, packet, strict=True):
                    residual[info.slot_of[inputs[name]]] = value
                result = eager.forward(residual)
                np.testing.assert_array_equal(
                    out, [round(result[info.slot_of[dims[n]]]) for n in control.COMPOSE_OUTPUTS]
                )
            actual[x:x + 8] = out[:8]
            occupied[x >> 3:(x >> 3) + 2] = out[8:]
            for c in range(min(8, 256 - x)):
                if not expected[x + c] and row[c]:
                    expected[x + c] = row[c]
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(occupied[:32], expected[:256].reshape(32, 8).any(axis=1))
            assert not occupied[32] and not actual[256:].any()


def test_evaluation_selects_the_first_eight_in_oam_order(engine):
    rng = np.random.default_rng(5)
    for _ in range(1000):
        line, ctrl = int(rng.integers(240)), int(rng.integers(256))
        height = 16 if ctrl & 0x20 else 8
        density = rng.random()  # from empty lines to overflowing ones
        ys = [
            (line - int(rng.integers(height))) & 0xFF
            if rng.random() < density
            else int(rng.integers(256))
            for _ in range(64)
        ]
        for i, y in enumerate(ys):
            engine.write(0x4000 + 4 * i, y)
        covering = [i for i in range(64) if 0 <= line - ys[i] < height]
        want = [min(len(covering), 8), *(covering[:8] + [64] * 8)[:8]]
        tall = f"tall{int(height == 16)}"
        for model in (engine.model_id("evaluate"), engine.model_id("evaluate", tall)):
            assert engine.step([1, line, ctrl, 0], model).tolist() == want
        # The ordinary expert must actually fold its start pin to zero.
        assert engine.step([1, line, ctrl, 64], engine.model_id("evaluate", tall)).tolist() == want
        start = int(rng.integers(65))
        remaining = [i for i in covering if i >= start]
        want = [min(len(remaining), 8), *(remaining[:8] + [64] * 8)[:8]]
        for model in (
            engine.model_id("evaluate"),
            engine.model_id("evaluate", f"remaining_{tall}"),
        ):
            assert engine.step([1, line, ctrl, start], model).tolist() == want


def test_overflow_reads_diagonally_including_non_y_bytes(engine):
    rng = np.random.default_rng(41)
    for trial in range(1024):
        line, ctrl, start = int(rng.integers(240)), int(rng.integers(256)), trial % 65
        height = 16 if ctrl & 0x20 else 8
        oam = rng.integers(256, size=256)
        if trial % 2:
            # Dense matches exercise every phase and the first-hit rule.
            oam[rng.random(256) < 0.5] = line
        for a, value in enumerate(oam):
            engine.write(0x4000 + a, int(value))
        want = next(
            (j for j in range(start, 64) if 0 <= line - int(oam[4 * j + (j - start) % 4]) < height),
            64,
        )
        for model in (
            engine.model_id("overflow"),
            engine.model_id("overflow", f"tall{int(height == 16)}"),
        ):
            found = 64
            # Include start=64: a completely empty batch must not match.
            for base in range(start, max(start + 1, 64), 8):
                batch = next(
                    (
                        j
                        for j in range(base, min(base + 8, 64))
                        if 0 <= line - int(oam[4 * j + (j - start) % 4]) < height
                    ),
                    64,
                )
                found = engine.step([1, line, ctrl, base], model).tolist()[0]
                assert found == batch
                if found != 64:
                    break
            assert found == want


def after_fetch(latch, address, wide_low):
    address &= 0x1FFF
    if 0x1FD8 <= address <= 0x1FDF:
        latch[1] = 0
    elif 0x1FE8 <= address <= 0x1FEF:
        latch[1] = 1
    elif (0x0FD8 <= address <= 0x0FDF) if wide_low else address == 0x0FD8:
        latch[0] = 0
    elif (0x0FE8 <= address <= 0x0FEF) if wide_low else address == 0x0FE8:
        latch[0] = 1


def native_fetch(memory, line, ctrl, count, index, latch, wide_low, latches=True):
    """load_sprites over the render view's CHR windows ($0000 FD, $5000
    FE; just $0000 for a cart without latches): the outputs that the PPU
    reads (pixels, x and A12 table of the selected sprites, fetch flags,
    latches)."""
    height = 16 if ctrl & 0x20 else 8
    latch = list(latch)
    pixels, xs, fetch, a12 = [0] * 64, [None] * 8, [0] * 8, [None] * 8
    for k in range(8):
        if k >= count:  # a dummy fetch
            fetch[k], a12[k] = 1, int(height == 16 or bool(ctrl & 8))
            continue
        i = index[k]
        y, tile, attr, xs[k] = (int(memory[0x4000 + 4 * i + j]) for j in range(4))
        row = line - y
        if not 0 <= row < height:
            continue
        if attr & 0x80:
            row = height - 1 - row
        if height == 16:
            address = ((tile & 1) << 12) | ((tile & 0xFE) << 4) | ((row & 8) << 1) | (row & 7)
        else:
            address = ((ctrl & 8) << 9) | (tile << 4) | row
        half = address >> 12
        window = 0x5000 if latch[half] else 0
        lo, hi = int(memory[window + address]), int(memory[window + address + 8])
        if latches:
            after_fetch(latch, address, wide_low)
            after_fetch(latch, address + 8, wide_low)
        fetch[k], a12[k] = 1, half
        if line >= 239:
            continue
        meta = 0x10 | (attr & 3) << 2 | (0x40 if attr & 0x20 else 0) | (0x80 if i == 0 else 0)
        for c in range(8):
            bit = c if attr & 0x40 else 7 - c
            value = (lo >> bit) & 1 | ((hi >> bit) & 1) << 1
            pixels[8 * k + c] = meta | value if value else 0
    return pixels, xs, fetch, a12, latch


def test_fetch_is_the_native_sprite_fetch(engine):
    rng = np.random.default_rng(3)
    memory = rng.integers(256, size=0x7000)
    for address, value in enumerate(memory):
        engine.write(address, int(value))
    for _ in range(1000):
        line, ctrl, count = int(rng.integers(240)), int(rng.integers(256)), int(rng.integers(9))
        index = sorted(rng.choice(64, 8, replace=False).tolist())
        height = 16 if ctrl & 0x20 else 8
        for i in index:  # mostly covering the line, often latch tiles
            sprite = 0x4000 + 4 * i
            if rng.random() < 0.9:
                memory[sprite] = (line - int(rng.integers(height))) & 0xFF
            memory[sprite + 1] = rng.choice([0xFD, 0xFE, 0xFC, 0xFF, int(rng.integers(256))])
            for j in range(4):
                engine.write(sprite + j, int(memory[sprite + j]))
        latch, wide_low = [int(rng.integers(2)), int(rng.integers(2))], int(rng.integers(2))
        inputs = [1, line, ctrl, count, *index, *latch, wide_low, *[0] * 8]
        # The full model, and the expert for the size on a cart without
        # latches (inputs 0, no flips).
        expert = f"tall{int(height == 16)}_count{count}"
        debug = f"debug_tall{int(height == 16)}"
        for name, latches in ((None, True), (debug, False), (expert, False)):
            if name == expert:
                inputs[12:14] = [0, 0]
            pixels, xs, fetch, a12, new_latch = native_fetch(
                memory, line, ctrl, count, index, inputs[12:14], wide_low, latches
            )
            out = engine.step(inputs, engine.model_id("fetch", name)).tolist()
            projected = engine.model_id("fetch_bus", name)
            assert engine.step(inputs, projected).tolist() == out[72:]
            status = engine.model_id("fetch_status", name)
            assert engine.step(inputs, status).tolist() == out[:8] + [out[64]] + out[72:]
            assert out[:64] == pixels
            assert out[72:80] == fetch
            for k in range(8):
                if k < count:
                    assert out[64 + k] == xs[k]
                if fetch[k]:
                    assert out[80 + k] == a12[k]
            assert out[88:90] == new_latch
