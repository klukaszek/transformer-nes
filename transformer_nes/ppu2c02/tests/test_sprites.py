"""The sprite unit's graphs against the hardware rules they replace
(nes/ppu.cpp: evaluate_sprites, load_sprites, ChrLatch::after_fetch),
on random OAM, CHR and latch states."""

import numpy as np
import pytest

from transformer_nes.ppu2c02.machine import ppu_engine


@pytest.fixture(scope="module")
def engine():
    return ppu_engine()


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
            assert engine.step([1, line, ctrl], model).tolist() == want


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
        expert = engine.model_id("fetch", f"tall{int(height == 16)}_count{count}")
        for model, latches in ((engine.model_id("fetch"), True), (expert, False)):
            if not latches:
                inputs[12:14] = [0, 0]
            pixels, xs, fetch, a12, new_latch = native_fetch(
                memory, line, ctrl, count, index, inputs[12:14], wide_low, latches
            )
            out = engine.step(inputs, model).tolist()
            assert out[:64] == pixels
            assert out[72:80] == fetch
            for k in range(8):
                if k < count:
                    assert out[64 + k] == xs[k]
                if fetch[k]:
                    assert out[80 + k] == a12[k]
            if latches:
                assert out[88:90] == new_latch
