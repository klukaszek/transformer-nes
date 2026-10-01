"""The PPU render graph: folded experts against the full graph, and
graph-rendered frames against the native PPU."""

import os

import numpy as np
import pytest
from carts import console_frames, latch_cart, mmc5_cart, register_cart

from transformer_nes.nes.backends import backend_factory
from transformer_nes.nes.build import load_hardware
from transformer_nes.ppu2c02 import render
from transformer_nes.ppu2c02.machine import ppu_engine

EXAMPLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "nes",
    "examples",
)


def random_slot(rng, pins: dict) -> list[int]:
    """SLOT inputs drawn at random where an expert's pins hold."""
    first, last = pins.get("first_slot"), pins.get("last_slot")
    slot = 0 if first else 31 if last else int(rng.choice([*range(1, 31), 32, 33]))
    mask = int(rng.integers(256))
    if "maskbit3" in pins:
        mask = (mask | 0x08) & ~0x01
    ctrl = int(rng.integers(256))
    if "ctrlbit4" in pins:
        ctrl = (ctrl & ~0x10) | (pins["ctrlbit4"] << 4)
    fine_x = int(rng.integers(8))
    if "fine_x_bit0" in pins:
        fine_x = sum(pins[f"fine_x_bit{b}"] << b for b in range(3))
    # Background pixels: 0 (transparent) or palette * 4 + value 1-3.
    bg = [
        0 if rng.random() < 0.3 else int(rng.integers(4)) * 4 + int(rng.integers(1, 4))
        for _ in range(16)
    ]
    sprites = [0] * 8
    if "sprite_pixel0" not in pins:
        sprites = [
            0
            if rng.random() < 0.4
            else 16 + int(rng.integers(16)) | int(rng.integers(2)) << 6 | int(rng.integers(2)) << 7
            for _ in range(8)
        ]
    v = [int(rng.integers(32)), int(rng.integers(32)), int(rng.integers(4)), int(rng.integers(8))]
    head = [1, slot, int(rng.integers(2)), fine_x, ctrl, mask]
    pages = [int(rng.integers(4)) for _ in range(4)]
    latches = [int(rng.integers(2)), int(rng.integers(2)), int(rng.integers(2))]
    if "latch" in pins:  # carts without CHR latches
        latches = [0, 0, 0]
    mmc5 = [0, int(rng.integers(4))]  # extended mode pinned off; chr_upper
    return head + pages + latches + mmc5 + v + bg + sprites


def test_experts_agree_with_the_full_graph():
    engine = ppu_engine()
    rng = np.random.default_rng(7)
    for lo, hi in ((0, 0x3000), (0x5000, 0x7000)):
        for addr in range(lo, hi):
            engine.write(addr, int(rng.integers(256)))
    for addr in range(0x3F00, 0x3F20):
        engine.write(addr, int(rng.integers(64)))
    for model, name in enumerate(engine.expert_names, start=1):
        for _ in range(300):
            inputs = random_slot(rng, render.EXPERTS[name])
            full, folded = engine.step(inputs, 0), engine.step(inputs, model)
            if "latch" in render.EXPERTS[name]:  # new latches unused without them
                full, folded = full[:-2], folded[:-2]
            assert np.array_equal(full, folded), (name, inputs, full, folded)



def whole_ppu(hardware, cart, count):
    """Frames with the PPU as one transformer: register file, rendering,
    sprite unit."""
    probe = hardware.Console(cart, backend_factory("weights"), 44100.0)
    engine = ppu_engine(probe.ppu_graph_chr_full)
    return console_frames(hardware, cart, count, lambda c: c.set_ppu_engine(engine))


@pytest.mark.parametrize(("mask_bits", "vertical"), [(0x18, False), (0x18, True), (0x1F, False)])
def test_graph_renders_the_synthetic_cart(mask_bits, vertical):
    """Every register: rendering toggles, scroll and address writes
    mid-frame, $2007 across CHR, nametables and palette, OAM, sprites,
    both mirrorings; with 0x1F also greyscale and left-edge clipping
    (the full render graph)."""
    hardware = load_hardware()
    cart = register_cart(hardware, vertical=vertical, mask_bits=mask_bits)
    assert np.array_equal(console_frames(hardware, cart, 6), whole_ppu(hardware, cart, 6))


@pytest.mark.parametrize("mapper", [9, 10])
def test_graph_renders_latch_mappers(mapper):
    """MMC2/MMC4: the render graph switches CHR through the latches its
    own fetches set (and the sprite fetches between lines)."""
    hardware = load_hardware()
    cart = latch_cart(hardware, mapper=mapper)
    assert np.array_equal(console_frames(hardware, cart, 6), whole_ppu(hardware, cart, 6))


@pytest.mark.parametrize("exram_mode", [0, 1, 2])
def test_graph_renders_mmc5(exram_mode):
    """MMC5: nametable maps over CIRAM, ExRAM and fill mode, both CHR
    sets, and (mode 1) extended attributes from the whole CHR ROM."""
    hardware = load_hardware()
    cart = mmc5_cart(hardware, exram_mode=exram_mode)
    assert np.array_equal(console_frames(hardware, cart, 6), whole_ppu(hardware, cart, 6))


@pytest.mark.slow
@pytest.mark.parametrize(
    "rom",
    [
        "smb.nes",
        "legendofzelda.nes",
        "smb3.nes",
        "megaman6.nes",
        "punchout.nes",
        "castlevania3.nes",
    ],
)
def test_graph_renders_games(rom):
    """Background, compositing and the register file all as transformer
    weights: frames identical to the native PPU."""
    path = os.path.join(EXAMPLES, rom)
    if not os.path.exists(path):
        pytest.skip(f"{rom} not present")
    hardware = load_hardware()
    cart = hardware.Cartridge.load(path)
    assert np.array_equal(console_frames(hardware, cart, 300), whole_ppu(hardware, cart, 300))


def test_slot_trace_explains_the_frame():
    """The X-ray's pixel dissection: a slot traced through the full model
    (on a copy of the engine) reports the color the frame shows."""
    from transformer_nes.nes.xray import PpuXray

    hardware = load_hardware()
    console = hardware.Console(
        latch_cart(hardware, mapper=9), backend_factory("weights"), 44100.0
    )
    engine = ppu_engine()
    console.set_ppu_engine(engine)
    xray = PpuXray(engine)
    for _ in range(3):
        console.run_frame()
    for line, x in ((40, 37), (120, 130), (200, 251)):
        console.request_ppu_slot_trace(line, x // 8)
        frame = console.run_frame()
        trace = console.take_ppu_slot_trace()
        assert trace is not None and (trace["line"], trace["slot"]) == (line, x // 8)
        d = xray.dissect(trace, x)
        assert d["title"].endswith(f"color ${frame[line][x]:02X}"), d["title"]
        assert any(kind == "attention" and "palette" in detail for _, kind, _, detail, _ in d["steps"])
    assert console.take_ppu_slot_trace() is None  # one trace per request


def test_knock_out_breaks_rendering_until_restored():
    """Ablation: the full model alone renders identically; forcing its
    attribute read to 0 changes the picture; restoring brings it back."""
    from transformer_nes.nes.xray import PpuXray

    hardware = load_hardware()
    cart = latch_cart(hardware, mapper=9)
    native = console_frames(hardware, cart, 8)
    console = hardware.Console(cart, backend_factory("weights"), 44100.0)
    engine = ppu_engine()
    console.set_ppu_engine(engine)
    xray = PpuXray(engine)
    out = []
    for i in range(8):
        if i == 2:
            console.ppu_force_full_render = True
        if i == 4:
            xray.knock_out(console, "attr")
        if i == 6:
            xray.restore(console)
        console.set_buttons(0)
        out.append(console.run_frame().copy())
    assert np.array_equal(native[:4], out[:4])
    assert not np.array_equal(native[4:6], out[4:6])
    # The first frame after restoring still shows the two tiles prefetched
    # for its line 0 while ablated; from the next frame on, it is gone.
    assert np.array_equal(native[7:], out[7:])
