"""The PPU register graph: its compiled engine is the exact transformer
forward, and as the console's register file it leaves every frame as
the native PPU draws it.

A synthetic cart hammers every register -- $2005/$2006 pairs, $2007 reads
and writes across CHR, nametables and palette, OAM access, status reads,
ctrl/mask changes that switch rendering on and off -- and feeds its
$2007 reads back into the scroll, so a wrong read shows on screen.
"""

import os

import numpy as np
import pytest
from carts import console_frames, latch_cart, mmc5_cart, register_cart

from transformer_nes.graph.core import InputDimension
from transformer_nes.model.graph_engine import construct
from transformer_nes.model.inference import ExactInference
from transformer_nes.nes.build import load_hardware
from transformer_nes.ppu2c02.machine import ENGINE_INPUTS, register_engine, register_graph
from transformer_nes.ppu2c02.registers import OUTPUT_FIELDS

EXAMPLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "nes",
    "examples",
)


def random_access(rng) -> dict[str, int]:
    """An ACCESS token over every register and register-file state."""
    ranges = {
        "reg": 8, "value": 256, "is_write": 2, "status": 256, "rendering": 2,
        "chr_writable": 2, "fine_x": 8, "w": 2, "ctrl": 256, "mask": 256,
        "oam_addr": 256, "read_buffer": 256,
        **{f"nt_page{k}": 4 for k in range(4)},
        **{f"{r}_{f}": n for r in "tv" for f, n in (("cx", 32), ("cy", 32), ("nt", 4), ("fy", 8))},
    }
    return {"is_access": 1, **{name: int(rng.integers(n)) for name, n in ranges.items()}}


def test_compiled_engine_is_the_exact_forward():
    """The native engine against model/inference.py's float64 forward of
    the same weights, on random memory writes and accesses."""
    graph = register_graph()
    model, _tokens, _rows, info = construct(graph)
    reference = ExactInference(model, info.d_model)
    engine = register_engine(graph)
    dims = {d.name: d for d in graph.all_dims}
    inputs = {d.name: d for d in graph.all_dims if isinstance(d, InputDimension)}

    def forward(values: dict[str, int]) -> np.ndarray:
        x = np.zeros(info.d_model)
        x[info.slot_of[graph.one]] = 1.0
        for name, value in values.items():
            x[info.slot_of[inputs[name]]] = value
        return reference.forward(x)

    rng = np.random.default_rng(3)
    for _ in range(300):
        for _ in range(4):
            addr, value = int(rng.choice([rng.integers(0x2000), 0x2000 + rng.integers(0x1000),
                                          0x3F00 + rng.integers(0x20), 0x4000 + rng.integers(0x100)])), int(rng.integers(256))
            engine.write(addr, value)
            forward({"is_mem_write": 1, "mem_addr": addr, "mem_value": value})
        access = random_access(rng)
        compiled = engine.step([access[name] for name in ENGINE_INPUTS])
        residual = forward(access)
        expected = [round(residual[info.slot_of[dims[name]]]) for name in OUTPUT_FIELDS]
        assert list(compiled) == expected, access



@pytest.mark.parametrize(
    "make",
    [
        lambda h: register_cart(h, vertical=False),
        lambda h: register_cart(h, vertical=True),
        lambda h: latch_cart(h, mapper=9),
        lambda h: mmc5_cart(h, exram_mode=0),
        lambda h: mmc5_cart(h, exram_mode=2),
    ],
    ids=["horizontal", "vertical", "mmc2", "mmc5_exram0", "mmc5_exram2"],
)
def test_compiled_engine_drives_the_console(make):
    """The transformer as the register file, called natively: CHR
    latches, MMC5's ExRAM and fill pages, both mirrorings."""
    hardware = load_hardware()
    cart = make(hardware)
    native = console_frames(hardware, cart, 6)
    engine = register_engine()
    driven = console_frames(hardware, cart, 6, lambda c: c.set_ppu_register_engine(engine))
    assert np.array_equal(native, driven)


@pytest.mark.slow
@pytest.mark.parametrize("rom", ["smb.nes", "legendofzelda.nes", "smb3.nes", "punchout.nes"])
def test_weights_drive_the_console(rom):
    """Game frames stay identical with the register file as weights."""
    path = os.path.join(EXAMPLES, rom)
    if not os.path.exists(path):
        pytest.skip(f"{rom} not present")
    hardware = load_hardware()
    cart = hardware.Cartridge.load(path)
    native = console_frames(hardware, cart, 240)
    engine = register_engine()
    compiled = console_frames(hardware, cart, 240, lambda c: c.set_ppu_register_engine(engine))
    assert np.array_equal(native, compiled)
