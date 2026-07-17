"""Battery-backed PRG RAM: loaded image is CPU-visible before the first
frame and the shutdown dump matches the live $6000-$7FFF contents."""

import os

from transformer_nes.nes.backends import OPCODES_NATIVE, backend_factory
from transformer_nes.nes.build import load_hardware

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")


def test_battery_prg_ram_round_trip():
    hardware = load_hardware()
    cart = hardware.Cartridge.load(os.path.join(EXAMPLES, "legendofzelda.nes"))
    assert cart.battery
    console = hardware.Console(cart, backend_factory("reference"), OPCODES_NATIVE, 44100.0)
    image = bytes((i * 7 + 3) & 0xFF for i in range(0x2000))
    console.load_prg_ram(image)
    assert console.read_ram(0x6000, 0x2000) == image
    console.run_frame()  # the game may rewrite save RAM (checksum init)...
    console.run_frame()
    saved = console.prg_ram()
    assert len(saved) == 0x2000
    assert saved == console.read_ram(0x6000, 0x2000)  # ...dump == live RAM
