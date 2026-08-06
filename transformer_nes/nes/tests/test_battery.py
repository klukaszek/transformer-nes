"""Battery-backed PRG RAM: loaded image is CPU-visible before the first
frame and the shutdown dump matches the live $6000-$7FFF contents."""

from transformer_nes.nes.backends import OPCODES_NATIVE, backend_factory
from transformer_nes.nes.build import load_hardware


def battery_cart(hardware):
    """Build a minimal battery-backed NROM without requiring a shipped ROM."""
    prg = bytearray(0x4000)
    prg[:3] = b"\x4c\x00\x80"  # JMP $8000
    prg[3] = 0x40  # RTI, for a possible NMI or IRQ
    prg[-6:] = bytes((0x03, 0x80, 0x00, 0x80, 0x03, 0x80))
    header = bytes((0x4e, 0x45, 0x53, 0x1a, 1, 0, 0x02, 0)) + bytes(8)
    return hardware.Cartridge.parse(header + bytes(prg))


def test_battery_prg_ram_round_trip():
    hardware = load_hardware()
    cart = battery_cart(hardware)
    assert cart.battery
    console = hardware.Console(cart, backend_factory("reference"), OPCODES_NATIVE, 44100.0)
    image = bytes((i * 7 + 3) & 0xFF for i in range(0x2000))
    console.load_prg_ram(image)
    assert console.read_ram(0x6000, 0x2000) == image
    console.run_frame()
    console.run_frame()
    saved = console.prg_ram()
    assert len(saved) == 0x2000
    assert saved == console.read_ram(0x6000, 0x2000)
