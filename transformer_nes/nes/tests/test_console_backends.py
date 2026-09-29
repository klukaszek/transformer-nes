"""Reference-vs-weights lockstep through the native console: identical
frames, cycles, and RAM under scripted input, and MMC1's RMW serial-port
quirk through the engine's watched-write path."""

import os

import numpy as np
import pytest

from transformer_nes.nes.backends import backend_factory
from transformer_nes.nes.build import load_hardware

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")

# Frame -> controller mask: Start taps at the title, then hold A+Right
# (exercises the strobe writes and the serial-read lease queue).
SCRIPT = {30: 0x08, 33: 0x00, 45: 0x08, 48: 0x00, 60: 0x81}
FRAMES = 90


def run_console(hardware, cart, backend, frames, script=SCRIPT):
    console = hardware.Console(cart, backend_factory(backend), 44100.0)
    frames_out = []
    mask = 0
    for i in range(frames):
        mask = script.get(i, mask)
        console.set_buttons(mask)
        frames_out.append(console.run_frame().copy())
    return console, frames_out


@pytest.mark.parametrize("rom", ["smb.nes", "legendofzelda.nes"])
def test_console_weights_matches_reference(rom):
    path = os.path.join(EXAMPLES, rom)
    if not os.path.exists(path):
        pytest.skip(f"{rom} not present")
    hardware = load_hardware()
    cart = hardware.Cartridge.load(path)
    ref, ref_frames = run_console(hardware, cart, "reference", FRAMES)
    wts, wts_frames = run_console(hardware, cart, "weights", FRAMES)
    for i, (a, b) in enumerate(zip(ref_frames, wts_frames, strict=True)):
        assert np.array_equal(a, b), f"frame {i} diverged"
    assert ref.cycles == wts.cycles
    # RAM, PRG RAM, and the mapped ROM window; MMIO ranges only mirror
    # serviced device reads, whose cadence legitimately differs.
    for lo, count in ((0x0000, 0x2000), (0x6000, 0xA000)):
        assert ref.read_ram(lo, count) == wts.read_ram(lo, count)


def mmc1_rmw_cart(hardware):
    """MMC1 cart whose reset code loads the PRG register with five INC
    $E000s: hardware clocks the ROM byte's bit (1), not the CPU's final
    value (0), so $8000 must map bank 1 afterwards."""
    code = bytes([
        0xEE, 0x00, 0xE0,  # INC $E000  x5 (serial load via RMW)
        0xEE, 0x00, 0xE0,
        0xEE, 0x00, 0xE0,
        0xEE, 0x00, 0xE0,
        0xEE, 0x00, 0xE0,
        0xAD, 0x00, 0x80,  # LDA $8000  (which bank landed?)
        0x85, 0x10,        # STA $10
        0x4C, 0x14, 0xC0,  # spin
    ])
    bank0 = bytes(0x4000)
    bank1 = bytearray(b"\x01" * 0x4000)
    bank1[: len(code)] = code
    bank1[0x3FFC:0x3FFE] = b"\x00\xC0"  # reset -> $C000 (fixed last bank)
    header = bytes([0x4E, 0x45, 0x53, 0x1A, 2, 0, 0x10, 0]) + bytes(8)
    return hardware.Cartridge.parse(header + bank0 + bytes(bank1))


@pytest.mark.parametrize("backend", ["reference", "weights"])
def test_mmc1_rmw_register_write_uses_rom_byte(backend):
    hardware = load_hardware()
    cart = mmc1_rmw_cart(hardware)
    console, _frames = run_console(hardware, cart, backend, 2, script={})
    # Register = 0b11111 -> bank 1 at $8000, whose first byte is the
    # code's 0xEE; the final-value bug would leave bank 0 (0x00).
    assert console.read_ram(0x10, 1) == b"\xee"
