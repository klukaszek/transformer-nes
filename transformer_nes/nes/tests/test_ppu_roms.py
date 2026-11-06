"""blargg's self-reporting PPU/MMC3 ROMs through transformer CPU and PPU.
These ROMs write a status byte to $6000 behind the signature
DE B0 61 at $6001; 0 is a pass. Run this module to fetch the pinned
fixtures. Local runs skip absent ROMs; CI requires them."""

import os

import pytest

from transformer_nes.nes.backends import backend_factory
from transformer_nes.nes.build import load_hardware

TESTROMS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "testroms"
)
ROMS = {
    # mmc3_test_2 (6-MMC3_alt targets the other chip revision)
    **{name: f"mmc3_test_2/rom_singles/{name}.nes" for name in (
        "1-clocking", "2-details", "3-A12_clocking", "4-scanline_timing", "5-MMC3",
    )},
    # ppu_vbl_nmi: vblank flag, NMI and odd-frame timing to the PPU dot
    **{name: f"ppu_vbl_nmi/rom_singles/{name}.nes" for name in (
        "01-vbl_basics", "02-vbl_set_time", "03-vbl_clear_time", "04-nmi_control",
        "05-nmi_timing", "06-suppression", "07-nmi_on_timing", "08-nmi_off_timing",
        "09-even_odd_frames", "10-even_odd_timing",
    )},
    "oam_read": "oam_read/oam_read.nes",
}


def fetch_roms():
    from pathlib import Path
    from urllib.request import urlopen

    revision = "95d8f621ae55cee0d09b91519a8989ae0e64753b"
    root = f"https://raw.githubusercontent.com/christopherpow/nes-test-roms/{revision}"
    for name, source in ROMS.items():
        with urlopen(f"{root}/{source}", timeout=30) as response:
            data = response.read()
        if data[:4] != b"NES\x1a":
            raise ValueError(f"{source}: not an iNES fixture")
        Path(TESTROMS, name + ".nes").write_bytes(data)
        print(f"fetched {name}")


def run_rom(path, frames=1200):
    hardware = load_hardware()
    console = hardware.Console(
        hardware.Cartridge.load(path), backend_factory("weights"), 44100.0
    )
    assert console.ppu_engine is not None
    for _ in range(frames):
        console.run_frame()
        if console.read_ram(0x6001, 3) == b"\xde\xb0\x61":
            status = console.read_ram(0x6000, 1)[0]
            if status < 0x80:
                text = console.read_ram(0x6004, 256).split(b"\0")[0].decode("latin1")
                return status, text.strip()
    return None, "no result"


@pytest.mark.slow
@pytest.mark.parametrize("name", ROMS)
def test_ppu_rom(name):
    path = os.path.join(TESTROMS, name + ".nes")
    if not os.path.exists(path):
        if os.environ.get("CI") == "true":
            pytest.fail(f"required CI fixture missing: {name}.nes")
        pytest.skip(f"{name}.nes not present")
    status, text = run_rom(path)
    assert status == 0, text


if __name__ == "__main__":
    fetch_roms()
