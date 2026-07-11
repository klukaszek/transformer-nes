"""Frame-level NES PPU (2C02): the native implementation's loader.

The PPU itself is `nes/ppu.cpp` -- registers $2000-$2007 behind the MMIO
bus, VRAM/OAM, and a frame-at-once renderer -- compiled once and cached by
content digest, exactly like the generated engines.  It was ported from a
Python reference and proven pixel-identical before that reference was
deleted (receipts in ROADMAP.md, 2026-07-11: register tests on both,
byte-identical scene renders across four ctrl modes, frame/cycle identity
over console runs, and a byte-identical nestest PPM).

Frame-level approximations (documented, revisit when a game needs them):
base-cycle frame pacing, sprite-0 hit raised at its scanline's cycle
offset from bg/sprite overlap of the *rendered* frame, sprite-overflow
flag always 0, no mid-frame scroll/bank changes (no raster effects), no
left-edge masking, no color emphasis/greyscale.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from transformer_nes.nes.cartridge import MIRROR_VERTICAL, Cartridge

PPU_CPP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppu.cpp")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ppu_cache")

# Standard 2C02 palette (RGB triples for the 64 NES colors).
NES_PALETTE = np.array([
    (84, 84, 84), (0, 30, 116), (8, 16, 144), (48, 0, 136),
    (68, 0, 100), (92, 0, 48), (84, 4, 0), (60, 24, 0),
    (32, 42, 0), (8, 58, 0), (0, 64, 0), (0, 60, 0),
    (0, 50, 60), (0, 0, 0), (0, 0, 0), (0, 0, 0),
    (152, 150, 152), (8, 76, 196), (48, 50, 236), (92, 30, 228),
    (136, 20, 176), (160, 20, 100), (152, 34, 32), (120, 60, 0),
    (84, 90, 0), (40, 114, 0), (8, 124, 0), (0, 118, 40),
    (0, 102, 120), (0, 0, 0), (0, 0, 0), (0, 0, 0),
    (236, 238, 236), (76, 154, 236), (120, 124, 236), (176, 98, 236),
    (228, 84, 236), (236, 88, 180), (236, 106, 100), (212, 136, 32),
    (160, 170, 0), (116, 196, 0), (76, 208, 32), (56, 204, 108),
    (56, 180, 204), (60, 60, 60), (0, 0, 0), (0, 0, 0),
    (236, 238, 236), (168, 204, 236), (188, 188, 236), (212, 178, 236),
    (236, 174, 236), (236, 174, 212), (236, 180, 176), (228, 196, 144),
    (204, 210, 120), (180, 222, 120), (168, 226, 144), (152, 226, 180),
    (160, 214, 228), (160, 162, 160), (0, 0, 0), (0, 0, 0),
], dtype=np.uint8)


def to_rgb(frame) -> np.ndarray:
    """Convert a frame of NES color indices to (240, 256, 3) RGB."""
    return NES_PALETTE[np.asarray(frame)]


ppu_module = None


def load_ppu_module():
    """Compile-or-load the native PPU extension, cached by source digest."""
    global ppu_module
    if ppu_module is None:
        from torch.utils.cpp_extension import load

        with open(PPU_CPP, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()[:16]
        build_dir = os.path.join(CACHE_DIR, f"build_{digest}")
        os.makedirs(build_dir, exist_ok=True)
        ppu_module = load(
            name=f"nes_ppu_{digest}",
            sources=[PPU_CPP],
            extra_cflags=["-O2", "-std=c++17"],
            build_directory=build_dir,
            verbose=False,
        )
    return ppu_module


def make_ppu(cart: Cartridge):
    """The PPU for a cartridge (device protocol: read/write/read_lease)."""
    module = load_ppu_module()
    chr_data = bytes(cart.chr) if cart.chr else bytes(0x2000)
    return module.PPU(
        chr_data, not cart.chr, cart.mirroring == MIRROR_VERTICAL
    )
