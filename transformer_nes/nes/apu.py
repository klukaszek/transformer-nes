"""NES APU (2A03 audio): the native implementation's loader.

The APU itself is `nes/apu.cpp` -- registers $4000-$4017 behind the MMIO
bus, the five channels, a frame sequencer, and a per-frame sample
generator -- compiled once and cached by content digest, exactly like the
PPU (`nes/ppu.py`).  Plain fixed silicon; nothing here is model-generated.

The console attaches it as a bus device and calls `end_frame(cpu_cycles)`
once per frame to pull that frame's mono samples; the native app pushes
them to `sokol_audio`.  Pulse x2 + triangle + noise are implemented; the
DMC (sample) channel is stubbed (registers accepted, no output).
"""

from __future__ import annotations

import hashlib
import os

APU_CPP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apu.cpp")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_apu_cache")

apu_module = None


def load_apu_module():
    """Compile-or-load the native APU extension, cached by source digest."""
    global apu_module
    if apu_module is None:
        from torch.utils.cpp_extension import load

        with open(APU_CPP, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()[:16]
        build_dir = os.path.join(CACHE_DIR, f"build_{digest}")
        os.makedirs(build_dir, exist_ok=True)
        apu_module = load(
            name=f"nes_apu_{digest}",
            sources=[APU_CPP],
            extra_cflags=["-O2", "-std=c++17"],
            build_directory=build_dir,
            verbose=False,
        )
    return apu_module


def make_apu(sample_rate: float = 44100.0):
    """The APU device (protocol: read/write/end_frame/irq_pending)."""
    return load_apu_module().APU(sample_rate)
