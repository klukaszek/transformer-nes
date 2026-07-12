"""NES APU (2A03 audio): the native implementation's loader.

The APU itself is `nes/apu.cpp` -- registers $4000-$4017 behind the MMIO
bus, the five channels, a frame sequencer, and a per-frame sample
generator -- compiled once and cached by content digest, exactly like the
PPU (`nes/ppu.py`).  Plain fixed silicon; nothing here is model-generated.

The console attaches it as a bus device and calls `end_frame(cpu_cycles)`
once per frame to pull that frame's mono samples; the native app pushes
them to `sokol_audio`.  All five channels are implemented; DMC fetches
read through the console's CPU-memory callback and report DMA stalls.
"""

from __future__ import annotations

import hashlib
import os

from transformer_nes.nes import build_dir

APU_CPP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apu.cpp")

apu_module = None


def load_apu_module():
    """Compile-or-load the native APU extension, cached by source digest."""
    global apu_module
    if apu_module is None:
        from torch.utils.cpp_extension import load

        with open(APU_CPP, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()[:16]
        cache_dir = build_dir("apu", digest)
        apu_module = load(
            name=f"nes_apu_{digest}",
            sources=[APU_CPP],
            extra_cflags=["-O2", "-std=c++17"],
            build_directory=cache_dir,
            verbose=False,
        )
    return apu_module


def make_apu(sample_rate: float = 44100.0):
    """The APU device (protocol: read/write/end_frame/irq_pending)."""
    return load_apu_module().APU(sample_rate)
