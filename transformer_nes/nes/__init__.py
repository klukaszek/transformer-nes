"""NES system layer: everything console-shaped lives here.

The pure 6502 implementation (CPU cores, graph, kernel, conformance
suites) stays in transformer_nes.cpu6502 and knows nothing about the NES.
This package wires devices to the CPU's machine interfaces over the MMIO
bus; the PPU, cartridge/mappers, and the frame loop land here.
"""

from __future__ import annotations

import os

NES_DIR = os.path.dirname(os.path.abspath(__file__))
NES_CACHE_DIR = os.path.join(NES_DIR, "_nes_cache")


def build_dir(component: str, digest: str) -> str:
    """Return and create one component's digest-keyed native build directory."""
    path = os.path.join(NES_CACHE_DIR, component, f"build_{digest}")
    os.makedirs(path, exist_ok=True)
    return path
