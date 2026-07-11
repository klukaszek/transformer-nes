"""Shared build-cache layout for the NES native extensions."""

from __future__ import annotations

import os

NES_DIR = os.path.dirname(os.path.abspath(__file__))
NES_CACHE_DIR = os.path.join(NES_DIR, "_nes_cache")


def build_dir(component: str, digest: str) -> str:
    """Return and create one component's digest-keyed build directory."""
    path = os.path.join(NES_CACHE_DIR, component, f"build_{digest}")
    os.makedirs(path, exist_ok=True)
    return path
