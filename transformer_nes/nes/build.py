"""Load the native extension modules the repo-root Makefile builds.

Python's role is one quiet `make` for dependency-checked freshness,
then a plain extension import from bin/.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

NES_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(NES_DIR, "..", ".."))
BIN_DIR = os.path.join(REPO_ROOT, "bin")


def _make(target: str):
    subprocess.run(
        ["make", "-s", "-C", REPO_ROOT, target, f"PYTHON={sys.executable}"], check=True
    )


def _import_extension(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BIN_DIR, f"{name}.so"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_hardware():
    _make("hardware")
    return _import_extension("nes_hardware")


def load_frontend():
    _make("frontend")
    return _import_extension("nes_app")
