"""The saved model runs nestest by its eager PyTorch forward alone, with
nothing but the standard library, Torch, safetensors and model/transformer.py
imported (cpu6502/eager.py)."""

import subprocess
import sys

import pytest

from transformer_nes.nes.model import NES, model_directory

GUARDED_RUN = """
import importlib.abc
import sys

ALLOWED = {
    "transformer_nes", "transformer_nes.cpu6502", "transformer_nes.cpu6502.eager",
    "transformer_nes.cpu6502.trace", "transformer_nes.cpu6502.spec",
    "transformer_nes.model", "transformer_nes.model.transformer",
}

class OnlyTheModel(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("transformer_nes") and fullname not in ALLOWED:
            raise AssertionError(f"the eager CPU imported {fullname}")

sys.meta_path.insert(0, OnlyTheModel())
from transformer_nes.cpu6502.eager import load_cpu, run

cpu = load_cpu(sys.argv[1])
image, rows, steps = open(sys.argv[2], "rb").read(), sys.argv[3], int(sys.argv[4])
log = [tuple(int(v, 16) for v in line.split()) for line in open(rows)]
for i, (state, want) in enumerate(zip(run(cpu, image, pc=0xC000, steps=steps), log)):
    got = tuple(state[k] for k in ("pc", "a", "x", "y", "p", "s"))
    assert got == want, (i, got, want)
print("matched", steps)
"""


@pytest.mark.parametrize("steps", [1000, pytest.param(8991, marks=pytest.mark.slow)])
def test_eager_cpu_runs_nestest_from_the_saved_model_alone(steps, tmp_path):
    from transformer_nes.cpu6502.conformance.nestest import load_image, load_log

    NES.load()  # saved, so the run below only loads it
    (tmp_path / "image").write_bytes(bytes(load_image()))
    (tmp_path / "log").write_text(
        "".join(" ".join(f"{v:x}" for v in row[:6]) + "\n" for row in load_log())
    )
    result = subprocess.run(
        [sys.executable, "-c", GUARDED_RUN, str(model_directory()), str(tmp_path / "image"),
         str(tmp_path / "log"), str(steps)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert f"matched {steps}" in result.stdout
