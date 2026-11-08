"""The NES as one PyTorch model.

``NES`` is an nn.Module holding every chip that runs as weights
(model/transformer.py). Its forward is exact eager PyTorch, the reference;
``compile()`` lowers the whole model into one native library
(model/backend.py), which the console runs. ``compiled()`` is the process's
NES: constructed from its circuits the first time, then loaded from
safetensors, and compiled once per model and backend.

The CPU still builds its own bank and engine (cpu6502/bank.py, cpu6502/engine.py).
"""

from __future__ import annotations

import functools
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path

from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from transformer_nes import build
from transformer_nes.apu2a03 import audio
from transformer_nes.model.backend import CompiledChip, CompiledModel, compile_model
from transformer_nes.model.transformer import Chip, Named, pack_state, unpack_state
from transformer_nes.nes import controller, mappers
from transformer_nes.ppu2c02 import control, machine

CACHE_DIR = Path(os.environ.get("TRANSFORMER_NES_CACHE", build.PROJECT_DIR / ".cache" / "nes"))


def recipes() -> dict[str, Callable[[], Chip]]:
    """How each chip is constructed from its circuits, by name."""
    return {
        "apu": audio.chip,
        "apu_mixer": audio.mixer_chip,
        "controller": controller.chip,
        **{name: functools.partial(mappers.chip, number) for number, name in mappers.CHIPS.items()},
        "ppu": machine.chip,
        "ppu_control": control.chip,
    }


def construction_digest() -> str:
    """The identity of the circuits and their construction: what a saved
    model was built from."""
    modules = {getattr(r, "func", r).__module__ for r in recipes().values()}
    return build.source_digest(*sorted(modules), "transformer_nes.model.construct")[:16]


class NES(nn.Module):
    """The console's chips as one model."""

    def __init__(self, chips: dict[str, Chip]):
        super().__init__()
        self.chips = Named(chips)

    def forward(self, chip: str, token: str, inputs: Tensor, memory: Tensor,
                specialization: str | None = None) -> Tensor:
        """One token through one chip: exact eager PyTorch."""
        return self.chips[chip](token, inputs, memory, specialization)

    @classmethod
    def construct(cls) -> NES:
        return cls({name: build_chip() for name, build_chip in recipes().items()})

    def save(self, directory: Path) -> None:
        """``nes.json``, each chip's structure, and ``nes.safetensors``, its
        weights (model/transformer.py:pack_state)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_file(pack_state(self.state_dict()), directory / "nes.safetensors")
        structure = {name: chip.structure() for name, chip in self.chips.items()}
        (directory / "nes.json").write_text(json.dumps(structure, separators=(",", ":")) + "\n")

    @classmethod
    def from_directory(cls, directory: Path) -> NES:
        directory = Path(directory)
        structure = json.loads((directory / "nes.json").read_text())
        model = cls({name: Chip.empty(chip) for name, chip in structure.items()})
        model.load_state_dict(unpack_state(model.state_dict(), load_file(directory / "nes.safetensors")))
        return model

    @classmethod
    def load(cls, cache_dir: Path = CACHE_DIR) -> NES:
        """The model for the current circuits: loaded if saved, else
        constructed and saved (published by rename, never half-written),
        replacing models of earlier circuits."""
        directory = Path(cache_dir) / f"model_{construction_digest()}"
        if (directory / "nes.json").exists():
            return cls.from_directory(directory)
        model = cls.construct()
        work = directory.with_name(f".{directory.name}.{os.getpid()}")
        shutil.rmtree(work, ignore_errors=True)
        model.save(work)
        try:
            work.rename(directory)
        except OSError:  # an identical concurrent construction published first
            shutil.rmtree(work, ignore_errors=True)
        for older in Path(cache_dir).glob("model_*"):
            if older != directory:
                shutil.rmtree(older, ignore_errors=True)
        return model

    def compile(self, cache_dir: Path = CACHE_DIR) -> CompiledNES:
        """The whole model as one native library."""
        compiled = compile_model(dict(self.chips), cache_dir)
        return CompiledNES(compiled.module, compiled.chips, compiled.library)


class CompiledNES(CompiledModel):
    """The compiled NES, with its chips by hardware role."""

    def mapper(self, number: int) -> CompiledChip:
        """The mapper chip of an iNES board family (BankControl::Chip)."""
        return self.chip(mappers.CHIPS[number])


@functools.cache
def compiled() -> CompiledNES:
    """The process's compiled NES, built on first use."""
    return NES.load().compile()
