"""Compile a model's chips into one native library, and run them.

Every circuit of every chip (model/transformer.py) is lowered through the
scalar compiler (model/compiler.py: exact FX arithmetic, then scalar SSA and
C++). One Torch extension holds the fixed runtime (native/runtime.cpp), each
chip's kernels with its descriptor (native/bank.h), and a table naming the
chips.

The library is keyed by content: the model's weights and the backend's own
sources. Nothing is locked: a build happens in a private directory that is
renamed into place, so a concurrent identical build is harmless, and it
replaces the libraries it supersedes (a process that loaded one keeps its
mapping). Loading a built library needs neither construction nor lowering.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import shutil
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from transformer_nes import build
from transformer_nes.model import ssa
from transformer_nes.model.compiler import bank_kernels
from transformer_nes.model.transformer import HALVES, MATRICES, Chip, Layer, Transformer

logger = logging.getLogger(__name__)
NATIVE_DIR = Path(__file__).with_name("native")
RUNTIME_FILES = (NATIVE_DIR / "runtime.cpp", NATIVE_DIR / "bank.h")
CFLAGS = ("-O2", "-std=c++17", "-fwrapv", "-ffp-contract=off", "-fvisibility=hidden")
INTERMEDIATE = (".o", ".d", ".ninja_deps", ".ninja_log", "build.ninja")
_LOADED: dict[str, object] = {}


def model_digest(chips: Mapping[str, Chip]) -> str:
    """The model's identity: its structure and every weight."""
    h = hashlib.sha256(json.dumps({n: c.structure() for n, c in chips.items()}, sort_keys=True).encode())
    for name, chip in chips.items():
        for key, tensor in sorted(chip.state_dict().items()):
            h.update(f"{name}.{key}".encode())
            h.update(tensor.contiguous().numpy().tobytes())
    return h.hexdigest()


def backend_digest() -> str:
    """The backend's identity: its sources (and everything they import),
    the runtime, the compiler flags and the Torch that builds them."""
    sources = build.source_digest(__name__, files=tuple(map(str, RUNTIME_FILES)))
    return hashlib.sha256(f"{sources}|{torch.__version__}|{CFLAGS}".encode()).hexdigest()


def layer_blocks(layer: Layer) -> dict[str, np.ndarray]:
    """A layer's sparse weights as the compiler's blocks (compiler.layer_arrays):
    each projection dense over the slots it touches."""
    def part(matrix, p):
        return getattr(layer, f"{matrix}_{p}").numpy()

    n_rows, n_channels, n_ffn = layer.dims.tolist()
    matrices = MATRICES + (HALVES if layer.has_halves else ())

    def slots(axis_of: dict[str, str]) -> np.ndarray:
        used = [part(m, p) for m, p in axis_of.items() if m in matrices]
        return np.unique(np.concatenate(used)).astype(np.int32) if used else np.zeros(0, np.int32)

    blocks = {
        "read_kind": layer.read_kind.numpy(),
        "in_slots": slots({"w_in": "cols"}),
        "out_slots": slots({"w_out": "rows", "h_out": "rows"}),
        "ff_in_slots": slots({"w_ff": "cols", "h_ff": "cols"}),
        "ff_out_slots": slots({"w_down": "rows", "h_down": "rows"}),
    }
    shape = {
        "w_in": (n_rows, "in_slots", 1), "w_out": (n_channels, "out_slots", 0),
        "w_ff": (2 * n_ffn, "ff_in_slots", 1), "w_down": (n_ffn, "ff_out_slots", 0),
    }
    for matrix in matrices:
        n, slot_name, axis = shape["w_" + matrix[2:]]
        slot_index = {int(s): i for i, s in enumerate(blocks[slot_name])}
        rows, cols, coef = part(matrix, "rows"), part(matrix, "cols"), part(matrix, "coef")
        block = np.zeros((n, len(slot_index)) if axis == 1 else (len(slot_index), n), dtype=coef.dtype)
        if axis == 1:
            block[rows, [slot_index[int(c)] for c in cols]] = coef
        else:
            block[[slot_index[int(r)] for r in rows], cols] = coef
        blocks[matrix] = block
    return blocks


def circuit_description(chip: Chip, token: str, circuit: Transformer) -> dict:
    """A circuit in the compiler's terms (model/compiler.py:weight_module)."""
    return {
        "layers": [layer_blocks(layer) for layer in circuit.layers],
        "d_model": circuit.d_model,
        "one_slot": circuit.one_slot,
        "input_slots": circuit.input_slots.tolist(),
        "output_slots": circuit.output_slots.tolist(),
        "fraction_bits": chip.f_bits,
        "dtype": "int32" if chip.dtype == torch.int32 else "float32",
        "reassociate_residuals": chip.tokens[token].reassociate_residuals,
    }


def chip_source(index: int, chip: Chip) -> str:
    """One translation unit: the chip's kernels (internal to it) and its descriptor."""
    circuits = chip.circuits()
    kernels, metadata, scalar = bank_kernels(
        [circuit_description(chip, token, c) for token, _spec, c in circuits],
        memory_size=chip.memory_size,
    )
    entries = ",\n    ".join(
        f"{{reinterpret_cast<const void*>(&circuit_{m['circuit']}), {m['width']}, {m['reads']}}}"
        for m in metadata
    )
    tokens = ", ".join(
        f"{{{len(chip.tokens[token].inputs)}, {len(chip.tokens[token].outputs)}}}"
        for token, _spec, _c in circuits
    )
    primary = circuits[0][2]
    kind = "Float32" if scalar == "float" else "Int32"
    return (
        f"// Chip {index}: {len(circuits)} circuits.\n"
        + ssa.CPP_PREAMBLE
        + '#include "bank.h"\n'
        + "namespace {\n"
        + kernels
        + f"\nconst transformer_nes_bank::Circuit circuits[] = {{\n    {entries}}};\n"
        + f"const transformer_nes_bank::Token tokens[] = {{{tokens}}};\n"
        + "}  // namespace\n"
        + f"extern const transformer_nes_bank::Bank chip_{index} = {{\n"
        + f"    TRANSFORMER_NES_BANK_ABI, transformer_nes_bank::{kind}, {len(circuits)}, circuits, tokens,\n"
        + f"    {chip.memory_size}, {chip.f_bits}, {len(primary.layers) + 1}, {primary.d_model}}};\n"
    )


def table_source(names: list[str]) -> str:
    banks = "".join(f"extern const transformer_nes_bank::Bank chip_{i};\n" for i in range(len(names)))
    entries = ", ".join(f'{{"{name}", &chip_{i}}}' for i, name in enumerate(names))
    return (
        '#include "bank.h"\n' + banks
        + f"const transformer_nes_bank::ChipEntry nes_chips[] = {{{entries}}};\n"
        + f"const int nes_chip_count = {len(names)};\n"
    )


def load_library(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_model(chips: Mapping[str, Chip], cache_dir: Path) -> CompiledModel:
    """The chips compiled into one library, built if this model and backend
    have not been built into ``cache_dir`` before."""
    chips = dict(chips)
    digest = hashlib.sha256(f"{model_digest(chips)}|{backend_digest()}".encode()).hexdigest()[:16]
    name = f"nes_{digest}"
    final = Path(cache_dir) / name
    library = final / f"{name}.so"
    if name in _LOADED:
        module = _LOADED[name]
    elif library.exists():
        module = load_library(name, library)
    else:
        module = build_library(name, chips, final)
    _LOADED[name] = module
    return CompiledModel(module, chips)


def build_library(name: str, chips: dict[str, Chip], final: Path):
    from torch.utils.cpp_extension import load

    start = time.perf_counter()
    work = final.with_name(f".{final.name}.{os.getpid()}")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    sources = [str(RUNTIME_FILES[0])]
    for i, chip in enumerate(chips.values()):
        path = work / f"chip_{i}.cpp"
        path.write_text(chip_source(i, chip))
        sources.append(str(path))
    (work / "chips.cpp").write_text(table_source(list(chips)))
    sources.append(str(work / "chips.cpp"))
    lowered = time.perf_counter()
    module = load(
        name=name,
        sources=sources,
        extra_cflags=list(CFLAGS),
        extra_ldflags=[f"-Wl,-rpath,{Path(torch.__file__).parent / 'lib'}"],
        extra_include_paths=[str(NATIVE_DIR)],
        build_directory=str(work),
        verbose=False,
    )
    for entry in work.iterdir():
        if entry.name.endswith(INTERMEDIATE):
            entry.unlink()
    try:
        work.rename(final)
    except OSError:  # an identical concurrent build published first
        shutil.rmtree(work, ignore_errors=True)
    for older in final.parent.glob("nes_*"):
        if older != final:
            shutil.rmtree(older, ignore_errors=True)
    logger.info(
        "built %s: %d chips lowered in %.1fs, compiled in %.1fs",
        name, len(chips), lowered - start, time.perf_counter() - lowered,
    )
    return module


class CompiledModel:
    """A model's chips compiled into one native library."""

    def __init__(self, module, chips: dict[str, Chip]):
        if list(module.chips()) != list(chips):
            raise RuntimeError("the library does not hold this model's chips")
        self.module, self.chips = module, chips

    def chip(self, name: str) -> CompiledChip:
        """A new engine, with its own memory, running the named chip."""
        return CompiledChip(self.module.engine(name), self.chips[name])


class CompiledChip:
    """One chip's native engine: its own memory, running the chip's circuits
    by model id (token by token, each full circuit before its specializations)."""

    def __init__(self, engine, chip: Chip):
        self.engine, self.chip = engine, chip
        self.graphs = dict(chip.tokens)  # name -> TokenType (.inputs, .outputs, .pins)
        self.model_ids = {(token, spec): i for i, (token, spec, _c) in enumerate(chip.circuits())}
        self.dtype = np.int32 if chip.dtype == torch.int32 else np.float32

    def model_id(self, token: str, specialization: str | None = None) -> int:
        return self.model_ids[token, specialization]

    def step(self, inputs, model: int = 0) -> np.ndarray:
        return self.engine.step(model, np.asarray(inputs, dtype=self.dtype))

    def write(self, addr: int, value: int) -> None:
        self.engine.write(addr, value)

    def abi(self) -> tuple:
        """(engine, write, step, trace, trace rows, trace width) for native
        callers; this object must outlive them."""
        return self.engine.abi()
