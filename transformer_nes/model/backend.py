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
from collections.abc import Callable, Mapping
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


def circuit_description(chip: Chip, token: str, circuit: Transformer,
                        outputs: list[int] | None = None) -> dict:
    """A circuit, read at all its outputs or those at positions `outputs`,
    in the compiler's terms (model/compiler.py:weight_module)."""
    view = chip.views.get(token)
    readout = circuit.output_slots if outputs is None else circuit.output_slots[outputs]
    return {
        "layers": [layer_blocks(layer) for layer in circuit.layers],
        "d_model": circuit.d_model,
        "one_slot": circuit.one_slot,
        "input_slots": circuit.input_slots.tolist(),
        "output_slots": readout.tolist(),
        "fraction_bits": circuit.f_bits,
        "dtype": "int32" if chip.dtype == torch.int32 else "float32",
        "reassociate_residuals": chip.tokens[view.base if view else token].reassociate_residuals,
    }


def chip_source(index: int, chip: Chip, *, observe: bool = False) -> str:
    """One translation unit: the chip's kernels (internal to it) and its
    descriptor; `observe` kernels also report their memory reads."""
    programs = chip.programs()
    kernels, metadata, scalar = bank_kernels(
        [circuit_description(chip, token, c, outputs) for token, _spec, c, outputs in programs],
        memory_size=chip.memory_size, reads=observe,
    )
    entries = ",\n    ".join(
        f"{{reinterpret_cast<const void*>(&circuit_{m['circuit']}), {m['width']}, {m['reads']}, {c.f_bits}}}"
        for m, (_token, _spec, c, _outputs) in zip(metadata, programs, strict=True)
    )
    tokens = ", ".join(
        f"{{{len(chip.io(token)[0])}, {len(chip.io(token)[1])}}}" for token, _spec, _c, _o in programs
    )
    primary = programs[0][2]
    kind = "Float32" if scalar == "float" else "Int32"
    return (
        f"// Chip {index}: {len(programs)} programs.\n"
        + ssa.CPP_PREAMBLE
        + '#include "bank.h"\n'
        + "namespace {\n"
        + kernels
        + f"\nconst transformer_nes_bank::Circuit circuits[] = {{\n    {entries}}};\n"
        + f"const transformer_nes_bank::Token tokens[] = {{{tokens}}};\n"
        + "}  // namespace\n"
        + f"extern const transformer_nes_bank::Bank chip_{index} = {{\n"
        + f"    TRANSFORMER_NES_BANK_ABI, transformer_nes_bank::{kind}, {len(programs)}, circuits, tokens,\n"
        + f"    {chip.memory_size}, {len(primary.layers) + 1}, {primary.d_model}}};\n"
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
    if name not in _LOADED:
        _LOADED[name] = load_library(name, library) if library.exists() else build_library(name, chips, final)
    return CompiledModel(_LOADED[name], chips, final)


def build_extension(name: str, final: Path, generated: Callable[[], dict[str, str]],
                    files: tuple[Path, ...] = ()):
    """Compile `files` and the sources `generated` returns (file name ->
    text) into the Torch extension ``final/name.so``, published by rename."""
    from torch.utils.cpp_extension import load

    work = final.with_name(f".{final.name}.{os.getpid()}")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    sources = [str(f) for f in files]
    for file_name, text in generated().items():
        (work / file_name).write_text(text)
        sources.append(str(work / file_name))
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
    return module


def build_library(name: str, chips: dict[str, Chip], final: Path):
    start = time.perf_counter()

    def generated() -> dict[str, str]:
        sources = {f"chip_{i}.cpp": chip_source(i, chip) for i, chip in enumerate(chips.values())}
        logger.info("lowered %d chips in %.1fs", len(chips), time.perf_counter() - start)
        return sources | {"chips.cpp": table_source(list(chips))}

    module = build_extension(name, final, generated, files=(RUNTIME_FILES[0],))
    # Older libraries, and the debug kernels built beside them.
    for older in final.parent.glob("nes_*"):
        if older != final:
            shutil.rmtree(older, ignore_errors=True)
    logger.info("built %s in %.1fs", name, time.perf_counter() - start)
    return module


DEBUG_MODULE = """
#include <pybind11/pybind11.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel", [] {
        return pybind11::make_tuple(reinterpret_cast<std::uintptr_t>(&circuit_0), @READS@);
    });
}
"""


def observing_module(library: Path, name: str, chip: Chip):
    """A library of one chip whose kernels report their memory reads (the
    X-ray's read counts), built on first use beside `library`, which it is
    pruned with. Its model ids are the chip's own."""
    module_name = f"{library.name}_{name}_observe"
    final = library.with_name(f"{library.name}.{name}_observe")
    if module_name not in _LOADED:
        so = final / f"{module_name}.so"
        if so.exists():
            _LOADED[module_name] = load_library(module_name, so)
        else:
            _LOADED[module_name] = build_extension(
                module_name, final,
                lambda: {"chip_0.cpp": chip_source(0, chip, observe=True), "chips.cpp": table_source([name])},
                files=(RUNTIME_FILES[0],),
            )
    return _LOADED[module_name]


def debug_kernel(library: Path, name: str, chip: Chip, trace: bool) -> tuple[int, int]:
    """(kernel address, reads) of a chip's model 0 compiled for diagnosis:
    taking an ablation (slot, row, value) after its inputs, and for `trace`
    writing every layer's residual stream. Built on first use beside the
    library, which it is pruned with."""
    kind = "trace" if trace else "ablate"
    final = library.with_name(f"{library.name}.{name}_{kind}")
    module_name = f"{library.name}_{name}_{kind}"
    if module_name not in _LOADED:
        so = final / f"{module_name}.so"
        if so.exists():
            _LOADED[module_name] = load_library(module_name, so)
        else:
            def generated() -> dict[str, str]:
                token, _spec, circuit = chip.circuits()[0]
                kernels, metadata, _scalar = bank_kernels(
                    [circuit_description(chip, token, circuit)],
                    memory_size=chip.memory_size, trace=trace, ablation=True,
                )
                module = DEBUG_MODULE.replace("@READS@", str(metadata[0]["reads"]))
                return {"debug.cpp": ssa.CPP_PREAMBLE + kernels + module}

            _LOADED[module_name] = build_extension(module_name, final, generated)
    return _LOADED[module_name].kernel()


class CompiledModel:
    """A model's chips compiled into one native library."""

    def __init__(self, module, chips: dict[str, Chip], library: Path):
        if list(module.chips()) != list(chips):
            raise RuntimeError("the library does not hold this model's chips")
        self.module, self.chips, self.library = module, chips, library

    def chip(self, name: str) -> CompiledChip:
        """A new engine, with its own memory, running the named chip."""
        chip = self.chips[name]
        engine = self.module.engine(name)
        library = self.library
        engine.set_debug_loader(lambda trace: debug_kernel(library, name, chip, trace))
        return CompiledChip(engine, chip, lambda: CompiledChip(
            observing_module(library, name, chip).engine(name), chip))


class CompiledChip:
    """One chip's native engine: its own memory, running the chip's circuits
    by model id (token by token, each full circuit before its specializations)."""

    def __init__(self, engine, chip: Chip, observing=None):
        self.engine, self.chip, self._observing = engine, chip, observing
        # name -> TokenType or View (.inputs, .outputs)
        self.graphs = {**chip.tokens, **chip.views}
        self.model_ids = {(token, spec): i for i, (token, spec, _c, _o) in enumerate(chip.programs())}
        for name, token in self.graphs.items():
            for spec, target in token.same.items():
                if (name, target) in self.model_ids:
                    self.model_ids[name, spec] = self.model_ids[name, target]
        self.dtype = np.int32 if chip.dtype == torch.int32 else np.float32

    def model_id(self, token: str, specialization: str | None = None) -> int:
        return self.model_ids[token, specialization]

    def step(self, inputs, model: int = 0) -> np.ndarray:
        return self.engine.step(model, np.asarray(inputs, dtype=self.dtype))

    def observing(self) -> CompiledChip:
        """A new engine running this chip with kernels that count their
        memory reads per address (``engine.take_reads()``)."""
        if self._observing is None:
            raise RuntimeError("this chip has no observing build")
        return self._observing()

    def write(self, addr: int, value: int) -> None:
        self.engine.write(addr, value)

    def abi(self) -> tuple:
        """(engine, write, step, trace, trace rows, trace width, memory) for
        native callers; this object must outlive them."""
        return self.engine.abi()
