"""Run a fixed transformer bank lowered from Torch FX through scalar SSA.

Compilation happens in a child process before play. Cache hits load one
immutable extension: no runtime weight tensors, online specialization or
per-forward Tensor/ArrayRef wrappers. The native engine retains budgets,
interrupts and device pauses; every expert's neural layers run in the bank.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import subprocess
import sys
from pathlib import Path

import torch

from transformer_nes.cpu6502.backends.weights import native
from transformer_nes.cpu6502.backends.weights.execution import readout_slots
from transformer_nes.files import write_atomic
from transformer_nes.model.build import (
    CACHE_DIR,
    artifact_lock,
    load_module,
    manifest_put,
    manifest_read,
)
from transformer_nes.model.sources import source_digest

logger = logging.getLogger(__name__)
BANK_CFLAGS = ("-O3",)


@functools.cache
def bank_code_digest() -> str:
    """Compiler identity as source data, available without loading weights."""
    sources = source_digest(
        "transformer_nes.model.torch_codegen",
        "transformer_nes.cpu6502.backends.weights.runner",
        files=(
            str(native.ROOT / "torch_hook.cpp"),
            str(Path(__file__).parents[3] / "model/native/torch_bank.h"),
        ),
    )
    return hashlib.sha256(f"{sources}|{torch.__version__}|{BANK_CFLAGS!r}".encode()).hexdigest()


def bank_digest(bcd: bool) -> str:
    return hashlib.sha256(f"{bank_code_digest()}|bcd={int(bcd)}".encode()).hexdigest()[:16]


def bank_path(bcd: bool) -> str:
    digest = bank_digest(bcd)
    return os.path.join(CACHE_DIR, f"build_{digest}", f"torch_bank_{digest}.so")


def compile_bank(bcd: bool):
    from transformer_nes.cpu6502.backends.weights.experts import build_core_experts
    from transformer_nes.cpu6502.backends.weights.folds import cached_fold
    from transformer_nes.cpu6502.construction import Cpu6502Graph
    from transformer_nes.model.torch_codegen import bank_source

    digest = bank_digest(bcd)
    name = f"torch_bank_{digest}"
    with artifact_lock(digest):
        if not os.path.exists(bank_path(bcd)):
            logger.info("compiling the fixed transformer bank through scalar SSA")
            experts, _ = cached_fold(
                f"core_bcd{int(bcd)}",
                lambda: build_core_experts(Cpu6502Graph(bcd=bcd, wide_bits=True).program_graph()),
            )
            from transformer_nes.cpu6502.backends.weights.runner import Cpu6502TransformerMachine

            union = Cpu6502TransformerMachine.__new__(Cpu6502TransformerMachine)
            union.bcd = bcd
            spec = union.step_spec()
            models = experts + [{"model": union.model, "step_spec": spec, "output_slots": readout_slots(spec)}]
            source = os.path.join(CACHE_DIR, f"build_{digest}.cpp")
            write_atomic(source, bank_source(models, native.source("torch_hook.cpp")).encode())
            load_module(name, digest, [source], extra_cflags=BANK_CFLAGS)
            os.remove(source)
        manifest_put(name, {"kind": "torch_bank", "digest": digest}, code=bank_code_digest())


def bank_module(bcd: bool):
    digest = bank_digest(bcd)
    name = f"torch_bank_{digest}"
    if (module := sys.modules.get(name)) is not None:
        # A mapped extension outlives its pruned disk artifacts. Reuse it
        # when constructing another CPU in the same process.
        return module
    # Check under the artifact lock: a concurrent publisher may prune the
    # file between a cache snapshot and load. Compile outside the lock so
    # the child can acquire it, then retry the cache read.
    while True:
        with artifact_lock(digest):
            if os.path.exists(bank_path(bcd)):
                return load_module(name, digest, [])
        subprocess.run([sys.executable, "-m", __name__, str(int(bcd)), digest], check=True)


class TorchForwards:
    """Bind one immutable neural bank to a machine's residual and memory.

    Startup completes before the first frame, so live play never races a
    partial hook publication or competes with a background compiler.
    """

    def __init__(self, machine):
        self.machine = machine
        self.hook = bank_module(machine.bcd)
        engine = machine.engine.engine
        if self.hook.union_id != machine.engine.union_id:
            raise RuntimeError("transformer bank and engine disagree on expert count")
        self.ctx = self.hook.Context(engine.ram_address(), *engine.xray_address())
        engine.set_forward_hook(self.hook.forward_address(), self.ctx.address)
        # The engine's raw context pointer must live for the machine's lifetime.
        machine._torch_forwards = self
        logger.info("native SSA runs every expert's transformer forward (fixed bank)")

    def trace_union(self):
        import numpy as np

        from transformer_nes.model.graph_engine import DEBUG_ADAPTER
        from transformer_nes.model.torch_codegen import bank_source

        code = hashlib.sha256((bank_code_digest() + DEBUG_ADAPTER).encode()).hexdigest()
        key = f"cpu_trace:{self.machine.bcd}"
        if not hasattr(self, "_trace_module"):
            entry = manifest_read().get(key)
            if entry is not None and entry.get("code") == code:
                with artifact_lock(entry["digest"]):
                    path = Path(CACHE_DIR) / f"engine_{entry['digest']}_trace.cpp"
                    if path.exists():
                        self._trace_module = load_module(
                            f"cpu_trace_{entry['digest']}", entry["digest"], [str(path)]
                        )
        if not hasattr(self, "_trace_module"):
            model = {"model": self.machine.model, "step_spec": self.machine.step_spec()}
            model["output_slots"] = readout_slots(model["step_spec"])
            source = bank_source(
                [model],
                DEBUG_ADAPTER.replace("@ROWS@", str(len(self.machine.model.attn) + 1)),
                trace=True,
            )
            digest = hashlib.sha256(source.encode()).hexdigest()[:16]
            with artifact_lock(digest):
                path = Path(CACHE_DIR) / f"engine_{digest}_trace.cpp"
                write_atomic(path, source.encode())
                self._trace_module = load_module(f"cpu_trace_{digest}", digest, [str(path)])
                manifest_put(key, {"kind": "cpu_trace", "digest": digest}, code=code)
        x = np.asarray(self.machine.engine.engine.union_inputs(), dtype=np.int32)
        return self._trace_module.evaluate(x, self.machine.engine.engine.ram_address())

    @property
    def counts(self) -> tuple[int, int]:
        return tuple(self.ctx.counts)


if __name__ == "__main__":
    from transformer_nes.model.build import exit_with_parent

    exit_with_parent()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    bcd = bool(int(sys.argv[1]))
    if len(sys.argv) > 2 and sys.argv[2] != bank_digest(bcd):
        raise SystemExit("transformer compiler sources changed during startup; restart the process")
    compile_bank(bcd)
