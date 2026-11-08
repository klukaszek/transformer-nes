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

from transformer_nes import build
from transformer_nes.cpu6502.engine import NATIVE_DIR
from transformer_nes.cpu6502.execution import readout_slots

logger = logging.getLogger(__name__)
BANK_CFLAGS = ("-O3",)


@functools.cache
def bank_code_digest() -> str:
    """Compiler identity as source data, available without loading weights."""
    sources = build.source_digest(
        "transformer_nes.model.compiler",
        "transformer_nes.cpu6502.runner",
        files=(
            str(NATIVE_DIR / "torch_hook.cpp"),
            str(build.PACKAGE_DIR / "model/native/torch_bank.h"),
        ),
    )
    return hashlib.sha256(f"{sources}|{torch.__version__}|{BANK_CFLAGS!r}".encode()).hexdigest()


def bank_digest(bcd: bool) -> str:
    return hashlib.sha256(f"{bank_code_digest()}|bcd={int(bcd)}".encode()).hexdigest()[:16]


def bank_path(bcd: bool) -> str:
    digest = bank_digest(bcd)
    return os.path.join(build.CACHE_DIR, f"build_{digest}", f"torch_bank_{digest}.so")


def compile_bank(bcd: bool):
    from transformer_nes.cpu6502.execution import step_spec_base
    from transformer_nes.cpu6502.experts import build_core_experts
    from transformer_nes.cpu6502.folds import cached_fold
    from transformer_nes.cpu6502.graph import RUNTIME_READ_DIMS, Cpu6502Graph
    from transformer_nes.graph.fold import GraphFolder
    from transformer_nes.model.compiler import bank_source
    from transformer_nes.model.construct import construct

    digest = bank_digest(bcd)
    name = f"torch_bank_{digest}"
    with build.artifact_lock(digest):
        if not os.path.exists(bank_path(bcd)):
            logger.info("compiling the fixed transformer bank through scalar SSA")
            experts, _ = cached_fold(
                f"core_bcd{int(bcd)}",
                lambda: build_core_experts(Cpu6502Graph(bcd=bcd, wide_bits=True).program_graph()),
            )
            graph = Cpu6502Graph(bcd=bcd, wide_bits=True).program_graph()
            union = GraphFolder(graph, root_names=RUNTIME_READ_DIMS, structural_reads=True).fold(
                {}, label="union"
            )
            model, _, info = construct(union.program_graph)
            slots = {name: info.slot_of[dim] for name, dim in union.dim_of_name.items() if dim in info.slot_of}
            spec = step_spec_base(slots.__getitem__)
            spec["outputs"] = {name: slots[name] for name in RUNTIME_READ_DIMS}
            models = experts + [{"model": model, "step_spec": spec, "output_slots": readout_slots(spec)}]
            source = os.path.join(build.CACHE_DIR, f"build_{digest}.cpp")
            build.write_atomic(source, bank_source(models, (NATIVE_DIR / "torch_hook.cpp").read_text()).encode())
            build.load_module(name, digest, [source], extra_cflags=BANK_CFLAGS)
            os.remove(source)
        build.manifest_put(name, {"kind": "torch_bank", "digest": digest}, code=bank_code_digest())


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
        with build.artifact_lock(digest):
            if os.path.exists(bank_path(bcd)):
                return build.load_module(name, digest, [])
        subprocess.run([sys.executable, "-m", __name__, str(int(bcd)), digest], check=True)


class ForwardBank:
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
        logger.info("native SSA runs every expert's transformer forward (fixed bank)")

    def trace_union(self):
        import numpy as np

        from transformer_nes.model.compiler import bank_source
        from transformer_nes.model.graph_engine import DEBUG_ADAPTER

        code = hashlib.sha256((bank_code_digest() + DEBUG_ADAPTER).encode()).hexdigest()
        key = f"cpu_trace:{self.machine.bcd}"
        if not hasattr(self, "_trace_module"):
            entry = build.manifest_read().get(key)
            if entry is not None and entry.get("code") == code:
                with build.artifact_lock(entry["digest"]):
                    path = Path(build.CACHE_DIR) / f"engine_{entry['digest']}_trace.cpp"
                    if path.exists():
                        self._trace_module = build.load_module(
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
            with build.artifact_lock(digest):
                path = Path(build.CACHE_DIR) / f"engine_{digest}_trace.cpp"
                build.write_atomic(path, source.encode())
                self._trace_module = build.load_module(f"cpu_trace_{digest}", digest, [str(path)])
                build.manifest_put(key, {"kind": "cpu_trace", "digest": digest}, code=code)
        x = np.asarray(self.machine.engine.engine.union_inputs(), dtype=np.int32)
        return self._trace_module.evaluate(x, self.machine.engine.engine.ram_address())

    @property
    def counts(self) -> tuple[int, int]:
        return tuple(self.ctx.counts)


if __name__ == "__main__":
    build.exit_with_parent()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    bcd = bool(int(sys.argv[1]))
    if len(sys.argv) > 2 and sys.argv[2] != bank_digest(bcd):
        raise SystemExit("transformer compiler sources changed during startup; restart the process")
    compile_bank(bcd)
