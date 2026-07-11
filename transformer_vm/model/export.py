"""Export the analytic transformer weights as a standard ML artifact.

Writes ``weights.safetensors`` (per-matrix CSR: the weights are ~99.9%
zeros, so dense would be ~11 GB where sparse is tens of MB; values stay
float64 for exactness) plus ``machine.json`` (the machine protocol: input
slot map, readout slots, output-head rows, model dims).  The artifact is
self-contained: a standalone runner needs only these two files and the
execution semantics documented in the JSON -- hardmax attention with the
LATEST tie-break, ReGLU FFN, residual stream, no normalization.

Usage: ``uv run python -m transformer_vm.model.export OUT_DIR [--bcd]``
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import scipy.sparse as sp

from transformer_vm.cpu6502.evaluator import _RUNTIME_READ_DIMS, Cpu6502Runtime
from transformer_vm.graph.core import InputDimension, ProgramGraph
from transformer_vm.model.weights import build_model
from transformer_vm.scheduler.milp import asap_schedule


def _csr_tensors(name: str, mat: np.ndarray, out: dict) -> None:
    m = sp.csr_matrix(mat)
    m.sort_indices()
    out[f"{name}.data"] = m.data.astype(np.float64)
    out[f"{name}.indices"] = m.indices.astype(np.int32)
    out[f"{name}.indptr"] = m.indptr.astype(np.int32)


def export_model(out_dir: str, *, bcd: bool = False) -> dict:
    """Build the union model and write the artifact; returns the metadata."""
    import os

    from safetensors.numpy import save_file

    rt = Cpu6502Runtime(bcd=bcd)
    dim_by_id = {d.id: d for d in rt.all_dims}
    output_tokens = dict(rt.output_tokens)
    readout = 0
    for name in _RUNTIME_READ_DIMS:
        readout = readout + 1e-6 * dim_by_id[rt.dim_id_by_name[name]]
    output_tokens["READOUT"] = readout
    pg = ProgramGraph(rt.input_tokens, output_tokens)
    sched = asap_schedule(pg.input_tokens, pg.output_tokens, program_graph=pg)
    model, _tokens, tok_to_idx, info = build_model(program_graph=pg, schedule=sched)

    tensors: dict[str, np.ndarray] = {
        "tok": model.tok.weight.astype(np.float64),
        "head": model.head.weight.astype(np.float64),
    }
    for li in range(info.n_layers):
        _csr_tensors(f"L{li}.in_proj", model.attn[li].in_proj_weight, tensors)
        _csr_tensors(f"L{li}.out_proj", model.attn[li].out_proj.weight, tensors)
        _csr_tensors(f"L{li}.ff_in", model.ff_in[li].weight, tensors)
        _csr_tensors(f"L{li}.ff_out", model.ff_out[li].weight, tensors)

    input_slots = {
        d.name: int(info.slot_of[d])
        for d in pg.all_dims
        if isinstance(d, InputDimension) and d in info.slot_of
    }
    meta = {
        "format": (
            "6502 transformer, analytically constructed (no training). "
            "Matrices are CSR (name.data f64 / name.indices i32 / "
            "name.indptr i32); tok and head are dense.  Execution: per "
            "layer, x -> fused in_proj (rows [0,D) Q, [D,2D) K, [2D,3D) V, "
            "2 dims per head); every token's K/V appends to a per-layer "
            "cache; each head attends by HARDMAX over qx*kx + qy*ky with "
            "exact ties won by the LATEST token (insert before query, the "
            "current token included); head outputs (2 dims at columns "
            "2h, 2h+1) map through out_proj into the residual; FFN is "
            "ReGLU: gates = ff_in[:F] @ x, values = ff_in[F:] @ x, "
            "act = gate > 0 ? gate*value : 0, x += ff_out @ act.  No "
            "normalization.  Position features are added at slots 0..2 of "
            "the embedding: pos, 1/ln(2) - 1/ln(pos+2), pos^2.  Readouts "
            "round half-to-even.  Precision: values are exact in float64; "
            "float32 weights/residuals are also exact for THIS construction "
            "(certified: residual values are dyadic rationals < 2^24, and "
            "grouped keys keep attention score gaps above f32 rounding), "
            "but the three position features must stay float64 -- pos^2 "
            "exceeds f32's exact-integer range past pos ~ 4096."
        ),
        "bcd": bcd,
        "d_model": int(info.d_model),
        "n_layers": int(info.n_layers),
        "d_ffn": int(model.ff_in[0].weight.shape[0] // 2),
        "n_heads": int(model.attn[0].num_heads),
        "one_slot": int(info.slot_of[rt._one]),
        "input_slots": input_slots,
        "readout_slots": {
            name: int(info.slot_of[dim_by_id[rt.dim_id_by_name[name]]])
            for name in _RUNTIME_READ_DIMS
        },
        "state_row": int(tok_to_idx["STATE"]),
        "halt_row": int(tok_to_idx["HALT"]),
    }

    os.makedirs(out_dir, exist_ok=True)
    save_file(tensors, os.path.join(out_dir, "weights.safetensors"))
    with open(os.path.join(out_dir, "machine.json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir")
    parser.add_argument("--bcd", action="store_true")
    args = parser.parse_args()
    t0 = time.perf_counter()
    meta = export_model(args.out_dir, bcd=args.bcd)
    print(
        f"exported d_model={meta['d_model']} n_layers={meta['n_layers']} "
        f"to {args.out_dir} in {time.perf_counter() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
