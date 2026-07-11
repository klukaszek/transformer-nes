"""Export the analytic transformer weights as a standard ML artifact.

Writes ``weights.safetensors`` (per-matrix CSR: the weights are ~99.9%
zeros, so dense would be ~11 GB where sparse is tens of MB; values stay
float64 for exactness) plus ``machine.json`` (the machine protocol: input
slot map, readout slots, output-head rows, model dims).  The artifact is
self-contained: a standalone runner needs only these two files and the
execution semantics documented in the JSON -- hardmax attention with the
LATEST tie-break, ReGLU FFN, residual stream, no normalization.

Usage: ``uv run python -m transformer_nes.model.export OUT_DIR [--bcd]``
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import scipy.sparse as sp

from transformer_nes.cpu6502.backends.evaluator import RUNTIME_READ_DIMS, Cpu6502Runtime
from transformer_nes.graph.core import InputDimension, ProgramGraph
from transformer_nes.model.weights import build_model
from transformer_nes.scheduler.milp import asap_schedule


def csr_tensors(name: str, mat: np.ndarray, out: dict, dtype) -> None:
    m = sp.csr_matrix(mat)
    m.sort_indices()
    out[f"{name}.data"] = m.data.astype(dtype)
    out[f"{name}.indices"] = m.indices.astype(np.int32)
    out[f"{name}.indptr"] = m.indptr.astype(np.int32)


def export_model(out_dir: str, *, bcd: bool = False, f64: bool = False) -> dict:
    """Build the union model and write the artifact; returns the metadata."""
    import os

    from safetensors.numpy import save_file

    rt = Cpu6502Runtime(bcd=bcd)
    dim_by_id = {d.id: d for d in rt.all_dims}
    output_tokens = dict(rt.output_tokens)
    readout = 0
    for name in RUNTIME_READ_DIMS:
        readout = readout + 1e-6 * dim_by_id[rt.dim_id_by_name[name]]
    output_tokens["READOUT"] = readout
    pg = ProgramGraph(rt.input_tokens, output_tokens)
    sched = asap_schedule(pg.input_tokens, pg.output_tokens, program_graph=pg)
    model, _tokens, tok_to_idx, info = build_model(program_graph=pg, schedule=sched)

    # f32 is the default: it loads on every GPU (WebGPU has no f64) and is
    # certified exact for this construction; --f64 ships the theorem-free
    # exact-constructed coefficients instead (see the precision prose).
    dtype = np.float64 if f64 else np.float32
    # The READOUT head row is scheduling scaffolding (1e-6 keep-alive
    # coefficients, ignored at prediction time -- only the STATE and HALT
    # rows compete); zeroed so every exported coefficient is an integer
    # or +-0.5.
    head = model.head.weight.copy()
    head[tok_to_idx["READOUT"]] = 0.0
    tensors: dict[str, np.ndarray] = {
        "tok": model.tok.weight.astype(dtype),
        "head": head.astype(dtype),
    }
    for li in range(info.n_layers):
        csr_tensors(f"L{li}.in_proj", model.attn[li].in_proj_weight, tensors, dtype)
        csr_tensors(f"L{li}.out_proj", model.attn[li].out_proj.weight, tensors, dtype)
        csr_tensors(f"L{li}.ff_in", model.ff_in[li].weight, tensors, dtype)
        csr_tensors(f"L{li}.ff_out", model.ff_out[li].weight, tensors, dtype)

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
            "round half-to-even.  Precision contract: EVERY coefficient "
            "is an integer or +-0.5 (verifiable by inspection -- the "
            "recency score term was retired; ties are broken by the "
            "LATEST rule alone), so storage is exact in any dtype with "
            "24 mantissa bits and every attention score is an "
            "integer-coefficient function of integer inputs -- no "
            "irrational arithmetic anywhere.  Memory-lookup scores stay "
            "below 2^24 (i32/f32-exact).  The state/start lookups key on "
            "position (scores in pos and pos^2): compute them in integer "
            "arithmetic (i64), or sidestep them entirely with a running "
            "argmax / positional-latest implementation as the reference "
            "engines do -- f32 score arithmetic there is exact only "
            "below pos ~ 4096."
        ),
        "dtype": "float64" if f64 else "float32",
        "bcd": bcd,
        "d_model": int(info.d_model),
        "n_layers": int(info.n_layers),
        "d_ffn": int(model.ff_in[0].weight.shape[0] // 2),
        "n_heads": int(model.attn[0].num_heads),
        "one_slot": int(info.slot_of[rt.one]),
        "input_slots": input_slots,
        "readout_slots": {
            name: int(info.slot_of[dim_by_id[rt.dim_id_by_name[name]]])
            for name in RUNTIME_READ_DIMS
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
    parser.add_argument("--f64", action="store_true",
                        help="store the exact constructed coefficients "
                             "(default f32 is certified exact but GPU-portable)")
    args = parser.parse_args()
    t0 = time.perf_counter()
    meta = export_model(args.out_dir, bcd=args.bcd, f64=args.f64)
    print(
        f"exported d_model={meta['d_model']} n_layers={meta['n_layers']} "
        f"to {args.out_dir} in {time.perf_counter() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
