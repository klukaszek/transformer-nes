"""Export the analytic transformer weights as a standard ML artifact.

Writes ``weights.safetensors`` (per-matrix CSR of ragged per-layer
matrices: dead heads/FFN units are pruned exactly and the remaining
weights are still ~99% zeros; values are float32, exact after query
rescaling) plus ``machine.json`` (the machine protocol: input
slot map, readout slots, output-head rows, per-layer dims).  The artifact is
self-contained: a standalone runner needs only these two files and the
execution semantics documented in the JSON -- hardmax attention with the
LATEST tie-break, ReGLU FFN, residual stream, no normalization.

Usage: ``uv run python -m transformer_nes.model.export OUT_DIR [--bcd]``
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp

from transformer_nes.cpu6502.backends.graph import RUNTIME_READ_DIMS, Cpu6502Runtime
from transformer_nes.graph.core import InputDimension, ProgramGraph
from transformer_nes.model.schedule import asap_schedule
from transformer_nes.model.weights import build_model


def csr_tensors(name: str, mat: np.ndarray, out: dict) -> None:
    m = sp.csr_matrix(mat)
    m.sort_indices()
    out[f"{name}.data"] = m.data.astype(np.float32)
    out[f"{name}.indices"] = m.indices.astype(np.int32)
    out[f"{name}.indptr"] = m.indptr.astype(np.int32)


def prune_layer(ip, op, fi, fo, d_model, num_heads, d_ffn):
    """Drop dead heads and FFN units from one layer's matrices, exactly.

    A head contributes 0 to the residual when its V rows or its out_proj
    columns are all zero (zero Q/K with live V is a passthrough -- kept);
    an FFN unit contributes 0 when its gate row, value row, or ff_out
    column is all zero.  Returns ragged matrices plus the kept counts.
    """
    D = d_model
    heads = [
        h for h in range(num_heads)
        if ip[2 * D + 2 * h:2 * D + 2 * h + 2].any()
        and op[:, 2 * h:2 * h + 2].any()
    ]
    hi = [r for h in heads for r in (2 * h, 2 * h + 1)]
    ip = np.vstack([ip[hi], ip[[D + r for r in hi]], ip[[2 * D + r for r in hi]]])
    op = op[:, hi]
    units = [
        j for j in range(d_ffn)
        if fi[j].any() and fi[d_ffn + j].any() and fo[:, j].any()
    ]
    fi = np.vstack([fi[units], fi[[d_ffn + j for j in units]]])
    fo = fo[:, units]
    return ip, op, fi, fo, heads, units


def rescale_head_queries(in_proj: np.ndarray, num_heads: int) -> np.ndarray:
    """Divide each head's Q row pair by its hardmax temperature.

    A positive per-head scalar on Q scales that head's scores uniformly, so
    the argmax and its exact ties are untouched; the construction's memory
    queries carry C = 1e10*sqrt(2), which this removes.  Only applied when
    the quotients land on integers (guarded), so the exported coefficients
    are integers -- the clear gates stay K-side and are never scaled.
    """
    out = in_proj.copy()
    for h in range(num_heads):
        rows = out[2 * h:2 * h + 2]
        nz = np.abs(rows[rows != 0.0])
        if nz.size == 0:
            continue
        s = nz.min()
        if s <= 1.0:
            continue
        scaled = rows / s
        ints = np.rint(scaled)
        if (np.abs(scaled - ints) <= 1e-6 * np.maximum(1.0, np.abs(ints))).all():
            out[2 * h:2 * h + 2] = ints
    return out


def export_model(out_dir: str, *, bcd: bool = False) -> dict:
    """Build the union model and write the artifact; returns the metadata."""
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

    # f32 storage loads on every GPU (WebGPU has no f64) and is exact:
    # after query rescaling every coefficient but the -1e30 clear gates
    # (argmax losers, exactness-free) is an integer below 2^24.
    # The READOUT head row is scheduling scaffolding (1e-6 keep-alive
    # coefficients, ignored at prediction time -- only the STATE and HALT
    # rows compete); zeroed so every exported coefficient is an integer
    # or +-0.5.
    head = model.head.weight.copy()
    head[tok_to_idx["READOUT"]] = 0.0
    tensors: dict[str, np.ndarray] = {
        "tok": model.tok.weight.astype(np.float32),
        "head": head.astype(np.float32),
    }
    layer_heads, layer_ffn = [], []
    for li in range(info.n_layers):
        ip, op, fi, fo, heads, units = prune_layer(
            model.attn[li].in_proj_weight, model.attn[li].out_proj.weight,
            model.ff_in[li].weight, model.ff_out[li].weight,
            info.d_model, model.attn[li].num_heads,
            model.ff_in[li].weight.shape[0] // 2)
        # Everything kept must be live by the model's own classification
        # (head_kinds/ffn_kinds 0 = unused); pruning MORE is fine -- the
        # kinds over-approximate (e.g. a kind-1 unit whose gate row folded
        # to zero never passes ReGLU).
        if any(model.head_kinds[li][h] == 0 for h in heads):
            raise AssertionError(f"L{li}: kept a head classified unused")
        if any(model.ffn_kinds[li][j] == 0 for j in units):
            raise AssertionError(f"L{li}: kept an FFN unit classified unused")
        layer_heads.append(len(heads))
        layer_ffn.append(len(units))
        csr_tensors(f"L{li}.in_proj", rescale_head_queries(ip, len(heads)), tensors)
        csr_tensors(f"L{li}.out_proj", op, tensors)
        csr_tensors(f"L{li}.ff_in", fi, tensors)
        csr_tensors(f"L{li}.ff_out", fo, tensors)
    for name, arr in tensors.items():
        if name.endswith((".indices", ".indptr")):
            continue
        vals = arr[arr != 0.0]
        bad = vals[(vals != np.trunc(vals)) & (np.abs(vals) != 0.5)]
        if bad.size:
            raise AssertionError(
                f"{name}: {bad.size} non-integer coefficients survive "
                f"query rescaling (e.g. {bad[:3]})")

    input_slots = {
        d.name: int(info.slot_of[d])
        for d in pg.all_dims
        if isinstance(d, InputDimension) and d in info.slot_of
    }
    meta = {
        "format": (
            "6502 transformer, analytically constructed (no training). "
            "Matrices are CSR (name.data f32 / name.indices i32 / "
            "name.indptr i32); tok and head are dense.  Layers are RAGGED "
            "-- dead heads and FFN units are pruned exactly, so layer li "
            "has H=layer_heads[li] heads and F=layer_ffn[li] units: "
            "in_proj is (6H, D) (rows [0,2H) Q, [2H,4H) K, [4H,6H) V, 2 "
            "dims per head), out_proj (D, 2H), ff_in (2F, D), ff_out "
            "(D, F).  Execution: per layer, x -> in_proj; every token's "
            "K/V appends to a per-layer cache; each head attends by "
            "HARDMAX over qx*kx + qy*ky with exact ties won by the LATEST "
            "token (insert before query, the current token included); "
            "head outputs (2 dims at columns 2h, 2h+1) map through "
            "out_proj into the residual; FFN is ReGLU: gates = "
            "ff_in[:F] @ x, values = ff_in[F:] @ x, "
            "act = gate > 0 ? gate*value : 0, x += ff_out @ act.  No "
            "normalization, no position encoding: order enters ONLY "
            "through the LATEST tie-break (passthrough heads tie every "
            "token with constant Q/K and take the latest = themselves).  "
            "Readouts round half-to-even.  Precision contract: EVERY "
            "coefficient is an integer or +-0.5 (enforced at export: each "
            "head's Q row pair is stored divided by its hardmax "
            "temperature -- a positive per-head score scale, invisible to "
            "the argmax and its exact ties), so storage is exact in any "
            "dtype with 24 mantissa bits and every attention score is an "
            "integer-coefficient function of integer inputs -- no "
            "irrational arithmetic anywhere.  Non-clear coefficients stay "
            "below 2^24; clear-gated keys are shifted by -1e30 and only "
            "need to lose the argmax, never to be exact."
        ),
        "dtype": "float32",
        "bcd": bcd,
        "d_model": int(info.d_model),
        "n_layers": int(info.n_layers),
        "layer_heads": layer_heads,
        "layer_ffn": layer_ffn,
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
    args = parser.parse_args()
    t0 = time.perf_counter()
    meta = export_model(args.out_dir, bcd=args.bcd)
    print(
        f"exported d_model={meta['d_model']} n_layers={meta['n_layers']} "
        f"heads={sum(meta['layer_heads'])} ffn={sum(meta['layer_ffn'])} "
        f"to {args.out_dir} in {time.perf_counter() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
