"""Export the analytic transformer weights as a standard ML artifact.

Writes ``weights.safetensors`` (per-matrix CSR of ragged per-layer
matrices: dead heads/FFN units are pruned exactly and the remaining
weights are still ~99% zeros; values are float32, exact after query
rescaling) plus ``machine.json`` (the machine protocol: input
slot map, readout slots, output-head rows, per-layer dims).  The artifact is
self-contained: a standalone runner needs only these two files and the
execution semantics documented in the JSON -- hardmax attention with the
LATEST tie-break, ReGLU FFN, residual stream, no normalization.

The exported model is the self-contained ``wide_bits=False`` build, which is
NOT the shape the generated engine runs; see ``export_model``.

Usage: ``uv run python -m transformer_nes.model.export OUT_DIR [--bcd]``
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp

from transformer_nes.cpu6502.backends.weights.experts import (
    UNION_FALLBACK_OPERATIONS,
    build_interrupt_experts,
    build_opcode_execution_experts,
)
from transformer_nes.cpu6502.construction import RUNTIME_READ_DIMS, Cpu6502Graph
from transformer_nes.cpu6502.spec import OPCODES
from transformer_nes.graph.core import InputDimension, ProgramGraph
from transformer_nes.model.schedule import asap_schedule
from transformer_nes.model.weights import build_model


def csr_tensors(name: str, mat: np.ndarray, out: dict) -> None:
    m = sp.csr_matrix(mat)
    m.sort_indices()
    out[f"{name}.data"] = to_fixed_point(m.data, f"{name}.data")
    out[f"{name}.indices"] = m.indices.astype(np.int32)
    out[f"{name}.indptr"] = m.indptr.astype(np.int32)


def prune_layer(ip, op, fi, fo, num_heads, d_ffn):
    """Drop dead heads and FFN units from one layer's matrices, exactly.

    A head contributes 0 to the residual when its V rows or its out_proj
    columns are all zero (zero Q/K with live V is a passthrough -- kept);
    an FFN unit contributes 0 when its gate row, value row, or ff_out
    column is all zero.  Returns ragged matrices plus the kept counts.
    """
    k_base, v_base = 2 * num_heads, 4 * num_heads
    heads = [
        h for h in range(num_heads)
        if ip[v_base + 2 * h:v_base + 2 * h + 2].any()
        and op[:, 2 * h:2 * h + 2].any()
    ]
    hi = [r for h in heads for r in (2 * h, 2 * h + 1)]
    ip = np.vstack(
        [ip[hi], ip[[k_base + r for r in hi]], ip[[v_base + r for r in hi]]]
    )
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


CLEAR_GATE_THRESHOLD = -1e29
SCALE_BITS = 1


def split_clear_gates(ip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lift the -1e30 clear gates out of a layer's Q/K/V projection.

    A clear gate exists only to lose the argmax, never to be exact, so it is
    a *predicate*, not a coefficient: token t is masked for head h when the
    gated input is non-zero.  Extracting it leaves every surviving
    coefficient finite and integral, which is what lets the artifact be
    int32 with no magnitude reasoning anywhere.
    """
    mask = ip <= CLEAR_GATE_THRESHOLD
    kept = np.where(mask, 0.0, ip)
    rows, cols = np.nonzero(mask)
    return kept, np.ascontiguousarray(rows, dtype=np.int32), \
        np.ascontiguousarray(cols, dtype=np.int32)


def to_fixed_point(arr: np.ndarray, name: str) -> np.ndarray:
    """Exact int32 in units of 2**-SCALE_BITS; raises if any value is lost."""
    scaled = arr * (1 << SCALE_BITS)
    ints = np.rint(scaled)
    if not np.array_equal(scaled, ints):
        bad = scaled[scaled != ints]
        raise AssertionError(
            f"{name}: {bad.size} coefficients are not exact at "
            f"SCALE_BITS={SCALE_BITS} (e.g. {bad[:3]})")
    if np.abs(ints).max(initial=0.0) > np.iinfo(np.int32).max:
        raise AssertionError(f"{name}: coefficient exceeds int32")
    return ints.astype(np.int32)


def export_model(out_dir: str, *, bcd: bool = False) -> dict:
    """Build the union model and write the artifact; returns the metadata."""
    from safetensors.numpy import save_file

    # wide_bits=False on purpose: the artifact must stand alone, so the model
    # derives a byte's bits itself.  The generated engine builds wide_bits=True
    # instead and has its harness serve those 16 bit inputs, trading a wider
    # token contract for less depth -- a different model, exported nowhere.
    rt = Cpu6502Graph(bcd=bcd)
    dim_by_id = {d.id: d for d in rt.all_dims}
    output_tokens = dict(rt.output_tokens)
    readout = 0
    for name in RUNTIME_READ_DIMS:
        readout = readout + 1e-6 * dim_by_id[rt.dim_id_by_name[name]]
    output_tokens["READOUT"] = readout
    pg = ProgramGraph(rt.input_tokens, output_tokens)
    sched = asap_schedule(pg.input_tokens, pg.output_tokens, program_graph=pg)
    model, _tokens, tok_to_idx, info = build_model(program_graph=pg, schedule=sched)

    # Every coefficient is an exact int32 in units of 2**-SCALE_BITS: query
    # rescaling makes them integers, and the only half-integers left
    # ((in_p_d - in_p_n)/2 in one FFN) are exact at SCALE_BITS=1.  The
    # READOUT head row is scheduling scaffolding, zeroed first.
    head = model.head.weight.copy()
    head[tok_to_idx["READOUT"]] = 0.0
    tensors: dict[str, np.ndarray] = {
        "tok": to_fixed_point(model.tok.weight, "tok"),
        "head": to_fixed_point(head, "head"),
    }
    layer_heads, layer_ffn = [], []
    # Consumers that can exploit the generated engine's structural fusion
    # need the classifications after pruning.  Keeping these in the
    # artifact makes the optimization available to standalone runners too;
    # old artifacts simply omit them and retain the generic path.
    layer_head_kinds, layer_ffn_kinds = [], []
    n_clear = 0
    for li in range(info.n_layers):
        ip, op, fi, fo, heads, units = prune_layer(
            model.attn[li].in_proj_weight, model.attn[li].out_proj.weight,
            model.ff_in[li].weight, model.ff_out[li].weight,
            model.attn[li].num_heads,
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
        layer_head_kinds.append([
            int(model.head_kinds[li][h]) for h in heads
        ])
        layer_ffn_kinds.append([
            int(model.ffn_kinds[li][j]) for j in units
        ])
        ip_kept, clear_rows, clear_cols = split_clear_gates(
            rescale_head_queries(ip, len(heads)))
        csr_tensors(f"L{li}.in_proj", ip_kept, tensors)
        tensors[f"L{li}.clear_rows"] = clear_rows
        tensors[f"L{li}.clear_cols"] = clear_cols
        csr_tensors(f"L{li}.out_proj", op, tensors)
        csr_tensors(f"L{li}.ff_in", fi, tensors)
        csr_tensors(f"L{li}.ff_out", fo, tensors)
        n_clear += int(clear_rows.shape[0])

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
            "Readouts round half-to-even.  Precision contract: THERE IS NO "
            "FLOATING POINT.  Every coefficient is an exact int32 in units "
            "of 2**-scale_bits, so a conforming runner is pure integer "
            "arithmetic: multiply, add, shift right by scale_bits after "
            "each matmul and after the ReGLU product, compare.  Two things "
            "make that possible.  (1) Each head's Q row pair is stored "
            "divided by its hardmax temperature -- a positive per-head "
            "score scale, invisible to the argmax and its exact ties -- "
            "which turns the queries into integers; the only half-integers "
            "left are one FFN's (in_p_d - in_p_n)/2, exact at "
            "scale_bits=1.  (2) The -1e30 clear gates are NOT stored as "
            "coefficients: a clear gate exists only to lose the argmax, "
            "never to be exact, so it is lifted into `clear_gates` as "
            "(row, col) pairs.  Token t is masked for the head owning that "
            "row when input col is non-zero at insert time.  That removes "
            "the last unbounded magnitude: the residual is int32, and "
            "a score (an address times a temperature, against another) "
            "accumulates in int64.  This artifact is "
            "the SELF-CONTAINED construction: it derives each byte's bits "
            "inside the graph, so input_slots is the whole token contract.  "
            "The repo's generated C++ engine executes a different build of "
            "the same semantics in which the harness serves those bits as "
            "16 extra inputs, trading token width for depth; its shape "
            "therefore differs from the one reported here."
        ),
        "dtype": "int32",
        "scale_bits": SCALE_BITS,
        "accumulator": "int32",
        "score_accumulator": "int64",
        "clear_gates": n_clear,
        "bcd": bcd,
        "d_model": int(info.d_model),
        "n_layers": int(info.n_layers),
        "layer_heads": layer_heads,
        "layer_ffn": layer_ffn,
        "layer_head_kinds": layer_head_kinds,
        "layer_ffn_kinds": layer_ffn_kinds,
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


def _expert_metadata(expert: dict, index: int) -> dict:
    """Return the JSON-only contract for one folded execution expert."""
    model = expert["model"]
    return {
        "index": index,
        "byte": None if "byte" not in expert else int(expert["byte"]),
        "name": expert["name"],
        "kind": expert["kind"],
        "nmi": bool(expert.get("nmi", False)),
        "d_model": int(model.tok.weight.shape[1]),
        "n_layers": len(model.attn),
        "one_slot": int(model.one_slot),
        "layer_heads": [int(a.num_heads) for a in model.attn],
        "layer_ffn": [int(w.weight.shape[0] // 2) for w in model.ff_in],
        "layer_head_kinds": [list(map(int, kinds))
                             for kinds in model.head_kinds],
        "layer_ffn_kinds": [list(map(int, kinds))
                            for kinds in model.ffn_kinds],
        "head_memory_groups": [list(map(int, groups))
                                for groups in model.head_memory_groups],
        "head_memory_outputs": [
            [None if outputs is None else list(outputs) for outputs in layer]
            for layer in model.head_memory_outputs
        ],
        "step_spec": expert["step_spec"],
    }


def expert_artifact(expert: dict, index: int) -> tuple[dict[str, np.ndarray], dict]:
    """One folded expert as checkpoint tensors (``E<index>.L<li>.*``: exact
    int32 CSR, queries rescaled, clear gates lifted) and its metadata."""
    tensors: dict[str, np.ndarray] = {}
    model = expert["model"]
    for li, (attn, ff_in, ff_out) in enumerate(
        zip(model.attn, model.ff_in, model.ff_out, strict=True)
    ):
        stem = f"E{index}.L{li}"
        ip, clear_rows, clear_cols = split_clear_gates(
            rescale_head_queries(attn.in_proj_weight.copy(), attn.num_heads)
        )
        csr_tensors(f"{stem}.in_proj", ip, tensors)
        tensors[f"{stem}.clear_rows"] = clear_rows
        tensors[f"{stem}.clear_cols"] = clear_cols
        csr_tensors(f"{stem}.out_proj", attn.out_proj.weight, tensors)
        csr_tensors(f"{stem}.ff_in", ff_in.weight, tensors)
        csr_tensors(f"{stem}.ff_out", ff_out.weight, tensors)
    return tensors, _expert_metadata(expert, index)


def export_experts(
    out_dir: str, *, bcd: bool = False, opcodes=None,
) -> dict:
    """Export the generated engine's folded one-opcode execution experts.

    The union artifact and the folded artifact intentionally have separate
    contracts.  These experts use ``wide_bits=True`` (the generated engine's
    shape), and each expert carries only its own small residual matrices plus
    the sequence ABI needed to apply state, control flow, and writes.
    """
    from safetensors.numpy import save_file

    rt = Cpu6502Graph(bcd=bcd, wide_bits=True)
    pg = ProgramGraph(rt.input_tokens, rt.output_tokens)
    if opcodes is None:
        opcodes = sorted(
            byte for byte in OPCODES
            if OPCODES[byte].operation not in UNION_FALLBACK_OPERATIONS
        )
    experts = build_opcode_execution_experts(pg, opcodes)
    experts.extend(build_interrupt_experts(pg))
    tensors: dict[str, np.ndarray] = {}
    metadata = []
    for ei, expert in enumerate(experts):
        expert_tensors, meta = expert_artifact(expert, ei)
        tensors.update(expert_tensors)
        metadata.append(meta)

    meta = {
        "format": "6502 folded opcode execution experts",
        "dtype": "int32",
        "scale_bits": SCALE_BITS,
        "accumulator": "int32",  # memory reads are structural: no scores
        "bcd": bcd,
        "n_experts": len(metadata),
        "experts": metadata,
    }
    os.makedirs(out_dir, exist_ok=True)
    save_file(tensors, os.path.join(out_dir, "expert_weights.safetensors"))
    with open(os.path.join(out_dir, "experts.json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir")
    parser.add_argument("--bcd", action="store_true")
    parser.add_argument(
        "--experts", action="store_true",
        help="export folded wide-bit opcode experts instead of the union",
    )
    args = parser.parse_args()
    t0 = time.perf_counter()
    if args.experts:
        meta = export_experts(args.out_dir, bcd=args.bcd)
        print(
            f"exported {meta['n_experts']} folded experts to {args.out_dir} "
            f"in {time.perf_counter() - t0:.1f}s"
        )
    else:
        meta = export_model(args.out_dir, bcd=args.bcd)
        print(
            f"exported d_model={meta['d_model']} n_layers={meta['n_layers']} "
            f"heads={sum(meta['layer_heads'])} ffn={sum(meta['layer_ffn'])} "
            f"to {args.out_dir} in {time.perf_counter() - t0:.1f}s"
        )


if __name__ == "__main__":
    main()
