"""Export the analytic transformer weights as a standard ML artifact.

Writes ``weights.safetensors`` (per-matrix CSR of ragged per-layer
matrices: dead heads/FFN units are pruned exactly and the remaining
weights are still ~99% zeros; coefficients are exact int32 fixed point)
plus ``machine.json`` (the machine protocol: input slot map, readout slots, output-head rows, per-layer dims).  The artifact is
self-contained: a standalone runner needs only these two files and the
execution semantics documented in the JSON -- softmax attention over a
KV cache holding one entry per key, ReGLU FFN, residual stream, no
normalization.

The exported model is the self-contained ``wide_bits=False`` build, which is
NOT the shape the generated engine runs; see ``export_model``.

Usage: ``uv run python -m transformer_nes.cpu6502.backends.weights.export OUT_DIR [--bcd]``
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from transformer_nes.cpu6502.backends.weights.experts import (
    build_interrupt_experts,
    build_opcode_execution_experts,
)
from transformer_nes.cpu6502.construction import RUNTIME_READ_DIMS, Cpu6502Graph
from transformer_nes.cpu6502.spec import OPCODES
from transformer_nes.graph.core import InputDimension
from transformer_nes.model.artifacts import (
    ATTENTION_TEMPERATURE,
    SCALE_BITS,
    csr_tensors,
    prune_layer,
    rescale_head_queries,
    split_clear_gates,
    to_fixed_point,
)
from transformer_nes.model.schedule import asap_schedule
from transformer_nes.model.weights import build_model


def export_model(out_dir: str, *, bcd: bool = False) -> dict:
    """Build the union model and write the artifact; returns the metadata."""
    from safetensors.numpy import save_file

    # wide_bits=False on purpose: the artifact must stand alone, so the model
    # derives a byte's bits itself.  The generated engine builds wide_bits=True
    # instead and has its harness serve those 16 bit inputs, trading a wider
    # token contract for less depth -- a different model, exported nowhere.
    rt = Cpu6502Graph(bcd=bcd)
    dim_by_id = {d.id: d for d in rt.all_dims}
    pg = rt.program_graph()
    sched = asap_schedule(pg)
    model, tok_to_idx, info = build_model(program_graph=pg, schedule=sched)

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
            "Analytically constructed 6502 decoder, no training, normalization or positional encoding. "
            "Dense tok/head and CSR matrices (data/indices/indptr int32). Per layer H=layer_heads, "
            "F=layer_ffn, D=d_model: in_proj (6H,D) packs Q (2H), "
            "K (2H), V (2H); out_proj (D,2H), ff_in (2F,D), ff_out (D,F). "
            "Insert before query, one KV entry per distinct key, replacing equal keys with the "
            "latest token. clear_rows/clear_cols are token-mask predicates on key rows: skip "
            "insertion when any named input is nonzero. Softmax(attention_temperature * dot(Q,K)), "
            "weighted V rounded to the fixed-point grid, then out_proj added to x. "
            "ReGLU: split ff_in @ x into gate/value, relu(gate)*value, ff_out, residual addition. "
            "Residual and matrix coefficients are int32 in units of 2**-scale_bits; shift after "
            "each matrix multiply and ReGLU product. Scores accumulate in int64; "
            "softmax runs in float64. "
            "Quadratic memory uses Q=[q,1], K=[4k,k-2k^2] in sixteen 4KiB groups. "
            "Keys have unique winners and an equality mask for unwritten addresses. "
            "All address/byte features are computed inside the graph; input_slots is the "
            "complete token contract. Constant-key heads return the current token's value."
        ),
        "dtype": "int32",
        "scale_bits": SCALE_BITS,
        "accumulator": "int32",
        "score_accumulator": "int64",
        "attention_temperature": ATTENTION_TEMPERATURE,
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
    pg = rt.program
    if opcodes is None:
        opcodes = sorted(OPCODES)
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
