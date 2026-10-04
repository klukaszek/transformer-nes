"""Project transformer layer weights into the exact integer runtime layout.

Attention queries are shared by identical projection. Whole and half
coefficients remain separate so each matrix keeps its own rounding point.
"""

from __future__ import annotations

import numpy as np
import torch

from transformer_nes.model.artifacts import rescale_head_queries, split_clear_gates, to_fixed_point

LOOKUP, PASSTHROUGH = 1, 2
NO_MEMORY_GROUP = -1
READ_CODES = {"value": -1, "address": -2, **{f"bit{n}": n for n in range(8)}}


def share_queries(w_in, w_out, read_kind):
    """Share query projections and merge identical live value channels."""
    queries, rows, routes, columns, channels = {}, [], [], [], {}
    for h, kinds in enumerate(read_kind.tolist()):
        pair = w_in[2 * h : 2 * h + 2]
        key = pair.numpy().tobytes()
        if key not in queries:
            queries[key] = len(rows)
            rows.append(pair)
        for c, kind in enumerate(kinds):
            column = w_out[:, 2 * h + c]
            if kind == -3 or not column.any():
                continue
            value = (queries[key], kind)
            if value not in channels:
                channels[value] = len(columns)
                columns.append(column.to(torch.int64))
                routes.append(value)
            else:
                columns[channels[value]] += column
    projected = torch.cat([*rows, w_in[2 * len(read_kind) :]])
    values = torch.stack(columns, 1) if columns else w_out[:, :0]
    out = torch.cat([values, w_out[:, 2 * len(read_kind) :]], 1).to(torch.int64)
    if ((out < -(2**31)) | (out >= 2**31)).any():
        raise ValueError("shared value projection exceeds int32")
    return projected, out.to(w_out.dtype), torch.tensor(routes, dtype=torch.int32).reshape(-1, 2)


def layer_arrays(model, li: int) -> dict:
    """Layer `li` of a folded model in expert_runner.layer's form (as
    expert_runner.ExpertLayer cuts it from the exported checkpoint)."""
    attn, heads = model.attn[li], model.attn[li].num_heads
    kinds, groups = model.head_kinds[li], model.head_memory_groups[li]
    if any(kinds[h] == LOOKUP and groups[h] == NO_MEMORY_GROUP for h in range(heads)):
        raise ValueError(f"layer {li}: a lookup head that reads no memory")
    # Head h's query is in-projection rows 2h, 2h + 1, its value rows
    # 4H + 2h, + 1; its output, out-projection columns 2h, 2h + 1.  No
    # head needs its keys.  In units of a half, the engine's fixed point.
    memory = [h for h in range(heads) if kinds[h] == LOOKUP and groups[h] != NO_MEMORY_GROUP]
    if any(groups[h] >= 0 for h in memory):
        raise ValueError("Torch models require structural memory reads before scheduling")
    reads = [2 * h + c for h in memory for c in range(2)]
    passes = [2 * h + c for h in range(heads) if kinds[h] == PASSTHROUGH for c in range(2)]
    w_in = split_clear_gates(rescale_head_queries(attn.in_proj_weight, heads))[0]
    w_in = to_fixed_point(w_in[reads + [4 * heads + c for c in passes]], "w_in")
    w_out = to_fixed_point(attn.out_proj.weight[:, reads + passes], "w_out")
    w_ff = to_fixed_point(model.ff_in[li].weight, "w_ff")
    w_down = to_fixed_point(model.ff_out[li].weight, "w_down")
    if (w_in % 2).any():
        raise ValueError(f"layer {li}: half coefficients in the in-projection")
    outputs = model.head_memory_outputs[li]
    w_in, w_out, read_kind = share_queries(
        torch.from_numpy(w_in),
        torch.from_numpy(w_out),
        torch.tensor([read_codes(outputs[h]) for h in memory], dtype=torch.int32).reshape(-1, 2),
    )
    w_in, w_out, read_kind = w_in.numpy(), w_out.numpy(), read_kind.numpy()
    in_slots, out_slots = touched(w_in, 0), touched(w_out, 1)
    ff_in_slots, ff_out_slots = touched(w_ff, 0), touched(w_down, 1)
    doubled = {
        "w_in": w_in[:, in_slots],
        "w_out": w_out[out_slots],
        "w_ff": w_ff[:, ff_in_slots],
        "w_down": w_down[ff_out_slots],
    }
    halves = {f"h_{name[2:]}": W - 2 * (W >> 1) for name, W in doubled.items() if name != "w_in"}
    arrays = {
        "in_slots": in_slots,
        "out_slots": out_slots,
        "ff_in_slots": ff_in_slots,
        "ff_out_slots": ff_out_slots,
        "read_kind": read_kind,
        **{name: W >> 1 for name, W in doubled.items()},
    }
    if any(h.any() for h in halves.values()):
        arrays |= halves
    arrays = {name: np.ascontiguousarray(a, dtype=np.int32) for name, a in arrays.items()}
    return arrays


def touched(W: np.ndarray, axis: int) -> np.ndarray:
    """The columns (axis 0) or rows (axis 1) of W holding a nonzero."""
    return np.flatnonzero((W != 0).any(axis=axis)).astype(np.int32)


def read_codes(outputs) -> list[int]:
    kinds = list(outputs or ("value", "address"))
    return [READ_CODES[k] for k in kinds] + [-3] * (2 - len(kinds))
