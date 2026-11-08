"""Construct analytic transformer matrices and serialize exact integer weights.

The schedule assigns residual slots; attention and ReGLU projections implement
the graph, including residual clears at slot reuse. BuildInfo describes that
layout. The final section provides lossless CSR checkpoint serialization.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp

from transformer_nes.graph.core import (
    MEMORY_GROUP_FUSED,
    Expression,
    InputDimension,
    LookUpDimension,
    MemoryRead,
    MemoryReadDimension,
    PersistDimension,
    ReGLUDimension,
)

logger = logging.getLogger(__name__)

HARD_K = 1e10  # construction temperature: attention at its argmax limit (export divides it out)


class BuildInfo(NamedTuple):
    """The schedule/layout facts a caller needs alongside the weights."""

    n_layers: int
    slot_of: dict
    d_model: int
    # (dim, slot, first, end): the dim holds `slot` in the residual stream
    # after the embedding (row 0) and each layer L (row L + 1), for rows
    # first <= row < end -- slots are recycled, so one slot carries
    # several dims over the stack (the debugger's layer-by-layer labels).
    lives: list


class WeightMatrix:
    """A named weight matrix (mimics the ``.weight`` attribute access that the
    construction and the codegen reader use)."""

    __slots__ = ("weight",)

    def __init__(self, shape):
        self.weight = np.zeros(shape, dtype=np.float64)


class AttentionWeights:
    """One attention layer's weights: a fused Q/K/V projection over this
    layer's own H heads (rows 0..2H-1 Q, 2H..4H-1 K, 4H..6H-1 V, two dims per
    head) and an output projection (d_model, 2H)."""

    __slots__ = ("in_proj_weight", "out_proj", "num_heads")

    def __init__(self, d_model, num_heads):
        self.in_proj_weight = np.zeros((6 * num_heads, d_model), dtype=np.float64)
        self.out_proj = WeightMatrix((d_model, 2 * num_heads))
        self.num_heads = num_heads


class WeightModel:
    """Numpy weight container for analytic construction, Torch compilation,
    export and the exact reference forward in model/inference.py."""

    # Per-layer classification metadata, attached by build_model: head/FFN
    # kinds (0 unused, 1 lookup, 2 passthrough), tie-breaks, memory
    # groups, lookup provenance, and erase masks.
    head_kinds: list[list[int]]
    ffn_kinds: list[list[int]]
    head_tiebreak: list[list[int]]
    head_memory_groups: list[list[int]]
    head_lookup_ids: list[list[int]]
    head_pairs: list[list[int]]
    head_memory_outputs: list[list[tuple[str, ...] | None]]
    one_slot: int

    def __init__(self, vocab, d_model, n_heads, n_layers, d_ffn):
        # n_heads / d_ffn are per-layer sequences: the layers are ragged.
        self.tok = WeightMatrix((vocab, d_model))
        self.attn = [AttentionWeights(d_model, n_heads[li]) for li in range(n_layers)]
        self.ff_in = [WeightMatrix((2 * d_ffn[li], d_model)) for li in range(n_layers)]
        self.ff_out = [WeightMatrix((d_model, d_ffn[li])) for li in range(n_layers)]
        self.head = WeightMatrix((vocab, d_model))


@dataclass
class SlotLayout:
    """Where each dim lives in the residual stream (assign_slots)."""

    slot_of: dict  # dim -> residual slot
    d_model: int
    # (layer, half) -> slots whose dead occupant must be erased there,
    # half 1 the attention block and 3 the FFN block.
    erased_at: dict
    lives: list  # BuildInfo.lives


def assign_slots(input_dims, output_tokens, alive_after, n_layers, recycle_slots) -> SlotLayout:
    """Interval coloring of the residual stream.  Inputs take the first
    slots; walking the schedule's half-layer boundaries, a dim born there
    takes the lowest slot freed before it (or a new one), and dims only the
    output tokens read come last.  A slot freed at a boundary is reusable
    from the next one."""
    slot_of = {d: i for i, d in enumerate(dict.fromkeys(input_dims))}
    next_slot = len(slot_of)
    free_slots: list[int] = []
    pending_free: list[int] = []  # freed at this boundary, usable at the next
    death_of_slot = {}  # slot -> boundary index where its occupant died
    # An erase only stops a stale value from polluting the additive update
    # that reuses the slot, so it is created on reuse, not on death: a dead
    # value nobody reallocates rides the residual unread and costs nothing.
    erase_events = []  # (death index, slot)
    boundaries = [(L, half) for L in range(n_layers) for half in (1, 3)]
    first_row = {d: 0 for d in input_dims}
    end_row = {}
    alive = frozenset(input_dims)
    for i, (L, half) in enumerate(boundaries):
        after = alive_after.get(4 * L + half, alive)
        free_slots.extend(pending_free)
        free_slots.sort()
        pending_free = []
        for d in alive - after:
            pending_free.append(slot_of[d])
            death_of_slot[slot_of[d]] = i
            end_row[d] = L + 1  # still readable after layer L - 1
        if not recycle_slots:
            free_slots.clear()
            pending_free.clear()
        for d in sorted(after - alive, key=lambda x: x.id):
            first_row[d] = L + 1
            if free_slots:
                slot_of[d] = free_slots.pop(0)
                erase_events.append((death_of_slot[slot_of[d]], slot_of[d]))
            else:
                slot_of[d] = next_slot
                next_slot += 1
        alive = after

    # An erase may sit at any boundary between death and reuse, but the death
    # boundary is cheapest on non-zeros; concentrating them wastes the spare V
    # slots and shared Q/K rows the genuine passthroughs already pay for.
    erased_at = {b: set() for b in boundaries}
    for death_i, slot in erase_events:
        erased_at[boundaries[death_i]].add(slot)

    output_only = {
        d
        for expr in output_tokens.values()
        if isinstance(expr, Expression)
        for d in expr.terms
        if d not in slot_of and d not in input_dims
    }
    for d in sorted(output_only, key=lambda x: x.id):
        if free_slots:
            slot_of[d] = free_slots.pop(0)
        else:
            slot_of[d] = next_slot
            next_slot += 1
    return SlotLayout(
        slot_of=slot_of,
        d_model=next_slot + next_slot % 2,  # even, for the 2-wide head projections
        erased_at=erased_at,
        lives=[
            (d, slot_of[d], first, end_row.get(d, n_layers + 1)) for d, first in first_row.items()
        ],
    )


def build_model(program_graph, schedule, recycle_slots=True):
    """Build transformer weight matrices from a computation graph.

    Args:
        program_graph: a ProgramGraph wrapping the input/output tokens.
        schedule: its schedule (model/schedule.py:asap_schedule).
        recycle_slots: reuse a dead dim's residual slot (interval coloring).
            Recycling narrows d_model but costs one erase per reuse, because
            the residual update is additive and the stale value has to be
            subtracted.  On the union, recycling wins (d_model 622 instead
            of 4284 for 15.8% fewer non-zeros); on the folded experts, which
            are what execute, a slot per dim wins (max d_model 36 -> 56 for
            13.6% fewer non-zeros), so they build without it
            (experts.py:schedule_and_build).
    """
    pg = program_graph
    input_tokens = pg.input_tokens
    output_tokens = pg.output_tokens
    ALL_DIMS = pg.all_dims
    one = pg.one
    std_layers = schedule["std_layers"]
    n_layers = schedule["num_layers"]
    alive_after = schedule.get("alive_after", {})

    input_dims = [d for d in ALL_DIMS if isinstance(d, InputDimension)]
    layout = assign_slots(input_dims, output_tokens, alive_after, n_layers, recycle_slots)
    slot_of, erased_at, D = layout.slot_of, layout.erased_at, layout.d_model

    # ── Internal dims (never in any alive set) ────────────────
    all_ever_alive = set()
    for s in alive_after.values():
        all_ever_alive |= s
    output_dims_set = set()
    for expr in output_tokens.values():
        if isinstance(expr, Expression):
            output_dims_set |= set(expr.terms.keys())
    non_internal = all_ever_alive | set(input_dims) | output_dims_set
    internal_reglus = {
        d for d in ALL_DIMS if isinstance(d, ReGLUDimension) and d not in non_internal
    }
    internal_lookups = {
        d
        for d in ALL_DIMS
        if isinstance(d, (LookUpDimension, MemoryReadDimension)) and d not in non_internal
    }
    internal_persists = {
        d for d in ALL_DIMS if isinstance(d, PersistDimension) and d not in non_internal
    }
    internal_dims = internal_reglus | internal_lookups | internal_persists

    # ── Expressions for passthrough/erase Q/K ─────────────────
    # Attend-to-self: constant query and key tie every token's score
    # exactly, and the LATEST tie-break selects the current token.
    one_expr = Expression({one: 1})
    erase_q2d = [one_expr, one_expr]
    erase_k2d = [one_expr, one_expr]

    def expr_to_tensor(expr):
        w = np.zeros(D, dtype=np.float64)
        for dim, coeff in expr.terms.items():
            if dim in internal_dims:
                continue
            if dim in slot_of:
                w[slot_of[dim]] += coeff
        return w

    # ── Count resources per layer ─────────────────────────────
    # Layers are RAGGED: each gets exactly the heads and FFN units its phase
    # needs.  Residual width is set by liveness, independent of head count.
    heads_per_layer = []
    ffn_per_layer = []

    for L in range(n_layers):
        attn, persist1, ffn, persist2 = std_layers[L]

        erased1 = erased_at[(L, 1)]

        lookup_dims_set = set()
        for lu in attn:
            lookup_dims_set |= set(lu.dims)
        n_lu_heads = sum((len(lu.dims) + 1) // 2 for lu in attn)

        pt_srcs = set(erased1)
        for pd in persist1:
            for d in pd.expr.terms:
                if d not in slot_of or d in internal_dims:
                    continue
                if not (
                    isinstance(d, (LookUpDimension, MemoryReadDimension)) and d in lookup_dims_set
                ):
                    pt_srcs.add(slot_of[d])

        heads_per_layer.append(n_lu_heads + (len(pt_srcs) + 1) // 2)

        erased3 = erased_at[(L, 3)]

        same_rg = set(ffn)
        pt_srcs_ffn = set(erased3)
        for pd in persist2:
            for d in pd.expr.terms:
                if d not in slot_of or d in internal_dims:
                    continue
                if d not in same_rg:
                    pt_srcs_ffn.add(slot_of[d])

        ffn_per_layer.append(max(1, len(ffn) + len(pt_srcs_ffn)))

    # ── Vocabulary ────────────────────────────────────────────
    all_tokens = sorted(set(input_tokens.keys()) | set(output_tokens.keys()))
    tok_to_idx_map = {t: i for i, t in enumerate(all_tokens)}
    vocab_size = len(all_tokens)

    # ── Create model ──────────────────────────────────────────────
    model = WeightModel(
        vocab=vocab_size,
        d_model=D,
        n_heads=heads_per_layer,
        n_layers=n_layers,
        d_ffn=ffn_per_layer,
    )

    # ── Populate weights ──────────────────────────────────────────
    sqrt_dh = math.sqrt(2.0)
    n_pt_h = n_pt_g = 0

    # Embedding
    for tok_name, expr in input_tokens.items():
        idx = tok_to_idx_map[tok_name]
        model.tok.weight[idx] = expr_to_tensor(expr)
        model.tok.weight[idx, 0] = 0.0
        model.tok.weight[idx, 1] = 0.0
        model.tok.weight[idx, 2] = 0.0

    # Output head
    for tok_name, expr in output_tokens.items():
        idx = tok_to_idx_map[tok_name]
        model.head.weight[idx] = expr_to_tensor(expr)

    # ── Per-layer weights ─────────────────────────────────────
    all_tiebreak = []
    all_ffn_kinds = []
    # Emit only the metadata consumed by inference, lowering, and export.
    # Provenance maps folded experts onto their union's attention state.
    model.head_kinds = []
    model.head_memory_groups = []
    model.head_lookup_ids = []
    model.head_pairs = []
    model.head_memory_outputs = []

    for layer_idx in range(n_layers):
        attn, persist1, ffn, persist2 = std_layers[layer_idx]

        ip = model.attn[layer_idx].in_proj_weight
        op_w = model.attn[layer_idx].out_proj.weight
        n_heads = model.attn[layer_idx].num_heads
        # Q/K/V row bases within this layer's own head block.
        k_base = 2 * n_heads
        v_base = 4 * n_heads

        fi = model.ff_in[layer_idx].weight
        fo = model.ff_out[layer_idx].weight
        d_ffn = model.ff_out[layer_idx].weight.shape[1]

        # ── ATTENTION HALF-LAYER ──────────────────────────────
        head_idx = 0
        lookup_dim_to_head = {}
        layer_tiebreak = []

        # 1) LookUp heads
        kinds = [0] * n_heads
        memory_groups = [-1] * n_heads
        lookup_ids = [-1] * n_heads
        pairs = [-1] * n_heads
        memory_outputs = [None] * n_heads

        for lu in attn:
            if isinstance(lu, MemoryRead):
                value_exprs = [Expression() for _ in lu.dims]
                query_exprs = [lu.address, one_expr]
                key_exprs = [Expression(), Expression()]
                tie_break = "latest"
                memory_group = MEMORY_GROUP_FUSED
            else:
                value_exprs = lu.value_exprs
                query_exprs = lu.query_exprs
                key_exprs = lu.key_exprs
                tie_break = lu.tie_break
                memory_group = lu.memory_group
            nv = len(value_exprs)
            for p in range((nv + 1) // 2):
                h = head_idx
                head_idx += 1
                layer_tiebreak.append(1 if tie_break == "latest" else 0)
                ip[h * 2] = expr_to_tensor(query_exprs[0]) * HARD_K * sqrt_dh
                ip[h * 2 + 1] = expr_to_tensor(query_exprs[1]) * HARD_K * sqrt_dh
                ip[k_base + h * 2] = expr_to_tensor(key_exprs[0])
                ip[k_base + h * 2 + 1] = expr_to_tensor(key_exprs[1])
                ip[v_base + h * 2] = expr_to_tensor(value_exprs[p * 2])
                if p * 2 + 1 < nv:
                    ip[v_base + h * 2 + 1] = expr_to_tensor(value_exprs[p * 2 + 1])

                d0 = lu.dims[p * 2]
                lookup_dim_to_head[d0] = (h, 0)
                if d0 not in internal_lookups:
                    op_w[slot_of[d0], h * 2] = 1.0
                if p * 2 + 1 < nv:
                    d1 = lu.dims[p * 2 + 1]
                    lookup_dim_to_head[d1] = (h, 1)
                    if d1 not in internal_lookups:
                        op_w[slot_of[d1], h * 2 + 1] = 1.0

                kinds[h] = 1
                memory_groups[h] = -1 if memory_group is None else memory_group
                lookup_ids[h] = lu.id
                pairs[h] = p
                if isinstance(lu, MemoryRead):
                    memory_outputs[h] = tuple(lu.output_kinds[p * 2 : p * 2 + 2])

        # 2) Passthrough contributions: src_slot -> {dst_slot: coeff}
        #    Persist terms referencing non-lookup dims need passthrough.
        pt = defaultdict(lambda: defaultdict(float))

        for pd in persist1:
            if pd not in slot_of:
                continue
            for d, c in pd.expr.terms.items():
                if d in lookup_dim_to_head:
                    h, comp = lookup_dim_to_head[d]
                    op_w[slot_of[pd], h * 2 + comp] += c
                elif d in slot_of:
                    pt[slot_of[d]][slot_of[pd]] += c

        for s in erased_at[(layer_idx, 1)]:
            pt[s][s] -= 1.0

        # 3) Pack passthroughs 2 per head (each reads one src_slot)
        pt_items = list(pt.items())
        for pair_idx in range(0, len(pt_items), 2):
            h = head_idx
            head_idx += 1
            ip[h * 2] = expr_to_tensor(erase_q2d[0]) * HARD_K * sqrt_dh
            ip[h * 2 + 1] = expr_to_tensor(erase_q2d[1]) * HARD_K * sqrt_dh
            ip[k_base + h * 2] = expr_to_tensor(erase_k2d[0])
            ip[k_base + h * 2 + 1] = expr_to_tensor(erase_k2d[1])

            src1, dsts1 = pt_items[pair_idx]
            ip[v_base + h * 2, src1] = 1.0
            for dst, coeff in dsts1.items():
                op_w[dst, h * 2] += coeff

            kinds[h] = 2
            if pair_idx + 1 < len(pt_items):
                src2, dsts2 = pt_items[pair_idx + 1]
                ip[v_base + h * 2 + 1, src2] = 1.0
                for dst, coeff in dsts2.items():
                    op_w[dst, h * 2 + 1] += coeff

        n_pt_h += len(pt_items)
        # Passthrough/erase heads rely on the LATEST tie-break (constant
        # keys tie every token; latest == self).
        while len(layer_tiebreak) < head_idx:
            layer_tiebreak.append(1)
        assert head_idx <= n_heads, f"L{layer_idx} attn: {head_idx} heads > {n_heads}"
        while len(layer_tiebreak) < n_heads:
            layer_tiebreak.append(1)
        all_tiebreak.append(layer_tiebreak)
        model.head_kinds.append(kinds)
        model.head_memory_groups.append(memory_groups)
        model.head_lookup_ids.append(lookup_ids)
        model.head_pairs.append(pairs)
        model.head_memory_outputs.append(memory_outputs)

        # ── FFN HALF-LAYER ────────────────────────────────────
        reglu_to_gate = {}
        layer_ffn_kinds = [0] * d_ffn
        j = 0

        # 1) ReGLU neurons
        for rg in ffn:
            fi[j] = expr_to_tensor(rg.b_expr)
            fi[d_ffn + j] = expr_to_tensor(rg.a_expr)
            reglu_to_gate[rg] = j
            layer_ffn_kinds[j] = 1
            if rg not in internal_reglus:
                fo[slot_of[rg], j] = 1.0
            j += 1

        # 2) Passthrough contributions: src_slot -> {dst_slot: coeff}
        pt_ffn = defaultdict(lambda: defaultdict(float))

        for pd in persist2:
            if pd not in slot_of:
                continue
            for d, c in pd.expr.terms.items():
                if d in reglu_to_gate:
                    fo[slot_of[pd], reglu_to_gate[d]] += c
                elif d in slot_of:
                    pt_ffn[slot_of[d]][slot_of[pd]] += c

        for s in erased_at[(layer_idx, 3)]:
            pt_ffn[s][s] -= 1.0

        # 3) One neuron per source slot
        for src, dsts in pt_ffn.items():
            fi[j] = expr_to_tensor(one_expr)
            fi[d_ffn + j, src] = 1.0
            for dst, coeff in dsts.items():
                fo[dst, j] += coeff
            layer_ffn_kinds[j] = 2
            j += 1
            n_pt_g += 1

        assert j <= d_ffn, f"L{layer_idx} ffn: {j} neurons > {d_ffn}"
        all_ffn_kinds.append(layer_ffn_kinds)

    model.head_tiebreak = all_tiebreak
    # FFN neuron classification per layer: 0 unused, 1 ReGLU, 2 linear
    # passthrough/erase neuron.  Fast engines can fuse kind-2 neurons into a
    # direct linear residual update without changing the represented model.
    # The fusion drops the gate, which is only sound because kind-2 gates are
    # the constant-one expression; one_slot lets consumers verify that.
    model.ffn_kinds = all_ffn_kinds
    model.one_slot = slot_of[next(iter(one_expr.terms))]
    n_persist = sum(len(p1) + len(p2) for _, p1, _, p2 in std_layers)
    logger.debug(
        "Built model: d_model=%d, n_layers=%d, heads/layer max=%d mean=%.1f, "
        "d_ffn/layer max=%d mean=%.1f",
        D,
        n_layers,
        max(heads_per_layer, default=0),
        sum(heads_per_layer) / max(len(heads_per_layer), 1),
        max(ffn_per_layer, default=0),
        sum(ffn_per_layer) / max(len(ffn_per_layer), 1),
    )
    logger.debug(
        "  vocab=%d, lookups=%d, persist=%d, pt_h=%d, pt_g=%d",
        vocab_size,
        sum(sum(kind == 1 for kind in kinds) for kinds in model.head_kinds),
        n_persist,
        n_pt_h,
        n_pt_g,
    )

    info = BuildInfo(n_layers=n_layers, slot_of=slot_of, d_model=D, lives=layout.lives)
    return model, tok_to_idx_map, info


def compact_integer_tensors(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Store exact integers in the narrowest signed type, without quantization.

    Checkpoint readers widen data to int32 and indices to int64 on load.
    Tensor shapes, fixed-point units and every integer value stay intact.
    """
    packed = {}
    for name, array in tensors.items():
        if array.dtype.kind != "i":
            raise TypeError(f"{name}: expected signed integer checkpoint tensor")
        lo, hi = int(array.min(initial=0)), int(array.max(initial=0))
        for dtype in (np.int8, np.int16, np.int32):
            bounds = np.iinfo(dtype)
            if bounds.min <= lo and hi <= bounds.max:
                packed[name] = array.astype(dtype, copy=False)
                break
        else:
            raise OverflowError(f"{name}: coefficient or index exceeds int32")
    return packed


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
        h
        for h in range(num_heads)
        if ip[v_base + 2 * h : v_base + 2 * h + 2].any() and op[:, 2 * h : 2 * h + 2].any()
    ]
    hi = [r for h in heads for r in (2 * h, 2 * h + 1)]
    ip = np.vstack([ip[hi], ip[[k_base + r for r in hi]], ip[[v_base + r for r in hi]]])
    op = op[:, hi]
    units = [j for j in range(d_ffn) if fi[j].any() and fi[d_ffn + j].any() and fo[:, j].any()]
    fi = np.vstack([fi[units], fi[[d_ffn + j for j in units]]])
    fo = fo[:, units]
    return ip, op, fi, fo, heads, units


def rescale_head_queries(in_proj: np.ndarray, num_heads: int) -> np.ndarray:
    """Divide each head's query rows by their construction temperature.

    The construction builds every head as softmax at a vast temperature
    (C = 1e10*sqrt(2) on memory queries), its argmax limit; dividing it
    out leaves integer scores, which the exported attention_temperature
    then scales just enough for softmax to be exact.  Only applied when
    the quotients land on integers (guarded), so the exported coefficients
    are integers -- the clear gates stay K-side and are never scaled.
    """
    out = in_proj.copy()
    for h in range(num_heads):
        rows = out[2 * h : 2 * h + 2]
        nz = np.abs(rows[rows != 0.0])
        if nz.size == 0:
            continue
        s = nz.min()
        if s <= 1.0:
            continue
        scaled = rows / s
        ints = np.rint(scaled)
        if (np.abs(scaled - ints) <= 1e-6 * np.maximum(1.0, np.abs(ints))).all():
            out[2 * h : 2 * h + 2] = ints
    return out


CLEAR_GATE_THRESHOLD = -1e29
# Softmax over integer scores whose best leads by >= 1: at this temperature
# the other weights (< e**-32 each) cannot move a rounded head output.
ATTENTION_TEMPERATURE = 32.0
SCALE_BITS = 1


def split_clear_gates(ip: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return (
        kept,
        np.ascontiguousarray(rows, dtype=np.int32),
        np.ascontiguousarray(cols, dtype=np.int32),
    )


def to_fixed_point(arr: np.ndarray, name: str) -> np.ndarray:
    """Exact int32 in units of 2**-SCALE_BITS; raises if any value is lost."""
    scaled = arr * (1 << SCALE_BITS)
    ints = np.rint(scaled)
    if not np.array_equal(scaled, ints):
        bad = scaled[scaled != ints]
        raise AssertionError(
            f"{name}: {bad.size} coefficients are not exact at "
            f"SCALE_BITS={SCALE_BITS} (e.g. {bad[:3]})"
        )
    if np.abs(ints).max(initial=0.0) > np.iinfo(np.int32).max:
        raise AssertionError(f"{name}: coefficient exceeds int32")
    return ints.astype(np.int32)
