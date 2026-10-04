"""Analytical weight construction for the transformer.

Takes an asap_schedule output and constructs the
weight matrices of the transformer from the computation graph.

Each half-layer is a transform: input dims -> latent (gates + passthrough) -> output dims.
The persist/output phase is a linear projection. For each output slot:
  delta[slot] = persist_expr evaluated over gate/passthrough outputs
              - old_value (if slot was reused from a dead dim)

The weights live in plain numpy arrays (WeightModel). model/inference.py
runs an exact reference forward; model/torch_codegen.py lowers their matrix
operations through PyTorch Inductor for CPU and PPU execution.
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

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


class _Weight:
    """A named weight matrix (mimics the ``.weight`` attribute access that the
    construction and the codegen reader use)."""

    __slots__ = ("weight",)

    def __init__(self, shape):
        self.weight = np.zeros(shape, dtype=np.float64)


class _Attn:
    """One attention layer's weights: a fused Q/K/V projection over this
    layer's own H heads (rows 0..2H-1 Q, 2H..4H-1 K, 4H..6H-1 V, two dims per
    head) and an output projection (d_model, 2H)."""

    __slots__ = ("in_proj_weight", "out_proj", "num_heads")

    def __init__(self, d_model, num_heads):
        self.in_proj_weight = np.zeros((6 * num_heads, d_model), dtype=np.float64)
        self.out_proj = _Weight((d_model, 2 * num_heads))
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
        self.tok = _Weight((vocab, d_model))
        self.attn = [_Attn(d_model, n_heads[li]) for li in range(n_layers)]
        self.ff_in = [_Weight((2 * d_ffn[li], d_model)) for li in range(n_layers)]
        self.ff_out = [_Weight((d_model, d_ffn[li])) for li in range(n_layers)]
        self.head = _Weight((vocab, d_model))


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
