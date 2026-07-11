"""Analytical weight construction for the transformer.

Reads a schedule plan (plan.yaml or MILP-generated) and constructs the weight
matrices of the transformer from the computation graph.

Each half-layer is a transform: input dims -> latent (gates + passthrough) -> output dims.
The persist/output phase is a linear projection. For each output slot:
  delta[slot] = persist_expr evaluated over gate/passthrough outputs
              - old_value (if slot was reused from a dead dim)

The weights live in plain numpy arrays (WeightModel); nothing runs a forward
here -- inference is entirely in the generated C++ engine.  Torch is used only
to compile that engine (torch.utils.cpp_extension), not as a tensor library.
"""

import logging
import math
from collections import defaultdict
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

HARD_K = 1e10  # softmax temperature scaling to approximate hardmax (argmax) attention


class BuildInfo(NamedTuple):
    """The schedule/layout facts a caller needs alongside the weights."""

    n_layers: int
    std_layers: list
    slot_of: dict
    d_model: int


def add_position_encoding(x, pos):
    """Add the three deterministic position features to a residual vector."""
    x[0] += pos
    x[1] += 1.0 / math.log(2) - 1.0 / math.log(pos + 2)
    x[2] += pos * pos


class _Weight:
    """A named weight matrix (mimics the ``.weight`` attribute access that the
    construction and the codegen reader use)."""

    __slots__ = ("weight",)

    def __init__(self, shape):
        self.weight = np.zeros(shape, dtype=np.float64)


class _Attn:
    """One attention layer's weights: a fused Q/K/V projection (rows 0..D-1 Q,
    D..2D-1 K, 2D..3D-1 V) and an output projection."""

    __slots__ = ("in_proj_weight", "out_proj", "num_heads")

    def __init__(self, d_model, num_heads):
        self.in_proj_weight = np.zeros((3 * d_model, d_model), dtype=np.float64)
        self.out_proj = _Weight((d_model, d_model))
        self.num_heads = num_heads


class WeightModel:
    """Numpy weight container the analytic construction populates and the
    generated C++ engine reads.  Not a torch module -- there is no forward."""

    def __init__(self, vocab, d_model, n_heads, n_layers, d_ffn, stop_token_id=0):
        self.stop_token_id = stop_token_id
        self.tok = _Weight((vocab, d_model))
        self.attn = [_Attn(d_model, n_heads) for _ in range(n_layers)]
        self.ff_in = [_Weight((2 * d_ffn, d_model)) for _ in range(n_layers)]
        self.ff_out = [_Weight((d_model, d_ffn)) for _ in range(n_layers)]
        self.head = _Weight((vocab, d_model))


def build_model(program_graph=None, schedule=None):
    """Build transformer weight matrices from a computation graph.

    Args:
        program_graph: a ProgramGraph wrapping the input/output tokens.
        schedule: a precomputed schedule dict (asap_schedule / milp_schedule).

    Slots are reused across dead dims (interval coloring) and freed slots are
    erased before reuse, so the represented computation is unchanged.
    """
    from transformer_nes.graph.core import (
        Expression,
        InputDimension,
        LookUpDimension,
        PersistDimension,
        ReGLUDimension,
    )

    if program_graph is not None:
        pg = program_graph
        input_tokens = pg.input_tokens
        output_tokens = pg.output_tokens
        ALL_DIMS = pg.all_dims
        _one = pg.one
        position = pg.position
        inv_log_pos = pg.inv_log_pos
        position_sq = pg.position_sq
    else:
        raise ValueError(
            "build_model requires a program_graph (the default WASM "
            "machine graph was removed; wrap your interpreter's "
            "input/output tokens in a ProgramGraph)"
        )

    # ── Load schedule ─────────────────────────────────────────
    if schedule is None:
        raise ValueError(
            "build_model requires a precomputed schedule "
            "(asap_schedule or milp_schedule output)"
        )
    std_layers = schedule["std_layers"]
    n_layers = schedule["num_layers"]
    alive_after = schedule.get("alive_after", {})

    # ── Slot assignment (interval coloring) ───────────────────
    input_dims = [d for d in ALL_DIMS if isinstance(d, InputDimension)]
    FIXED = {position: 0, inv_log_pos: 1, position_sq: 2}
    protected_slots = set(FIXED.values())
    slot_of = dict(FIXED)
    next_slot = 3
    for d in input_dims:
        if d not in slot_of:
            slot_of[d] = next_slot
            next_slot += 1

    free_slots = []
    pending_free = []  # freed this half-layer, available next
    reused_at = {}  # (L, sp) -> slots written (already 0)
    erased_at = {}  # (L, sp) -> slots being erased to 0

    cur = frozenset(input_dims)
    for L in range(n_layers):
        for sp in (1, 3):
            c = 4 * L + sp
            nxt = alive_after.get(c, cur)
            dying = cur - nxt
            born = nxt - cur

            free_slots.extend(pending_free)
            free_slots.sort()
            pending_free = []

            erased = set()
            for d in dying:
                s = slot_of[d]
                if s in protected_slots:
                    continue
                pending_free.append(s)
                erased.add(s)

            reused = set()
            for d in sorted(born, key=lambda x: x.id):
                if free_slots:
                    s = free_slots.pop(0)
                    slot_of[d] = s
                    reused.add(s)
                else:
                    slot_of[d] = next_slot
                    next_slot += 1

            reused_at[(L, sp)] = reused
            erased_at[(L, sp)] = erased
            cur = nxt

    # Assign slots to output-referenced dims not yet covered
    output_dims_need = set()
    for expr in output_tokens.values():
        if isinstance(expr, Expression):
            for d in expr.terms:
                if d not in slot_of and d not in input_dims:
                    output_dims_need.add(d)
    for d in sorted(output_dims_need, key=lambda x: x.id):
        if free_slots:
            s = free_slots.pop(0)
            slot_of[d] = s
        else:
            slot_of[d] = next_slot
            next_slot += 1

    D = next_slot
    D += D % 2

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
        d for d in ALL_DIMS if isinstance(d, LookUpDimension) and d not in non_internal
    }
    internal_persists = {
        d for d in ALL_DIMS if isinstance(d, PersistDimension) and d not in non_internal
    }
    internal_dims = internal_reglus | internal_lookups | internal_persists

    # ── Expressions for passthrough/erase Q/K ─────────────────
    one_expr = Expression({_one: 1})
    pos_expr = Expression({position: 1})
    erase_q2d = [pos_expr, one_expr]
    erase_k2d = [pos_expr * 2, one_expr]

    def expr_to_tensor(expr):
        w = np.zeros(D, dtype=np.float64)
        for dim, coeff in expr.terms.items():
            if dim in internal_dims:
                continue
            if dim in slot_of:
                w[slot_of[dim]] += coeff
        return w

    # ── Count resources (n_heads, d_ffn) ──────────────────────
    max_heads = 0
    max_ffn = 1

    cur = frozenset(input_dims)
    for L in range(n_layers):
        attn, persist1, ffn, persist2 = std_layers[L]

        c1 = 4 * L + 1
        nxt1 = alive_after.get(c1, cur)
        erased1 = erased_at[(L, 1)]

        lookup_dims_set = set()
        for lu in attn:
            lookup_dims_set |= set(lu.dims)
        n_lu_heads = sum((len(lu.value_exprs) + 1) // 2 for lu in attn)

        pt_srcs = set(erased1)
        for pd in persist1:
            for d in pd.expr.terms:
                if d not in slot_of or d in internal_dims:
                    continue
                if not (isinstance(d, LookUpDimension) and d in lookup_dims_set):
                    pt_srcs.add(slot_of[d])

        max_heads = max(max_heads, n_lu_heads + (len(pt_srcs) + 1) // 2)
        cur = nxt1

        c3 = 4 * L + 3
        nxt3 = alive_after.get(c3, cur)
        erased3 = erased_at[(L, 3)]

        same_rg = set(ffn)
        pt_srcs_ffn = set(erased3)
        for pd in persist2:
            for d in pd.expr.terms:
                if d not in slot_of or d in internal_dims:
                    continue
                if d not in same_rg:
                    pt_srcs_ffn.add(slot_of[d])

        max_ffn = max(max_ffn, len(ffn) + len(pt_srcs_ffn))
        cur = nxt3

    if 2 * max_heads > D:
        D = 2 * max_heads
        D += D % 2
    n_heads = D // 2
    d_ffn = max_ffn

    # ── Vocabulary ────────────────────────────────────────────
    all_tokens = sorted(set(input_tokens.keys()) | set(output_tokens.keys()))
    tok_to_idx_map = {t: i for i, t in enumerate(all_tokens)}
    vocab_size = len(all_tokens)

    # ── Create model ──────────────────────────────────────────────
    model = WeightModel(
        vocab=vocab_size,
        d_model=D,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ffn=d_ffn,
        stop_token_id=tok_to_idx_map.get("halt", 0),
    )

    # ── Populate weights ──────────────────────────────────────────
    sqrt_dh = math.sqrt(2.0)
    n_hulls = n_pt_h = n_pt_g = 0
    head_map = {}  # layer_idx -> {head_idx -> info}

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
    cur = frozenset(input_dims)
    all_tiebreak = []
    all_ffn_kinds = []

    for layer_idx in range(n_layers):
        attn, persist1, ffn, persist2 = std_layers[layer_idx]

        ip = model.attn[layer_idx].in_proj_weight
        op_w = model.attn[layer_idx].out_proj.weight

        fi = model.ff_in[layer_idx].weight
        fo = model.ff_out[layer_idx].weight

        # ── ATTENTION HALF-LAYER ──────────────────────────────
        c1 = 4 * layer_idx + 1
        nxt1 = alive_after.get(c1, cur)
        head_idx = 0
        lookup_dim_to_head = {}
        layer_tiebreak = []

        # 1) LookUp heads
        layer_heads = {}

        def expr_sig(expr):
            return tuple(sorted((d.id, c) for d, c in expr.terms.items()))

        for lu in attn:
            nv = len(lu.value_exprs)
            for p in range((nv + 1) // 2):
                h = head_idx
                head_idx += 1
                n_hulls += 1
                layer_tiebreak.append(1 if lu.tie_break == "latest" else 0)
                ip[h * 2] = expr_to_tensor(lu.query_exprs_2d[0]) * HARD_K * sqrt_dh
                ip[h * 2 + 1] = expr_to_tensor(lu.query_exprs_2d[1]) * HARD_K * sqrt_dh
                ip[D + h * 2] = expr_to_tensor(lu.key_exprs_2d[0])
                ip[D + h * 2 + 1] = expr_to_tensor(lu.key_exprs_2d[1])
                ip[2 * D + h * 2] = expr_to_tensor(lu.value_exprs[p * 2])
                if p * 2 + 1 < nv:
                    ip[2 * D + h * 2 + 1] = expr_to_tensor(lu.value_exprs[p * 2 + 1])

                d0 = lu.dims[p * 2]
                lookup_dim_to_head[d0] = (h, 0)
                if d0 not in internal_lookups:
                    op_w[slot_of[d0], h * 2] = 1.0
                if p * 2 + 1 < nv:
                    d1 = lu.dims[p * 2 + 1]
                    lookup_dim_to_head[d1] = (h, 1)
                    if d1 not in internal_lookups:
                        op_w[slot_of[d1], h * 2 + 1] = 1.0

                vals = [d0.name]
                if p * 2 + 1 < nv:
                    vals.append(lu.dims[p * 2 + 1].name)
                v1_sig = expr_sig(lu.value_exprs[p * 2 + 1]) if p * 2 + 1 < nv else ()
                q_terms = {d.name: c for d, c in lu.query_exprs_2d[0].terms.items()}
                q_terms.update(
                    {"_qy_" + d.name: c for d, c in lu.query_exprs_2d[1].terms.items()}
                )
                k_terms = {d.name: c for d, c in lu.key_exprs_2d[0].terms.items()}
                k_terms.update(
                    {"_ky_" + d.name: c for d, c in lu.key_exprs_2d[1].terms.items()}
                )
                layer_heads[h] = {
                    "type": "lookup",
                    "lookup": lu.name or f"lookup_{lu.id}",
                    "lookup_id": lu.id,
                    "pair": p,
                    "values": vals,
                    "tie_break": lu.tie_break,
                    "memory_group": getattr(lu, "memory_group", -1),
                    "cache_key": (
                        expr_sig(lu.key_exprs_2d[0]),
                        expr_sig(lu.key_exprs_2d[1]),
                        expr_sig(lu.value_exprs[p * 2]),
                        v1_sig,
                    ),
                }
        head_map[layer_idx] = layer_heads

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
            ip[D + h * 2] = expr_to_tensor(erase_k2d[0])
            ip[D + h * 2 + 1] = expr_to_tensor(erase_k2d[1])

            src1, dsts1 = pt_items[pair_idx]
            ip[2 * D + h * 2, src1] = 1.0
            for dst, coeff in dsts1.items():
                op_w[dst, h * 2] += coeff

            pt_info = {"type": "passthrough", "slot_v0": src1}
            if pair_idx + 1 < len(pt_items):
                src2, dsts2 = pt_items[pair_idx + 1]
                ip[2 * D + h * 2 + 1, src2] = 1.0
                for dst, coeff in dsts2.items():
                    op_w[dst, h * 2 + 1] += coeff
                pt_info["slot_v1"] = src2
            layer_heads[h] = pt_info

        n_pt_h += len(pt_items)
        while len(layer_tiebreak) < head_idx:
            layer_tiebreak.append(0)
        assert head_idx <= n_heads, f"L{layer_idx} attn: {head_idx} heads > {n_heads}"
        while len(layer_tiebreak) < n_heads:
            layer_tiebreak.append(0)
        all_tiebreak.append(layer_tiebreak)

        cur = nxt1

        # ── FFN HALF-LAYER ────────────────────────────────────
        c3 = 4 * layer_idx + 3
        nxt3 = alive_after.get(c3, cur)
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

        cur = nxt3

    model.attn_erase = [sorted(erased_at[(li, 1)]) for li in range(n_layers)]
    model.ffn_erase = [sorted(erased_at[(li, 3)]) for li in range(n_layers)]
    model.head_tiebreak = all_tiebreak
    # FFN neuron classification per layer: 0 unused, 1 ReGLU, 2 linear
    # passthrough/erase neuron.  Fast engines can fuse kind-2 neurons into a
    # direct linear residual update without changing the represented model.
    # The fusion drops the gate, which is only sound because kind-2 gates are
    # the constant-one expression; one_slot lets consumers verify that.
    model.ffn_kinds = all_ffn_kinds
    model.one_slot = slot_of[next(iter(one_expr.terms))]
    # Head classification per layer (0 unused, 1 lookup, 2 passthrough).
    # Passthrough/erase heads provably attend to self (their key is 2*pos,
    # strictly increasing), which lets a fast engine skip their hulls.
    model.head_kinds = []
    model.head_cache_groups = []
    model.head_memory_groups = []
    # Lookup provenance per head: the graph LookUp id and value-pair index
    # (-1 for non-lookup heads).  An engine hosting several models built
    # from folds of one union graph uses these to map every model's heads
    # onto the union's shared attention state (memory structure, registers).
    model.head_lookup_ids = []
    model.head_pairs = []
    for layer_idx in range(n_layers):
        kinds = [0] * n_heads
        cache_groups = [-1] * n_heads
        memory_groups = [-1] * n_heads
        lookup_ids = [-1] * n_heads
        pairs = [-1] * n_heads
        cache_group_of_key = {}
        for h, info in head_map.get(layer_idx, {}).items():
            if info["type"] == "lookup":
                kinds[h] = 1
                key = info["cache_key"]
                if key not in cache_group_of_key:
                    cache_group_of_key[key] = len(cache_group_of_key)
                cache_groups[h] = cache_group_of_key[key]
                memory_groups[h] = info["memory_group"]
                lookup_ids[h] = info["lookup_id"]
                pairs[h] = info["pair"]
            else:
                kinds[h] = 2
        model.head_kinds.append(kinds)
        model.head_cache_groups.append(cache_groups)
        model.head_memory_groups.append(memory_groups)
        model.head_lookup_ids.append(lookup_ids)
        model.head_pairs.append(pairs)

    n_persist = sum(len(p1) + len(p2) for _, p1, _, p2 in std_layers)
    logger.info(
        "Built model: d_model=%d, n_layers=%d, n_heads=%d, d_ffn=%d",
        D,
        n_layers,
        n_heads,
        d_ffn,
    )
    logger.info(
        "  vocab=%d, hulls=%d, persist=%d, pt_h=%d, pt_g=%d",
        vocab_size,
        n_hulls,
        n_persist,
        n_pt_h,
        n_pt_g,
    )

    info = BuildInfo(n_layers=n_layers, std_layers=std_layers,
                     slot_of=slot_of, d_model=D)
    return model, all_tokens, tok_to_idx_map, info

