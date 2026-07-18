"""Metal megakernel runner: the whole 6502 run loop in ONE GPU dispatch.

The answer to "can one 6502 run on the GPU at a respectable rate": the
per-step math is smaller than a kernel launch, so the run loop itself
must live on the GPU.  One dispatch executes up to max_steps whole
instructions -- STATE embed, 75 layers, readout, RAM writes, next state
-- against device-resident RAM/registers, returning only at the step
budget or halt (the C++ engine's run() protocol, minus MMIO).

The linear algebra is pre-scheduled into per-lane sparse streams at pack
time (packed 16-bit dst/col plus coeff), whole rows per lane, loads
balanced -- the SIMT-correct analogue of the C++ engine's straight-line
codegen (per-lane immediates would diverge and serialize).  With --threads
32 the kernel is a single simdgroup and every barrier is a near-free
simdgroup_barrier. Attention is the licensed O(1) forms, identical to
gpu_runner.py.

v4 adds EXPERT ROUTING, the C++ engine's MoE lever on the GPU: every
opcode's rebuilt expert model (fold + asap + build_model, exactly
engine_gen's machinery) is packed alongside the union, and each step
dispatches on ram[pc] to that expert's layer range -- the median expert
is ~5x shallower than the 75-layer union, and the measured wall (one
unhidden memory-latency chain per phase) scales with phase count.
Registers resolve through fold provenance onto the union's allocation;
expert memory reads are the folds' fused 64K-span form.  NMI and
non-expert bytes fall back to the union stack (descriptor 0).

v5 adds SEQUENCE WINDOWS: closed micro-loops fold as ONE forward
retiring K instructions (fold_sequence, bytes re-verified from RAM at
dispatch).  Benching gotcha: a one-threadgroup kernel never ramps the
GPU clock (2-3x swings) -- gpu_heat() pins it with a matmul load first.

v8 opts GPU experts into fold-time linear persist inlining: pure linear
chains become wider rows instead of extra layers, while lookup-facing
coordinates stay materialized for the O(1) attention licenses.  The
all-opcode median falls from 20 to 11 layers.

v9 makes licensed register lookups table-free by certifying their affine
per-layer row layout, and caches positive sequence-window verification
with exact invalidation on overlapping writes.

v10 adds guarded countdown superwindows: DEX/BNE and DEY/BNE loops run
several iterations in one forward whenever the current counter certifies
that every intermediate branch is taken.  The final branch remains live,
so the same expert handles both continuing and exiting the loop exactly.

v11 ports canonical shape experts from the emitted C++ backend, with the
GPU's linear-inlining depth construction. One symbolic model serves every
matched PC/operand instantiation. Profile-selected compare/branch, DEX,
NOP, and setup/store shapes use an O(1) head table and compact opcode-only
verification. RAM-store shapes carry static address bounds; one direct
absolute-store family uses a live operand guard proving target < $2000 and
disjoint from its own code.

v12 adds PGO TRACE WINDOWS: flow-following windows built from a weights-
engine profile, crossing JSR/RTS/JMP with non-contiguous byte checks.
Store-to-load forwarding in the fold licenses stack idioms (PHA/PHP ..
PLA/PLP) and self-modifying operands; pinning the profiled entering S
(dispatch guards S == pin) folds every stack address to a constant the
forwarding can decide.  Mode-certified read bounds prove indexed reads
disjoint from pending stores.  Speculated interior branches (runtime
next_pc verification with exact prefix commit) are supported but default
off: they measured slower than branch-final windows on Klaus.  Klaus's
check-macro block packs into ~50 windows covering ~97% of its steps;
threads=128 is the measured optimum for these wide window models.

Usage:
  uv run python experiments/gpu_single/metal_runner.py --verify
  uv run python experiments/gpu_single/metal_runner.py --bench 100000 [--threads 32]
  uv run python experiments/gpu_single/metal_runner.py --suite nestest --bench 8991 --check
  uv run python experiments/gpu_single/metal_runner.py --suite klaus --bench 200000 --check
  uv run python experiments/gpu_single/metal_runner.py --suite klaus --bench 1000000 --pgo-windows 8
  ... [--experts all|none|A2,CA,D0,4C] [--no-windows]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import sysconfig
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gpu_runner import (  # noqa: E402
    BENCH_ORG,
    STATE_FIELDS,
    bench_image,
    build_union,
    check_row,
    ref_machine,
)

from transformer_nes.cpu6502.backends.graph import RUNTIME_READ_DIMS  # noqa: E402
from transformer_nes.cpu6502.backends.weights.experts import (  # noqa: E402
    build_opcode_experts,
    build_sequence_expert,
    build_shape_experts,
    build_trace_window_expert,
    decode_ops,
    head_row,
)
from transformer_nes.cpu6502.backends.weights.windows import (  # noqa: E402
    STACK_OPERATIONS,
    match_shape_windows,
    propose_trace_window,
    scan_sequence_windows,
)
from transformer_nes.cpu6502.spec import OPCODES  # noqa: E402
from transformer_nes.graph.core import MEMORY_GROUP_FUSED  # noqa: E402
from transformer_nes.model.classify import (  # noqa: E402
    integer_score_terms,
    memory_query,
    register_query,
)

HERE = os.path.dirname(os.path.abspath(__file__))

READOUT_ORDER = (
    "current_pc", "next_pc", "new_a", "new_x", "new_y", "new_s", "new_p",
    "new_halted", "fetched_opcode", "fetched_operand_lo",
    "fetched_operand_hi", "ea", "data_out",
    "is_store", "write_addr_out", "write_value_out",
    "is_store2", "write_addr2_out", "write_value2_out",
    "is_store3", "write_addr3_out", "write_value3_out",
)

COUNTDOWN_REPEATS = (4, 2)
SEQ_ROW = 11
SEQ_GUARD_NONE = 0
SEQ_GUARD_X = 1
SEQ_GUARD_Y = 2
SEQ_GUARD_STORE_ABS = 3
SEQ_GUARD_S = 4

# Profile-derived canonical opcode shapes, ordered longest/most profitable
# first when several match one head. PCs and operands remain symbolic; the
# fold licenses control flow and memory effects exhaustively.
GPU_SHAPE_LIBRARY = (
    ("lda_abs_cmp_beq", bytes.fromhex("ad0000c900f000"), 1),
    ("lda_zp_cmp_beq", bytes.fromhex("a500c900f000"), 1),
    ("and_cmp_beq", bytes.fromhex("2900c900f000"), 1),
    ("nop4_beq", bytes.fromhex("eaeaeaeaf000"), 1),
    ("dey_tya_tax_bpl", bytes.fromhex("8898aa1000"), 1),
    ("iny_dex_bne", bytes.fromhex("c8cad000"), 1),
    ("dex6", bytes.fromhex("cacacacacaca"), 1),
    ("nop6", bytes.fromhex("eaeaeaeaeaea"), 1),
    ("nop4", bytes.fromhex("eaeaeaea"), 1),
    ("bit_clc_lda", bytes.fromhex("240018a900"), 1),
    ("bit_sec_lda", bytes.fromhex("240038a900"), 1),
    ("cmp_imm_bne", bytes.fromhex("c900d000"), 1),
    ("cmp_imm_beq", bytes.fromhex("c900f000"), 1),
    ("cpx_imm_bne", bytes.fromhex("e000d000"), 1),
    ("cpy_imm_bne", bytes.fromhex("c000d000"), 1),
    ("cpy_imm_beq", bytes.fromhex("c000f000"), 1),
    ("cmp_zp_bne", bytes.fromhex("c500d000"), 1),
    ("cmp_absx_bne", bytes.fromhex("dd0000d000"), 1),
    ("cmp_absy_bne", bytes.fromhex("d90000d000"), 1),
    ("eor_zp_bmi", bytes.fromhex("45003000"), 1),
    ("eor_zp_bpl", bytes.fromhex("45001000"), 1),
    ("dex_bpl", bytes.fromhex("ca1000"), 1),
)

GPU_NESTEST_EXTRA_SHAPES = (
    ("lda_sta_zp_x3", bytes.fromhex("a9008500a9008500a9008500"), 1),
    ("lda_sta_zp_x2_ldy", bytes.fromhex("a9008500a9008500a000"), 1),
    ("clc_bit_lda", bytes.fromhex("182400a900"), 1),
    ("ldy_lda_sta_abs", bytes.fromhex("a000a9008d0000"), 1),
    ("lda_sta_abs_lda", bytes.fromhex("a9008d0000a900"), 1),
    ("iny_lda_sta_abs", bytes.fromhex("c8a9008d0000"), 1),
    ("lda_sta_abs", bytes.fromhex("a9008d0000"), 1),
    ("nop2_php_pha_ldy", bytes.fromhex("eaea0848a000"), 1),
    ("nop_php_pha_ldy", bytes.fromhex("ea0848a000"), 1),
    ("iny_lda_sta_zp", bytes.fromhex("c8a9008500"), 1),
    ("lda_sta_zp", bytes.fromhex("a9008500"), 1),
    ("lda_sta_zp_ldy", bytes.fromhex("a9008500a000"), 1),
    ("clv_clc_lda", bytes.fromhex("b818a900"), 1),
    ("bit_lda", bytes.fromhex("2400a900"), 1),
    ("nop3", bytes.fromhex("eaeaea"), 1),
    ("sta_zp_php", bytes.fromhex("850008"), 1),
)
GPU_NESTEST_SHAPES = GPU_SHAPE_LIBRARY + GPU_NESTEST_EXTRA_SHAPES


def stack_rows(rows):
    """Pack a list of (cols, vals) rows into global CSR arrays."""
    indptr = [0]
    cols: list[int] = []
    vals: list[float] = []
    for c, v in rows:
        cols.extend(int(i) for i in c)
        vals.extend(float(x) for x in v)
        indptr.append(len(cols))
    return (np.array(indptr, dtype=np.int32),
            np.array(cols, dtype=np.int32),
            np.array(vals, dtype=np.float32))


def csr_rows_of(mat):
    return [(np.nonzero(mat[r])[0], mat[r][np.nonzero(mat[r])[0]])
            for r in range(mat.shape[0])]


class Stream:
    """Per-lane sparse streams: rows assigned whole to lanes (no write
    races), greedily balanced, one contiguous segment per (layer, lane)."""

    def __init__(self, T):
        self.T = T
        self.dst: list[int] = []
        self.col: list[int] = []
        self.val: list[float] = []
        self.seg: list[int] = []

    def add_layer(self, rows):
        """rows: list of (dst, cols, vals).  Appends T+1 seg offsets."""
        lanes: list[list] = [[] for _ in range(self.T)]
        loads = [0] * self.T
        for dst, cols, vals in sorted(rows, key=lambda r: -len(r[1])):
            lane = loads.index(min(loads))
            lanes[lane].append((dst, cols, vals))
            loads[lane] += len(cols)
        for lane in lanes:
            self.seg.append(len(self.dst))
            for dst, cols, vals in lane:
                for c, v in zip(cols, vals, strict=True):
                    self.dst.append(int(dst))
                    self.col.append(int(c))
                    self.val.append(float(v))
        self.seg.append(len(self.dst))
        # seg holds T+1 entries per layer: lane t's segment is
        # [seg[li*(T+1)+t], seg[li*(T+1)+t+1]); the trailing entry doubles
        # as the next layer's base, so drop the duplicate on append.
        if len(self.seg) % (self.T + 1) != 0:
            self.seg.pop()

    def arrays(self, n_layers):
        seg = np.array(self.seg, dtype=np.int32)
        assert seg.size == n_layers * (self.T + 1)
        # All current live-vector indices fit in 16 bits. Pack (dst, col)
        # into one word beside the coefficient: 8 rather than 12 bytes per
        # sparse term, while retaining the same per-lane schedule.
        dst = np.array(self.dst, dtype=np.uint32)
        col = np.array(self.col, dtype=np.uint32)
        assert np.all(dst <= 0xffff) and np.all(col <= 0xffff)
        pair = (dst << np.uint32(16)) | col
        trip = np.empty(2 * len(self.dst), dtype=np.int32)
        trip[0::2] = pair.view(np.int32)
        trip[1::2] = np.array(self.val, dtype=np.float32).view(np.int32)
        return trip, seg


def spec_state_slots(spec) -> list[int]:
    """A fold's step-spec inputs in the kernel's 23-value embed order
    (STATE_FIELDS + is_state; the p_flags list is c..n, matching)."""
    ins = spec["inputs"]
    pf = [s for s, _m in spec["p_flags"]]
    return ([ins[n] for n in ("pc_lo", "pc_hi", "a", "x", "y", "s")] + pf +
            [ins[n] for n in ("ir", "oplo", "ophi", "ea_lo", "ea_hi",
                              "data", "halted", "nmi", "is_state")])


def countdown_guard(head: int, code: bytes) -> int:
    """Return the state register guarding a two-op decrement/BNE loop.

    For R loop iterations, all R-1 interior BNEs are taken exactly when
    the entering byte counter is 0 or >= R.  The last BNE stays dynamic.
    The byte/PC checks here are construction certificates; dispatch still
    re-verifies the code bytes before using the expert.
    """
    if len(code) != 3:
        return SEQ_GUARD_NONE
    dec, branch = OPCODES.get(code[0]), OPCODES.get(code[1])
    if (
        dec is None
        or dec.operation not in ("dex", "dey")
        or dec.length != 1
        or branch is None
        or branch.operation != "bne"
        or branch.mode != "rel"
        or branch.length != 2
    ):
        return SEQ_GUARD_NONE
    off = code[2] - (256 if code[2] & 0x80 else 0)
    if (head + len(code) + off) & 0xFFFF != head:
        return SEQ_GUARD_NONE
    return SEQ_GUARD_X if dec.operation == "dex" else SEQ_GUARD_Y


def pack_tables(bcd: bool = False, T: int = 32, expert_opcodes=(),
                windows=(), shapes=(), shape_images=(), trace_windows=()):
    """Build the union model plus per-opcode experts and flatten everything
    into kernel buffers.  Layers of all models share the streams and layer
    table (indexed by global layer); each model gets a 12-int descriptor:
    [lay0, n_layers, d_model, one_slot, state_slots, readout_slots,
     shead_col, shead_val, shead_n, hhead_col, hhead_val, hhead_n]
    (offsets local to the slots/heads tables; pack_buffers absolutizes).
    optab maps each opcode byte to its descriptor, 0 (the union) default."""
    model, info, tok_to_idx, input_slots, readout_slots, _rt, pg = build_union(bcd)
    assert set(READOUT_ORDER) == set(RUNTIME_READ_DIMS)

    astream, fstream, gstream = Stream(T), Stream(T), Stream(T)
    mem_rows, mem_meta = [], []   # meta: (group base | -1 = fused 64K, ho_pos)
    reg_srows, reg_grows, reg_vrows, reg_meta = [], [], [], []
    layer_tab = []
    reg_of: dict[tuple, int] = {}
    slots_tab: list[int] = []
    heads_col: list[int] = []
    heads_val: list[float] = []
    desc: list[list[int]] = []

    def add_slots(vals):
        off = len(slots_tab)
        slots_tab.extend(int(v) for v in vals)
        return off

    def add_head(cols_vals):
        cols, vals = cols_vals
        co, vo = len(heads_col), len(heads_val)
        heads_col.extend(int(c) for c in cols)
        heads_val.extend(float(v) for v in vals)
        return co, vo, len(cols)

    def pack_model(mdl, n_layers, label, state_slots, readout, shead, hhead,
                   union_lookup_of=None):
        dm = mdl.tok.weight.shape[1]
        is_union = union_lookup_of is None
        lay0 = len(layer_tab)

        def fused_rows(direct, extra):
            rows = []
            for r in range(dm):
                nz = np.nonzero(direct[r])[0]
                cols = list(nz)
                vals = list(direct[r][nz])
                if extra is not None and extra.shape[1]:
                    ec = np.nonzero(extra[r])[0]
                    cols.extend(dm + int(j) for j in ec)
                    vals.extend(extra[r][ec])
                if cols:
                    rows.append((r, cols, vals))
            return rows

        for li in range(n_layers):
            ip = mdl.attn[li].in_proj_weight
            op = mdl.attn[li].out_proj.weight
            fi = mdl.ff_in[li].weight
            fo = mdl.ff_out[li].weight
            ffn = fo.shape[1]
            kinds = mdl.head_kinds[li]
            nh = mdl.attn[li].num_heads
            lookup = [h for h in range(nh) if kinds[h] == 1]
            pt = [h for h in range(nh) if kinds[h] == 2]

            pt_cols = [2 * h + c for h in pt for c in range(2)]
            ad = op[:, pt_cols] @ ip[[2 * dm + c for c in pt_cols]]
            lu_cols = [2 * h + c for h in lookup for c in range(2)]
            astream.add_layer(fused_rows(ad, op[:, lu_cols] if lookup else None))

            fk = mdl.ffn_kinds[li]
            pt_j = [j for j in range(ffn) if fk[j] == 2]
            fd = fo[:, pt_j] @ fi[[ffn + j for j in pt_j]]
            live = [j for j in range(ffn) if fk[j] == 1 and fo[:, j].any()]
            fstream.add_layer(fused_rows(fd, fo[:, live] if live else None))
            grows = []
            for jj, j in enumerate(live):
                for dst, row in ((2 * jj, fi[j]), (2 * jj + 1, fi[ffn + j])):
                    nz = np.nonzero(row)[0]
                    grows.append((dst, list(nz), list(row[nz])))
            gstream.add_layer(grows)

            mem0, reg0 = len(mem_meta), len(reg_meta)
            for pos, h in enumerate(lookup):
                qx = [(c, i) for i, c in enumerate(ip[2 * h]) if c]
                qy = [(c, i) for i, c in enumerate(ip[2 * h + 1]) if c]
                g = mdl.head_memory_groups[li][h]
                if g >= 0 or g == MEMORY_GROUP_FUSED:
                    terms = memory_query(qx, qy, one_id=mdl.one_slot)
                    if terms is None:
                        raise RuntimeError(
                            f"{label} L{li} h{h}: memory head is not "
                            "structure-eligible")
                    if g == MEMORY_GROUP_FUSED and is_union:
                        raise RuntimeError(
                            f"{label} L{li} h{h}: fused reads are expert-only")
                    mem_rows.append(([i for _c, i in terms],
                                     [c for c, _i in terms]))
                    mem_meta.append((-1 if g == MEMORY_GROUP_FUSED
                                     else g * 4096, 2 * pos))
                    continue
                rq = register_query(qx, qy, one_id=mdl.one_slot,
                                    latest=bool(mdl.head_tiebreak[li][h]))
                if rq is None:
                    raise RuntimeError(f"{label} L{li} h{h}: unlicensed head")
                kx = [(c, i) for i, c in enumerate(ip[dm + 2 * h]) if c]
                ky = [(c, i) for i, c in enumerate(ip[dm + 2 * h + 1]) if c]
                sc = integer_score_terms(rq[0], rq[1], kx, ky)
                if sc is None:
                    raise RuntimeError(f"{label} L{li} h{h}: unlicensed score")
                score_terms, gate_ids = sc
                lu_id, pair = mdl.head_lookup_ids[li][h], mdl.head_pairs[li][h]
                if is_union:
                    key = (lu_id, pair)
                    assert key not in reg_of
                    r = reg_of[key] = len(reg_of)
                else:
                    # Experts resolve onto the union's register through the
                    # fold's lookup provenance -- never allocate (a private
                    # register would accumulate divergent history).
                    r = reg_of.get((union_lookup_of.get(lu_id, -1), pair))
                    if r is None:
                        raise RuntimeError(
                            f"{label} L{li} h{h}: no union register for "
                            f"lookup {lu_id} pair {pair}")
                reg_srows.append(([i for _c, i in score_terms],
                                  [c for c, _i in score_terms]))
                reg_grows.append((gate_ids, [1.0] * len(gate_ids)))
                reg_vrows.append(csr_rows_of(ip[2 * dm + 2 * h:2 * dm + 2 * h + 2]))
                reg_meta.append((r, 2 * pos))
            nreg = len(reg_meta) - reg0
            if nreg:
                score_cols = tuple(reg_srows[reg0][0])
                gate_col = int(reg_grows[reg0][0][0])
                assert len(score_cols) == 2
                assert all(tuple(reg_srows[r][0]) == score_cols
                           and int(reg_grows[r][0][0]) == gate_col
                           for r in range(reg0, reg0 + nreg))
                first_values = reg_vrows[reg0]
                assert len(first_values[0][0]) == len(first_values[1][0]) == 1
                value_base = int(first_values[0][0][0])
                rid_base, ho_base = reg_meta[reg0]
                pair_count = sum(
                    bool(len(reg_vrows[r][1][0]))
                    for r in range(reg0, reg0 + nreg)
                )
                for j, r in enumerate(range(reg0, reg0 + nreg)):
                    v0, v1 = reg_vrows[r]
                    assert list(v0[0]) == [value_base + 2 * j]
                    assert list(v0[1]) == [1.0]
                    assert list(v1[0]) in ([value_base + 2 * j + 1], [])
                    assert list(v1[1]) in ([1.0], [])
                    assert reg_meta[r] == (rid_base + j, ho_base + 2 * j)
                assert all(len(reg_vrows[reg0 + j][1][0])
                           for j in range(pair_count))
                assert all(not len(reg_vrows[reg0 + j][1][0])
                           for j in range(pair_count, nreg))
            else:
                score_cols, gate_col = (-1, -1), -1
                value_base = pair_count = rid_base = ho_base = -1
            layer_tab.append([
                len(lookup), len(live), mem0, len(mem_meta) - mem0,
                nreg, *score_cols, gate_col,
                value_base, pair_count, rid_base, ho_base,
            ])

        sc_, sv_, sn_ = add_head(shead)
        hc_, hv_, hn_ = add_head(hhead)
        desc.append([lay0, n_layers, dm, mdl.one_slot,
                     add_slots(state_slots), add_slots(readout),
                     sc_, sv_, sn_, hc_, hv_, hn_])
        return len(desc) - 1

    pack_model(
        model, info.n_layers, "union",
        [input_slots[n] for n, _ in STATE_FIELDS] + [input_slots["is_state"]],
        [readout_slots[n] for n in READOUT_ORDER],
        head_row(model, tok_to_idx["STATE"]), head_row(model, tok_to_idx["HALT"]))

    optab = [0] * 256
    for e in build_opcode_experts(
        pg, sorted(expert_opcodes), inline_linear=True,
    ):
        spec = e["step_spec"]
        optab[e["byte"]] = pack_model(
            e["model"], e["layers"], f"expert {e['name']}",
            spec_state_slots(spec),
            [spec["outputs"][n] for n in READOUT_ORDER],
            spec["state_head"], spec["halt_head"],
            union_lookup_of=e["union_lookup_of"])

    # Sequence windows: one forward retires the whole window.  The kernel
    # re-verifies the bytes at dispatch, so windows from any image are
    # safe to carry (mismatching ones are inert).
    seqtab = []
    skipped_seq = 0
    for head, code, repeat in windows:
        guard = countdown_guard(head, bytes(code))
        variants = []
        if guard:
            base_steps = len(tuple(decode_ops(bytes(code))))
            for super_repeat in COUNTDOWN_REPEATS:
                taken = tuple(
                    base_steps * r + base_steps - 1
                    for r in range(super_repeat - 1)
                )
                variants.append((super_repeat, guard, super_repeat, taken))
        variants.append((repeat, SEQ_GUARD_NONE, 0, ()))
        for variant_repeat, guard_kind, guard_min, taken_steps in variants:
            try:
                e = build_sequence_expert(
                    pg, head, bytes(code), variant_repeat, inline_linear=True,
                    assume_taken_steps=taken_steps,
                )
            except RuntimeError:
                # Window discovery proposes candidates; the fold licenses
                # arbitrate.  In particular, NES MMIO stores must not be
                # swallowed by a persistent GPU forward.
                skipped_seq += 1
                continue
            spec = e["step_spec"]
            k = e["n_steps"]
            mdesc = pack_model(
                e["model"], e["layers"], e["name"],
                spec_state_slots(spec),
                [spec["step_outputs"][s][n]
                 for s in range(k) for n in READOUT_ORDER],
                spec["state_head"], spec["halt_head"],
                union_lookup_of=e["union_lookup_of"])
            seqtab.append([
                head, k, mdesc, len(e["window"]),
                add_slots([
                    (offset << 8) | byte
                    for offset, byte in enumerate(e["window"])
                ]),
                len(e["window"]), guard_kind, guard_min,
                head, len(e["window"]), -1,
            ])

    # Canonical shapes compile once with symbolic PC/operands, then share
    # that descriptor across every matching image head. A -1 pattern byte
    # is an operand wildcard; opcode bytes are still reverified live.
    shape_matches = 0
    if shapes:
        shape_experts, shape_meta = build_shape_experts(
            pg, shapes, inline_linear=True, allow_stores=True,
            allow_runtime_store_guard=True,
        )
        shape_desc = []
        shape_checks = []
        for e, meta in zip(shape_experts, shape_meta, strict=True):
            if meta["close"] == "jmp":
                raise RuntimeError(
                    "Metal canonical shapes do not yet encode JMP self guards"
                )
            spec = e["step_spec"]
            k = e["n_steps"]
            shape_desc.append(pack_model(
                e["model"], e["layers"], e["name"],
                spec_state_slots(spec),
                [spec["step_outputs"][s][n]
                 for s in range(k) for n in READOUT_ORDER],
                spec["state_head"], spec["halt_head"],
                union_lookup_of=e["union_lookup_of"],
            ))
            shape_checks.append(add_slots([
                (int(offset) << 8) | int(byte)
                for offset, byte in meta["sig"]
            ]))

        matches = set()
        for image in shape_images:
            for sid, head in match_shape_windows(
                image, shape_meta, limit=16384,
            ):
                bounds = shape_experts[sid]["store_bounds"]
                if bounds is not None:
                    lo, hi = bounds
                    length = int(shape_meta[sid]["length"])
                    if head <= hi and head + length > lo:
                        continue
                matches.add((sid, head))
        # Stable shape priority at one head follows GPU_SHAPE_LIBRARY order.
        for sid, head in sorted(matches, key=lambda item: (item[1], item[0])):
            e, meta = shape_experts[sid], shape_meta[sid]
            store_guard = e["store_guard_offset"]
            seqtab.append([
                head, e["n_steps"], shape_desc[sid], meta["length"],
                shape_checks[sid], len(meta["sig"]),
                SEQ_GUARD_STORE_ABS if store_guard is not None
                else SEQ_GUARD_NONE,
                0 if store_guard is None else store_guard,
                head, meta["length"], -1,
            ])
            shape_matches += 1

    # Trace windows: PGO-selected flow-following windows (JSR/RTS crossed,
    # stack idioms forwarded, entering S pinned and guarded).  Byte checks
    # carry signed pc deltas; the span fields bound cache invalidation for
    # non-contiguous code.
    trace_packed = 0
    for tw in trace_windows:
        if "model" in tw:
            e = tw  # prebuilt by the PGO proposal loop
        else:
            try:
                e = build_trace_window_expert(
                    pg, tw["steps"], s_pin=tw.get("s_pin"),
                    symbolic=tw.get("symbolic", frozenset()),
                    inline_linear=True,
                )
            except RuntimeError:
                # PGO proposes; the fold licenses arbitrate.
                skipped_seq += 1
                continue
        spec = e["step_spec"]
        k = e["n_steps"]
        mdesc = pack_model(
            e["model"], e["layers"], e["name"],
            spec_state_slots(spec),
            [spec["step_outputs"][s][n]
             for s in range(k) for n in READOUT_ORDER],
            spec["state_head"], spec["halt_head"],
            union_lookup_of=e["union_lookup_of"])
        span_lo, span_hi = e["span"]
        spec_next = e["spec_next"]
        seqtab.append([
            e["head"], k, mdesc, span_hi - span_lo,
            add_slots([(delta << 8) | byte for delta, byte in e["checks"]]),
            len(e["checks"]),
            SEQ_GUARD_S if e["s_pin"] is not None else SEQ_GUARD_NONE,
            0 if e["s_pin"] is None else e["s_pin"],
            span_lo, span_hi - span_lo,
            add_slots(spec_next) if any(p >= 0 for p in spec_next) else -1,
        ])
        trace_packed += 1

    tables = {}
    for name, stream in (("a", astream), ("f", fstream), ("g", gstream)):
        trip, seg = stream.arrays(len(layer_tab))
        tables[f"{name}trip"] = trip
        tables[f"{name}seg"] = seg
    p, c, v = stack_rows(mem_rows)
    tables["memq_ptr"], tables["memq_col"], tables["memq_val"] = p, c, v

    assert all(list(vals) == [2.0, -1.0] for _cols, vals in reg_srows)
    assert all(list(vals) == [1.0] for _cols, vals in reg_grows)

    tables["layer_tab"] = np.array(layer_tab, dtype=np.int32)
    tables["mem_meta"] = np.array(mem_meta, dtype=np.int32).reshape(-1, 2)
    tables["reg_meta"] = np.array(reg_meta, dtype=np.int32).reshape(-1, 2)
    tables["slots_tab"] = np.array(slots_tab, dtype=np.int32)
    tables["heads_col"] = np.array(heads_col, dtype=np.int32)
    tables["heads_val"] = np.array(heads_val, dtype=np.float32)
    tables["desc"] = np.array(desc, dtype=np.int32)
    tables["optab"] = np.array(optab, dtype=np.int32)
    seqtab.sort(key=lambda row: row[0])
    tables["seqtab"] = np.array(seqtab, dtype=np.int32).reshape(-1, SEQ_ROW)
    # O(1) miss filter and per-head descriptor range, mirroring the emitted
    # C++ engine's seq_head table.  Encoding: ((start + 1) << 8) | count;
    # zero means this pc has no candidate.  Rows are sorted by head above.
    seqhead = np.zeros(65536, dtype=np.int32)
    start = 0
    while start < len(seqtab):
        head = seqtab[start][0]
        end = start + 1
        while end < len(seqtab) and seqtab[end][0] == head:
            end += 1
        count = end - start
        if count >= 256:
            raise RuntimeError(f"sequence head {head:04X} has {count} variants")
        seqhead[head] = ((start + 1) << 8) | count
        start = end
    tables["seqhead"] = seqhead
    params = {
        "d_model": max(d[2] for d in desc),
        "n_layers": info.n_layers,
        "n_regs": len(reg_of),
        "n_seq": len(seqtab),
        "shape_matches": shape_matches,
        "trace_packed": trace_packed,
        "skipped_seq": skipped_seq,
        "max_lookup": max(t[0] for t in layer_tab),
        "max_live": max(t[1] for t in layer_tab),
        "max_k": max((row[1] for row in seqtab), default=1),
    }
    return tables, params


def pack_buffers(tables, params, threads):
    """Concatenate every table into one int and one float buffer; build the
    kernel's P[] offset array (order is the kernel's contract)."""
    iparts, fparts = [], []
    ioff, foff = {}, {}

    def addi(name, arr):
        ioff[name] = sum(a.size for a in iparts)
        iparts.append(np.ascontiguousarray(arr, dtype=np.int32).ravel())

    def addf(name, arr):
        foff[name] = sum(a.size for a in fparts)
        fparts.append(np.ascontiguousarray(arr, dtype=np.float32).ravel())

    for s in ("a", "f", "g"):
        addi(f"{s}trip", tables[f"{s}trip"])
        addi(f"{s}seg", tables[f"{s}seg"])
    addi("memq_ptr", tables["memq_ptr"])
    addi("memq_col", tables["memq_col"])
    addf("memq_val", tables["memq_val"])
    for t in ("layer_tab", "mem_meta"):
        addi(t, tables[t])
    addi("slots_tab", tables["slots_tab"])
    addi("heads_col", tables["heads_col"])
    addf("heads_val", tables["heads_val"])
    # Descriptor slot/head offsets are local to their tables; absolutize
    # into I/F now that the table bases are known.
    dsc = tables["desc"].copy()
    dsc[:, 4] += ioff["slots_tab"]
    dsc[:, 5] += ioff["slots_tab"]
    dsc[:, [6, 9]] += ioff["heads_col"]
    dsc[:, [7, 10]] += foff["heads_val"]
    addi("desc", dsc)
    addi("optab", tables["optab"])
    sqt = tables["seqtab"].copy()
    if sqt.size:
        sqt[:, 4] += ioff["slots_tab"]
        spec = sqt[:, 10] >= 0
        sqt[spec, 10] += ioff["slots_tab"]
    addi("seqtab", sqt)
    addi("seqhead", tables["seqhead"])

    parr = np.array([
        params["d_model"], params["n_layers"], params["n_seq"],
        params["n_regs"],
        ioff["atrip"], ioff["aseg"],
        ioff["ftrip"], ioff["fseg"],
        ioff["gtrip"], ioff["gseg"],
        ioff["memq_ptr"], ioff["memq_col"], foff["memq_val"],
        ioff["layer_tab"], ioff["mem_meta"],
        ioff["desc"], ioff["optab"], ioff["seqtab"], ioff["seqhead"],
    ], dtype=np.int32)
    simd = threads <= 32
    defines = (
        f"#define TG_D {params['d_model']}\n"
        f"#define TG_LIVE {max(params['max_live'], 1)}\n"
        f"#define TG_HO {max(2 * params['max_lookup'], 1)}\n"
        f"#define TG_REG {max(params['n_regs'], 1)}\n"
        f"#define TG_OUT {22 * max(params['max_k'], 1)}\n"
        f"#define SEQ_ROW {SEQ_ROW}\n"
        f"#define SEQ_GUARD_X {SEQ_GUARD_X}\n"
        f"#define SEQ_GUARD_Y {SEQ_GUARD_Y}\n"
        f"#define SEQ_GUARD_STORE_ABS {SEQ_GUARD_STORE_ABS}\n"
        f"#define SEQ_GUARD_S {SEQ_GUARD_S}\n"
        + ("#define BARRIER() simdgroup_barrier(mem_flags::mem_threadgroup)\n"
           "#define DBARRIER() simdgroup_barrier(mem_flags::mem_threadgroup"
           " | mem_flags::mem_device)\n" if simd else
           "#define BARRIER() threadgroup_barrier(mem_flags::mem_threadgroup)\n"
           "#define DBARRIER() threadgroup_barrier(mem_flags::mem_threadgroup"
           " | mem_flags::mem_device)\n")
    )
    return (np.concatenate(iparts), np.concatenate(fparts), parr, defines)


def build_extension() -> str:
    """Compile the Metal host (once per source hash)."""
    import hashlib
    src = os.path.join(HERE, "metal_host.mm")
    with open(src, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:16]
    out_dir = os.environ.get("METAL_HOST_BUILD", os.path.join(HERE, "_build"))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"metal_host_{digest}.so")
    if not os.path.exists(out):
        root = os.path.dirname(os.path.dirname(HERE))
        cmd = [
            "c++", "-ObjC++", "-std=c++20", "-O2", "-fPIC",
            "-fvisibility=hidden",
            f"-I{os.path.join(root, 'vendor', 'pybind11', 'include')}",
            f"-I{sysconfig.get_paths()['include']}",
            "-shared", "-undefined", "dynamic_lookup",
            "-framework", "Metal", "-framework", "Foundation",
            src, "-o", out,
        ]
        subprocess.run(cmd, check=True)
    return out


def load_host():
    import importlib.util
    path = build_extension()
    spec = importlib.util.spec_from_file_location("metal_host", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class MetalMachine:
    """One 6502 stepped by the megakernel; run() is one GPU dispatch."""

    def __init__(self, threads: int = 32, expert_opcodes=(), windows=(),
                 bcd: bool = False, shapes=(), shape_images=(),
                 trace_windows=()):
        tables, params = pack_tables(
            bcd=bcd, T=threads, expert_opcodes=expert_opcodes, windows=windows,
            shapes=shapes, shape_images=shape_images,
            trace_windows=trace_windows,
        )
        ibuf, fbuf, parr, defines = pack_buffers(tables, params, threads)
        host = load_host()
        with open(os.path.join(HERE, "kernel.metal")) as f:
            source = defines + f.read()
        self.native = host.Machine(source, ibuf, fbuf, parr, threads)
        if self.native.threads != threads:
            raise RuntimeError(
                f"pipeline clamped threads to {self.native.threads}; "
                f"streams are packed for {threads}")
        self.params = params

    def boot_image(self, image: bytes, *, start_pc: int):
        self.native.boot(bytes(image), start_pc)
        self.halted = False

    def run(self, max_steps: int, trace: bool = False):
        res = self.native.run(max_steps, trace)
        self.halted = bool(res["halted"])
        return res

    def trace_rows(self, res) -> list[dict]:
        rows = []
        arr = np.asarray(res["trace"]).reshape(-1, len(READOUT_ORDER))
        for r in arr:
            rd = dict(zip(READOUT_ORDER, (int(v) for v in r), strict=True))
            writes = []
            for st, ad, va in (("is_store", "write_addr_out", "write_value_out"),
                               ("is_store2", "write_addr2_out", "write_value2_out"),
                               ("is_store3", "write_addr3_out", "write_value3_out")):
                if rd[st] & 0xFF:
                    writes.append((rd[ad] & 0xFFFF, rd[va] & 0xFF))
            rows.append({
                "pc": rd["current_pc"] & 0xFFFF, "a": rd["new_a"] & 0xFF,
                "x": rd["new_x"] & 0xFF, "y": rd["new_y"] & 0xFF,
                "s": rd["new_s"] & 0xFF, "p": rd["new_p"] & 0xFF,
                "ir": rd["fetched_opcode"] & 0xFF,
                "oplo": rd["fetched_operand_lo"] & 0xFF,
                "ophi": rd["fetched_operand_hi"] & 0xFF,
                "ea": rd["ea"] & 0xFFFF, "data": rd["data_out"] & 0xFF,
                "writes": tuple(writes),
            })
        return rows


def parse_experts(spec: str) -> tuple:
    if spec == "none":
        return ()
    if spec == "all":
        return tuple(OPCODES)
    return tuple(int(b, 16) for b in spec.split(","))


def verify(threads: int, experts=()):
    from transformer_nes.cpu6502.backends.weights.runner import (
        VERIFY_START,
        verify_program,
    )
    invalidation_image = bytearray(65536)
    invalidation_image[BENCH_ORG:BENCH_ORG + 13] = bytes([
        0xA2, 0x01,              # LDX #1
        0xCA, 0xD0, 0xFD,        # cached DEX/BNE window
        0xA9, 0xEA,              # LDA #NOP
        0x8D, 0x02, 0x06,        # STA $0602: overwrite cached window
        0x4C, 0x02, 0x06,        # JMP $0602
    ])
    invalidation_image = bytes(invalidation_image)
    windows = (
        scan_sequence_windows(verify_program())
        + scan_sequence_windows(bench_image())
        + scan_sequence_windows(invalidation_image)
    )
    image = verify_program()
    m = MetalMachine(
        threads, experts, windows, shapes=GPU_SHAPE_LIBRARY,
        shape_images=(image, bench_image(), invalidation_image),
    )
    m.boot_image(image, start_pc=VERIFY_START)
    res = m.run(40, trace=True)
    rows = m.trace_rows(res)
    ref = ref_machine(image, VERIFY_START)
    for i, row in enumerate(rows):
        check_row(row, ref.step(), i)
    print(f"[metal t={threads}] verify_program: {len(rows)} steps "
          f"lockstep-exact (halted={m.halted})")

    m.boot_image(bench_image(), start_pc=BENCH_ORG)
    res = m.run(3000, trace=True)
    rows = m.trace_rows(res)
    ref = ref_machine(bench_image(), BENCH_ORG)
    for i, row in enumerate(rows):
        check_row(row, ref.step(), i)
    print(f"[metal t={threads}] bench loop: {len(rows)} steps lockstep-exact")

    m.boot_image(invalidation_image, start_pc=BENCH_ORG)
    res = m.run(24, trace=True)
    rows = m.trace_rows(res)
    ref = ref_machine(invalidation_image, BENCH_ORG)
    for i, row in enumerate(rows):
        check_row(row, ref.step(), i)
    print(f"[metal t={threads}] sequence cache invalidation: "
          f"{len(rows)} steps lockstep-exact")


def gpu_heat(seconds: float = 2.0):
    """Pin the GPU clock before timing.  A single-threadgroup kernel is
    ~1% utilization, so macOS never ramps for it -- measurements swing
    2-3x with whatever ran before.  A saturating matmul load does ramp."""
    import torch
    if not torch.backends.mps.is_available():
        return
    a = torch.rand(2048, 2048, device="mps")
    until = time.perf_counter() + seconds
    while time.perf_counter() < until:
        a = (a @ a).clamp(0.5, 1.5)
    torch.mps.synchronize()


def bench(n: int, threads: int, null_model: bool = False, experts=(),
          windows=True):
    wins = scan_sequence_windows(bench_image()) if windows else ()
    m = MetalMachine(threads, experts, wins)
    m.boot_image(bench_image(), start_pc=BENCH_ORG)
    m.run(2000)
    gpu_heat()
    t0 = time.perf_counter()
    res = m.native.run(n, 2) if null_model else m.run(n)
    dt = time.perf_counter() - t0
    steps = int(res["steps"]) if not null_model else n
    tag = " NULL-MODEL" if null_model else ""
    print(f"[metal t={threads}{tag}] {steps} steps in {dt:.3f}s = "
          f"{steps / dt:,.0f} steps/s ({dt / steps * 1e6:.1f} us/step)")


def bench_suite(name: str, n: int, threads: int, check: bool, experts=(),
                windows=True, pgo_windows: int = 0,
                pgo_min_hits: int = 512, shapes: bool = True,
                trace_len: int = 10, spec_min: float = 2.0):
    """Benchmark a conformance workload from boot on the megakernel."""
    if name == "nestest":
        from transformer_nes.cpu6502.examples.nestest import (
            START_PC,
            load_image,
        )
        image, bcd = bytes(load_image()), False
    elif name == "klaus":
        from transformer_nes.cpu6502.examples.klaus.harness import (
            START_PC,
            load_image,
        )
        image, bcd = load_image(), True
    else:
        raise ValueError(f"unknown suite {name!r}")

    wins, traces = (), ()
    if pgo_windows:
        # Profile through the C++ weights engine (the reference machine is
        # for lockstep checks, not traces), then propose flow-following
        # trace windows: JSR/RTS crossed, stack idioms forwarded, entering
        # S pinned from the dominant profiled value.
        from transformer_nes.cpu6502.backends.weights.runner import (
            Cpu6502TransformerMachine,
        )
        eng = Cpu6502TransformerMachine(bcd=bcd)
        pcs_parts, s_parts, w_parts = [], [], []
        for out in eng.iter_run_arrays(image, start_pc=START_PC,
                                       max_steps=n):
            pcs_parts.append(np.asarray(out["pc"], dtype=np.int64))
            s_parts.append(np.asarray(out["s"], dtype=np.int64))
            w_parts.append(np.asarray(out["w_addr"], dtype=np.int64))
        pcs = np.concatenate(pcs_parts)
        s_arr = np.concatenate(s_parts)
        written = frozenset(
            int(a) for a in np.unique(np.concatenate(w_parts))
        )
        mem = eng.dump_ram()
        heads, counts = np.unique(pcs, return_counts=True)
        order = np.argsort(-counts, kind="stable")
        # Per-transition counts: speculation past a branch is proposed
        # only when the profile shows it falls through almost always (a
        # rollback wastes the rest of the forward).
        pair, pair_n = np.unique(pcs[:-1] * 65536 + pcs[1:],
                                 return_counts=True)
        succ: dict[int, dict[int, int]] = {}
        for p, c in zip(pair, pair_n, strict=True):
            succ.setdefault(int(p) >> 16, {})[int(p) & 0xFFFF] = int(c)

        def spec_ok(pc, fall):
            outs = succ.get(pc)
            if not outs:
                return False
            return outs.get(fall, 0) >= spec_min * sum(outs.values())
        # Build each candidate here so only LICENSED windows claim
        # coverage: a rejected head must not shadow proposals at its
        # interior pcs.
        pg = build_union(bcd)[-1]
        covered: set[int] = set()
        proposed = []
        rejected = 0
        for head, hits in zip(heads[order], counts[order], strict=True):
            head, hits = int(head), int(hits)
            if len(proposed) >= pgo_windows or hits < pgo_min_hits:
                break
            if head in covered:
                continue
            prop = propose_trace_window(mem.__getitem__, written, head,
                                        max_len=trace_len, spec_ok=spec_ok)
            if prop is None:
                continue
            steps, symbolic, uses_stack = prop
            s_pin = None
            if uses_stack:
                at = np.nonzero(pcs == head)[0]
                at = at[at > 0]
                if not at.size:
                    continue
                vals, vc = np.unique(s_arr[at - 1], return_counts=True)
                s_pin = int(vals[np.argmax(vc)])
            # A license failure names its failing step; the prefix before
            # it is still a valid window, and the suffix head gets its own
            # proposal on a later iteration.
            expert = None
            trunc = list(steps)
            while len(trunc) >= 2:
                use_s = s_pin if any(
                    OPCODES[b[0]].operation in STACK_OPERATIONS
                    for _pc, b in trunc
                ) else None
                try:
                    expert = build_trace_window_expert(
                        pg, trunc, s_pin=use_s, symbolic=symbolic,
                        inline_linear=True,
                    )
                    break
                except RuntimeError as exc:
                    st = getattr(exc, "step", None)
                    if st is None or st < 2:
                        print(
                            f"[metal {name}] PGO reject ({hits} hits): {exc}"
                        )
                        break
                    trunc = trunc[:st]
            if expert is None:
                rejected += 1
                continue
            proposed.append(expert)
            covered.update(pc for pc, _b in trunc)
        traces = tuple(proposed)
        print(
            f"[metal {name}] PGO built {len(traces)} trace windows, "
            f"{rejected} rejected (cap={pgo_windows}, "
            f"min_hits={pgo_min_hits})"
        )
    elif windows and not shapes:
        # Canonical shapes supersede the cold static-loop scan. Concrete
        # models enter representative runs only through explicit PGO; the
        # old scan has no nestest/Klaus coverage win.
        wins = scan_sequence_windows(image)
    suite_shapes = (
        GPU_NESTEST_SHAPES if name == "nestest" else GPU_SHAPE_LIBRARY
    )
    m = MetalMachine(
        threads, experts, wins, bcd=bcd,
        shapes=suite_shapes if shapes else (),
        shape_images=(image,) if shapes else (),
        trace_windows=traces,
    )
    if check:
        m.boot_image(image, start_pc=START_PC)
        res = m.run(n, trace=True)
        ref = ref_machine(image, START_PC, bcd=bcd)
        rows = m.trace_rows(res)
        for i, row in enumerate(rows):
            check_row(row, ref.step(), i)
        print(f"[metal {name} t={threads}] {len(rows)} steps lockstep-exact")

    gpu_heat()
    m.boot_image(image, start_pc=START_PC)
    t0 = time.perf_counter()
    res = m.run(n)
    dt = time.perf_counter() - t0
    steps = int(res["steps"])
    print(
        f"[metal {name} t={threads}] {steps} steps in {dt:.3f}s = "
        f"{steps / dt:,.0f} steps/s ({dt / steps * 1e6:.1f} us/step); "
        f"{m.params['n_seq']} windows packed, "
        f"{m.params['shape_matches']} shape matches, "
        f"{m.params['trace_packed']} trace windows, "
        f"{m.params['skipped_seq']} rejected"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--bench", type=int, default=0)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--suite", choices=("micro", "nestest", "klaus"),
                   default="micro")
    p.add_argument("--check", action="store_true")
    p.add_argument("--pgo-windows", type=int, default=0,
                   help="profile this suite and pin up to N hot prefixes")
    p.add_argument("--pgo-min-hits", type=int, default=512,
                   help="minimum profile executions for a pinned prefix")
    p.add_argument("--trace-len", type=int, default=10,
                   help="max instructions per proposed trace window")
    p.add_argument("--spec-min", type=float, default=2.0,
                   help="min profiled fall-through rate to speculate past "
                        "a branch (default > 1: branch-final windows; "
                        "measured better on Klaus)")
    p.add_argument("--null-model", action="store_true")
    p.add_argument("--experts", default="all",
                   help="all | none | comma-separated opcode hex bytes")
    p.add_argument("--no-windows", action="store_true")
    p.add_argument("--no-shapes", action="store_true")
    args = p.parse_args()
    experts = parse_experts(args.experts)
    if args.verify:
        verify(args.threads, experts)
    if args.bench:
        if args.suite == "micro":
            bench(args.bench, args.threads, args.null_model, experts,
                  windows=not args.no_windows)
        else:
            bench_suite(args.suite, args.bench, args.threads, args.check,
                        experts, windows=not args.no_windows,
                        pgo_windows=args.pgo_windows,
                        pgo_min_hits=args.pgo_min_hits,
                        shapes=not args.no_shapes,
                        trace_len=args.trace_len,
                        spec_min=args.spec_min)


if __name__ == "__main__":
    main()
