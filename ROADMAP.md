# Roadmap

Status snapshot and prioritized next steps.  Written 2026-07-06 as the
handoff point for fresh working sessions; the design details live in
[`transformer_nes/cpu6502/README.md`](transformer_nes/cpu6502/README.md);
the architecture-agnostic optimization catalog (for reuse on future
targets) lives in [`PLAYBOOK.md`](PLAYBOOK.md).

## 2026-07-19 metal v14 — lowering rung 2: every routed tier scalar,
570.7K Klaus (1.8 µs) / 275.7K nestest (3.6 µs), 100% coverage

Rung 1's capture generalizes to every pack_model call, so `--lower N`
now lowers trace windows (hottest-first), sequence windows, canonical
shapes, and opcode experts; only the union stays interpreter-only.
Both suites route 100.0% of forwards through scalar bodies — nestest
has zero trace windows, and this is the first rung that reaches its
shape/expert mix at all.

- Klaus 1M, t=128: **570,703 steps/s (1.8 µs/instruction)**, full
  trace lockstep-exact, 298 models lowered (120,879 ops).  Ladder:
  t=32 546K, t=64 563K, t=128 optimal.  1.81x v15, 1.20x rung 1;
  real-time NES (~592K) is 4% away.
- Nestest 8,991 steps, t=128: **275,666 steps/s (3.6 µs)**, lockstep-
  exact, 336 models lowered — **1.91x v15's 144.5K** with zero new
  composition: the same 1.68 instr/fwd, each forward just costs less.
- Replayed layer stack (paired probe, t=128): 456 ns/step, vs 866
  (rung 1) and 1875 (interpreter) — a 75.7% total cut.  The adjacent
  paired base measured 574.5K.
- Verify suite runs lockstep-exact with 257 lowered models live
  (`--verify --lower 512`); whole build ~15 s including Metal compile
  of ~173K-op generated source; suite 86 green; ruff clean.

The cost anatomy has flipped: with stacks at 456 ns/step, the
per-forward FIXED floor (dispatch scan, TG_D zeroing + embed wave,
readout gather, commit) is ~5.3 µs of every 7.2 µs forward.  Rung 3 is
whole-forward lowering: a lowered route needs no threadgroup residual
at all — lane 0 can read stg[] directly, skip the zero/embed wave,
compute the readout and head sums scalar, and write outk straight from
expressions.  Liveness already knows the embed slots; non-embed leaves
are exact zeros.  That collapses the remaining barriers per forward to
~1 and attacks both suites equally (nestest's 6.1 µs/fwd is nearly all
floor).  After that: predicated regions (assessment item 2) to raise
nestest's 1.68 instr/fwd.

## 2026-07-19 metal v13b — structural lowering rung 1: window stacks as
scalar Metal, 474K Klaus (2.1 µs), gate passed at 53.8%

Item 3 of the post-v15 assessment, in its trivial-recognizer form: a
v14+ trace-window model has no register rows and only fused/constant
reads, so its ENTIRE layer stack is already an exact scalar program —
every residual slot a dyadic-linear expression, every FFN unit a ReGLU
of two, every read a guarded ram[] load.  scalargen.py emits that
program as straight-line expressions (literal weight evaluation,
identical numbers; dyadic exactness licenses the reordering) executed
by lane 0 behind the same uniform route switch, with liveness pruned
backward from the readout and head slots.  No stream walks, no
per-layer barriers, no tables — the two costs the tile experiment
proved dominant simply do not exist on the lowered path.
`--lower-windows N`; interpreter remains the fallback and serves the
6-8 attribution probes.  The row-tile backend was removed the same day
(dead surface); its paired-probe harness and the io[6] coverage counter
stay.

Measured on 1M Klaus, 49 windows, 83.1% forward coverage:

- Replayed layer stack (probe-5 adjacent pairs, t=256): 1875 → 866
  ns/step — a 53.8% cut, PASSING the pre-registered ≥40% gate.
- End-to-end: 341K → 469K steps/s adjacent at t=256; thread ladder
  moved DOWN as predicted for serial bodies: **t=128 474,067 steps/s
  (2.1 µs/instruction)**, t=64 465K, t=256 ~464-469K.  1.50x v15's
  315.7K; real-time NES (~592K) is now 1.25x away.
- Lowered bodies: 29,026 ops / 11,790 temps / 4,018 stores across 49
  models (~600 ops per lowered forward, ~590 avg terms — liveness
  keeps roughly everything because a 22·k readout consumes the width).

CORRECTNESS: full 1M Klaus lockstep-exact with all 49 windows lowered
(t=256), and the deeper dev-config windows (median 5 layers, t=32)
lockstep-exact; `--verify` green; ruff clean.

Next rungs, in order: (1) lower shape models and opcode experts the
same way — nestest routes EVERYTHING through those tiers (zero trace
windows) and still pays 6.9 µs; (2) whole-forward lowering — with the
stack scalar, the ~1.6 µs dispatch/embed/readout floor around it is
the next largest term; (3) GPU-aware scheduling (item 4) matters only
for what stays on the interpreter.

## 2026-07-19 metal v13 — compiled row tiles: NEGATIVE result, the
sparse walk is a latency/barrier floor, not interpreter overhead

Experiment 1 from the post-v15 assessment: generate straight-line Metal
for the hottest trace-window models (`--compile-tiles N`, tilegen.py) —
homogeneous unrolled rounds over constant-memory term tables, uniform
coefficients folded to immediates, no seg loads / packed decode / tail
branches, dispatched by a uniform route switch with the interpreter as
fallback.  Hard gate pre-registered: ≥40% reduction in replayed
layer-stack time (probe-5 paired delta; `--replay-probe` now runs its
own adjacent base for the thermal rule, and the kernel counts
tile-routed forwards in io[6]).

Measured on 1M Klaus, 49 PGO windows:

- t=256, top-4 tiles: 1.9% forward coverage — the window hit
  distribution is FLAT (hottest head ~4.7K of 244K forwards); a
  "hottest four" prototype cannot move the total.
- t=256, all 49 tiles: 83.1% coverage, but base REGRESSED 342K → 291K
  steps/s.  The [round][lane] tables are 96% padding at t=256 (4.2 MB
  vs the interpreter's 225 KB packed stream) and every cold forward
  pays the constant-cache misses; the warm replay hides them.
- t=128, all 49 tiles (fair pair): replayed layer stack 1874 → 1738
  ns/step = 7.3% warm reduction; base 340K → 334K (net loss).

Verdict: gate FAILED by 5x on the favorable metric.  Eliminating the
interpreter machinery (decode, seg bounds, loops, tails) is worth ~7%
because at t=128-256 each lane walks only 1-2 terms per phase: the
phase cost is one unhidden device-load latency plus the barrier wave —
a floor ANY term-table executor pays.  The 1.46 µs sparse-walk share is
memory latency and barrier structure, not instruction overhead, so
per-term codegen variants (packing, encodings, arity buckets) are a
dead end at these widths.  Per the pre-registered decision rule the
next lever is STRUCTURAL CIRCUIT LOWERING (recognize wrap/compare/mux/
flag-assembly subgraphs, execute fused integer ops, prove slot-by-slot
equivalence), which removes rounds and barriers instead of decorating
them.  Fewer, wider layers (GPU-aware scheduling) attacks the same
floor.

CORRECTNESS: full 1M Klaus lockstep-exact WITH 49 tiles live at t=256;
`--verify` green; default build identical (tiles are opt-in); ruff
clean.  The tile backend was removed the same day once the verdict was
in (see v13b above); the paired probe and io[6] coverage counter stay
as measurement infrastructure.

## 2026-07-18 construction v15 — flag-circuit flattening + constant-side
multiply fold, 315.7K Klaus (3.2 µs) / 144.5K nestest

Same session as v14; the census (now 5/7/4 after v14) named the last
serial stages and three exact rewrites removed them:

- **Constant-side multiply fold** (core.make_multiply tags its persist
  with `multiply_of`; fold.py partial_eval, structural folds only).
  `make_multiply(a, b)` is `a·relu(b) − a·relu(−b)`, and partial_eval
  only had a rule for constant b — so every `multiply(gate, data)` with
  the GATE in the a-position (all of v14's gated bit selects, plus the
  interpreter's `multiply(op_adc, carry)`-style flag updates and
  data_ref sums) survived as a two-ReGLU identity: one wasted layer and
  width at every site.  The tagged fold collapses the product to an
  exact linear map when EITHER side pins.  Census 7/8/7 → 5/7/4.
- **Flat N/Z/V flag circuits** (interpreter flat_nz / overflow_wide,
  wide_bits builds only).  N and Z of `wrap8(raw)` are parallel step
  ladders on the PRE-wrap difference (N = [raw≥128] + [−128≤raw<0]),
  so CMP/CPX/CPY/DCP flags land one stage after their operands instead
  of serially behind wrap8.  ADC/SBC overflow computes in ONE stage
  from the served sign bits: V = [s<−128] + [s>127] on the signed sum
  s = raw − 256·(a7 + b7) — the old sign-of-wrapped-result chain was
  the 4-serial-reglu tail of every arithmetic window.
- **Pruned bit-state feedback** (fold_sequence bit_feedback_until;
  builders compute last_a_bit_step from the window's opcodes).  Feedback
  outputs were seeded live at every step, which kept v14's arithmetic
  bit ladder alive even when no later step consumed A's bits.  Seeding
  stops after the last consuming opcode; wrong pruning fails loudly at
  build (tr KeyError), never silently.

Census 12/15/12 (v13) → **4/5/4**: a window's critical path is now
read (lookup + value persist) → one ALU stage → one flag stage → P
assembly — the structural floor for serial dependent reads.

**RESULT:** heated 1M Klaus at t=256: **315,681 steps/s
(3.2 µs/instruction)** — 3.39x v13's 93.0K in ONE session, and above
v10's 311K DEX/BNE MICROBENCHMARK tier on the real representative
suite.  t=128 measures flat (314.0K adjacent); t=256 stays the
default.  Nestest **144,520 (6.9 µs)** at t=128 (t=32 flat at 144.2K),
+75% on the session.  CORRECTNESS: FULL 1M Klaus trace lockstep-exact
(49 trace windows + flag pins live); full 8,991-step nestest
lockstep-exact; both suites exact at t=32 AND t=128/256; suite 85
green; ruff clean.

Nestest is now the laggard tier (6.9 vs 3.2 µs): it has zero trace
windows (its hottest pc runs 10 times — PGO correctly proposes
nothing), so it retires 1-3 instructions per forward through shapes and
opcode experts and pays the per-forward fixed cost.  The next
representative levers are therefore (a) the per-forward floor itself
(dispatch/embed/readout, ~1.6 µs of every forward) and (b) longer
certified composition for shape-tier workloads.  Real-time NES
(~592K instr/s) is now less than 2x away on Klaus-shaped code.

## 2026-07-18 construction v14 — structural serving,
188.7K Klaus / 138.6K nestest (2.0x in one session)

The v13 census said the remaining path was "the lookup fabric itself";
this rung noticed the fabric is REDUNDANT on the GPU: the Metal kernel
executes every fused read structurally (`h0 = wr[addr] ? ram[addr] : 0`)
and embeds the current state directly, so graph circuits that re-derive
what the kernel already serves exactly are pure critical-path drag.
Four rungs landed, all behind fold/build flags the metal packer opts
into (`structural_reads`, plus the existing GPU-only `wide_bits`); the
C++ tier and every default build are byte-identical (suite 85 green).

- **Structural reads** (fold.py masked_exact_read(structural=True)).
  The exact-address mask (2 layers) is dropped — the kernel's fused
  read already returns the exact masked value — and the lookup carries
  EMPTY key rows, so the scheduler stops serializing every read behind
  the key-square coordinate (`make_multiply(k, k)`, 2 more layers) that
  no structural backend computes.  A read now lands in the residual at
  its lookup's own layer.  Census 12/15/12 → 8/11/8; Klaus 93.0 → 107.8K.
- **Flag-pinned trace windows** (PIN_FLAGS, seqtab p_guard, 12th row
  int).  Housekeeping P bits (I/D/B/U) whose profiled entering value is
  constant at a window head pin at fold time; dispatch guards
  (P & mask) == values, mirroring the S pin.  A D=0 pin deletes the
  decimal ADC/SBC circuit and its operand bit extraction from the whole
  window (35B7: 11 → 9).  Width regressions from pin-shifted inlining
  fall back to the unpinned build per head.
- **State-fetch bypass** (LookUp.state_fetch tag).  The latest-STATE
  fetch is an identity on the kernel (the embed IS the latest state),
  but its key-square coordinate held every step-0 consumer 3 layers
  deep.  Structural folds substitute the fetch's values with the raw
  state inputs; expert models now contain ZERO register rows.  Census
  8/8/8; Klaus ~124K.
- **Served bits** (wide_bits interpreter + kernel embed/read contract).
  Bits of any embedded or loaded byte are pure scalar work for the
  kernel, so they cost zero graph depth at the source: STATE embeds A's
  bits (31 embed values), fused reads serve their byte's bits as extra
  value outputs (WRITE-token wbit dims; mem_meta gains a value count and
  the kernel writes the bit ho slots), and A's bits flow as state
  (new_a_bit* fed back like new_a; gated sums pick each writer's natural
  bit list, arithmetic shares one ladder on new_a that liveness prunes).
  PLP's 255-step ladder and the operand/pull nibble ladders leave every
  structural fold; bit-annotated ladder persists stay materialized so
  the annotation survives inline_linear, and pinned operand bytes still
  const-fold their bits without keeping the raw read alive.  Census
  7/8/7 — critical paths now START at a served bit value.

**RESULT:** heated 1M Klaus at t=256: **188,738 steps/s
(5.3 µs/instruction)** — 2.03x v13's 93.0K record in one session;
nestest **138,614 (7.2 µs)**, +68% over 82.7K.  The thread optimum
moved again: Klaus t=256 (t=128 96.8→87.5→102.3 bracketed by t=256 hot;
t=512 regresses), nestest stays t=128.  CORRECTNESS: the FULL 1M-step
Klaus trace lockstep-exact with all 51 trace windows + flag pins live;
full 8,991-step nestest lockstep-exact; both suites exact at t=32 AND
t=128/256; suite 85 green; ruff clean.  Same-session thermal note
confirmed again: back-to-back 1M runs throttle ~15%; only adjacent runs
compare.

What remains on the path per the census: the read→ALU→flags→branch
chains themselves (wrap8/carry stepglus and the branch mux) — genuine
instruction serialization, ~5 reglu stages per window — plus the
non-layer floor.  The next depth lever is likely cross-instruction
carry/flag composition inside the fold, or serving more derived scalars
(nibbles, carry chains) from the kernel the way bits are served now.

## 2026-07-18 construction v13 — critical-path circuit diet,
93.0K Klaus / 82.7K nestest

The v12 decomposition said the layer stack is ~83% of GPU step time, so
this rung attacked the folded circuits themselves.  A critical-path
census over three representative trace-window folds (walk the folded
graph, ASAP depth per dim, witness longest chain) showed one consistent
shape: ~10 layers of masked-read fabric, then 8 layers of serial
extract_bits, then ~4 layers of actual instruction logic.  Two exact
rewrites landed:

- **Depth-lean exact-address mask** (fold.py masked_exact_read, shared
  by every expert/window rebuild on BOTH backends).  For integer diff,
  [diff > 0] is relu(diff) - relu(diff - 1), so the exact-hit indicator
  is a LINEAR combination of four ReGLUs with no intermediate persists,
  and the masked value is one gated ReGLU (the mask is 0/1, so
  reglu(value, exact) is the product exactly).  Three layers past the
  lookup instead of seven, and fewer units than the old
  abs-square/step/multiply chain — the C++ tier wins on both axes.
  Window graph depths 22/25/22 → 18/21/18; Klaus 86.7K → 91.0K
  (adjacent A/B).
- **Nibble-hybrid bit extraction** (interpreter extract_bits_wide,
  behind a `wide_bits` build flag the GPU union opts into; the C++
  engine keeps the 8-unit serial form because its cost is unit count).
  Split on the high nibble via a persist-free 15-step relu-difference
  ladder, then two 15-step nibble ladders: bits land ~4 layers after
  the value instead of 8, at ~60 ReGLUs.  The nibble values are wrapped
  in unit-gated ReGLUs so linear inlining cannot smear their 30-term
  ladder sums into every consumer row.  Depths fall to 12/15/12.
  Measured against the alternatives on Klaus 1M, adjacent runs:
  serial 91.0K, nibble-hybrid **93.0K**, and the full 255-step ladder
  (depth 11/14/11 but ~500 ReGLUs per live site) **66.0K** — a hard
  rejection that sharpens the cost model: past ~4-8 rows/lane the walk
  length dominates, so depth trades are only profitable while the added
  width stays under roughly a hundred units per site.

**RESULT:** three heated 1M Klaus runs at t=128: **93,029 / 93,019 /
93,020, mean 93,023 (10.8 µs/instruction)** — +10.3% over v12's 84.3K,
**2.05x over v11's 45.3K record from the same morning**.  Nestest moves
to **82,848 / 82,710 / 82,608, mean 82,722 (12.1 µs)** at t=128, +8.0%
over its long-standing 76.6-76.9K plateau (the shallower bitwise
experts pay there too; t=32 gives 82.1K, so the thin-model/one-simdgroup
era is over on both suites).  CORRECTNESS: verify + bench +
cache-invalidation programs lockstep-exact at t=32 AND t=128; full
nestest and the FULL 1M Klaus trace lockstep-exact (the wide-bits union
is exercised end-to-end by both suites); suite 85 green (the mask
rewrite is also covered by the C++ expert tests); ruff clean.

The census probe (scratchpad probe_critpath.py) is the reusable method:
witness chains name the circuit family to attack next.  What remains on
the path is the lookup fabric itself plus genuine instruction
serialization (ADC→flags→CMP chains), and the ~17% non-layer floor —
diminishing structural room at this tier without a different execution
mapping.

## 2026-07-18 megakernel v12 — store-to-load forwarding + PGO trace
windows, 84.3K Klaus (+86%)

The v11 conclusion ("broader certified cross-step composition") landed as
store-to-load forwarding in the fold plus flow-following PGO windows.  A
weights-engine profile (iter_run_arrays; the Python reference is for
lockstep checks only) showed Klaus's real structure: **94.5% of 1M steps
sit in 213 pcs** — one ~160-instruction straight-line check-macro block
plus its caller loop and two JSR trampolines — and the block is dense
with exactly what windows could not cross: PHP/PLA/PLP stack idioms,
reads-after-writes, JSR/RTS, and a self-modifying `STA $0212` that
rewrites a trampoline's ADC immediate before calling it.  Neither suite
has any hot self-closing loop, so v10-style guarded repetition has zero
representative coverage; composition had to come from forwarding.

- **fold_sequence(forward_stores=True)**: every step's store roots must
  fold to constant flags/addresses; a later read at a matching constant
  address takes the written value instead of a lookup.  Constant values
  forward through partial_eval itself, so a JSR/RTS round trip folds the
  return address into a constant next_pc (the license needs it);
  live values forward as the already-materialized write-value root
  persist.  Reads at distinct constant addresses keep the global lookup;
  non-constant reads with pending writes are admitted only when
  caller-certified mode bounds (zp,X and indirect pointers wrap in page
  zero, pulls stay in page one, abs,X adds ≤255 to a pinned base) prove
  disjointness.  Everything else raises a step-tagged SequenceRejection.
- **Trace windows** (build_trace_window_expert + propose_trace_window):
  K profiled instructions along the executed control flow, JSR/RTS/JMP
  crossed with non-contiguous signed-offset byte checks.  Pinning the
  profiled entering S (unique at every hot head; dispatch guards
  S == pin, SEQ_GUARD_S) folds every stack address to a constant.
  Profile-written operand bytes stay symbolic and read through the live
  memory structure — the self-modifying trampoline folds through
  forwarding exactly.  seqtab rows grew 8 → 11 ints: span_lo/span_len
  bound cache invalidation for non-contiguous code, plus a speculation
  table offset.  PGO builds experts inside the proposal loop (a rejected
  head must not shadow proposals at its interior pcs) and retries with
  the window truncated before the failing step, so an aliasing read
  splits a segment into two licensed windows instead of killing it.
- **Width policy, measured the hard way.**  The first full pack REGRESSED
  Klaus to 27.7K: a mid-window symbolic PLP fans its extracted flags
  into every downstream consumer under inline_linear, blowing one layer
  to 10,570 nnz (41,447 total, ~25x its peers) — ~10x the cost of the
  whole rest of the forward.  Policy now: full inline, fall back to the
  materialized fold when the worst layer exceeds 1,200 nnz, reject past
  2,600.  The PLP windows land at 29 layers / 2,076 worst.  A budgeted
  middle rung (inline only persists ≤16 terms: 18 layers at 2,350 worst)
  measured SLOWER than deep/narrow at t=128 and was removed.
- **Speculated interior branches: built, exact, default-off.**  The
  block's never-taken `BNE $fe` traps invited crossing: interior next_pc
  that folds non-constant is licensed for branches, the kernel verifies
  each speculated step's next_pc readout against the trace and commits
  exactly the prefix before the first violation (rollback retires what a
  branch-final window would have).  Full 1M lockstep-exact — and
  consistently ~12-15% SLOWER than branch-final windows at every window
  length (73-78K vs 86.5K): rollbacks waste deep forwards on ~50/50
  branches, and trap-crossing de-aligns window boundaries and pushes PLP
  decodes mid-window into materialized builds.  Kept behind
  --spec-min (default disables); the machinery is the right primitive
  for workloads with genuinely-never-taken guards.
- **Threads re-laddered.**  Trace-window Klaus: 73.5K / 80.7K / 86.4K /
  79.6K at t=32/64/128/256 — the wide window models moved the optimum to
  FOUR simdgroups, retiring v7's "one simdgroup is optimal" for this
  tier.  Nestest (thin shapes, no trace windows) stays best at t=32.

**RESULT:** three heated 1M Klaus runs at t=128 with 51 trace windows
(~97% of steps retired by windows): **84,390 / 84,443 / 84,202
steps/s, mean 84,345 (11.9 µs/instruction) — +86% over v11's 45,312.**
Session ladder: 42.9K shapes-only → 27.7K naive pack → 51.3K width
policy → 57.4K read bounds → 61.9K deeper PGO cap → 73.5K truncate-retry
(t=32) → 84-86K at t=128.  Nestest is unchanged at **76,415 / 76,689 /
76,606, mean 76,570** (its hottest pc runs 10 times; PGO correctly
proposes nothing).  CORRECTNESS: verify + bench + cache-invalidation
programs lockstep-exact at t=32 AND t=128; full 8,991-step nestest and
the FULL 1M-step Klaus trace lockstep-exact with all 51 windows live;
suite 85 green; ruff clean.

**Same-day follow-up — post-commit micro-rungs, 86.8K Klaus.**  Three
adjacent A/B experiments after the v12 commit: (1) rejecting the two
wide PLP windows outright (width cap 2,600 → 1,500) so their interiors
re-propose as thin windows regressed 86.3K → 64.6K — the wide windows
earn their keep, REVERTED; (2) the lane-0 readout's per-slot RO loads
were the last serial device chains per forward — gathering
rint(x[RO[i]]) across all lanes into a threadgroup staging array before
lane 0's ordered commit gained +1.8%, KEPT; (3) 4-wide batching of the
dispatch byte-check loads measured flat (+0.1%) and was reverted.
Receipts after the kept change: verify + bench + invalidation programs
exact at t=32 AND t=128, nestest 8,991 and Klaus 200K lockstep-exact;
three heated 1M Klaus runs **86,831 / 86,777 / 86,801, mean 86,803
(11.5 µs/instruction, +92% over v11)**; nestest 76,311 / 75,621 /
75,975 (mean 75,969, within noise); ruff clean.

**Decomposition receipt (same day, replay probe — added, measured,
removed).**  A temporary kernel mode replayed each committed step's
routed layer stack with results discarded (x is re-embedded every
dispatch; register updates suppressed), so the adjacent A/B delta
isolates the layer stack on the REAL Klaus trajectory: normal 11.54s vs
replay 21.14s over 1M steps.  **The layer stack is ~9.6 µs of the
11.5 µs step — ~83% (warm-biased, so a floor); dispatch + verification
+ embed + readout + writes are ~17% combined.**  This settles the
direction: kernel fixed costs cap at +20% even if zeroed, so the next
representative gain must come from shallower/cheaper folded circuits —
the check macro's serial ADC→flags→CMP chains — on the construction
side.  Post-probe restore verified: verify programs exact, re-bench
86,727 (matches the 86.6-86.8K receipts).

Two operational notes for future sessions.  Sustained back-to-back
benching thermally throttles the SoC: identical configs read 86.4K cold
and 75.6K hot, so only adjacent runs compare — the receipts above were
run consecutively on a rested machine.  And the remaining rejected
families are principled, not accidental: indirect (zp,X)/(zp),Y data
reads are unbounded before dispatch, and abs,X reads whose pinned base
overlaps the stack page genuinely may alias in-window pushes.  The
next depth lever is the window models themselves (~3-4 layers per
retired instruction; the macro's serial ADC→flags→CMP chains bound it),
not coverage — and 84K is still ~7x from realtime NES.

## 2026-07-18 megakernel v11 — symbolic shape routing,
76.9K nestest / 45.3K Klaus PGO

The next representative gain comes from circuit depth, not another table
tweak. Canonical shape experts from the emitted C++ backend now run in the
Metal megakernel: one fold keeps PC and operand bytes symbolic, compiles one
model, and shares it across every matching code address. Metal's
`inline_linear` construction was extended to shape folds while preserving
the symbolic PC bytes needed by the exhaustive interior-control-flow
license. This changes the original hand-written shape library from mostly
unprofitable (for example DEX/BNE 11 layers versus 10 cumulative opcode
layers) to consistently profitable (8 versus 10).

Representative profiles selected common opcode shapes rather than hot
addresses. The main coverage is compare/branch pairs at hundreds of
nestest PCs, `CMP zp/BNE` at 60 Klaus PCs, and six straight DEX operations
per forward. Longer setup/store and NOP shapes fill nestest's repeated
conformance scaffolding. Every image head is indexed through the existing
O(1) sequence-head table. Dispatch descriptors now hold compact
`(offset, expected opcode)` checks: concrete windows verify every byte,
while symbolic shapes verify only opcodes and read operands live.

Store shapes gained two explicit licenses:

- statically bounded RAM/stack stores carry their certified address interval,
  and the packer admits only heads disjoint from the entire interval;
- one direct absolute store may carry a live operand-offset guard. The
  builder proves it is the only write root and every step is non-halting;
  dispatch reconstructs the target from resident bytes and requires
  `target < $2000` and no overlap with the shape's code.

Both licenses preserve the existing default CPU policy—stores remain
rejected unless a router explicitly opts into and enforces the proof.
Full nestest and a 1M-step Klaus trace exercise the routed models
lockstep-exactly.

**RESULT:** three heated full nestest runs with canonical shapes and no
concrete windows are **76,886 / 76,667 / 77,162 steps/s**, mean **76,905
(13.0 µs/instruction)**. That is **+10.3%** over v10's packed-stream
69,755 baseline and +12.6% over the original 68.3K representative
correction. Three 1M-step Klaus shape-only runs are **43,058 / 43,090 /
43,107**, mean **43,085 (23.2 µs/instruction)**, +5.6% over the packed
40,815 baseline. Composing the existing eight operand-pinned PGO windows
with the symbolic shapes gives **45,312 / 45,319 / 45,306**, mean
**45,312 (22.1 µs/instruction)**: **+11.0%** over packed v10 and +12.9%
over the first 40.1K Klaus baseline. The current 200K combined route is
lockstep-exact.

Measured limits remain important. Expanding the profitable catalog past
the concentrated families has sharply diminishing returns. A 16-DEX shape
reduced forwards but regressed Klaus; its extra model width/depth outweighed
dispatch savings. Threads 16 and 64 regress full nestest to 67.7K and
75.0K, so one 32-lane simdgroup remains optimal. Static concrete-loop
scanning stays disabled for representative defaults: it had no suite
coverage win; concrete models enter only through the measured 512-hit PGO
policy.

This is a real cross-suite construction gain, but **45-77K is still nowhere
near realtime NES throughput**. The remaining route is broader certified
cross-step composition or a different execution mapping—not more sparse
metadata compression.

## 2026-07-18 megakernel v10 — guarded countdown superwindows,
311K microbenchmark; representative suites 40.8-69.8K

The next circuit-depth lever landed as guarded cross-step overlap.  A closed
`DEX/BNE` or `DEY/BNE` window now gets 4-iteration and 2-iteration variants
in addition to its original one-iteration expert.  For an R-iteration
variant, construction pins only the R-1 interior BNE decisions to taken; the
final BNE remains live, so one expert handles both loop continuation and the
exit exactly.  Dispatch proves the assumption from the resident byte state:
the entering counter must be zero (wraparound) or at least R.  It still
re-verifies the window bytes, NMI and step-budget conditions exactly as
before, and falls through 4x -> 2x -> 1x near the exit.

This turns four independent executions of the old 2-instruction / 4-layer
window into one 8-instruction / 7-layer forward.  The 2x cleanup expert is
4 instructions / 5 layers.  A temporary route-count probe caught one
important cache interaction: after a loop exit, the positive cache retained
the unguarded 1x fallback across the surrounding `JMP/LDX` and prevented the
guarded variants from being reconsidered on re-entry.  The cache now expires
as soon as execution leaves its cached head.  That keeps the hot in-loop hit
while restoring priority on every re-entry.

**RESULT:** three heated 100K-step runs are **311,449 / 311,850 / 311,454
steps/s = 3.207-3.211 µs/instruction**, versus the exact v9 construction's
**210,885 / 211,145 / 210,880 = 4.736-4.742 µs/instruction**: a stable
**1.48x** gain.  The repeat ladder mapped the width/depth crossover:
2x ~282K, 3x ~305K, **4x ~311K**, 5x ~293K, 8x ~266K, and 16x ~190K.
More speculative depth eventually loses to the wider streams, so production
keeps only 4x and 2x.

Correctness receipts: verify program, 3,000-step full-byte countdown loop,
and 24-step self-modifying cache invalidation are lockstep-exact at t=32 and
t=128; full suite **82 passed**; ruff clean.  This is deliberately a narrow,
certified first instance of cross-step overlap.  The next extension should
look for other loop idioms with a cheap resident-state guard rather than
blindly increasing repeat.

**Representative correction (same day, after explicit user challenge):**
the headline above is a DEX/BNE tier, not a general megakernel speed.  Full
8,991-step nestest is lockstep-exact but runs only **~68.3K steps/s
(14.6 µs/instruction)** with all opcode experts and no windows.  Its six
licensed static windows slightly regress to ~67.7K; all DEX/DEY countdown
loops in that ROM write `$2007` and are correctly rejected as NES MMIO, so
v10 has zero nestest coverage.  Nestest deliberately executes 225/227
opcodes: packing exactly those 225 is flat, top-64 falls to 28.5K, and
top-16 to 7.3K because 75-layer union fallbacks dominate.

Klaus required a newly exposed BCD-enabled Metal construction.  A 200K
prefix is lockstep-exact; three heated 1M-step no-window runs are
**40,139 / 40,087 / 40,142 steps/s (~24.9 µs/instruction)**.  Klaus's hot
loops use compare/BCS shapes, not v10's decrement/BNE guard, so v10 has zero
coverage there too.  The full 30.6M timing was stopped after 16M at the
user's request once its cumulative rate had converged to ~39.1K/s.

Mining the emitted C++ backend did produce one representative win:

- Metal's sequence router linearly scanned every descriptor on every miss;
  the C++ sequence module instead uses a 64K pc-to-range index.  Porting that
  O(1) head index recovered most of nestest's static-window penalty and makes
  larger window sets viable.
- The C++ runner's next-boot PGO policy transfers when its **512-hit**
  profitability threshold is preserved.  Eight concrete Klaus prefixes,
  each seen ~4,656 times in the 1M profile, are 200K-prefix lockstep-exact
  and improve three 1M runs to **42,324 / 42,340 / 42,326 steps/s**:
  **+5.5%** over no windows.  The same policy proposes zero nestest windows
  (its hottest eligible pc appears only 10 times), correctly avoiding
  pollution.  `metal_runner.py --suite {nestest,klaus} --check` now makes
  the representative receipts routine; `--pgo-windows N` performs the
  two-pass, next-boot-style construction.

The next C++-construction pass found a second, workload-independent win.
The all-expert Metal pack contains 30,510 attention, 36,963 FFN-output, and
88,372 gate terms.  Their largest destination/source indices are only
1,019/1,111, so carrying two 32-bit indices per coefficient was pure traffic.
Packing `(dst, col)` as two certified 16-bit fields cuts every sparse term
from 12 to 8 bytes without adding a dependent load or per-layer decision.
Three full no-window nestest runs are now **69,843 / 69,759 / 69,662
steps/s (mean 69,755, +2.1%)**, with the checked 8,991-step run lockstep
exact.  Three 1M-step Klaus runs are **40,791 / 40,823 / 40,830 steps/s
(mean 40,815, +1.7%)**, with one full 1M trace lockstep exact.  The existing
eight-window Klaus PGO path composes with the layout win at **43,045
steps/s**; its 200K checked prefix is exact.

Rejected constructions sharpen the GPU cost model.  Compact 16-bit segment
bounds were neutral/slightly slower and were removed.  C++-style row
accumulation regressed nestest to 65.0K because dynamic row transitions cost
more than repeated threadgroup updates at only 1.3 terms/row.  Specializing
26,497 pure `x += -x` rows into zero assignments required per-layer erase
metadata/control and regressed to 63.8K even with O(1) packed offsets.
Finally, folding the 624 distinct coefficients into each 32-bit term halved
the stream again but its dependent coefficient-table load reduced nestest
to ~69.0K.  The retained lesson is precise: reduce independent stream loads;
do not add dependent loads or per-layer control to save static footprint.

The standing conclusion is therefore not “311K viable NES.”  It is:
guarded overlap works when it has coverage, while representative performance
is currently **40.8-69.8K instructions/s**.  The next GPU construction work
should extend high-value guarded shapes from real profiles and keep the
C++ backend's profitability/isolation rules, rather than adding generic
windows to the global table.

## 2026-07-18 megakernel v9 — table-free register rows + cached windows,
208K steps/s

Construction v8 changed the four-layer sequence path enough to justify a
fresh kernel decomposition and one more measured ladder.  Two changes landed:

- **Table-free affine register lookups.**  A full all-model census proved
  that every one of the 2,749 register rows has exactly two score terms with
  coefficients `[2, -1]`, one exact-zero gate coordinate, and zero/one
  coefficient-1 term for each returned value.  More strongly, all 230 layers
  containing register rows are affine runs: one shared score/gate triple per
  layer, consecutive value pairs, consecutive union-register IDs, consecutive
  `ho` pairs, and (for every 12-row run) only the final second value absent.
  The packer now certifies those invariants loudly and stores four bases/counts
  in the layer descriptor.  The kernel's five CSR chains + metadata table,
  then its intermediate fixed-width row table, are gone entirely: a register
  row performs **zero device-table loads**.  The temporary decomposition's
  lookup component fell from ~1.52 to ~1.09 µs/instr after the first compact
  rung; the final affine form is smaller again.
- **Positive sequence-dispatch cache.**  Once a window's pc/bytes are
  verified, the persistent kernel retains its head/K/descriptor/length in
  lane-0 scalars and reuses them while pc matches and the remaining step
  budget covers K.  Any GPU RAM write overlapping the cached byte range
  invalidates it immediately; a new verification receipt warms a DEX/BNE
  cache, overwrites DEX with NOP through `STA`, jumps back, and remains
  lockstep-exact.  Cache lifetime is one dispatch, so host boot or external
  mutation cannot bypass verification.

**RESULT:** heated DEX/BNE is **4.8 µs/instr = 209,602 steps/s**, versus
construction v8's 5.4 µs / 185,309 (**1.13x**).  No-windows is
**10.0 µs = 100,344 steps/s**, versus 10.6 µs / 93,966 (**1.07x**).
The combined construction-v8 + kernel-v9 jump over v7 is
123,969 → 209,602 (**1.69x**) windowed and ~66K → 100K (**1.52x**)
per-instruction.

Measured rejections stayed out: simdgroup-parallel readout was
lockstep-exact but regressed 5.4→5.6 µs (the extra rendezvous costs more than
44 lane-0 threadgroup loads); caching the routed `(lay0, NL, D)` descriptor
in per-lane registers was flat at 200.4K vs 200.6K; spelling each existing
12-byte triple as a Metal `packed_int3` load regressed 209.6K→204.8K.
Temporary walks-off probes perturbed compiler scheduling (removing walks made
the kernel slower) and were removed after isolating the useful lookup delta.

Receipts: verify program, 3,000-step loop, and the 24-step self-modifying
cache-invalidation program lockstep-exact at t=32 AND t=128; full suite
**82 passed**; ruff clean.  Standing tiers: DEX/BNE window 210K,
per-instruction 100K, prior JMP-spin baseline 225K (not remeasured in this
rung).  The four-layer window's remaining dependency chain is genuinely
nonlinear (state lookup followed by three chained ReGLU layers); the next
large lever is construction/circuit depth or cross-step overlap, not another
metadata cache.

## 2026-07-18 construction v8 — linear persist inlining, 185K steps/s

The first construction-side depth diet landed immediately after the v7
megakernel floor was mapped.  Expert folds now have an opt-in
`inline_linear` mode that substitutes pure linear `PersistDimension`s into
their consumers before scheduling.  This trades row width for dependency
depth — exactly the GPU cost model, where 32 lanes sit mostly idle but every
layer pays several serial device-load chains.  Direct lookup operands stay
materialized because the O(1) attention backends classify those residual
coordinates structurally (in particular, expanding a clear-gate coordinate
back into `1e30 * gate - 1e30 * one` destroys the integer-score license).
Annotated memory-read persists also stay materialized so the fold can still
rebuild their 16-group fabric as one fused lookup.  CPU expert construction
is unchanged; the Metal packer opts into the depth-shaped fold for opcode and
sequence experts.

- **Depth result:** across all 227 opcode experts, min/median/max layers
  **4/20/36 → 2/11/26**, mean **19.15 → 11.76**.  The bench opcodes move
  CA **8→3**, D0 **11→7**, 4C **10→7**, and A2 **19→6**.  Residual width
  stays essentially flat (the four probes remain d_model 52-54), while
  folded term counts also fall.  The DEX/BNE sequence expert falls
  **10→4 layers**, d_model 60→56.
- **RESULT:** heated DEX/BNE bench with the K=2 window is
  **5.4 µs/instr = 185,309 steps/s**, up from v7's 8.1 µs / 123,969
  (**1.49x**).  Per-instruction/no-windows is **10.6 µs = 93,966
  steps/s**, up from 15.2 µs / ~66K (**1.43x**).  This validates the v7
  conclusion directly: depth, not stream width or barrier count, is what
  this single-simdgroup GPU mapping pays for.
- **CORRECTNESS:** the full all-opcode Metal pack succeeds under the O(1)
  attention licenses; verify program + 3,000-step bench loop are
  lockstep-exact at both t=32 and t=128.  Fold receipts now exercise both
  materialized and linear-inlined opcode/sequence graphs (23 focused tests
  green); ruff clean.

Next megakernel baseline is 5.4 µs/instr windowed / 10.6 no-windows.  Kernel
work should now target the fixed/layer-independent tail and the four-layer
sequence expert specifically: after construction removed six of ten layers,
old phase-chain decompositions are stale and must be remeasured before
choosing the next load/layout change.

## 2026-07-18 megakernel v7 — seg hoist landed, the chain floor mapped,
window-K optimum found

Systematic pass over the remaining kernel structure, one measured rung
at a time (each verified lockstep + heated-benched against the 9.0/17.1
v5 baselines):

- **LANDED: per-layer seg-bound hoist (+12%).**  All three stream
  phases' seg loads (bounds + nonempty flags, pure functions of li) now
  issue at the layer top and fly during phase A; each phase starts at
  its barrier with bounds in registers.  DEX bench 9.0 → **8.1 µs/instr
  (123,969 steps/s — arc total 2,840 → 124K = 43.6x)**; no-windows
  17.1 → 15.2.  The walk was also restructured into Trip4/load4 +
  STREAM_APPLY (byte-equivalent semantics, same perf, cleaner).
- **REJECTED, measured: phase fusion** (act inlined into SRC_FFN, acc
  zero into phase A, x=xa commit overlapped with the gate walk — 6 → 4
  phases/layer): −1.5%.  The fused phases carried NO device chains —
  copies/act were nearly free all along, and the inline doubled acc
  reads.  Confirms v6's lesson from the other side: neither barriers
  nor copy phases are the wall.
- **REJECTED, measured: first-batch triple prefetch + software-
  pipelined walk** (batch n+1 loads issued before batch n applies):
  −5%.  Suspected compiler load-sinking defeats the hoist while the 36
  always-issued loads per layer are real cost.
- **The decomposition, from two temporary probe modes (walks-off,
  walks+lookups-off; added, measured, removed):** no-windows 15.2
  µs/instr = **walks 7.4 + lookups 1.3 + copies/fences/seg 4.8 +
  fixed 1.7**.  Combined with the fusion/v6 null results, the model is:
  every phase pays a ~0.25 µs dependent-access chain (threadgroup and
  L1/L2 latency in ONE non-latency-hiding simdgroup), and nothing
  inside the phase structure moves it — three independent probes this
  session tried (register residency, fusion, prefetch) and all measured
  flat-to-negative.
- **Window-K optimum: K≈4.**  Spin unroll ladder (repeat 4/8/16):
  4.43 / 4.68 / 5.25 µs/instr — the folded model's depth grows faster
  with K than the amortization pays.  Spin tier stands at **225,812
  steps/s**; raising repeat is NOT a lever.

Standing tiers after this pass: DEX/BNE micro-loop 124K, JMP spin
225K, per-instruction (no windows) 66K.  A wait-heavy real mix (SMB is
~78% spin+poll) sits at ~4-6 µs/instr ≈ 150-250K — half-speed-playable
territory; full-speed NES (~592K instr/s) needs ~3x more, and the
measured chain floor says that will NOT come from kernel
micro-structure.  The levers that remain are construction-side —
fewer layers per instruction (expert depth diet, per the standing
focus-weights-construction directive; depth is the ONLY thing the GPU
pays for, width is free at 32 idle lanes) — or a different execution
mapping (cross-step speculation overlapping forwards).  Receipts:
verify + bench lockstep-exact at t=32 AND t=128 after every rung and
after probe removal; ruff clean.

## 2026-07-17 megakernel v5 — sequence windows on GPU + the clock-state
measurement trap, 39x cumulative

Second rung of the GPU expert arc, same session as v4: sequence-window
experts (fold_sequence via `build_sequence_expert`, the C++ tier's own
machinery) are packed like any model plus a 5-int seq router entry
(head, K, descriptor, window bytes); the kernel checks windows before
opcode dispatch — pc == head, bytes re-verified from RAM (stale windows
inert, self-modifying code safe), whole window inside the step budget
(boundary falls back to per-instruction, step-for-step equivalence) —
and one forward retires K instructions: K trace rows, ordered writes,
final state, `step += K`.

- **RESULT: DEX/BNE bench 9.0 µs/instr = 110,500 steps/s** (the CA D0 FD
  window, K=2), t=128 within 1%; JMP-self spin (K=4 unroll) 4.9 µs/instr
  = **205,800 steps/s**.  Session ladder: 2,840 → 58,900 (v4 routing) →
  110,500 (v5 windows) = **39x**, spin tier 205K.  CORRECTNESS: verify
  program + 3,000-step bench loop lockstep-exact vs reference with
  windows live (trace rows per instruction, boundary chunks included).
- **THE MEASUREMENT TRAP, now understood and fixed — READ THIS BEFORE
  ANY FUTURE SINGLE-STREAM GPU BENCH.**  A one-threadgroup kernel is ~1%
  GPU utilization, so macOS NEVER ramps the GPU clock for it: identical
  runs swing 2-3x with whatever ran before (47 vs 18 µs cold/warm; the
  user reproduced 21.4 vs 9.1 back-to-back).  An in-kernel warmup loop
  does NOT fix it — the load is still invisible to the governor.  Fix:
  `gpu_heat()` saturates the GPU with 2 s of MPS matmuls (that load DOES
  ramp), then measure; three-run reproducibility after the heater is
  0.01%.  A whole probe round mid-session (staging "regression", seq
  "regression", 46.7 µs inert-window readings) was clock artifacts.
- **Phase-count economics, measured.**  Cost ≈ 1.6 µs/forward fixed +
  ~0.4 µs per NON-EMPTY phase (empty stream phases skip barrier and
  walk).  Non-empty phase totals: union 381; singles CA 38 / D0 53 /
  4C 48 / A2 90; seq DEX/BNE 46 for TWO instructions; seq spin 40 for
  FOUR.  The seq DEX/BNE model is structurally identical to one single
  expert (379 triples, 10 layers, 12 register rows, same score/gate/
  value table sizes) while retiring 2 instructions — phase count per
  instruction is the entire cost model; layers, triples, and d_model
  are all non-predictive at this scale.
- Next rungs, ranked by that model: (1) **register-resident residual**
  — expert d_model ≈ 50-60 ≤ 2 floats/lane at t=32, so x/xa fit in
  simdgroup registers with simd_shuffle for column reads; within one
  simdgroup the phase barriers dissolve entirely, leaving the triple
  streams' cache-hot loads (est 3-5x: DEX ~300-500K, spin ~1M — the
  jump that would pass the C++ per-instruction tier); (2) act-phase
  inlining + copy diet (6 → 4-5 phases/layer, bounded ~1.3x);
  (3) window coverage: the scanner already finds any ROM's idioms;
  repeat-unrolling stays spin-only (a mid-window branch exit is a
  license rejection, correctly).  Null probe note: probe modes now skip
  the halt compare (a layerless forward's garbage readout halted the
  timing loop at step one).

**Same-session follow-up (2026-07-18) — register-resident residual (v6)
BUILT, MEASURED, REJECTED, REVERTED.**  Rung (1) above was implemented
in full: narrow models (d ≤ 64 at t=32) held x in two registers/lane,
attn/ffn rows packed to owner lanes (dst mod 32), column reads via
simd_shuffle, phase barriers replaced by simdgroup lockstep, ReGLU
inlined into the ffn walk.  It reached lockstep-exactness (verify +
bench programs, all experts + windows) and then measured **1.7x
SLOWER** than the threadgroup path: no-windows 28.5 vs 17.1 µs/instr,
windowed 15.2 vs 9.0.  The projection above was WRONG about the wall:
removing every barrier and threadgroup-residual read bought nothing —
synchronization was never the cost — while the register form made the
real cost (device table loads) worse: its walks stall load→use per
triple (the wide path's 4x batching is the fix the narrow walk gave
up), and its 12 register-lookup rows run SEQUENTIALLY (redundant on
all lanes, shuffles need everyone converged) where the wide path does
all rows in parallel across lanes.  Verdict: attack DEVICE LOADS, not
synchronization; the threadgroup path stays.  Two hard-won findings
recorded for any future Metal work:
- **The divergent-shuffle law**: a simd_shuffle inside a branch that
  differs per lane reads from lanes that took the other branch —
  inactive there, result undefined (observed zero).  The correct form
  computes every shuffle unconditionally on all lanes and selects
  among the RESULTS.  This produced a wrong-but-plausible forward
  (state passthroughs silently zeroed) that survived until a
  slot-by-slot residual diff against a numpy replay of the same
  tables.
- **Debug-harness discipline**: two full rounds of divergence analysis
  were invalidated by the harness itself — script `replace()` patches
  that silently no-op on stale text (every patch must assert its match
  count) and debug dumps sharing trace-buffer regions (each probe
  needs its own range, verified).  The numpy table-replay oracle plus
  binary-search-by-dump was what actually found both real bugs.
Receipts after revert: verify + bench lockstep-exact at t=32 AND
t=128; heated bench 110.4K windowed / 58.4K no-windows — the v5
records, byte-identical behavior.  metal_host gained a `trace_raw`
debug accessor (kept; the kernel carries no debug code).

## 2026-07-17 megakernel v4 — expert routing on GPU, 21x

The recorded "expert-fold-on-GPU arc" landed in one rung: the fold/plan
machinery is now in the Metal packer
(`experiments/gpu_single/{metal_runner.py, kernel.metal}`), and the
megakernel routes every step to a per-opcode expert model.

- **Mechanics.**  `build_opcode_experts` (the C++ engine's own fold →
  asap → build_model pipeline, unchanged) supplies all 227 rebuilt
  models; the packer packs union + experts into the SAME global streams
  and layer table (layers indexed globally), one 12-int descriptor per
  model (layer range, d_model, one_slot, embed/readout slot tables,
  STATE/HALT head rows) and a 256-entry opcode→descriptor table.  The
  kernel reads the fetched byte ram[st_pc] masked by the written bit
  (exactly the graph's memory read) and runs that expert's layer range;
  NMI and non-expert bytes fall back to the union (descriptor 0).
  Expert register heads resolve onto the union's running-argmax
  allocation through the fold's lookup provenance (never allocate —
  engine_gen's rule); expert memory heads are the folds' fused 64K-span
  reads (group base −1 beside the union's 4K group dispatch).  Whole
  build (union + 227 folds + pack) ≈ 8 s post-diet.
- **RESULT: 2,840 → 58,900 steps/s (17.0 µs/step) on the DEX/BNE
  bench**, t=32 and t=128 within noise of each other — the
  single-simdgroup question dissolved with the phase count.  Steady
  state runs experts of 8-11 layers / d_model ~52 (CA 8, D0 11, 4C 10)
  vs the union's 75/632; expert stats overall: 4-36 layers, median 20.
  Null-model probe (loop+embed+dispatch+readout, no layers) 1.6 µs/step.
  CORRECTNESS: verify program (40 steps) + bench loop (3,000 steps)
  lockstep-exact vs reference at t=32 AND t=128 with all experts routed;
  --experts none reproduces v3 exactly (2.0K steps/s at t=32).
- **Threadgroup staging probed and REJECTED** (the recorded rung (b)):
  three variants — triples staged, batched bulk copy, seg tables +
  triples staged (whole expert ≈ 6.5 KB median, fits the 32 KB budget)
  — measured 18.5-19.5 µs vs 17.0 unstaged, lockstep-exact throughout,
  reverted.  Diagnosis: a 2-3 opcode loop keeps the routed experts'
  tables cache-hot after the first iteration, so the per-phase
  "device chains" are mostly L1 hits and the copy is pure added cost.
  The v2 lesson generalizes: traffic dieting loses whenever the wall is
  latency to WARM lines, and routing already shrank the working set to
  cache size.
- Remaining structure at 17 µs: ~60 phases across ~10 layers ≈
  0.26 µs/phase (chains through warm cache + barrier + in-order
  restart), plus the 1.6 µs serial lane-0 embed/readout tail.  Next
  rungs, in expected order: sequence experts on the GPU (fold the whole
  micro-loop as ONE forward — the C++ tier that took this same loop to
  15.8M; the packer now has everything it needs), in-place phase fusion
  (~3 phases/layer), and the lane-0 tail only after those.  Real-time
  NES (~592K instr/s) is now 10x away, not 200x.

## 2026-07-17 weight census + export integerization — block sparsity measured and rejected

Structural census of the union model's weights (d_model=632, 75 layers,
316 heads, d_ffn=686), toward GPU-shaped weights, per explicit user
call.  The numbers that decide everything:

- **41,012 nonzeros across 217M dense layer-matrix entries (99.98%
  zeros)**; value alphabet 82% ±1, four ±0.5, the rest small integers
  plus (pre-fix) the query temperatures below.
- **93.5% of (head, layer) slots are completely unused** — 125 lookup
  heads + 1,423 passthroughs in the whole stack; FFN unit-layer slots
  88.8% unused; the median layer touches 89 of 632 residual slots
  (min 38, max 484).  Head/FFN allocation is peaked: median 13 used
  heads/layer, max 137.
- **Block sparsity: measured, rejected.**  Even under a bipartite-RCM
  reordering upper bound, 8x8 blocks reach only ~8-14% fill (16x16
  ~4-7%) — a block-sparse kernel would do ~10x wasted work.  The nnz is
  scattered circuit wiring, not clustered structure.
- The exact structural lever instead: **ragged per-layer compaction**
  (live slots × used heads/units) is 5.3M dense entries = **2.4% of the
  allocated shapes**.  A batched GPU runner would gather live slots, run
  small dense per-layer matmuls over the batch, scatter back.

**Ragged pruning LANDED in the export** (`prune_layer`): the artifact's
layers now carry only live heads and FFN units — layer li has
H=layer_heads[li] heads (in_proj (6H, D), out_proj (D, 2H)) and
F=layer_ffn[li] units (ff_in (2F, D), ff_out (D, F)); d_model stays the
global 632 (the residual is shared across layers — slot compaction is a
runner-side gather, not an artifact shape).  Kept: 1,548 of 23,700
allocated head slots, 5,747 of 51,450 FFN unit slots; declared dense
shapes 217M → 18.7M entries (11.6x), torch_runner's per-token K/V cache
94,800 → 6,192 values (15.3x), artifact 0.87 MB.  The pruning license is
pattern-exact, not classification-based: a head is dead iff its V rows
OR out_proj columns are all zero (zero contribution either way; zero-Q/K
heads with live V are passthroughs and stay), an FFN unit iff its gate
row, value row, or ff_out column is all zero — and this found the kinds
tables OVER-approximating: L44 unit 14 is kind 1 with a zero gate row
(gate ≡ 0 never passes ReGLU's strict >), so pattern-pruning drops 2
units classification keeps.  The export asserts one-directionally that
everything KEPT is kinds-live; pruning more is legitimate.

Export integerization landed on the back of the census
(`model/export.py:rescale_head_queries`): each head's Q row pair is
stored divided by its hardmax temperature — a positive per-head score
scale, invisible to the argmax and its exact ties (the generated engine
already runs the divided-out integer queries, Klaus-verified).  The
artifact prose's "every coefficient is an integer or ±0.5" claim was
previously FALSE: 3,262 non-integer coefficients survived in in_proj Q
rows (C = 1e10·√2 on 3,097 entries plus a ±C·group_lo ladder for the
16 memory-group bases).  Now enforced by an export-time assertion.
Preconditions verified before the change: all 125 −1e30 clear gates are
K-side (never rescaled, only need to lose the argmax), every head passes
the integer-quotient guard, and the max non-clear coefficient
post-rescale is 122,880 < 2^24 — so **f32 storage is genuinely exact**,
and the recorded "GPU blocked on f64 score construction / key
rescaling" item is retired artifact-side.  The f64 export mode was
dropped entirely (user call: no reason to keep it once f32 is exact);
torch_runner still executes in f64 internally.

Receipts: full suite 74 green (incl. `test_export` lockstep: export →
standalone TorchMachine vs reference, 40 instructions, through the
ragged shapes); ruff clean.  Census scripts live in the session
scratchpad (weight_census{,2}.py, check_qk_sides.py).
**Same-day follow-up — single-machine GPU runner, measured
(`experiments/gpu_single/gpu_runner.py`), per explicit user call: ONE
6502 on the GPU (latency), not a batch.**  The runner lifts the
generated engine's O(1) attention forms to torch tensors, licensed by
model/classify.py exactly as engine_gen is: passthrough heads compose
into per-layer (I + attn_direct) matrices, kind-2 FFN units into
(I + ffn_direct), memory heads become a RAM-tensor gather
((ram[addr], addr) on a written in-group hit, else (0,0) = the
empty-group result), constant-query heads become running-argmax
registers, and WRITE tokens are scatters, never forwards.  All 125
union lookup heads licensed (zero hull heads).  CORRECTNESS: 40-step
verify program lockstep-exact on cpu AND mps; 2,000 bench steps
lockstep-checked vs reference ON MPS IN f32 — the rescaled artifact
semantics are demonstrably f32/GPU-exact end-to-end, not just by
argument.  PERFORMANCE (DEX/BNE micro-loop, M-series): mps eager 73
steps/s (13.6 ms/step), cpu eager 295, torch.compile: inductor-metal
fails shader codegen (experimental), cpu-inductor 88 (slower than
eager — guard overhead beats fusion at these sizes).  The C++ engine
on the SAME loop: 15.8M steps/s (its boot scan folds the loop into a
sequence window); union per-instruction tier ~420K (full Klaus).
DIAGNOSIS: a step is ~40K FLOPs across ~450 tiny sequential kernels —
>99.9% of GPU time is launch latency, and the math is smaller than one
launch.  The 2026-07-04 "GPU is the wrong tool for ONE stream" verdict
SURVIVES the fp64-blocker retirement: even a hand-fused Metal
megakernel with a GPU-resident run loop (one dispatch per run(), the
only remaining single-stream design) is bounded by one-threadgroup
execution — ~75 layers × barrier latency ≈ 30-60 µs/step ≈ 15-35K
steps/s, an order below the CPU engine and below real-time NES
(~592K instr/s).  Single-machine stays on the C++ engine; the GPU's
lane is batch, where the same O(1)-forms runner design applies
unchanged (state tensors gain a leading batch dim).

**Same-day follow-up 2 — Metal megakernel prototype, the GPU-resident
run loop (`experiments/gpu_single/{metal_runner.py, kernel.metal,
metal_host.mm}`), per user call ("will we just have to generate
megakernels like the cpp engine?" — yes, and here is rung one).**  One
Metal dispatch executes up to max_steps WHOLE instructions in a single
threadgroup: STATE embed, all 75 layers, readout with half-even rint,
RAM writes, next state — RAM/registers/state persist in device buffers
(the C++ engine's run() protocol minus MMIO).  Data-driven: CSR tables
walked from device memory; the host is a ~200-line pybind ObjC++
extension compiling kernel.metal at runtime (built via clang++ per the
Makefile's conventions, cached by source digest in _build/).
CORRECTNESS: verify program (40 steps) + bench loop (3,000 steps)
lockstep-exact vs reference, executed inside single dispatches.
PERFORMANCE ladder on the DEX/BNE loop: torch eager MPS 73 →
megakernel v1 (row-major CSR incl. identity diagonals) 2,193 →
traffic-lean v2 (identity as threadgroup copy, live-row lists, x/ho
and xa/act columns fused per row) 2,746 steps/s (364 µs/step; threads
512≈256>128).  The v1→v2 diet cutting ~6x traffic but only 20% of time
is the diagnosis: the wall is DEPENDENT-LOAD LATENCY (ptr→col→val
chains from device memory, one non-latency-hiding threadgroup) plus
the ~300-barrier/step sequential floor — not bandwidth.  Recorded next
rungs with estimates: (1) engine_gen-style Metal CODEGEN — coefficients
and slots as instruction immediates, no table loads (est 3-8x);
(2) single-simdgroup execution — simdgroup_barrier ≈ free vs
~0.3-0.5 µs threadgroup barriers (kills the barrier floor; only pays
after codegen shrinks per-phase work); (3) expert routing on GPU
(median expert ~16 layers → ~4x fewer barriers).  Combined projection
~100-200K steps/s — plausibly real-time-NES capable, still at-or-below
the CPU engine; the megakernel arc is a thesis/coolness artifact, not
a perf win over CPU.

**Same-day follow-up 3 — megakernel v3 (per-lane triple streams) + the
probe series that found the REAL wall.**  User call: continue the GPU
arc for its own sake.  First, a SIMT correction to the recorded codegen
plan: per-lane straight-line code with immediates DIVERGES and
serializes on a GPU — the SIMT-correct analogue of engine_gen's
codegen is a pack-time pre-scheduled per-lane TRIPLE STREAM (dst, col,
coeff interleaved 12B, whole rows per lane so no races, greedy
load-balanced, one contiguous segment per (layer, phase, lane)), plus
single-simdgroup execution turning barriers into simdgroup_barrier.
Landed as kernel v3 with 4x-batched loads (in-order GPUs stall
load→use per term otherwise).  Lockstep-exact as ever (verify 40 +
bench 3,000 steps).  RESULT: 2,840 steps/s at t=128 (352 µs/step) —
BARELY better than v2's CSR, and t=32 single-simd is WORSE (498 µs).
Probe series (do_trace modes 2/3, kept in the kernel): null model
(loop+embed+readout, no layers) 0.1-1.2 µs/step; barriers-only +450
barriers ≈ +1-2.6 µs — a barrier is ~2-6 ns, NOT the wall; three
disjoint table layouts within 10% of each other.  DIAGNOSIS, now
measured not guessed: each of the ~500 phases/step pays ONE unhidden
device-memory latency chain (~0.7 µs) reaching its tables (seg +
first triples); a single threadgroup has no other warps to hide
latency behind, and in-order lanes can't run ahead through a barrier.
Next rungs, re-ranked by this evidence: (a) FEWER PHASES — in-place
phase fusion using the engine's capture/apply conflict analysis
(~3 phases/layer, ~1.5-2x) and above all EXPERT ROUTING (median
expert ~16 layers → ~5x fewer phases AND less work) + sequence
windows (K instructions per forward); (b) threadgroup-staged seg
tables (ushort, fits at t=32 only, ~1.3x); (c) per-layer table
prefetch past barriers (loads are barrier-independent constants).
Realistic union-tier ceiling ~30-80 µs/step (12-30K steps/s);
expert+sequence tiers maybe 100-200K on hot mixes.  The GPU 6502 is
real and exact today at 2.8K steps/s; making it respectable is the
expert-fold-on-GPU arc (port the fold/plan machinery into the Metal
packer), a fresh session's work.

## 2026-07-17 weights-console correctness pass — audit + three seam fixes

Full audit of the C++ runtime's weights-backend seams, per explicit
user call: confirm the transformer drives all CPU/memory state and no
native code bridges gaps.  Verdict: clean — step_core is the emitted
forward (state embed → layer stack → readout → WRITE tokens into the
memory-attention structures), every C++-side mutation routes through
the engine protocol (inject_write / map_region / poke), and the C++
side owns only hardware: frame timing, MMIO devices, DMA, IRQ line
levels (delivery itself runs in the weights).  Three seams tightened,
none observed misbehaving on the tapes:

- **Undrained journal writes now throw.**  PythonEngineMachine never
  consumed wj_addr/wj_value; the native write journal and PRG banking
  are dormant (no callers anywhere), but re-arming them would have
  silently dropped mapper CHR/IRQ/mirroring side effects.  run_cycles
  fails loudly instead; wiring a real drain (with a wj_cycle stamp)
  stays the recorded perf lever.
- **RMW device writes reach write_rmw on the engine path.**  Only
  Bus::step routed RMW opcodes to write_rmw, so an MMC1 game doing INC
  on $8000-$FFFF got the final value clocked into the serial port
  instead of hardware's first write (the original bus byte).  run()
  now returns the pausing step's opcode (pause_ir — "ir" was taken by
  the per-step trace array); service_writes routes through write_rmw
  when is_memory_rmw (moved to bus.h).  The original parameter left
  the whole chain: Mmc1 computes it as rom_byte(addr), provably the
  pre-write byte since repairs keep the image ROM-accurate, and
  write_rmw carries the cycle stamp instead — MMC3-class RMW register
  writes now stamp like plain writes on both paths.
- **MMC5 locked $6000-$7FFF writes repair the machine image.**  The
  stray byte landed engine-side before the pause and was only healed
  by watched reads; OAM DMA and shadow decode read the image directly
  and would have seen it.

Known by-design, documented not changed: the base-cycle CYCLES table
is shared by all backends (no branch/page-cross penalties — the
console clock was never transformer-derived), leases are
receipt-verified invisible, shadow decode only decides pausing.
Stale "EngineBus" references in run_protocol.py/runner.py updated to
PythonEngineMachine (nes/bindings.cpp), its actual consumer.

Coverage: new nes/tests/test_console_backends.py — 90-frame scripted
reference-vs-weights lockstep through the native console (SMB and
Zelda: frames, cycles, and RAM byte-equal) plus an MMC1 RMW quirk
regression on a synthetic cart (five INC $E000s must select the ROM
byte's bank), run on both backends.  The suite previously never drove
Console(weights) at all.

Receipts: 74 tests green (was 70), native hardware test green (RMW
asserts reworked: ROM-byte clocking, stamped write_rmw), ruff clean;
all five tape cycle counts byte-identical post-change — LoZ
67,156,156, SMB 151,674,634, SMB3 260,464,627, spade 310,407,366, CV3
157,660,617; LoZ reference-backend replay matches weights exactly
(2255 frames, same cycles).

## 2026-07-17 NES C++ runtime de-OO pass — behavior-identical

Handmade-hero-lens refactor of the native runtime, per explicit user
call.  Five commits, each independently suite-green; net -305 lines
across 22 files.  Nothing behavioral: every stage's receipt is the
unchanged test suites, and the arc's receipt is tape cycle identity.

- **nes/chr.h owns the PPU-side mapper halves**: the ChrBanks types,
  Mmc5Ppu, and one plain ChrLatch struct replacing the virtual
  ChrLatchMapper interface and its two near-identical implementations
  (wide_low selects MMC4's full-row trigger ranges) — devirtualizing
  chr_index in the render inner loops.  Mmc2/Mmc4 themselves share one
  LatchMapper class (mappers/mmc2_mmc4.h; mmc2.h/mmc4.h deleted).
- **pImpl removed from Ppu and Apu**: state is plain public fields,
  ~100 forwarding stubs and the heap hop are gone; accessor pairs
  (sprite0_scanline/sprite0_hit/sprite_limit/mirroring, the
  nametables/palette/oam span getters) became field access.  APU
  channel structs and tables moved to apu.h unchanged.
- **One stamped write per layer**: Device::write(addr, value, cycle)
  and Mapper::write(addr, value, scanline) replace the write/write_at/
  write_stamped triple (-1 = no stamp); every dispatch site already
  carried the stamp, so the unstamped names were dead.
- **MapperCallbacks (11 std::function slots, five named after specific
  mappers — the "generic" layer knew every concrete type on both sides)
  replaced by MapperHost{Ppu*, ConsoleMachine*, IrqLine*}**: mappers
  call the PPU and machine directly.  IrqLine is a small value type in
  bus.h owning the shared level-sensitive line's source bits plus the
  mapper's sched-dirty replan signal; Console's set_irq/sync_irq
  bookkeeping dissolved into it.  Console holds Ppu/Apu/adapters/OamDma
  by value (seven unique_ptrs gone) and exposes cycles/frame_index/
  halted/last_audio/ppu/mapper as fields.  Dead code deleted along the
  way: Console::read_cpu/mapper()/sprite0_scanline(), Ppu tile_pixel/
  scroll_px_x/scroll_px_y, the F1-era MMC5 exram/split-bank debug
  accessors, ChrLatchMapper::latches().
- **std::function trims**: OamDma is a console adapter with direct
  ConsoleMachine&/Ppu& (was two std::function hops per DMA'd byte); the
  APU DMC memory reader threads through as a parameter (member +
  exception-guard struct gone, same frame-scoped lifetime); InputTape
  is a plain struct; the 8x16/8x8 sprite pattern decode is one function
  (was written out three times in ppu.cpp).
- **Kept virtual deliberately**: ConsoleMachine (Bus vs
  PythonEngineMachine — two real machines), Mapper (real hardware
  variance; only its callback channel was rotten), Device
  (address-dispatched MMIO + the native test's seam), SteppedRuntime
  (Python core + test fake).
- `*.sav` is gitignored now: battery saves are user data next to the
  ROM, and a working-copy snapshot had silently swept the live Zelda
  save into a commit mid-refactor (extracted, bytes preserved on disk).

Receipts: 70 Python tests + the native hardware test green after every
stage, ruff clean; all five recorded tape cycle counts byte-identical
post-refactor — LoZ 67,156,156 (861 fps), SMB 151,674,634 (1139 fps),
SMB3 260,464,627 (584 fps), spade 310,407,366 (585 fps), CV3
157,660,617 (561 fps).

## 2026-07-17 debug overlay + battery saves + frontend portability

QoL/debug infrastructure arc, motivated by the MM6 investigation below
(hardware-verified authentic — these tools make the next report cheaper
to chase):

- **Tilde debug overlay** (`nes/ui/debug_ui.{h,cpp}`, Dear ImGui via
  sokol_imgui; both vendored since 2026-07-11, now compiled in).  `~`
  toggles; Esc closes; game input is suppressed while open.  Contents:
  display fps + emulation ms/frame (last/EMA), frame index, cycles,
  backend; QoL toggles — pause + single-frame step, remove the
  8-per-scanline sprite cap (Ppu::set_sprite_limit; default stays
  hardware-exact because TLoZ's door masking NEEDS the cap), overscan
  crop on/off (the blit now samples a uniform sub-rectangle, so full
  256x240 is one checkbox), mute; and a clipper-based hex inspector
  over the CPU address space (enter a hex address to jump).  app.cpp
  wraps a Python `DebugBridge` (app.py) as std::function hooks; the
  overlay TU is pybind-free.  Gotcha that cost one crash: app.cpp casts
  the bridge's list-of-tuples returns to `std::vector<std::pair<...>>`,
  which silently does nothing without `#include <pybind11/stl.h>`.
- **Battery saves**: iNES flag6 bit 1 was already parsed; now exposed.
  Interactive runs load `<rom>.sav` into $6000-$7FFF before the first
  frame (one native map_region — no engine WRITE tokens) and write it
  back from the quit callback.  Tape runs stay save-blind on purpose: a
  .sav at boot would desync recorded inputs.  Receipt:
  `test_battery_prg_ram_round_trip` + a live Zelda save/load cycle.
- **Header tree**: `nes/include/nes/{*.h}` + `nes/include/nes/mappers/`,
  includes are `"nes/..."`-qualified, app.py passes the include dir;
  `.clangd` carries each repo -I at both depths (relative flags resolve
  against the edited file's directory).  pybind11 v3.0.1 (torch's
  bundled version) is vendored via install_deps.sh for clangd — builds
  still compile against torch's copy; only Python.h remains an absolute
  machine path (uv keeps CPython headers outside the venv).
- **Frontend portability**: backend selection lives in
  `ui/sokol_platform.h` (`__APPLE__` → SOKOL_METAL, else SOKOL_GLCORE);
  app.cpp carries #ifdef'd MSL and GLSL 410 shader pairs and
  backend-conditional reflection; app.py compiles sokol_impl as ObjC on
  darwin / plain C elsewhere and links GL/X11 libs on the else-branch.
  The Linux path is plumbed but UNTESTED (no Linux machine here).

**Same-day follow-up — Makefile build to bin/, torch out of the NES
build.**  Vendoring pybind11 removed the last thing
torch.utils.cpp_extension supplied (pybind includes, module naming,
digest caching), so the NES extensions now build from the repo-root
Makefile: `make hardware` → bin/nes_hardware.so, `make frontend` →
bin/nes_app.so (imgui objects warning-silent, sokol_impl as
ObjC/darwin or C/else), `make test-native` builds and runs the native
hardware test; -MMD dependency files replace content digests, and the
PYBIND11_MODULE names are literal (no TORCH_EXTENSION_NAME).  app.py's
loaders are now one quiet `make` (PYTHON=sys.executable; the Makefile
defaults to .venv/bin/python for bare `make`) plus a plain extension
import from bin/ — _nes_cache is gone.  vendor/pybind11 joined
sokol/imgui in .gitignore (fetched by install_deps.sh, untracked).
Only Python.h stays a machine path anywhere (.clangd): pybind11 wraps
the CPython C API, and uv keeps those headers outside the repo.
GNU-make gotcha for next time: `%` never matches an empty stem, so
`imgui%.o` cannot build imgui.o — the vendor rule is a plain `%.o`
pattern, applied only where vendor/imgui/%.cpp exists.
The weights-engine codegen build (model/_engine_cache) stays
torch-driven, untouched.  app.py then split along its seams: build.py
(make + extension import), backends.py (core adapters,
backend_factory, OPCODES_NATIVE), debug.py (DebugBridge); app.py
keeps only run_app + main (the nes-run entry point).  The NTSC
palette moved to include/nes/palette.h outright — it only ever
traveled Python → run() → memcpy (a fossil of app.py owning
presentation data), so the numpy round trip, the (64,3) validation,
and run()'s palette parameter are gone.  Python-side access is a
binding, not a copy: `nes_hardware.NES_PALETTE` exports the header
table as a (64, 3) uint8 array for frame-analysis scripts.

Receipts: 70 Python tests green (69 + battery), native hardware test
green (also via make test-native), ruff clean; LoZ tape
cycle-identical (67,156,156), MM6 tape clean at 570 fps; app boots,
runs, and quits cleanly through the new init/overlay/cleanup path with
the sprite-limit default verified inert (render paths identical when
the cap is on); post-Makefile re-receipts: suite 70 green, LoZ
67,156,156 again, app smoke 3.0 ms/frame, no-op `make` is silent.

## 2026-07-17 MM6 sprite reports — both symptoms are game truth, no engine change

Two play reports on Mega Man 6 (`tapes/megaman6_tape`, Wind Man stage),
both reproduced with numpy frame analysis and diagnosed from the
temporary env-gated PPU register/OAM logs (added, used, removed;
ppu.cpp byte-identical after).  NOTHING was changed — both are
authentic hardware behavior:

- **Left-edge garbage sliver.**  The stage scrolls horizontally under
  HORIZONTAL mirroring (MMC3 $A000=1; the register log shows mirr=3,
  mask=$18 throughout), so the nametable is a 256-px cylinder: the
  column streamed for the right edge is the SAME column leaving the
  left edge, and its tile+attribute writes land while 8−fine_x pixels
  are still visible past the left-8 mask.  Fingerprint receipt: on
  every glitch frame the sliver width equals 8−fine_x exactly (fx=6 →
  2 px, fx=7 → 1 px, 11/11 frames); the periodicity is one glitch per
  tile crossing, and the recolored tail is the shared attribute byte
  (2x2-tile granularity forces recoloring the old columns).  Real
  hardware shows the identical sliver in the first visible pixels;
  deep CRT overscan hid it — our 8-px crop is a tight CRT.  The whole
  Mega Man series does this.
- **Big-sprite flicker when the player shares scanlines.**  Two stacked
  mechanisms, both authentic.  (1) The GAME hides whole objects: on the
  panda mini-boss's blink frames the OAM dump has 32 live sprites vs
  43 — Mega Man's ten entries are simply absent (software flicker
  manager, 2-on/2-off), and single-frame enemy vanishes coincide with
  buster hits (damage blink).  (2) On the crowded frames, recomputing
  NESDEV first-8-per-scanline evaluation over the dumped OAM predicts
  exactly the dropped half-tile slivers in the render (panda rows are
  5-6 sprites wide; player+bullet push lines past 8).  Emulation-side
  levers that could have skewed the game's own flicker decision were
  checked: the OAM-DMA 513-cycle stall is charged on the engine path
  (bindings take_stall), and the $2002 overflow flag (which we don't
  set) is broken silicon no game relies on.
- METHOD, for the next sprite report: dump OAM at render time and
  compare against the drawn frame — sprites missing FROM OAM are the
  game's choice (authentic), sprites in OAM but not drawn must be
  explained by first-8/priority/clip rules or it's our bug.  And the
  seam-authenticity fingerprint is width == 8−fine_x.
- A "no 8-sprite limit" toggle was considered and NOT added: TLoZ
  depends on the limit for its dungeon-door masking (2026-07-16 round
  3), so lifting it globally is a known regression; if ever wanted it
  must be per-ROM opt-in.

## 2026-07-16 PPU raster fidelity — Zelda vertical scroll + SMB3 spade minigame

Two frame-model defects, both reported from play and reproduced on input
tapes (`tapes/loz_tape`, `tapes/smb3_spade`), both diagnosed from
env-gated PPU/mapper register text logs (added temporarily, removed):

- **Sprite-0 splits now honor the mid-span $2006 v reload and the real
  write row.**  Zelda scrolls its playfield under the fixed HUD by
  reloading all 15 bits of v after the sprite-0 hit; the render's live
  band ignored the load (y came from t, relative to screen row 0) and
  started at the hit line (48) instead of the write row (~64) — vertical
  room transitions tore the HUD bottom and showed the wrong playfield
  rows.  The PPU now records visible-span scroll writes and the last
  $2006 pair's 15-bit load; after a sprite-0 event (only when no mapper
  IRQ is scheduled) the console runs scanline-sized slices until the
  journal drains the game's post-hit write batch, then snaps that line
  up to the next tile-row boundary (slice attribution is ±a line; games
  time splits to tile edges: Zelda 61→64).  The live band starts there,
  y pinned to the loaded row.  SMB-class immediate writers stop the scan
  after 1-2 slices; a game that never writes costs at most 48 slices.
- **MMC3 IRQ latch is a delta, not an absolute line.**  The counter
  reloads from the latch on the line after each expiry, so chained
  same-frame IRQs fire latch+1 apart.  SMB3's spade minigame arms
  latch=0x20 in the NMI and each handler re-arms latch=0x34: the chain
  fires 32/85/138/191 (three 53-row picture strips + bottom HUD at 192).
  The old model read the latch as an absolute scanline, so the chain
  died after two fires and rows 53-239 (strips 2-3 AND the lower HUD)
  shared one band — the reported garbage.  `irq_scanline(last_fired)`
  now takes the previous expiry line (-1 at frame start; MMC5 ignores it,
  its $5203 compare is absolute).  Level frames still fire once at
  latch=192/193.
- **Ordered raster snapshots serve every mapper.**  The per-split
  journal (`raster_splits`, captured in mark_split for all mappers since
  the CV3 arc) was only consumed when MMC5 was active; non-MMC5 frames
  with >1 split now render through it too, each band fetching through
  its snapshot's CHR banks (single-split frames keep the mid_* path
  byte-identical).

Same-day cleanup pass over the fresh C++ console port: the consumer-less
event journal (JournalEntry/add_event, grew unboundedly and was never
read) and the F1 BTN_DEBUG bit (its app.py debug-dump consumer left in
the port) are gone; run_frame's event picking is two scalars instead of
a per-iteration vector+sort.  bus.cpp/tape.cpp/app.cpp reviewed, no
other changes warranted; the mid_*-vs-raster_splits duplication in
ppu.cpp is deliberate (the mid path serves the sprite-0+IRQ combo the
raster path doesn't model).

**Follow-up (2026-07-16): stamped write rows + full MMC3 counter.**
Every watched device write pauses the engine in the C++ port (the
2026-07-13 write journaling was not re-wired -- re-enabling it is the
recorded perf lever, and needs a wj_cycle stamp when it lands), so each
drained write already knows its cycle: run_cycles carries a cycle base,
Device::write_at stamps, and the console converts to a visible-span
scanline.  The sprite-0 split row now comes from the stamped row of the
first post-hit scroll write (tile-snapped -- the clock is base-cycles
only, so stamps are exact for OUR time, a few lines early vs hardware);
the scanline-slicing scan this replaces is deleted.  MMC3 grew the rest
of its counter semantics per NESDEV: latch 0 fires next-scanline when
chained, a mid-frame $C001 anchors the reload at its stamped row
(begin_frame resets the anchor), and $C000/$C001/$E001 writes raise an
irq_changed callback -- run_until exits early on it and run_frame
re-plans its events, so a mid-frame arm can't be run past (suspected
mechanism of the reported HUD corruption in SMB3's scrolling
cutscenes, spade-identical but tapeless so far; user-confirmed fixed).
The early exit's first cut applied to EVERY run_until and truncated
the end-of-frame vblank run on each MMC3 frame (the NMI writes
$C000/$C001/$E001), slipping the NMI tail into the next frame's
visible span -- user-reported as general random flicker, found by
elimination after a $C001-anchor histogram came back clean (all level
IRQs at 192/193, zero mid-span reloads).  run_until(target, replan)
now completes plain runs unconditionally; only the event-planning
loop's runs stop early.  Receipt: every tape's cycle count returned
byte-identical to its pre-arc value (SMB3 260464627, spade 310407366,
LoZ 67156155, SMB 151674636, CV3 157660617).

Third round, from play reports (Zelda room transitions):
- **Sprite/detection gates use the frame-start mask.**  The live $2001
  at render time can already be blanked by a spilled NMI whose effect
  lands after the visible span on hardware; gating the sprite pass and
  sprite-0 detection on it made Zelda's HUD items vanish on alternating
  room-draw frames (2,890-px flicker, caught by extending the HUD
  invariant to the room-draw window) and killed the split on those
  frames.  All sprite enable/edge gates now use the end_vblank latch.
- **The post-sprite-0 band renders the split batch's captured scroll**
  (ctrl/t/fine_x recorded at each post-hit scroll write) instead of the
  live registers, so a spilled NMI's scroll setup for the NEXT frame
  cannot clobber the split.  Its mask is the frame-start latch -- the
  first cut captured the live mask and rendered the band as backdrop
  (mostly-black transitions, user-caught).
- **Sprite evaluation's 8-per-scanline limit** (NESDEV sprite
  evaluation): first eight in-range sprites per line in OAM order take
  the slots, transparent pixels included; both sprite paths honor it
  (the ordered MMC2 path also stops fetching after eight -- dropped
  sprites cannot clock the latches).  Crowded lines now drop out the
  way OAM-rotating games expect.
- **Backdrop override** (NESDEV palettes): a fully blanked frame with v
  parked in $3F00-$3FFF shows that palette entry, not $3F00.
Receipts: suite/native test/invariants green; room-draw window shows a
contiguous 3-frame authentic blank instead of oscillation.

Fourth round -- update blanking, settled from side-by-side hardware
footage (user) + `tapes/tloz_flicker_tape`.  The decisive measurement:
Zelda writes $2001=00 a few rows INTO the visible span (stamped rows
2-23) at NMI end whenever the coming vblank carries VRAM work -- ON THE
REFERENCE BACKEND TOO, so it is game truth, not an engine-shortcut
artifact.  Model: a span's last on->off row is recorded
(span_blank_row; an in-span re-enable cancels); a frame that ends
blanked renders content above that row and backdrop below (post-sprite
memset, backdrop-override aware).  Partial blanks preserve the HUD rows
above the cut, exactly like hardware's menu/letter updates; a span that
STARTS blanked is a whole-frame blank, and leaving a whole-frame blank
takes one clean span (prev_full_blank holdover) because a load's tail
can alias one fully-enabled span over a half-drawn nametable -- that
was the single flicker frame at a room load's end.  Two failed
intermediate models are recorded here deliberately: ends-blanked ->
whole-frame-black (30Hz alternation: loads re-enable every vblank) and
off-observation hysteresis (aliases at every threshold; also blanked
the sword-cave typewriter, whose letter updates blank every 7th frame).
Receipt machinery that settled it: flicker_metric.py -- classify every
frame BLANK/content over a tape, flag short content runs inside blank
episodes and 1-frame black blinks; multi-frame black holds at loads are
hardware behavior.

The last flicker (user: cave entry chopped the HUD) was OURS, not the
game's: Zelda's NMI polls $2002 for the sprite-0 hit before blanking
BELOW the HUD (reference backend: every load blank stamps row 56; the
HUD survives, matching hardware footage).  The blank early-outs had
been resetting sprite0_scanline, so during loads the console scheduled
no sprite-0 event, the $2002 read lease never refreshed, the poll spun
to a budget boundary, and the blank landed at rows 2-12 -- through the
HUD.  Blanked renders now keep the sprite-0 prediction (the rows above
the blank row still render and hit on hardware).  After the fix the
weights backend stamps row 56 like the reference; the residual low-row
blanks are the menu screens (no sprite 0 -- authentic early blanks).
METHOD, for the next such bug: when weights-vs-hardware behavior
diverges, replay the same tape on --backend reference (no leases, no
windows, same base-cycle clock) -- identical rows mean game truth,
different rows mean an engine-shortcut artifact.  Receipts: both Zelda
tapes 0 short runs in the flicker metric, cave-entry probe shows HUD
intact through the load partials with a 3-frame authentic full black
before the reveal; suite/native test/invariants green; tape cycles:
spade/CV3 byte-identical, LoZ +1 / SMB -2 (sprite-0 events on blanked
frames now persist rather than vanish).  Also this round:
8-per-scanline sprite evaluation validated by TLoZ's deliberate
masking (dungeon doors hide Link behind 8 earlier-OAM sprites) and
greyscale ($2001 bit 0).  A forced-blank
render pass (mask-off rows as backdrop) was tried and REVERTED:
mid-frame $2001 blank rows inherit a whole frame of accumulated
base-cycle clock error, the blanks that are invisible on hardware
(extended-vblank room draws) landed mid-screen, and Zelda transitions
flickered -- user-verified regression.  Receipts: 69 tests + native
hardware test (new latch-0/reload-anchor assertions) green, ruff
clean, HUD/spade invariants green, tapes: spade 580, LoZ 837, SMB
1040, CV3 552 fps (SMB/LoZ cycle counts unchanged).  Zelda's transient
transition color errors remain open -- no mid-span palette writes in
the tape, mirroring flips bracket horizontal scrolls (1-frame class),
next step needs a precise symptom location from play.

Receipts: 69 Python tests + the native hardware test green (new MMC3
delta-chain assertions), ruff clean; numeric frame invariants on the
tapes — Zelda HUD rows 0-63 byte-stable across both transition windows
(and the check fails if row 64 joins it: the split lands exactly at the
HUD edge), spade header rows 0-30 + HUD rows 193-239 byte-stable while
all three strips scroll every frame.  Tape fps (post-C++-console-port
baselines; the port itself landed unlogged the day before): SMB 1037,
SMB3 563-572, spade 563, CV3 546, LoZ 809; smokes: Punch-Out 522,
Zelda2 743, MM6 822, FF 970.  Diagnosis rule that worked: a transient
glitch that settles correctly is mid-frame split state, not steady-state
logic.

**START HERE (2026-07-13, fused memory reads — the construction diet):**

- The census that reordered everything: ~80% of a median opcode expert's
  emitted lines were `memory_at`'s 16-group fabric (range gates, exact
  masks, combine multiplies — ~300 dims per logical read), and RTI/PLP
  were dominated by `low_bit`'s per-flag 256-value match ladders.  Both
  levers landed:
- **Fused memory reads** (`graph/fold.py:masked_exact_read`): expert folds
  rebuild every `memory_at` site as ONE 64K-span lookup (memory_group
  FUSED) plus the exact-address mask, from raw-ingredient annotations the
  construction leaves on the site persist.  Identical output by receipt
  (an exact hit answers from its own group either way; the mask zeroes
  everything else).  The union keeps its 16 groups — their purpose is the
  f32 attention-score range of the exported artifact, which the exact-int
  expert engines never compute.  LDA zp fold: 830 → 140 live dims.
- **Pulled-P threshold ladder** (`interpreter.pulled_flag_bits`): all
  eight flag bits of a pulled byte from one shared 255-stepglu ladder
  (each bit a telescoping run sum, depth one) instead of eight separate
  256-value match ladders.  The union model more than halved: d_model
  1762 → 702, heads 881 → 351 (layers 74 → 75); RTI's expert fell
  14.5K → 2.4K emitted lines.
- Core totals: 1,080,420 → 260,135 emitted lines (median opcode expert
  3,922 → 863, max 14,557 → 2,783); engine build 84 → 24 s.
- Receipts: 128 tests + fold receipts green, ruff clean; **full Klaus
  MATCHED through the weights at 406K steps/s (75.4 s — the record, was
  344K)**; full nestest MATCHED; **full-tape lockstep weights vs reference
  byte-identical on SMB3 (8,745 frames) and CV3 (5,293 frames)**.
  Gameplay tapes (segment fps): **SMB3 753-906, every segment** (was
  170-275), **CV3 386-485** (was 150-207), **SMB ~1,050** (was 392-475);
  SMB3 fully-warm tape total 745 fps (11.7 s).  A tape total can hide
  ~20 s of one-time boot folding whenever the proposal sidecar changed —
  report segments and totals separately.  The interactive app reads
  ~4 ms/frame for the same work because 60 Hz pacing runs 1.3 ms bursts
  between 15 ms idles (DVFS: measured 1.30 → 4.30 ms/frame on identical
  frames with a 14.5 ms sleep inserted); its stats line now says "at
  60 Hz pacing" instead of claiming uncapped headroom.
- **Same-day follow-up — hygiene emission peephole + MILP realignment.**
  The post-fusion census showed ~half of every expert forward was slot
  hygiene: erase/move rows emitted as capture+apply PAIRS (the two-phase
  form exists only so reads precede the phase's writes).  plan_model now
  scans each phase in apply order and collapses any row whose operands
  are provably untouched into one statement (an erase with an unwritten
  target becomes plain assignment): emitted erase pairs 38,977 → 28,
  total 255,604 → 190,084 lines (−26%), hot experts −23%, **Klaus 406K →
  420K steps/s (72.9 s)**; tapes unchanged within noise (frames are no
  longer forward-bound).  The scheduler MILP also got its objective
  realigned: the old "hygiene" objective priced only attention-side
  passthroughs and measured as a WASH against the greedy sweep (the
  solver dumped carries into the unpriced, 3x-larger FFN pool);
  `objective="lifetime"` (total dim lifetime, linear in existing
  variables) prices both sides and wins −10-20% passthrough entries per
  expert at 15-80 s/expert on post-diet sizes.  It stays the opt-in
  certificate (`expert_schedule="milp"`): the schedulable pool (moves)
  is only ~7% of expert lines now, so default builds keep the greedy.
  A lifetime-cost greedy variant measured mixed (one expert −13%) and
  was reverted.
- **Scheduler de-noised (same day):** the greedy hygiene sweep and the
  MILP's width/attention-hygiene objectives were excised.  The sweep,
  re-measured on the honest metric, was the partial-objective trap in
  greedy form: fewer attention passthroughs, MORE total passthroughs
  (union 3,561 vs 3,480 without it) and a WIDER union (d_model 702 vs
  634).  Emitted lines post-peephole were mixed (experts +6%, union
  −7% without it) and Klaus was a wash — DEMONSTRATED with 3 runs per
  config: no-sweep 417.2-423.9K vs sweep 413.8-424.0K steps/s, fully
  overlapping at ±1% run noise, no-sweep ahead on means — so simplicity
  and the smaller exported model won: **union d_model is now 634
  (317 heads)**.  milp_schedule keeps ONE objective (total lifetime +
  attention-passthrough tie-break, width-capped); the width objective,
  attention-only hygiene objective, and max_ffn machinery are gone, and
  so is the width-era diagnostic stratum (per-boundary matrix-rank
  "linear width" analyses, per-layer/cut report dumps, the plan.yaml
  writer, and the consumer-less interval_coloring): milp.py 962 → 654
  lines.  Receipts: 128 tests green, full Klaus MATCHED, SMB3 tape
  segments ~906 fps, milp_schedule exercised end-to-end post-slim
  (A5: 120 → 104 passthrough entries, 28 s).
- Two smaller landings the same day: device **write journaling** (PPU regs
  1-7 + APU $4000-$400F apply natively and drain in order at run()
  returns instead of pausing; NMI-edge/IRQ-ack/DMA/strobe registers still
  pause; journaled writes natively kill their own device's read leases,
  mirroring drop_device_leases.  Measured neutral-to-small on the tapes,
  kept for the pause-count reduction), and `interpreter.py` joined the
  engine cache digest (it shapes the emitted model; a graph edit
  previously reused stale engines silently).
- **Punch-Out weights crash, fixed (user report; latent since the
  2026-07-12 native-banking arc, NOT this session's changes -- it
  reproduces at that commit).**  The runner's write-position mirror
  disagreed with the engine in both region ops: `inject_region` inserts
  one WRITE token per byte engine-side but never advanced `self.pos`,
  and `map_region` advanced `self.pos` though `map_write` stamps state
  without touching the sequence.  Every native run() resyncs the mirror,
  which masked both -- until a mapper does a region op AND a single-byte
  poke inside ONE pause servicing.  MMC2/MMC4 are the only such mappers
  left (bank switch + clobbered-register repair) after MMC1/MMC3 went
  native and MMC5 moved to map_region, so only Punch-Out could trip the
  engine's strict `seq == pos` check.  Receipts: Punch-Out 240 frames
  byte- AND cycle-identical weights vs reference; CV3 full tape still
  byte-identical (the map_region side); regression test
  (`test_region_ops_keep_write_position_in_sync`); 129 tests green.
  Coverage hole exposed: no MMC2/MMC4 example has an input tape --
  record a Punch-Out tape next time the app is up.

- **The C++ kernel backend is retired (supersedes the 2026-07-11 "none
  are deletable" decision, per explicit user call).**  Its two roles were
  subsumed: for verification, reference ⇔ weights receipts are strictly
  stronger (the oracle shares no code with the graph pipeline, while the
  kernel shared the union graph, the evaluator's schedule, run_protocol,
  classify, and the hull header with the weights engine — whole bug
  classes were invisible to kernel ⇔ weights agreement); for speed, the
  weights engine became the fastest backend once the expert/window/fusion
  work landed, and the kernel got none of it.  The cost was real: 805
  lines emitting a parallel engine (every protocol change made twice —
  the write journal landed identically in kernel.py and engine_gen.py the
  same day), 51 MB of _kernel_cache, and a second bcd=True build.
  Deleted kernel.py/test_kernel.py/_kernel_cache; test_trace_convention,
  test_mmio, and test_console retargeted to the weights engine (device
  and IRQ scenarios keep three-backend coverage: reference/graph/
  weights); klaus --runner cpp/cpp-full and nestest --runner cpp removed;
  Console default backend is now "weights"; **KernelBus renamed
  EngineBus** (it drives the weights engine).  The chain remains
  complete: golden logs ⇔ reference ⇔ graph ⇔ weights.  Follow-up
  de-noising (2026-07-14): `backends/evaluator.py` renamed
  `backends/graph.py` (matching the backend's name everywhere else),
  `run_protocol.py` moved into `backends/weights/` (its only consumer
  since the kernel left), and the dead per-step fallbacks in the Klaus
  and nestest weights runners removed (no engine without a native run
  loop can exist).  PLAYBOOK's chain line updated.
- **Refactor sweep (2026-07-14, ongoing before the final phase):** a
  function-length census ranked the spaghetti; the worst offender was
  engine_gen.generate_source (821 lines of closures-in-closures sharing
  a dozen nonlocals).  plan_model/fuse_direct/row_int hoisted to module
  level with an explicit EmitterContext; generate_source is 446 lines of
  linear orchestration now.  **Receipt: emitted source byte-identical
  pre/post** (technique, recreate as needed: wrap generate_source with a
  capturing shim, build a tiny 2-expert machine so both the core and a
  sequence module emit, save the sources, diff after each stage).  Also:
  EngineBus.run's drains
  extracted to methods, app.py's dead stats path removed.  The
  fold()/fold_sequence() liveness+materialization dedup landed with the
  same byte-identity receipt: walk_live() + GraphMaterializer (counters
  and lookup provenance accumulate across per-step calls; same-lookup
  dedup stays per call); fold 182 → 97 lines, fold_sequence 316 → 231.
  Deliberately NOT refactored: interpreter.build (its statement order IS
  the schedule's dim-creation order, and auto_name(locals()) derives dim
  names from its locals) and weights.build_model (linear staged
  construction, not closure spaghetti -- long but readable).

**Previous START HERE (2026-07-12, mapper + hot-shape optimization landed):**

- MMC1/MMC3 PRG writes now decode inside the native run loop.  The ROM is
  resident engine-side, changed slots are mapped without Python pauses, and
  only CHR/IRQ/mirroring side effects drain to the Python mapper.  MMC5 and
  other Python-serviced bank changes use an in-place mapped-memory operation:
  a bank switch no longer manufactures 8/16K fictional transformer WRITE
  tokens.  Measured token growth fell from ~45K→7K/frame on SMB3 and
  ~69K→12K/frame on CV3.
- Canonical shape experts are now the zero-compile baseline, not the final
  tier.  Their executions feed the existing cumulative JIT heat counter;
  sustained-hot concrete instances persist as safe-prefix proposals and fold
  into operand-pinned per-ROM experts on the next boot.  This was the major
  mapper-game lever: SMB3's dominant symbolic `LDA zp; BPL` expert was 24
  layers / width 280; its two hot pinned instances are 16 layers / width 58.
  The controlled tape window rose from ~238 to ~469 fps.  CV3's safe-prefix
  promotion reached ~398 fps in the controlled window (the interactive app's
  own whole-session ms/frame remains the authoritative gameplay number).
- Window discovery now considers six instructions and admits later reads only
  when folded address bounds prove they stay outside watched devices and no
  earlier write needs intra-forward feedback.  The proposal generator stops
  at that hazard and retains the longest safe prefix instead of losing the
  whole hot region.
- Generated compilation is split into 12 balanced translation units with two
  compiler jobs by default (`TRANSFORMER_NES_COMPILE_SHARDS` and
  `TRANSFORMER_NES_COMPILE_JOBS` tune them independently).  Compile layout is
  cache metadata, so stale giant-shard artifacts are regenerated.  This fixes
  both invalid C++ brace emission and the previous ~30 GB parallel frontend
  blow-up mechanism.
- Temporary route/shape/expert benchmark counters were removed after finding
  the promotion lever.  Production runtime retains only counters that drive
  JIT decisions.  `KernelBus.run(trace=False)` also no longer retains result
  dictionaries needed only by trace concatenation.

Next performance work, in evidence order:

1. Re-profile the tapes post-fusion: the per-instruction mass is ~4x
   cheaper, so the wall has moved — measure before choosing the next lever
   (candidate suspects: remaining pause round trips, state-register
   argmax traffic, hygiene erase/move share of the surviving lines).
2. Promote more hot canonical instances and improve proposal ranking/capacity
   using persisted execution value, without reintroducing passive telemetry.
3. Move coherent MMC5 CPU register/PRG-RAM state native only where boundary
   profiles show a real remaining cost; mapped-byte copying itself measured at
   only microseconds and is not currently the main wall.
4. DONE 2026-07-13 (fused memory reads + pulled-P ladder, above) — reduce
   folded expert depth/width.  True PRG pointer indirection remains open
   if remap volume ever becomes a measurable share.

## 2026-07-12 inter-ROM performance — every fast path made bank-aware

Five landings, each verified by the full suite (127 tests green, ruff
clean), 120/120 frame identity kernel⇔weights on ALL EIGHT example ROMs,
and a 2,000-frame SMB3 soak (481 fps, zero checkpoints, 95.7M tokens,
RSS byte-flat):

- **trace=False run mode** (run_protocol, both engines): the shared run
  loop's 12 per-step trace vectors + write arrays and KernelBus's
  per-frame np.concatenate were pure overhead for gameplay (the console
  reads only reason/cycles; banked ROMs pause 60-90x/frame on mapper and
  PPU register writes).  Console.run_until passes trace=False; run()
  always returns "steps" so the bus no longer needs the arrays to count.
  Lockstep/receipt paths keep the traced default.
- **Checkpoint reseeds retired for the native engines.**  Verified in
  the compiled artifacts that NOTHING grows with the token log anymore:
  the weights engine's insert_write is O(1) memstruct only and its
  declared hulls[] were never inserted into or queried (they are no
  longer emitted; hull_seen survives as the live-insert guard), and the
  kernel's g_set clears per step.  Console checkpoint_tokens=None now
  means auto: OFF for kernel/weights, 2M for the graph evaluator whose
  Python log does grow.  boot_image seeds through a native seed_image
  (one call, nonzero bytes) instead of ~50K per-byte pybind calls.
- **Anchored shape windows everywhere** (root cause of the "jit window"
  log spam: match_shape_windows required loop closure to head, so
  straight-line matches could only ever arrive via JIT warmup — 512
  fallback steps + poll latency per pc, re-paid every boot).  Dispatch
  is correct and byte-verified for straight-line matches, and a cold
  table entry costs nothing, so the boot scan now adds EVERY anchored
  match (JMP self-spins keep operand == head); inject_region adds the
  injected region's matches immediately (memoized per bank content).
  The JIT tier now only ever fires for code copied to RAM at runtime.
- **All-bank sequence windows**: mappers expose prg_scan_regions()
  (every ROM bank at every base it can map: MMC1 16K x2 bases, MMC2/4
  the $8000 switch slot, MMC3 three 8K slots, MMC5 four); the console
  scans them all (0.05-0.25s, SMB3 72 / CV3 60 / Zelda 41 unique windows
  vs 9 boot-only before), dedupes on (head, code), caps at 48, and the
  per-ROM module folds the lot.  Module dispatch became a switch on
  st_pc (window bytes arbitrate within a head across banks; expert names
  carry the window hex to keep symbols unique).  A window for an
  unmapped bank is inert by byte-verification.
- **Queued multi-slot read leases**: the pause profile showed controller
  serial reads dominate device-read pauses on every ROM (32/frame at
  $4016/$4017; MMC5 $5204 was only 2/frame — lease idea rejected).  The
  lease grew a 4-slot table (interleaved $4016/$4017 bit pairs thrashed
  the single slot — CV3 stayed at 38 read pauses/frame until the second
  slot landed) where each slot repeats one value ($2002, $4017 open bus
  via an ApuFrameCounter shim) or serves a finite queue (the controller
  leases its remaining report bits; a served-read counter + bus flush
  advances the Python device model before its next read, so partial
  consumption stays exact).  Read pauses: SMB3 37 → 9.2/frame, CV3 38 →
  10.5/frame.  Receipt beyond frame identity: the interactive nestest
  suite (Start-button driven through the lease path) stays green.

`nes.console --profile MIN` lands as the shape-library gate: zero
fallback pcs >= 2K steps/300 frames on cv3/mm6/zelda2/smb3, so no
library additions are currently justified — the residual CPU time on
the 200-240 fps ROMs is flat per-frame game logic (~85% native
execution, ~10K instructions/frame), which is exactly the recorded
background pinned-module JIT tier's target.

fps receipts (240-frame runs, module caches warm; previous = 2026-07-11
logs): SMB 479.8 (~350), SMB3 479.6 (~192), FF 756.2, Zelda 597.0,
Punch-Out 327.8, Mega Man 6 238.1, Zelda II 230.8, CV3 202.8.

## 2026-07-12 store-licensed windows + warm-boot JIT — the write-free wall, down

The window license was write-free, and gameplay code stores constantly;
this arc replaces it with a PROVABLE-BOUND license and adds runtime
window discovery that persists across sessions.

- **The license** (`experts.check_writes_licensed`): a store is legal
  when its folded write address provably stays below $2000, shown by
  interval evaluation over byte-valued inputs — through ReGLUs, persists,
  and LOOKUPS (registers arrive via state attention, so the bound of a
  lookup dim is the bound of its insert-side value expression, widened
  by 0 for the no-match case).  Constants, zp/zp,X, stack pushes
  ($100+sp), and low abs,X all pass; pointer-indirect targets reject
  (memory-data address values bound to 64K).  Pinned windows must also
  prove store DISJOINTNESS from their own byte range (the fold pinned
  those bytes; self-modification would execute stale semantics), which
  is why store SHAPES stay banned: a symbolic head cannot prove it.
  Non-halting and first-read-only checks stay; leading RMW is now legal
  (its read leads the window, its write is licensed).
- **Dispatch safety is static, not truncation**: emit_step_core already
  applied seq-row writes natively in order; what write-freedom really
  protected was the watch_write pause.  Licensed targets < $2000 plus a
  dispatch guard (`watch_write_lo >= 0x2000`, maintained in the shared
  run state) make a watched write inside a window impossible rather
  than handled.  Traced runs now emit seq-row w_* entries; a core-side
  seq_head bitmap keeps 128-window module dispatch misses at one array
  load.  Receipts: store-window lockstep (writes + trace + final RAM
  byte-identical to per-instruction), decline-under-low-watch (all 256
  INC $20 writes pause once a $20 watch exists, windows fall back), and
  license-rejection tests ($2007 store, self-modifying store); 128
  tests green.
- **Warm-boot JIT proposals**: sustained-hot fallback pcs (3 jit_hot
  reports, re-earning 512 steps each) with no library shape decode into
  straight-line/branch-tailed pinned-window PROPOSALS (closure not
  required — hotness is the profitability argument), persisted per ROM
  (`_engine_cache/jitwin_<sha>.json`, saved by app/tape/console at
  shutdown) and folded into the pinned module at the NEXT boot — a
  mid-run fold+compile would stall the frame loop.  Budgets bound the
  fold cost: 96 proposals/run, 64 folded/boot, 192/sidecar, overlap
  pruned.  A ROM's sidecar converges in 2-3 sessions (SMB3: 96 → 39 →
  3 → 0 new).
- **Measured honestly on the tapes**: CV3 166.0 → 173.2 fps (+4%; its
  profile had CONCENTRATED hot pcs — the MMC5 IRQ prologue, RMW loops —
  exactly what proposals catch); SMB3 182.3 vs 184.1 baseline and SMB
  403.1 vs 418.2 — neutral within session noise, because their
  fallback tier was only ~6-15% of steps spread over thousands of pcs,
  so even a total win caps below 10% and 64 windows catch half of it.
  An A/B (all windows / write-free only / none) on an SMB3 gameplay
  slice measured IDENTICAL 6.1 ms/frame across all three: the
  per-instruction tier is no longer where banked-ROM frame time lives.
  The remaining lever is pause round trips — native PRG banking, START
  HERE item 2.
- **Cross-substrate receipt**: 8,745 frames byte-identical weights vs
  kernel under the full SMB3 gameplay tape with store windows and
  proposals live; full Klaus MATCHED through the weights post-change
  (30,646,177 steps, decimal, 92.9 s — 330K steps/s).

## 2026-07-12 input tapes — benchmarking gameplay instead of title screens

Per explicit user call: from-boot fps runs bench the START SCREEN (idle
poll loops — the best case for the window tiers), not moment-to-moment
play.  Two pieces landed:

- **`app --record PATH`**: serializes every input transition (press AND
  release) as `<frame> <mask hex> <names>` — a complete tape; `--debug`
  button logs also carry the frame index now.  Recorded tapes live in
  `tapes/` (smb ~83 s, smb3 ~144 s, cv3 ~86 s of real play).
- **`nes-run ROM --tape TAPE`**: the tape parser, frame scheduler,
  controller application, and uncapped replay timer now live in native
  `tape.cpp`. `app.py` only selects the runtime backend and reports the
  result; headless tape runs do not initialize the Sokol frontend. Use
  `--tail N` after the final event and `--frames N` to cap a smoke run.

Gameplay numbers (clean, idle machine): **SMB 418 fps** (segments
392-475), **SMB3 184** (170-275; 10.3K instr/frame), **CV3 166**
(150-207).  The SMB3 gameplay breakdown that reordered the lever list:
6.0 ms/frame = 4.9 native + 1.1 Python, at **163 run-loop pauses/frame**
(148 watched writes; 15.3 PRG bank switches) — pause round trips, not
instruction execution, are the top remaining cost; fallback is 15-25%
of instructions as a wide flat store-heavy mass (histogram in the START
HERE item).  Two pause-cost trims landed with it: untraced run() results
skip the 12 empty per-step arrays, and lease invalidation is per-device
(a mapper write no longer clears the controller/PPU leases; native
clear_read_lease_addr).  Worth ~0.4 ms/frame on SMB3 — the structural
fixes are the two START HERE items.

Receipts: 127 tests green, ruff clean; **5,092 frames byte-identical
weights vs reference under the real SMB tape** (reference runs the
Python bus with NO lease machinery — the strongest lease-invisibility
receipt); tape totals above; CV3 118-fps outlier reproduced as
bench-environment contention, 166 on the idle re-run.

## 2026-07-11 MMC5 PRG-RAM into $8000-$DFFF

The `$5114-$5116` PRG windows now honor their bit-7 ROM/RAM select: with
the bit clear a window maps a PRG-RAM bank (low bits, same 64 KiB space as
the `$6000` window) instead of ROM; `$5117`/`$E000` stays ROM. A per-mode
slot→source table drives which register feeds each 8 KiB slot, so RAM
windows compose with all four PRG banking modes. Injection identity is now
`(bank, is_ram)` per slot: switching a window ROM↔RAM or changing its RAM
bank re-injects the right region into the CPU image, and stray writes to a
locked RAM window are repaired from RAM rather than ROM. Writes still gate
on the `$5102/$5103` unlock. Receipt: `$8000` RAM store, ROM/RAM toggle
with RAM persistence, and lock-time repair. 52 NES tests green, Ruff clean.

Still deliberately pending for full MMC5 coverage: vertical split, precise
repeated-IRQ/in-frame timing, and the two pulse + PCM expansion-audio
channels.

Castlevania's raster split is itself chained: four visible IRQs at
41/42/52/53 create the HUD, a one-line transition, an intentional blank
separator, a new `$2006/$2005` playfield origin, and restored live CHR.
The singleton `mid_*` state discarded the first three captures, allowing
temporary `$7F` banks to leak beneath the HUD and applying upward-scroll
state to the wrong rows. The PPU now keeps ordered raster snapshots;
frame-start, every intermediate band, and final live state retain their
own loopy registers, `$5105` map, and MMC5 background/sprite CHR windows.

## 2026-07-11 MMC5 foundation — Castlevania III reaches its title

Mapper 5 now has all four PRG and CHR banking modes, `$5100-$5130`
configuration/bank registers, ExRAM CPU access, scanline IRQ registers,
and the 8×8 multiplier. `mmc5.h` owns the native PPU-side substitutions:
four independent `$5105` nametable sources, ExRAM and fill-mode tile/
attribute data, plus independent sprite/background CHR windows used by
8×16 mode. Castlevania III runs continuously and reaches a recognizable,
correctly composed title scene; before native ExRAM/fill it rendered only
a sparse border, and before independent CHR views its title patterns were
garbled.

Extended-attribute mode now supplies per-tile palette and 4 KiB CHR
selection from ExRAM, and the fixed `$6000-$7FFF` PRG-RAM window supports
bank selection plus the `$5102/$5103` write-unlock sequence.

## 2026-07-11 MMC4 — FxROM completion + native latch interface

Mapper 10 adds its switchable 16 KiB `$8000-$BFFF` PRG bank, fixed final
16 KiB bank, fixed `$6000-$7FFF` RAM, four 4 KiB CHR registers, and
runtime mirroring. Its PPU-side rules live in header-only `mmc4.h`:
unlike MMC2, both pattern-table halves respond symmetrically across all
eight rows of the `$FD/$FE` switching tiles.

The PPU now talks through one `ChrLatchMapper` interface in
`chr_latch.h`; `mmc2.h` and `mmc4.h` independently own mapping, latch
state, and post-fetch trigger decoding. The renderer retains only the
mapper-agnostic ordered PPU fetch path. Receipts cover MMC4 PRG switching,
fixed-bank preservation, write repair, CHR registers, mirroring, and the
low-pattern row-range difference from MMC2.

## 2026-07-11 MMC2 — Punch-Out!! and fetch-ordered CHR latches

Mapper 9/PxROM adds its switchable 8 KiB PRG window, three fixed upper
windows, four 4 KiB CHR registers, and runtime mirroring. The PPU owns
the hardware FD/FE latches: pattern fetches in `$0FD8/$0FE8` and
`$1FD8-$1FDF/$1FE8-$1FEF` select the corresponding lower/upper bank.
Unlike fixed-bank rendering, these fetches are order-dependent, so MMC2
uses a scanline path that fetches background then OAM-order sprites. It
also performs the 2C02's 34 background tile fetches (two beyond the
visible edge): Punch-Out deliberately places latch tiles in that hidden
prefetch area. The trigger tile's two pattern bytes use the old bank and
only subsequent reads see the new latch. Omitting those hidden fetches
left the wrong bank selected and produced garbage across the boxing
screen's top rows. F1 dumps include all four mapper CHR registers and
both live latch states.

The native lifetime audit was also closed out: APU DMC callbacks clear
on exceptional exits, and PPU debug arrays retain their native PPU owner
instead of exposing ownerless memoryviews. Receipts: mapper PRG/fixed-bank/
repair, CHR-register/mirroring, exact post-fetch latch ranges, a hidden-
prefetch-to-next-scanline scene, owner retention, and exceptional callback
release tests.

## 2026-07-11 APU completion — DMC samples + shared IRQ line

The fifth channel is live: `$4010-$4013` program the NTSC DMC rate,
direct DAC load, sample address, and sample length; `$4015` starts/stops
playback and reports activity; the output unit shifts fetched bytes into
the nonlinear TND mixer. Address wrap, looping, the 7-bit saturating DAC,
DMC IRQ acknowledgement, and four-cycle DMA steals are implemented. DMC
fetches read the console's current CPU image, so mapper bank changes and
backend handoffs remain transparent. Because audio is still generated in
one end-of-frame batch, DMA steals and IRQ visibility occur at that
boundary (and reduce the following frame's CPU budget).

Frame and DMC IRQ flags now drive the machine model's existing IRQ input.
The console tracks independent sources for the shared level-sensitive
line, so acknowledging MMC3 cannot clear a pending APU IRQ and
acknowledging the APU cannot clear MMC3.

The follow-up code review covered APU/PPU state ownership, buffer
lifetimes, duplicated responsibilities, IRQ ownership, and Python/C++
lifetime boundaries. Two concrete APU smells were removed: returned audio
now owns its sample storage instead of viewing a reusable C++ vector, and
the DMC reader callback is frame-scoped instead of forming a retained
Python/C++ ownership cycle. The PPU's output was already owned and its
loopy/split-band state has one clear owner, so no speculative rewrite was
made. Receipts: 3 new DMC tests plus a shared-IRQ-source regression; 40 NES
tests green, and Ruff clean.

## 2026-07-11 APU — native audio, sound across games

The last console subsystem, built in the PPU's mold: `nes/apu.cpp` is
fixed silicon (register file $4000-$4017, the channels, a frame
sequencer, a per-frame sample generator with the 2A03's nonlinear DAC +
a one-pole DC blocker), `nes/apu.py` is the digest-caching loader, and
the console attaches it as a bus device and calls `end_frame(cycles)`
once per frame to pull that frame's mono float samples.  The app pushes
them to `sokol_audio` (push mode, mono, AudioToolbox) from the frame
callback -- sokol's own audio thread drains the ring buffer, so no GIL
or audio-thread hazard.

MMIO layout worked out with no bus changes: the Controller was already
"port 1 only" ($4017 reads returned open-bus $40), so the APU took
$4000-$4013, $4015, and $4017 with zero range conflicts (controller
narrowed to $4016).

Channels: pulse x2 (duty, envelope, sweep, length), triangle (linear
counter + length), noise (envelope, LFSR, length) -- essentially all NES
music.  Frame sequencer 4/5-step with the NTSC trigger cycles; the frame
IRQ flag is readable at $4015 (read-clear) but deliberately NOT driven
onto the CPU IRQ line (the frame-batch model can't time it precisely,
and most games inhibit it at boot).  DMC (sample) channel stubbed:
registers accepted, no output/DMA/IRQ.

Frame-level approximation, documented: register writes take effect at
their instant but a frame's samples generate in one end-of-frame batch,
so once-per-frame (NMI) music -- the overwhelming majority -- is exact
while a rapid intra-frame sweep resolves at frame granularity.

Receipts: 7 new APU tests (A440 tone within 5 Hz + DC-centered, length
counter silences / halt sustains / $4015 disable, $4015 status +
frame-IRQ read-clear, IRQ inhibit); 36 nes tests green (attaching the
APU left frame identity and nestest byte-identical -- it drives no CPU
state); user-confirmed audio across multiple games.  Next: wire the DMC
(SMB's percussion is DMC PCM) and, if a game needs it, the frame/DMC
IRQ onto the CPU line.

## 2026-07-11 IRQ through the machine model + the PPU's v register — SMB3 fully playable

The MMC3 arc's second half: the scanline IRQ, delivered as graph logic
like the NMI, plus every PPU model gap SMB3's status bar exposed.
Debugged live against gameplay with an F1 state-dump loop (app.py
`debug_dump`: frame + nametables + band latches + mapper state + the
IRQ handler's captured register writes); each fix below was pinned by a
dump receipt before it was written.

- **IRQ in the machine model.**  The graph gains `in_irq`: one shared
  interrupt sequence (`is_int = is_nmi + is_irq`), vectors muxed
  $FFFA/$FFFE — everything else (pushes, I=1, B-clear) identical, so
  the union grew one input and no new machinery.  Delivery is gated at
  the step boundary by `irq_ready()` (pending, no NMI ahead, I clear —
  faithful 6502 polling); the flag is level-held until deliverable.
  All four backends deliver identically: reference (`interrupt_step`),
  evaluator, kernel, weights — with every expert/shape/sequence
  dispatch tier guarded so a deliverable IRQ always reaches the
  per-instruction path.  Receipts: 3 new lockstep tests (graph/kernel/
  weights vs reference) including SEI-deferral; a shape-window spin
  test (IRQ into a JIT-dispatched DEX/BNE loop) delivers on both
  engines.
- **The zero-cycle hang class, dead.**  Non-opcode bytes had CYCLES=0,
  so garbage execution (a bad pointer) could spin the native run loop
  forever inside one call — unbounded trace growth (observed: 12 GB)
  with Python never regaining control.  Unknown bytes now cost 2
  cycles: any runaway still returns at the frame boundary.  A 100K-
  pause runaway guard in KernelBus.run stays as a permanent tripwire.
- **MMC3 IRQ pacing** (console): the counter fires once per frame at
  the latch scanline, re-validated at delivery (rendering still on,
  mapper still armed — on silicon the counter only clocks while
  rendering); $E000 acks through to the machine line.
- **The PPU learns v.**  The write-capture receipt showed SMB3's IRQ
  handler loads the HUD's vertical position with a mid-frame $2006
  pair — on hardware the visible y comes from loopy **v** (self-
  incrementing, reloaded from t only by that pair), while $2005/$2000
  writes only arm t for the next frame.  The band model now latches v
  at the split and pins the post-split band's y to it (absolute),
  x from t — the ground-level HUD had only ever worked because t.y
  happened to be 239.  Bands also carry their own $2001 mask bits,
  mirroring, and CHR banks (the handler swaps all three for the HUD).
- **Hardware sprite compositing.**  Sprites composite lowest-index-
  first with per-pixel ownership (the claiming sprite's priority bit
  alone decides front-vs-behind — emerging powerups stay obscured),
  fetch through the banks of the band their row falls in (SMB3 blanks
  sprites under the HUD by switching to an empty CHR page), and honor
  the $2001 left-edge masks, sprite-0 included.
- **App presentation + pacing**: NTSC overscan crop (visible 240x224
  — the masked left column and SMB3's right-edge tile seam are gone),
  and a wall-clock budget (half the display cadence) per frame
  callback so a backend slower than realtime degrades to a slow game
  instead of starving the Cocoa event loop (the beach-ball class).

Receipts: 29 nes tests green (nestest PPM byte-identical throughout),
93 total; SMB3 played through world 1 with correct HUD at ground and
in the sky, sprites obscured correctly, no hang — plus SMB/FF/nestest
unchanged.  MMC1-class games keep byte-identical rendering (the new
band fields collapse to the old single pass when nothing splits).

## 2026-07-11 mappers — MMC1 + MMC3, banked ROMs on every backend

The cartridge stops being NROM-only.  The design constraint: the
machines boot from a flat 64 KiB image and run natively, so PRG bank
switching must rewrite machine memory at runtime.  Mapper registers
are writes into ROM space, served by a **write-only bus watch** on
$8000-$FFFF (`attach(..., writes_only=True)` on both buses — a read
watch there would pause on every ROM table read).  The run loop
applies a store before pausing, so the handler repairs the clobbered
PRG byte, then injects any bank change; `nes/mappers.py` owns the
protocols, the console wires per-backend hooks (poke/poke_region/
mirroring/CHR banks) after every boot and handoff.

- **MMC1** (serial shift register): PRG modes (32K / fix-first /
  fix-last), mirroring incl. single-screen (the PPU's nametable
  arrangement is runtime-switchable now), CHR-RAM carts only so far.
  Receipt: Final Fantasy boots, runs, and is playable in the app.
- **MMC3**: 8K PRG banks both modes, 2K/1K CHR banks via the PPU's new
  eight-slot CHR window (`set_chr_banks`; full CHR ROM of any size
  lives behind it), $A000 mirroring.  The scanline IRQ counter is
  accepted but NOT delivered — the machine model has NMI only — so
  SMB3's status-bar raster split renders garbage (warned once at
  runtime).  IRQ delivery is the recorded next step.
- **inject_region** (both generated engines + kernel): a bank switch
  was 8-16K Python→C++ `inject_write` calls (~1.1 ms each switch);
  SMB3 switches ~5x/frame and measured 21.3 ms/frame on the kernel.
  One native bulk call now: kernel 16.5 ms/frame, **weights 5.2
  ms/frame — SMB3 through the transformer at 60 fps with 3x headroom**
  (the token log still grows 8K/switch; the checkpoint reseed absorbs
  it, ~8 per 240 frames).

Receipts: 93 tests green (7 new mapper-protocol tests: MMC1 serial/
modes/reset/repair, MMC3 modes/CHR arrangement/mirroring); FF and SMB3
run 240+ frames on kernel and weights without halting; SMB unchanged.

## 2026-07-11 native app + PPU scroll model — SMB is playable

Two landings, verified by play: SMB through the transformer weights on
a real keyboard, world 1-1 cleared into 1-2.

- **The Sokol frontend.**  `vendor/install_deps.sh` pins sokol
  (3743ea6, 2026-07-08) plus Dear ImGui v1.92.8 (vendored now for the
  future debug overlay; not yet compiled in).  `nes/app.cpp` is one
  Objective-C++ TU (`SOKOL_IMPL`, Metal, `SOKOL_NO_ENTRY`),
  digest-cached like the PPU: window + keyboard + one nearest-filtered
  textured quad, NES-rate pacing by accumulator against
  sapp_frame_duration (burst cap 4 frames), letterboxed viewport.
  Python stays the orchestrator, per the recorded decision:
  `python -m transformer_nes.nes.app ROM [--backend weights] [--scale N]
  [--frames N] [--debug]` — each display frame calls back into
  Console.run_frame with the controller bitmask (bit i =
  console.BUTTONS[i]; arrows = d-pad, X/Z = A/B, Enter = Start, RShift
  = Select, Esc = quit).  macOS gotcha, recorded in app.py: quitting
  goes through [NSApp terminate:], so the PROCESS EXITS INSIDE
  sapp_run — a quit_cb invoked from the sokol cleanup callback (GIL
  still held) is the last Python to execute on every quit path; the
  run stats log from there.  Receipt: interactive SMB at 60 fps with
  emulation at 4.5 ms/frame — 221 fps of headroom through the weights.

- **The PPU scroll model** — the documented "no mid-frame scroll"
  approximation, cashed in the moment a game scrolled.  Symptoms: the
  HUD scrolled with the camera; at the camera's 256-pixel page
  crossing every backend froze for good (backend-identical, so a
  console-layer defect by construction).  Root cause, pinned by a
  $2000/$2005/$2006 write log at the crossing: SMB's NMI leaves $2000
  holding mirror|incr32 (nametable 1 after the crossing) and relies on
  its closing $2006=$0000 pair + $2005 0,0 to reset the frame-start
  scroll — on real silicon those share the loopy-t register, which our
  separate scroll_x/scroll_y model didn't capture.  So the frame
  rendered from nametable 1, the status bar vanished, sprite-0
  detection returned None, the console never raised the hit, and SMB
  spun forever in its NMI at the $8150 sprite-0 poll with NMI disabled
  (ctrl=$15) — a permanent wedge.  Fix, two halves in ppu.cpp: (1)
  **loopy t + fine_x** — $2000/$2005/$2006 update one shared
  scroll/nametable state and rendering derives from it; (2) the
  **sprite-0 split render** — end_vblank latches the span-start scroll
  state, rows through the hit line render from the latch, rows below
  from the live registers (identical states collapse to the old single
  pass, and the direct-render path with no end_vblank is untouched).
  One scene test now resets $2006 before rendering, as real games do.
  Receipts: 86 tests green (nestest PPM byte-identical — the loopy
  model changes nothing for well-behaved scroll writers); scripted SMB
  with a stall-triggered jump driver crosses the page boundary and
  keeps going (camera 506+, sprite-0 stable at scanline 30 every
  frame); played through 1-1 into 1-2 interactively.

**Previous snapshot (2026-07-10, 350 fps + universal shapes):** SMB gameplay
through the transformer weights runs at **~333 fps (3.0 ms/frame)**
with its per-ROM pinned module, and at **155 fps with ZERO
ROM-specific compilation** — the shape-expert tier serves any ROM,
homebrew included, from the shipped core alone.  The levers of the
arc, each with its own log below: the all-opcode MoE (227 rebuilt
experts, no game-derived tuning set), the engine build cache
(construction ~1 s, distributable compiled artifacts), the MMIO read
lease ($2002 polls served natively instead of pausing to Python
~1,420x/frame), group-dispatch fusion (one memory query per read
instead of 16), **sequence experts** (a whole pinned instruction
window as ONE forward: the poll trio at 0.07 µs/instr), the **per-ROM
sequence module** (the compile skirt: the core compiles once per code
digest, a new ROM's pinned windows cost a ~160 KB TU — cold ~7 s, warm
~1 s), and **canonical shape experts** (15 idiom shapes folded with pc
and operands symbolic, dispatched via a runtime window table the
machine populates from its own image at boot).  Full Klaus (30,646,177
steps, decimal) MATCHED in 94.8 s; 79 tests green; frames
byte-identical across backends and dispatch tiers.  Remaining
candidates: the profile-guided JIT counter (runtime window discovery;
both dispatch substrates are landed), shape-library growth, the kernel
backend's hull-log growth (optional cleanup), the Klaus harness cost
(lockstep reference + trace marshalling, no longer the forwards).  The
performance numbers in the older sections below are historical.

**Position features excised (2026-07-11):** the graph never referenced
position/position_sq/inv_log_pos after the recency retirement (probe:
zero references across every dim expression, key, value, and query in
the bcd=True graph) — the sole live consumer was build_model's
passthrough/erase gadget, whose attend-to-self used the position
parabola.  Replaced with constant Q/K + the LATEST tie-break (every
token ties; latest == self), which the torch runner's flip+argmax
already implements.  Everything downstream died with them: the three
builtin InputDimensions and the position-keyed k_abs branch (graph/
core.py), evaluator/eval.py token seeding, the kernel's new_token
setg's and builtin ids, classify's banned_ids params, weights.py's
fixed slots 0-2 + add_position_encoding, engine_gen's pf/uses_pf
machinery and position-slot ban, the export contract's "positions stay
f64" caveat, and the torch runner's per-token log().  **The model now
has NO position encoding: token order enters only through the LATEST
tie-break.**  d_model 1766 → 1762, 881 heads.  Receipts: 85 tests
green (incl. the torch-runner export lockstep); full nestest weights
MATCHED; full Klaus kernel PASSED 45.5 s; **full Klaus weights
(decimal) MATCHED 30,646,177 steps in 92.9 s**.

**Refactor receipt (2026-07-11):** plan items 2, 4, and 5 landed (details
in the plan below).  The weights runtime forward is integer END-TO-END:
register argmax folds integer scores behind exact clear-gate zero tests,
the memory-guard f64 key test became an integer gate test, and the
position features (with their per-forward log()) left the hot path — no
double survives in the emitted step.  Receipts: 85 tests green; full
nestest MATCHED through the weights (~194K steps/s); **full Klaus MATCHED
30,646,177 steps (decimal) in 89.0 s (344K steps/s)** — the record run.

**Refactor receipt (2026-07-10):** generated runtime is exact-int32 only;
the unreachable Python step/write paths and duplicated run-protocol tables
are gone. Full automated suite: 85 passed. Full nestest matched 8,991 steps
through reference/graph/kernel/weights. Full Klaus passed 30,646,177 steps
through the kernel (40.60 s) and weights (91.3 s, 336K steps/s). Scripted SMB
matched kernel⇔weights for 180/180 frames plus RAM, cycles, PPU state, and CPU
registers; measured 70.1 fps kernel and 257.9 fps weights in the final run.

## The refactor plan (written 2026-07-10 after the dead-code excision)

Same-day excisions already landed: LATEST_ALPHA + the recency machinery,
KEY_OFFSET, the average tie-break constructor, the vestigial
matmul/dtype/np_dtype attrs (jj lylmpmvy).  What remains is structural,
in priority order — the lockstep chain is the safety net for all of it:

1. **DONE — one classification module.**  The shared numeric licenses for
   memory structures and running-argmax registers now live in
   `model/classify.py`; `engine_gen.plan_model` (matrix rows) and
   `kernel._KernelSpec` (compiled graph expressions) only adapt their local
   representations into its linear terms.  Boundary tests cover scaling,
   positional-slot bans, coefficient limits, non-finite values, and LATEST
   ties.  Receipt: all 85 tests green across the exact evaluator, generated
   kernel, generated weights engine, export runner, MMIO, and console.
   The superseded `experiments/moe_probe/` scripts and frozen JSON results
   were removed; fold receipts now compare directly with the live union graph
   instead of treating research output as a test oracle.
2. **DONE — split weights_runner.**  `experts.py` owns the three fold-and-
   build expert builders with their licenses (write-free/non-halting
   const_root checks, first-read-only, _eval_over_heads) plus the shared
   step-spec builders — the input/p-flag slot maps that were triplicated
   are one function now.  `windows.py` owns SHAPE_LIBRARY, the scanners,
   and the anchored matcher the JIT uses (match_shape_at).  The machine
   module keeps orchestration, the machine protocol, the JIT tick, and the
   CLI (632 lines, from 1,057).  The manifest/cache logic already lived
   in engine_gen; the machine only computes keys.  `experts.py` joined the
   engine-cache digest.  Receipt: 85 tests green (jj kutolxul).
3. **DONE — one runtime arithmetic.**  The generated runtime is exact int32.
   The slower int64 and expert-incompatible f32 diagnostic modes, their CLI
   and cache-key surface, dual emission branches, and the unreachable Python
   `machine_step`/WRITE fallbacks are gone. The exact evaluator remains the
   semantics diagnostic; portable f32 safetensors export is unaffected.
   Receipt so far: all six generated-weights tests pass from fresh artifacts.
4. **DONE — one C++ run loop.**  `run_protocol.py` now emits the shared
   protocol for both generated engines: the exact nearest-address memory
   structure (mem_query, parameterized only by output type), the shadow
   decoder, in_ranges, the persistent run-state members with the
   READ_MODE/CYCLES tables, the machine API, and the whole run() skeleton
   (prologue, watch/lease check, result dict) — each engine supplies only
   its locals and per-iteration step body.  Receipt: 85 tests green with
   both engines freshly recompiled (jj rrqzmomk).
5. **DONE — integer register scores** (the weights runtime's f64
   retirement, complete).  The license lives in
   `classify.integer_score_terms`: qx == qy > 0 makes the score a positive
   multiple of kx + ky, so the integer sum preserves the argmax and its >=
   ties exactly; clear-gate terms (-1e30) split out as exact zero tests on
   the gate slots, replacing every "ky > -1e29" double compare (register
   folds AND the memory-head live-insert guard).  reg_s is i64; with no
   hull head left in the model, the position features (and their
   per-forward log()) are not emitted at all — the forward is integer
   end-to-end.  Anything unlicensed keeps a hull and pf comes back, so
   classification stays a speed decision.  The KERNEL deliberately stays
   f64: it is the emission mirror of the evaluator (same IEEE doubles in
   the same order), and that definitional role is worth one double
   backend.  Receipts: 85 tests; full nestest weights MATCHED; full Klaus
   decimal MATCHED 89.0 s — the record (jj osnuzrvw).
6. **DONE — docs pass.**  Both READMEs rewritten around the artifact
   boundary (graph -> schedule -> weights vs engines vs console): retired
   references gone (weights_engine.cpp, dense/sparse paths, the recency
   term as a numerics rule), the components tables cover the new modules,
   speed/fps claims match the 2026-07-11 receipts, and the MMIO section
   documents the single run-loop protocol + read lease.
7. **DONE (2026-07-11) — PPU to C++.**  `nes/ppu.cpp` (same registers,
   same frame-at-once renderer, same documented approximations), compiled
   once and digest-cached, is THE PPU; the Python PPU was deleted once
   identity was proven.  Pre-deletion receipts, all against the Python
   reference: every register test ran on both implementations; a
   register-built scene (CHR patterns, both nametables, attributes,
   scroll, flips, priority, sprite-0) rendered byte-identical across four
   ctrl modes; a 4-frame console run matched frames/sprite-0/cycles; and
   60 frames of the real nestest ROM produced a byte-identical PPM
   (63.8 fps on the kernel backend).  Post-deletion the suite pins the
   native renderer with concrete pixel/priority/sprite-0 assertions.
   The native frontend landed same day (log above) — this item is done.

## Recorded decision: Sokol over SDL3 for the native app (2026-07-11)

Per explicit user call ("pick the lightest of the two"): the frontend is
**Sokol** — single-file, header-only libraries (`sokol_app.h` window +
input, `sokol_gfx.h` one textured quad for the 256x240 framebuffer,
`sokol_audio.h` callback audio when the APU exists).  No build system, no
vendored dylib: the headers compile into the same one-TU-and-cache style
every native piece of this repo already uses.  SDL3 is a full library
build for the same three capabilities.  App shape (next step): a frame
loop hosting the existing backends — the PPU is already native
(`nes/ppu.cpp`), the machines are dlopened C++; the remaining design
decision is whether the app embeds Python for orchestration (cheapest:
the console layer already paces frames) or the engines grow a C ABI for
a fully Python-free binary.  Start with the embedded/orchestrated form;
measure; only then consider the C ABI.  **Landed 2026-07-11 in the
orchestrated form** (log above) — 4.5 ms/frame of emulation against the
16.6 ms frame budget leaves no case for the C ABI yet.

## Recorded decision: package rename + module placement (2026-07-11)

The package is **transformer_nes** (was transformer_vm); distribution name
transformer-nes.  Receipt: 85 tests green post-rename; the emitted C++
contains no package names, so the digest-addressed compiled artifacts were
reused (one refold/emission per config, zero recompiles).  GitHub URLs in
pyproject still point at Percepta-Core/transformer-vm until the repo is
renamed.

Module placement, decided while answering "why isn't weights_runner in
model/": the architecture is three rings — target-agnostic core (graph/,
attention/, scheduler/), THE MACHINE (cpu6502/ + model/, coupled through
run_protocol and the step spec), and the console (nes/).  weights_runner
stays in cpu6502/ because it is a machine BACKEND (peer of reference/
evaluator/kernel, same machine protocol); model/ holds the graph→weights→
engine mechanisms.  Landed same day: the four backends moved into
`cpu6502/backends/` (with `_kernel_cache/` following kernel.py), and the
weights backend's support modules nested under `backends/weights/`
(runner, experts, windows) so backends/ holds the four machines plus only
the shared run_protocol.  The harnesses moved next to their assets:
`cpu6502/examples/klaus/harness.py` and `nes/examples/nestest.py`.  The
cpu6502/ top level is now just the ISA: spec, interpreter, trace.  The one acknowledged wart: engine_gen imports
cpu6502/run_protocol (and hashes cpu6502 files into its cache key), so
model/ is not truly target-agnostic — accepted rather than hidden behind
parameter injection; the emitted engine IS the 6502 machine protocol.
Backend roles, restated: reference = oracle, evaluator = the graph's
definitional semantics and the kernel's frontend, kernel = the graph at
speed + the independent twin for frame-identity receipts, weights = the
product.  None are deletable; the chain is the writeup's credibility
spine.

## Recorded decision: retire f64 (CLOSED 2026-07-11)

Both items landed: (2) the recency term left the construction on
2026-07-10 (jj ttlvorys), and (1) integer register scores landed as
refactor item 5 above — for the weights runtime.  The kernel keeps f64 as
the evaluator's emission mirror (recorded there).  Original note:

Per explicit user call: runtime f64 is vestigial.  The last hot-path
doubles are the register scores, the K-row clear-gate guards, and the
position features -- and the integer argument covers all of them (score
integer parts fit i32; replace-on->= in insertion order IS a positional
LATEST tie-break, so the fractional recency term contributes nothing at
runtime).  Two items, receipts = the full lockstep chain:

1. Integer register scores + positional ties in both generated engines
   (weights + kernel); position features drop out of the hot path with
   them.
2. Remove the 0.3·inv_log_pos recency term from the graph construction
   itself.  Every coefficient becomes integer/±0.5: the model is dyadic
   end-to-end, f32 storage is exact by inspection (no certification
   needed), the evaluator can run integer, and "hardmax + LATEST
   tie-break" becomes an explicit architecture rule -- which it already
   effectively is (the recency term vanishes below one double ulp at
   score scale; that is why TieBreak::LATEST exists at all).  Until (2)
   lands, f64 survives only as the definitional medium of the current
   graph in the evaluator/verification layer.

## 2026-07-10 standard-ML artifact — safetensors + standalone PyTorch runner

The weights are now a portable artifact.  `model/export.py`
(`python -m transformer_nes.model.export OUT_DIR`) writes
**weights.safetensors** -- per-matrix CSR (dense would be ~11 GB;
sparse is ~4 MB) plus tok/head dense -- and **machine.json** (input
slot map, readout slots, output-head rows, dims, and a prose spec of
the execution semantics INCLUDING the precision contract).  The union
model only: experts/shapes/engines are execution strategies, not the
model.  **Storage dtype defaults to f32** (loads on every GPU; WebGPU
has no f64) -- certified exact for this construction and receipt-
verified: the f32 round-trip perturbs 2,005 non-dyadic coefficients
(0.3·ilp hull-key terms) yet locksteps the oracle exactly; `--f64`
ships the theorem-free exact-constructed coefficients.  The contract's
sharp edges, recorded in the metadata: attention SCORES must not be
f32 (|ky| ~ 2^24 at a group's top addresses where the f32 ulp is 2 and
gaps are 1) -- resolve in f64, or on GPU use the exact integer route
(integer score part fits i32; the fractional recency term only breaks
ties, which a positional argmax scan does exactly); and the three
position features stay f64 (pos^2 outgrows f32 integers past ~4096).

`experiments/torch_reference/torch_runner.py` is the proof of
portability: a ~200-line standalone runner importing ONLY
torch/numpy/json/safetensors (nothing from transformer_nes) that
re-implements the semantics from the JSON -- fused Q/K/V (2 dims per
head), per-layer append-only K/V logs, **hardmax attention with exact
ties won by the latest token** (`flip + argmax`), ReGLU FFN, residual,
no normalization, half-to-even readouts -- and drives it as a 6502
machine.  Receipt (`test_export.py`, 81 tests green): the exported
weights run `verify_program` **lockstep-exact with the reference
oracle for all 40 steps** at ~16 instr/s in float64 sparse torch.
Loops, stores, JSR/RTS, stack, arithmetic, flags -- vanilla PyTorch
executing 6502 machine code off a 4.2 MB safetensors file.

This is the ablation substrate (zero heads / perturb weights / cast
dtypes and watch which instructions break) and the port spec for a
jax-js browser demo (same ~200 lines in JAX; dense-per-layer WebGPU
would run demo-speed -- the fast browser path remains compiling the
generated C++ engine to WASM).  Not yet done: an ablation example
notebook; a JAX twin of the runner; exporting bcd=True.

## 2026-07-10 kernel O(1) attention state — the hull-log growth, fixed

The "C++ problem" from the START HERE list, closed by porting the
weights engine's proven design back into `kernel.py` (the emission
mirror of engine_gen's classification):

- **Memory lookups** (all 112, over the 16 4-KiB groups) are served by
  the exact nearest-written-address structure: `insert_write` becomes
  three stores + a bit set instead of 16 hull-chain inserts, and each
  read is ONE O(1) query behind the group-dispatch check (`qi >> 12 ==
  0`) instead of 16 hull queries — the kernel never had that fusion.
- **Constant-query LATEST lookups** (start, state) fold into
  running-argmax registers at insert time; the state hull log stops
  growing per instruction entirely.  No hull group survives: the
  kernel's attention state is now O(1) in the token log, like the
  weights engine.  Eligibility is checked at generation (query a
  positive scalar multiple of [local_address, 1], values exactly
  (byte_value, addr), integer coefficients, LATEST); anything failing
  keeps its hull — classification is a speed decision, never a
  correctness one.

Receipts: 80 tests green (kernel⇔evaluator lockstep, MMIO trace
identity across backends, console pixel identity), ruff clean; full
nestest MATCHED 0.012 s; **full Klaus PASSED 30,646,177 steps in
38.9 s (788K steps/s, was ~48 s)**; **SMB on the kernel: 18.6 →
73.5 fps (13.6 ms/frame, 4.0x)** — the graph oracle now runs the
console comfortably past realtime, and every frame-compare receipt
gets ~4x cheaper on its kernel side.  Frames byte-identical kernel vs
weights 120/120, unchanged.

## 2026-07-10 profile-guided JIT — runtime window discovery, landed

The recorded JIT idea, built on the two dispatch substrates.  The run
loop counts every per-instruction (fallback) step by pc (one array
increment); `jit_hot(min_count)` reports pcs past the CUMULATIVE
threshold that have no window yet, rearming only those.  The runner
polls every `jit_interval` native run() returns and matches each hot
pc against the shape library ANCHORED at that pc -- hotness replaces
loop-closure as the profitability filter (dispatch is correct for
straight-line matches by construction), except the JMP self-spin whose
dispatch predicate demands operand == head.  Matched pcs go straight
into the window table; misses land in a rejected set cleared at
boot/reseed.  This discovers what the boot scan cannot see: **code the
program copies into RAM at runtime**, plus hot straight-line idioms.

Two bugs found by the receipt test: (1) zeroing ALL counters at each
poll meant no pc could ever accumulate past the threshold between
polls -- counts are cumulative now, only reported pcs rearm; (2) the
matcher first read `self._ram`, but the native run loop's writes never
reach the Python mirror -- a runtime-copied loop exists only
engine-side, so the matcher reads engine RAM (`read_ram`).

Receipts: 80 tests green, ruff clean; `test_jit_discovers_runtime_
window` -- a program that STAs a DEX/BNE/JMP-self loop into $0300 and
jumps there gets its window added mid-run (boot scan provably missed
it) with the trace byte-identical to the per-instruction machine
across the hot-swap; full suite unchanged; **fallback-path cost of the
counter: none** (LDA abs;JMP 0.370-0.372 µs/step vs the 0.377 record).
SMB fps read 286-317 in this session's trials vs 331-335 earlier --
attributed to bench-session thermals, since the step-time microbench
is unchanged; re-measure cold for the canonical number.  Remaining JIT
tier (recorded, not started): background *pinned-module* builds for
hot windows -- fold/compile a per-ROM module off-thread and attach_seq
when ready, upgrading shape-tier windows (~0.2 µs/instr) to pinned
(~0.07) without pausing the run.

## 2026-07-10 shape-expert engine — any ROM, zero per-ROM compile, landed

The canonical shape folds (below) are now in the engine.  A curated
**15-shape library** (`weights_runner.SHAPE_LIBRARY`: JMP self-spin,
counter/branch pairs, zp and abs load-polls, BIT $2002 idioms, the
LDA/AND/BEQ+BNE trios) compiles ONCE into the core — the shapes are
code, covered by the code digest — and dispatches through a **runtime
window table**: `shape_of_pc[65536]` + generated per-shape checks that
re-verify the opcode bytes (and the JMP self-target) against live RAM
at dispatch, so a stale entry only loses the fast path.  The machine
populates the table itself at every boot/reseed by scanning its own
image (`match_shape_windows`; closure to head is a profitability
filter, not a correctness one — the fold licenses any operands).
Dispatch precedence: pinned per-ROM module first (cheaper forward),
then the shape table, then per-instruction.

Licenses are checked, not assumed (`_build_shape_experts`, hard errors
— the library is code): only the first instruction may read memory
(spec), every step write-free + non-halting (folded constants), and
each interior step's next_pc == head + offset verified **exhaustively
over every head the window fits at** (`_eval_over_heads`, vectorized;
possible exactly because the folded next_pc may depend on nothing but
the step-0 pc persists — the evaluation cuts where the head enters via
the state-attention lookup).  The JMP spin's closure is a dispatch
check instead (operand bytes == head), sound because the window is
write-free and dispatch-verified.

Receipts: 79 tests green, ruff clean; the three-tier lockstep test
(pinned module / shape table / per-instruction) byte-identical across
polls, flag flips, fall-throughs, spin, and mid-window chunk
boundaries — all four idioms in its program match library shapes; full
nestest MATCHED 0.1 s; **full Klaus MATCHED 30,646,177 steps in
94.8 s**; SMB frames byte-identical kernel vs weights **120/120 on
both tiers**; pinned+shapes **~333 fps** (unchanged — the table costs
nothing when the module wins); **shapes-only: 154.8 fps (6.5 ms/frame)
with ZERO ROM-specific compilation** — any NES ROM, homebrew included,
gets this tier from the shipped core alone.  Micro: DEX/BNE through
its shape expert 0.23 µs/instr vs 5.8 through the union step.  Full
core build (227 opcode + 15 shape experts) 52 s, one-time, cached.

Next: the profile-guided JIT counter (count dispatch-fallback pcs in
the run loop, propose windows at runtime, match shapes / build pinned
modules in the background); shape-library growth (4-instr timing
loops, LDX/LDY polls, INC zp delays) — each addition is a one-time
core rebuild shared by every ROM.

## 2026-07-10 canonical shape folds — symbolic pc/operands, probed GREEN

Path (b)'s recorded blocker is broken.  `fold_sequence` gains
`pc_offsets`: a window folded as a *shape* pins only the opcode bytes —
pc and every operand byte stay symbolic.  Step k's pc derives from the
symbolic head as head + offset with ONE stepglu carry per step (exact
on integer bytes; license: no $FFFF wrap), so the general 16→8+8
byte-split gadget the roadmap feared is never needed.  A non-constant
byte-split feedback source (ea with a symbolic address) now leaves the
input *unbound* instead of erroring: dead under the fold is fine, a
live reference fails loudly in tr().  `sequence_shape_pins(code)`
builds the pin list + offsets.

Receipts (test_shape_fold_shadows_union, 79 tests green): ONE fold of
the DEX/BNE shape reproduces the union byte-identically at two heads
including a page-crossing window (DEX@$04FF/BNE@$0500 — the carry
fires), loop iterations and exit both; ONE fold of the poll-trio shape
(LDA abs / AND imm / BEQ rel — poll address, mask, and rel all
symbolic) matches at two instantiations ($0700/#$40 and $0703/#$80),
spin and fall-through.  Shape sizes: DEX/BNE 2,436 terms; poll trio
10,414 terms / 97 lookups (symbolic addresses keep the fetch reads
live — a shape forward will cost more than a pinned window's but far
less than per-instruction routing).

Why this matters: a *fixed library* of shape experts compiles once into
the core — zero per-ROM compile, any ROM including homebrew — and the
runtime dispatch table (head, shape) is data.  JIT "discovery" becomes:
count dispatch-fallback pcs, decode candidate windows, match against
the shape library, add a table entry.  The per-ROM pinned module
remains the opt-in turbo tier (0.07 µs/instr vs an estimated
~0.2 µs/instr for shapes).  Still to build (next session): engine-side
shape experts + runtime window table (`add_window(shape_id, head)`
after Python-side license checks: opcode bytes at offsets, write-free
ops, watch-range exclusion, no $FFFF wrap, intermediate next_pc ==
head+offset — the last is exhaustively checkable over all 64K heads
since the folded expression's support is pc-only for non-branch steps);
JMP-spin shapes additionally need an operand==head router check (the
self-target is a dispatch property, not a fold property); then the
fallback-pc profiling counter and the scan-at-boot table population.

## 2026-07-10 per-ROM sequence module — the compile skirt, landed

The recorded path (a), built: sequence experts no longer compile into
the engine TU.  The **core engine** (union + 227 opcode experts) is
emitted with a sequence-dispatch *hook* — a raw function pointer + a
`seq_out[21*32]` buffer + `attach_seq`/`detach_seq` — and its cache key
deliberately excludes the windows, so it compiles once per code digest
and every ROM shares the dlopen fast path.  A ROM's windows become a
**per-ROM module** (`SequenceModule`): the folded window experts
emitted as a tiny TU that `#include`s the core's generated header and
reaches all shared machine state (st_* registers, RAM mirror, 16x4KiB
memory structure, running-argmax registers, seq) through a typed
`Engine_<digest>&` — public data members and in-header inline methods
only, so there is no cross-dylib symbol dependency and the digest-named
struct keeps ODR honest.  Module-private state is scratch only (xbuf,
the hull_seen guards).  The union planning pass re-runs at module
emission purely to reproduce the core's register allocation
(deterministic given the model); a mismatched core digest refuses to
attach.  The seq-module cache key covers (code digest, core digest,
bcd, arith, window set).

Receipts: 77 tests green, ruff clean; sequence lockstep test
byte-identical through the module path; full nestest MATCHED 0.1 s;
Klaus 200K decimal prefix MATCHED; **full Klaus MATCHED 30,646,177
steps in 90.0 s**; SMB frames byte-identical kernel vs weights 120/120
at **332.3 fps (3.0 ms/frame)** — the function-pointer indirection is
within run-to-run noise.  **Per-ROM cost: novel window set ~7.1 s
(sequence folds + one ~160 KB TU compile), cached ~0.95 s** — vs the
full ~45-60 s core re-emission + recompile it replaces.  Gotcha for
fps probes: the console handoff constructs the weights machine inside
the timed window, so a ROM's *first* run pays the module build there
(observed 30.8 "fps" cold vs 78.2 warm on the same nes-run command,
both dominated by kernel warmup frames); per-frame numbers need a
frame-level probe with the module cache warm.  This is the JIT
substrate: hot-swapping a window set is now build-module + attach, no
core recompile.

## 2026-07-10 sharded engine compile — 107 s → 17 s, per-ROM included

The 43 MB engine TU compiled single-threaded while nine cores idled.
The emission is now post-processed into a header (struct, state, small
inline methods, declarations) plus ~10 content-balanced .cpp shards with
out-of-class `layer_*`/`step_core*` definitions, compiled in parallel by
ninja (the split is deterministic text processing; the digest still
covers the original single source, and the dlopen fast path is
unchanged).  Receipts: **fresh compile 106.9 → 16.7 s (6.4x)**; a full
bcd=True machine build (folds + emission + compile) now fits in 30 s;
step times hold or improve (no cross-shard inlining loss measured); 77
tests green; SMB ~337 fps byte-identical 120/120; full Klaus MATCHED
90.8 s.  `-O1` was measured and rejected: ~12% compile win (parse/
codegen dominates, not optimization) for unquantified step-time risk.
Note: the hygiene sweep widened the *union* d_model 1564 → 1766 (its
guard counts per-phase ops, not boundary width) while narrowing the hot
top-end experts 1696 → 1584 — harmless, the union is cold under
all-opcode routing.

Recorded (landed same day, see the per-ROM sequence module log above):
the complete per-ROM compile skirt, path (a) — a stable core module
(union + opcode experts, compiled once per code digest) plus a tiny
per-ROM sequence module attached through a function-pointer hook.
Path (b) — canonical shape experts with symbolic pc (zero per-ROM
compile) — remains open; it needs byte-split feedback gadgets
fold_sequence currently requires to be constant.

## 2026-07-10 greedy hygiene scheduling — the MILP's win at zero solver cost

`asap_schedule` gains a hygiene sweep (default on): after the balance
pass, a deterministic two-round local search moves each persist within
its exact dependency window minimizing (over-width, new-passthrough-
layers, load, -phase) — preferring persist-2 slots (no attention
passthrough) and co-locating persist-1 consumers of shared deps, with
the same-layer-lookup exemption the MILP's pd_var has and a width guard
so no phase exceeds the balanced max load.

Receipts: **LDA abs;JMP 0.422 → 0.377 µs/step (1.12x), JSR;RTS 0.380 →
0.327 (1.16x)** — the greedy captures essentially the whole hygiene-MILP
result (0.375 at 632 s of solver) in milliseconds, so the MILP opt-in is
now purely a certificate.  SMB 350.6 fps, frames byte-identical 120/120;
full Klaus MATCHED 91.4 s; nestest MATCHED; 77 tests green.  Probe
gotcha for the record: monkeypatch-A/B configs must salt the engine
cache key — `functools.partial(key, extra=...)` is silently overridden
by the call site's `extra=` kwarg, and the collision surfaces as an
"x size != d_model" mismatch against the cached artifact.

## Recorded idea: profile-guided JIT window discovery (not started)

The general successor to the boot-time loop scanner: the run loop counts
dispatch-fallback pcs (one array increment where the opcode switch runs
today); the runner periodically takes the hottest pcs, decodes candidate
windows around them (straight-line bodies included — shapes the closed-
loop scanner cannot see), folds the ones that pass the sequence license,
and hot-swaps the engine (the build cache makes rebuilds ~1 s plus a
one-time compile per new window set; state carries over exactly like a
console handoff: st_* registers + RAM mirror + memstruct + running-
argmax registers).  Workload-independent by construction — it discovers
any ROM's hot idioms at runtime — and the self-profiling MoE that grows
its own experts is a thesis chapter on its own.  Prerequisites are all
landed: sequence experts, license checks, config-key caching, cheap
construction.

## 2026-07-10 generic loop scanner + written-bit fast path (any-ROM work)

Two generality changes for arbitrary ROMs (SMB itself unchanged at
~343 fps, frames byte-identical):

- **scan_sequence_windows is now a generic closed-micro-loop finder**,
  not a pattern list: decode forward from every address; a window closes
  when its final control flow (conditional branch or JMP abs) targets
  the head; cheap pre-filters (no writers, no mid-window branches,
  only-first-reads), with `_build_sequence_expert`'s license checks as
  the backstop — a rejected window is skipped with a log line, never a
  crash.  JMP-closed loops still fold spin_repeat iterations.  On SMB it
  finds **9 windows where the pattern list found 3**, including idioms
  the list missed: the canonical `LDA $2002 / BPL` vblank waits, the
  BNE-variant sprite-0 trio, a `DEY/BNE` delay, and two 4-instruction
  timing loops (`SEC/SBC/DEY/BPL`, `INY/LSR/BCC`).  Any game's idle
  idioms now qualify without anyone naming them.
- **mem_query written-bit fast path**: if the queried address's own bit
  is set, the nearest-written-address is the address itself — return
  `latest[q]` directly (below == above == q collapses the tie-break).
  Honest receipt: ~2% on a read-heavy loop — the scans were already
  cheap on dense bitsets; kept because it is five exact lines that are
  never slower.

Receipts: 77 tests green (the sequence lockstep test now drives four
idiom windows — abs trio, DEX/BNE delay, zero-page BPL poll, JMP spin —
across two mid-run flag flips); SMB 343 fps, frames byte-identical
120/120; full Klaus MATCHED 90.6 s; nestest MATCHED.

## 2026-07-10 hygiene-objective MILP — validated, kept as opt-in

Approach #3, probed and closed.  `milp_schedule` gains
`objective="hygiene"` (minimize the passthrough-persistence indicator
sum — total dying is schedule-invariant, so passthrough co-scheduling
and the persist1/persist2 choice are the schedule-variable hygiene
cost) plus `width_bound` (cap d_model at the balanced-ASAP width);
`expert_schedule="milp"` now uses it.  Receipts: the LDA abs;JMP loop
runs 0.429 (ASAP) → **0.375 µs/step (1.14x)** — the realignment works
where the width objective bought only ~7% — but costs 632 s of solver
per two experts (vs 12 s ASAP), incumbents don't appear at all inside
120 s for the bigger experts (JSR: 884K rows), and at 227 experts it is
~19 h per machine config.  Verdict: correct objective, wrong price for
a default — kept as the opt-in it already was, for maximum-performance
builds of chosen experts.  At 350 fps the logic tail it targets is
~0.1 ms/frame, so it is a thesis receipt, not a frame-rate lever.

## 2026-07-10 sequence experts — a pinned window is one forward

Approach #2 from the 2026-07-09 handoff, landed as construction + engine:

- **graph/fold.py `fold_sequence`**: folds a K-step composition of the
  union graph into one materialized graph.  Per-step pins come from the
  router (pc, every window byte, is_nmi=0), state chains through
  `SEQUENCE_FEEDBACK` (expression feedback for A/flags/data; constants
  propagate as pins; byte-split entries must fold constant), and each
  step's roots materialize as `name__s<k>`.  Under sequence pins the
  internal pc feedback is constant, so no byte-split gadgets are needed.
- **Runner `_build_sequence_expert`**: the license is checked, not
  assumed — every step write-free and non-halting (folded constants),
  straight-line control flow inside the window (folded `next_pc__sk` ==
  the next step's pinned pc), only the first instruction may read memory
  (the shadow-watch check covers exactly one), no insert-side hazards.
- **engine_gen**: `emit_step_core` generalizes to K out-buffers (a plain
  expert is the K=1 case); a generated `seq_dispatch` matches pc == head
  plus every window byte and requires the step/cycle budget that keeps
  dispatch step-for-step equivalent to the per-instruction path (no
  dispatch under stop_on_trap or a pending NMI); the run loop emits K
  trace rows per forward.  machine_step never dispatches sequences.
- **Windows come from an ISA pattern scan, not a workload**
  (`scan_sequence_windows`): closed `LDA abs / AND imm / BEQ -7` poll
  loops and `JMP`-to-self spins (folded 4 iterations per forward),
  found in the boot image; the router re-verifies every byte at
  dispatch, so self-modified code only loses the fast path.  The
  console passes the scan of the cartridge image; the cache key covers
  the windows.

Receipts (76 tests green, ruff clean):
- New fold shadow tests: one composed forward reproduces the union's K
  consecutive readouts exactly (trio across taken/not-taken; 3-step
  spin).  New run-loop lockstep test: traces byte-identical with and
  without dispatch across poll, mid-run flag flip, fall-through, and
  spin, including chunk boundaries that land mid-window (the budget
  guard falls back to single steps).
- **Poll trio 0.40 → 0.07 µs/instr; JMP spin 0.36 → 0.05 µs/instr.**
- **SMB gameplay: 182.4 → 349.9 fps (2.9 ms/frame)**, frames
  byte-identical kernel vs weights 120/120.  Fps arc: 28.24 → 29.68 →
  32.47 → 43.51 → 47.65 → 123.5 → 182.4 → **349.9**.
- Full Klaus re-validated MATCHED (90.7 s; the Klaus harness passes no
  windows, so this re-verifies the shared emission changes); nestest
  MATCHED.

## 2026-07-10 group-dispatch fusion — one memory query per read, not 16

The step-anatomy finding, fixed.  Each memory-head site's query is
already the *local* address for its group (`qi = address − 4096·g` by
construction — the structure-eligibility check verifies the query row is
a positive scalar multiple of `[address − lo, 1]`), so the dispatch is
purely local: serve `mem_query` only when `qi` lands in `[0, 4096)`
(one unsigned compare), else substitute `(0, 0)` — exactly the
empty-group result, a value every consumer already handles for real
(the documented masked-by-zero-multiplication case; contributions of
non-matching groups are killed by the exact-zero `query_in_group` gate).
No family detection, no emission reordering: a four-line change at the
query site in engine_gen.py, identical for union and expert bodies.

Receipts (74 tests green incl. the codegen-vs-generic identity test —
the substitution never reaches a readout; ruff clean):
- **JMP hot path 0.87 → 0.37 µs/step (2.7M instr/s); poll trio 0.88 →
  0.39; LDA abs ~1.27 → ~0.49** — a read now costs ~0.1 µs, down from
  ~0.4.
- **SMB gameplay: 123.5 → 182.4 fps (5.5 ms/frame)**, frames
  byte-identical kernel vs weights 120/120.  Fps arc: 28.24 → 29.68 →
  32.47 → 43.51 → 47.65 → 123.5 → **182.4**.
- Full Klaus MATCHED (92.1 s, ~333K instr/s) and nestest MATCHED —
  Klaus wall time is now harness-bound (lockstep reference + 30M-row
  trace marshalling), so step-level gains no longer move it.

## 2026-07-10 step-anatomy probe — the forward is ~90% grouped-lookup queries

Self-contained loop benches through the all-opcode machine decompose the
expert step: CLC/INY steps cost **~0.05 µs** (the per-forward fixed cost
is nearly nothing, and chunk length / trace vectors don't matter), while
cost tracks memory reads at **~0.4 µs per architectural read** — JMP abs
(2 operand reads) 0.87 µs, LDA abs (3 reads) ~1.27 µs.  The cause, read
off the emitted source: a read whose address group isn't statically
known emits **all 16** 4 KiB-group `mem_query` calls, with one-hot gates
zeroing 15 results downstream (~25 ns per query — there are just 16x too
many).  Next lever (user-confirmed priority, ahead of the sequence
expert): **group-dispatch fusion** — verify per read family at
generation that `qi_g == qi_0 - g*4096` as a linear-form identity, then
emit `group_sel = addr >> 12` and a single query.  Same construction
principle as the opcode router, applied per read; helps every workload.
Open design question to resolve first: where the group-selection
arithmetic lives downstream (gated products vs ungated out_proj — the
int32 wrap-masking argument leans on exact zero gates), so the fusion
can be guarded generation-time.  Probe scripts in the session scratchpad
(probe_step_anatomy.py; note its first version hung on a bare
`boot_image` + a zero-step-blind bench loop — `load_image` seeds the run
state, and `iter_run_arrays`' `if n == 0: break` guard is mandatory).

## 2026-07-10 MMIO read lease — the frame was pause-bound, now it isn't

Profiling the timed gameplay window overturned the plan: 78% of its
instructions are the wait construct (JMP-to-self spin 35.1%, poll trio
42.9%), the trio polls **$2002**, and every poll iteration paused the
C++ run loop back to Python for the device read — 1,420 pauses/frame at
~6-8.5 µs each ≈ 10 ms of the 21 ms frame, bigger than all transformer
forwards combined.  (The "render/MMIO ~5%, not a lever" note from
2026-07-08 predates all-opcode routing and a sprite-0-polling window;
it was stale.)  The kernel backend was pause-bound the same way.

The fix is a read lease, honoring the frame-level PPU's own model:
during the visible span, $2002 can only change at the sprite-0 and
vblank cycles the console already computes, and its read side effects
(vblank flag clear, write-latch reset) are idempotent after the first
read.  So when the bus services a $2002 pause it asks the device for a
lease (`PPU.read_lease`); the run loop then serves further reads of the
leased address natively — the same `inject_write` the Python path would
do — without pausing.  Invalidation is conservative: any watched write
clears it (bus), and the console clears at sprite-0 set, vblank begin,
and vblank end.  Both generated run loops (kernel.py and engine_gen.py)
carry the identical mechanism; devices that don't offer `read_lease`
(controllers, the MMIO test devices) behave exactly as before.

Receipts (74 tests green, ruff clean):
- **SMB gameplay: 47.65 → 123.46 fps (8.1 ms/frame)** — pauses 1,420 →
  ~32/frame; **frames byte-identical kernel vs weights, 120/120** in the
  timed window.  Fps arc: 28.24 → 29.68 → 32.47 → 43.51 → 47.65 →
  **123.5** — the 60 fps goal is exceeded 2x.
- Kernel backend, same window A/B: 12.9 fps (lease off) → 15.5 fps
  (lease on) — no regression; it is slow here because its hull token
  log grows with the ~1,500 injected poll reads/frame (the weights
  engine's O(1) memory structure is immune).  **The transformer weights
  now run SMB 8x faster than the C++ graph kernel.**
- Full Klaus re-validated (lease dormant — no watches): **MATCHED,
  30,646,177 steps in 91.5 s (334.9K instr/s)**; nestest MATCHED.

## 2026-07-09 engine build cache — construction is ~1 s, artifacts ship

The all-opcode default made warm machine construction ~30 s (227 folds +
43 MB emission), all of it reproducing a source file already sitting in
`model/_engine_cache/`.  Now cached at the config level: the key is
sha256 of (every source file that shapes the emission: graph/, scheduler/,
model/, attention/, plus spec/evaluator/reference/weights_runner) with
(bcd, arith, expert set) — computable without folding anything — and
`_engine_cache/manifest.json` maps it to the emitted-source digest plus
the engine metadata.  On a hit the runner skips `_build_expert_models`
and `generate_source` entirely and `GeneratedWeightsEngine.from_cache`
dlopens the compiled `.so` directly (no ninja/clang on the hit path).
Compiles now land in `_engine_cache/build_<digest>/` instead of torch's
home-dir extension cache, so the cache directory is a self-contained
distributable bundle: same-platform users load with zero compiles, other
platforms rebuild from the shipped `.cpp` automatically.  Any edit to a
generator-relevant source file changes the key — over-invalidation is
the safety argument.  MILP expert schedules are excluded (incumbents
depend on solve time).

Receipts: construction 30 s → **0.99 s** (hit), console handoff to
weights 29.3 s → **0.4 s**, slow test suite 303 s → **26.7 s** (11x);
74 tests green, ruff clean; SMB gameplay 47.28 fps (unchanged, as it
must be — the loaded engine is byte-identical).

## 2026-07-09 all-opcode experts — the expert set stops being a knob

Per explicit user decision ("default expert opcodes is game dependent...
when we try other games, this will probably fall apart"), the
game-derived default expert set is gone: `DEFAULT_EXPERT_OPCODES` is now
every opcode in `spec.OPCODES` (227, documented + implemented
undocumented) — one rebuilt expert model per opcode, NMI still falling
back to the exact union step.  Routing is workload-independent; nothing
about the default is derived from SMB (or any game) anymore.

The premise that made a subset necessary was stale: measured this
session, fold+asap+build_model is ~0.05 s per expert (all 227 fold with
zero insert-side hazards in 10.8 s), and the roadmap's "~25 s" top-48
build was actually ~7.4 s warm — the C++ digest cache already absorbed
the compile.  What the wider default really costs: emission 16.3 s, a
43 MB TU, one-time compile ~107 s (digest-cached), warm construction
~30 s per machine.  If the 30 s starts to hurt (tests, console handoff),
the fold-digest artifact cache from the old approach-#1 note is the
known fix — cache the per-expert schedule/model arrays keyed by a
content digest of the folded graph.  (Done same day: see the engine
build cache log above — cached at the whole-engine config level
instead, which also skips emission.)

One generator fix made it possible: the step_core variants share one
residual buffer, previously sized to the union's d_model with a hard
error if an expert exceeded it — and PLP's rebuilt model does (d_model
1696 vs union 1564; the `_low_bit` 256-way-gadget outlier).  The buffer
is now sized to the widest model and the guard is retired.

Receipts (74 tests green, ruff clean; all lockstep-exact):
- Full nestest MATCHED (~148K instr/s); every opcode swept through its
  expert by nestest + Klaus as before.
- **Full Klaus: MATCHED — 30,646,177 steps (decimal) in 92.6 s
  (1.54 min, 331K instr/s sustained incl. the lockstep reference),
  checkpoint-free — 2.25x over the 3.48 min record**, because the
  decimal ALU and the whole opcode tail now route.  Full-suite arc:
  57 → 18.9 → 15.3 → 9.4 → 6.3 → 6.2 → 4.55 → 3.83 → 3.60 → 3.48 →
  **1.54 min**.
- JMP-to-self 0.91 µs/step (1.10M instr/s); frame-poll loop 0.96 µs/step
  (1.04M instr/s).
- **SMB gameplay: 47.65 fps (21.0 ms/frame)** — arc: 28.24 → 29.68 →
  32.47 → 43.51 → **47.65**.  Remaining to 60 fps: ~1.26x, busy-frame
  logic through the mid-size experts.

## 2026-07-09 reference-aligned trace convention — the million-instruction mark

Per explicit decision (option: "align with reference"), the graph's trace-
context fields now follow reference.py's conventions *exactly*, field for
field: operands gated by instruction length (raw on an NMI step, which
reports the not-executed instruction bytes), ea already matched everywhere
(measured first: 0 mismatches over nestest before any change), and data as
the operation's primary result — loads report the loaded value, stores/RMW
the written byte (write_value already equals it for every writing family),
compares the wrapped difference, PHP live P, pure control flow 0, built as
a parallel gated sum over the mutually exclusive family one-hots (depth 2,
not a mux chain).  Previously these fields reported the raw pc+1/pc+2/[ea]
reads, which kept those memory lookups alive in every routed expert even
when the instruction never used them.

Union cost: terms 29,136 → 29,704 (+2%), depth 71 → 74 (the data sum
hangs off new_p, the critical-path tail); union fallback ~8.3 → ~8.9 µs.
Expert payoff: the fold now kills the operand reads and the operand-memory
read wherever the instruction doesn't use them.

The verification story got *stronger*, not weaker: the old caveat
("internal fetch/execute fields may differ between implementations") is
retired — test_graph_vs_reference now compares full states, and a new
test_trace_convention checks kernel-vs-reference on ALL trace fields over
full nestest and a 200K Klaus decimal prefix, instruction by instruction.
Measured before/after with a field-by-field differ: 15,177 mismatches in
331 (field, op, mode) classes before; **0 after**.  moe_probe results
regenerated against the new graph (baseline 74 layers / 29,704 terms;
checked-in JSONs updated).

Receipts (74 tests green, ruff clean; all lockstep-exact):
- Full nestest MATCHED ~177K instr/s; Klaus 200K decimal MATCHED ~137K/s.
- **Full Klaus re-validated: MATCHED — 30,646,177 steps (decimal) in
  208.7 s (3.48 min, ~147K instr/s sustained incl. the lockstep
  reference), checkpoint-free.**  Full-suite arc across the engine
  generations: 57 → 18.9 → 15.3 → 9.4 → 6.3 → 6.2 → 4.55 → 3.83 → 3.60 →
  **3.48 min**.
- **JMP-to-self hot path: 1.22 → 0.99 µs/step — 1.01M instr/s, the first
  million-instruction-per-second receipt through genuine 71-layer-model
  transformer inference.**
- **Frame-poll loop: 1.54 → 1.03 µs/step (973K instr/s).**
- **SMB gameplay: 33.33 fps (top-16), 43.51 fps (top-48, 23.0 ms/frame)**
  — the arc so far: 28.24 → 29.68 → 32.47 → 43.51.  Remaining to 60 fps:
  ~1.39x, still busy-frame logic through the mid-size experts.

### Next steps (handoff, written 2026-07-09): the three approaches to 60 fps

Current state: hot loops >1M instr/s; SMB gameplay 43.51 fps top-48 /
33.33 fps top-16 (default); the gap is ~1.39x, concentrated in busy-frame
logic through the mid-size experts plus the expert-set build cost.

**1. Expert-set tuning + build caching (engineering, do first).**
Top-48 (99.1% gameplay coverage, from the checked-in
`experiments/moe_probe/smb_opcode_hist_gameplay.json`) is clearly worth it
for play but costs ~25 s of fold+schedule+build per machine construction,
which is why the default stays top-16.  Fix by caching, then flip the
default: cache per-expert artifacts keyed by a content digest of the
folded graph (deterministic given bcd + opcode + pin set) — either the
solved schedule as {dim id -> phase} (rebuild via `_layers_and_liveness`;
ids are deterministic because fold materializes in id order) or the built
WeightModel arrays.  `model/_engine_cache/` is the pattern to follow.
Also reconsider the expert set itself: gameplay-weighted, per-opcode for
the hot ~16, and check whether the 9,870-instr busy-frame mix (broader
than the idle-frame poll mix) wants different members — measure with the
histogram before choosing.  Expected: top-48 for free at machine build,
+~10 fps over top-16 today.

**2. Poll-loop sequence expert (the interesting construction).**
Fold a multi-instruction pin vector: the SMB frame-wait loop
`LDA abs / AND imm / BEQ rel` (plus the `JMP abs` idle spin) as ONE step
— one forward consumes three instructions.  The fold machinery is ready
(pins are just names -> constants; pin all three opcodes' bytes and the
operand bytes, which the router verifies); the graph construction needs a
3-step composition of the union graph before folding (state chained
through, one token emitted).  Router: pc equals the loop head AND the
next ~7 RAM bytes match the pinned sequence AND no NMI pending; bail to
per-instruction steps otherwise.  The poll loop writes nothing, which
sidesteps intermediate-write semantics entirely (a general sequence
expert must not defer writes past a step that could observe them — keep
to write-free sequences first).  Trace rows must still be emitted per
instruction (three rows from one forward; the readout needs all three
states, or relax to emitting the last row only inside the native run loop
where no watch fires — decide with receipts).  Payoff: the ~87% idle-poll
fraction collapses ~3x; busy frames gain less, so measure at the frame
level.  Verification: the fold shadow receipt generalizes (replay the
union token stream, compare the 3-step composition), then the usual
lockstep chain.  This is also a genuinely novel artifact for the write-up
(trace-routed super-instruction expert).

**3. MILP objective realignment (bounded, do last).**
The exact solve is tractable at expert scale but the width objective buys
~7% (see the 2026-07-09 probe log).  Realign the objective to the
engine's cost model: minimize the passthrough/erase indicator sum (the
`pd_var`/dying terms `milp_schedule` already builds) subject to
`D_half <=` the balanced-ASAP width, per expert, with `time_limit` and
incumbent acceptance (both already wired:
`expert_schedule="milp"` / `expert_milp_seconds`).  Slot-hygiene work is
~20-30% of an expert's nnz, so the honest ceiling is ~1.2-1.3x on expert
steps.  One probe on the poll-loop experts decides it.

## 2026-07-09 fetched-byte pin — the router predicate, fully spent

What was built (`graph/fold.py` + tests; roadmap next-step #2 from the
rebuilt-experts log): `opcode_pins(..., pin_fetched_byte=True)` (now the
default) additionally pins the ``fetched_opcode`` persist to the routed
byte.  The router dispatches on ``ram[pc] == byte``, which is strictly
stronger than ``is_op = 1``, so the memory read at PC provably returns the
byte -- the fold deletes that read (16 grouped lookup sites per expert) and
the remaining decode residue.  Per expert: terms 0.82x, lookups 65 -> 49.
The existing fold receipts re-verify the stronger pin unchanged (the
expert-vs-union shadow lockstep runs on the default pin set); the probe
comparison test pins pin_fetched_byte=False for apples-to-apples.

Receipts (72 tests green; all lockstep-exact):
- Full nestest MATCHED ~167K instr/s; Klaus 200K decimal MATCHED ~140K/s.
- **JMP-to-self hot path: 1.75 -> 1.22 µs/step (817K instr/s)**.
- **Frame-poll loop (LDA/AND/BEQ): 2.15 -> 1.54 µs/step (650K instr/s)** —
  the hot loops are now individually past SMB real-time (~592K needed).
- **SMB gameplay: 32.47 fps (top-16), 38.68 fps (top-48, 25.9 ms/frame)**
  — from 28.24 at the start of the rebuilt-experts arc.
- **Full Klaus re-validated: MATCHED — 30,646,177 steps (decimal) in
  221.6 s (~138K instr/s incl. the lockstep reference), checkpoint-free**
  (measured while a MILP solve shared the machine; within noise of the
  previous 216.3 s).

Remaining to 60 fps (~1.56x): busy-frame logic mix through the mid-size
experts — per-expert width compaction, probed next.

## 2026-07-09 per-expert MILP: solves now, but width is the wrong objective

At expert scale the exact-width MILP actually solves — the union blowup
(2026-07-06: presolve alone exceeded an hour) was formulation-times-size,
not the solver:
- **CLC expert: 273K rows presolved to 74K / 33K binaries; optimal in
  118 s; d_model 250 → 154 (0.62x)** (260 → 162 with the byte pin).
- **LDA abs (22 layers / ~1,700 ops): 881K rows presolved to 456K;
  incumbents 152 @ 4 min → 105 @ 8 min → 98 @ 22 min vs ASAP 272**
  (bound 91, gap 7.2% when stopped) — usable incumbents come early,
  proofs are slow.

**But width does not buy step time**: the CLC expert at 0.62x width runs
only ~7% faster (CLC*30+JMP loop: 1.26 → 1.19 µs/step avg; extracted
CLC step ~1.03 → 0.95 µs).  The codegen engine is load-bound on *nnz*
(plus the lookup structures), and minimizing max width neither reduces
nnz nor the slot-hygiene work — tighter slot reuse can even add erase
churn.  Wiring landed anyway (`milp_schedule(time_limit=...)`,
`Cpu6502TransformerMachine(expert_schedule="milp", expert_milp_seconds=...)`)
as an experimental option; default stays ASAP.

Implication: if the MILP is to earn real step time, the *objective* must
be realigned to the engine's cost model — minimize total passthrough/
erase bookkeeping (the `pd_var`/dying indicators the formulation already
builds), subject to width ≤ the balanced-ASAP width, rather than
minimizing width itself.  Slot-hygiene work is ~20-30% of an expert's
nnz, so the honest ceiling of that realignment is ~1.2-1.3x — worth one
probe, not a research arc.

The bigger measured lever for busy-frame logic cost is the **trace-
context fetch reads**: fetched_operand_lo/hi and operand_mem/data_out
stay live in every expert only because the trace convention reports the
*raw* pc+1/pc+2/[ea] reads even for instructions that never use them
(1-2-byte opcodes, non-memory ops).  The reference core's convention is
already "0 when not fetched/used"; aligning the graph's trace fields
with it would let the fold kill up to two more grouped reads per expert
(the byte pin killed one such read for 1.75 → 1.22 µs on JMP).  This is
a semantics/verification-story decision — it changes trace-context
fields that have been byte-identical across every engine generation
(architectural lockstep fields and writes are untouched) — so it needs
an explicit call, not a silent redefinition.

## Where things stand (all verified, all tests green)

**Four backends, one 6502 semantics**, each lockstep-verified against the
next (`nestest golden log ⇔ reference ⇔ graph evaluator ⇔ C++ kernel ⇔
weights transformer`):

| Backend | Speed (instr/s) | Status |
|---------|-----------------|--------|
| `cpu6502/reference.py` | ~2.5M | oracle; full Klaus + nestest pass |
| `cpu6502/evaluator.py` (exact graph) | ~6K | lockstep-exact |
| `cpu6502/kernel.py` (generated C++) | ~600K | full Klaus: 30,646,177 steps exact, ~48 s |
| `cpu6502/weights_runner.py` (real transformer) | ~98-116K after boot | default generated engine (codegen: exact fixed-point int32, C++ machine_step, O(1) attention state); full nestest MATCHED in ~0.1 s; **full Klaus MATCHED: 30,646,177 steps exact in ~6 min, checkpoint-free** |

The weights transformer is genuine inference (d_model≈1550, 71 layers,
analytic weights, no training).  The default path is the generated
straight-line C++ engine (`model/engine_gen.py`) with an exact
fixed-point int32 residual, which runs the whole machine step per call;
the generic CSR engine (`model/weights_engine.cpp`) and the dense/sparse
Python paths remain diagnostics.

**NES layer** (`nes/`): MMIO bus with shadow-decode read injection,
in-graph NMI, frame-level PPU, OAM DMA, controllers, cycle-budget frame
pacing, invisible snapshot-and-rebuild checkpointing, and mid-game
backend handoff.  nestest boots and passes its suite interactively; SMB
plays at ~45 fps on the kernel; SMB frames render **through the
transformer weights** byte-identical to the kernel at ~0.09 s/frame
(~11 fps with the int32 generated engine + O(1) memory attention;
`nes-run --warmup-frames 60 --backend weights`).

**Scheduling**: the optimal-width MILP does not scale (formulation
blowup, not the HiGHS engine); `scheduler/milp.py:asap_schedule` is a
deterministic load-balanced critical-path scheduler (71 layers, D≈1550
with the grouped-memory lookup).

**Numerics facts** (each has a receipt in tests or the README):
- In the original scalar 16-bit memory lookup, float64 was needed **only**
  for the K-side attention-score construction (`ky = -k²…` reaches 2³²
  with gap 1; fp32 diverged at nestest step 0, but matched low-address
  programs).  The grouped-lookup experiment below fixes this for hull
  hard attention.
- All residual/FFN values are dyadic rationals < 2²⁴ with at most one
  fractional bit (certified 2026-07-07), so float32 execution is exact and
  the default engine runs an exact fixed-point int32 residual (F=1).
- Same-address rewrites are resolved by the hull LATEST sequence tie-break.
  True softmax still averages exact ties, so `--cache standard` remains a
  separate latest-write semantics problem.

## 2026-07-06 fp32 / grouped-lookup research log

Tracked in jj:
- `kxsxypym` — checkpoint current 6502 transformer performance work
- `qownuxsp` — make float32 engine the default weights runtime

What was tried:
- Scalar rescaling of the 16-bit address parabola was rejected.  It lowers
  `ky`, but also shrinks the adjacent-address score gap by the same squared
  factor, so the fp32 error-to-gap ratio does not improve.  With the old
  recency term, exact graph tests failed because newer neighboring bytes beat
  exact older bytes; scaling the recency term restored exact graph/kernel
  tests but still failed high-address float32 at nestest step 0.
- Splitting memory into 16 groups of 4 KiB worked.  Each lookup uses a
  12-bit local-address parabola, keeping K-side address scores below 2²⁴.
  The grouped graph passes exact graph tests and the C++ kernel test.
- The C++ weights engine now has `dtype="float32"` support for the engine
  path (float32 CSR storage/accumulation; hull cache still resolves using
  double after Q/K/V have been float32-rounded).
- The model now emits lookup `head_cache_groups`, and the C++ engine shares
  one hull cache for lookup heads with identical K/V expressions.  In the
  grouped model, 124 lookup heads collapse to 92 shared hulls; the busiest
  attention layer drops from 42 lookup insertions to 16 shared hull inserts.

Current receipt:
- `Cpu6502TransformerMachine(bcd=False)` defaults to the float32 C++ engine
  and matches all 8,991 high-address nestest instructions.
- The engine now directly inserts WRITE tokens into grouped memory hulls,
  skipping full transformer passes for ROM/RAM seeding and instruction
  writes.  `nestest` boot dropped from seconds to milliseconds.
- The engine iterates only active lookup heads; passthrough attention heads
  are now handled by the structural fusion below.
- The engine now also fuses structural passthrough/erase work: attention
  passthrough heads and linear FFN passthrough neurons become direct D->D
  CSR residual deltas, their projection rows/output columns are pruned, and
  the FFN activation loop only visits live non-passthrough ReGLU neurons.
- Measured on this machine after those optimizations: full nestest total
  ~1.2 s, ~7.7K instr/s; Klaus 20K prefix ~2.7 s, ~7.4K instr/s.
- Decimal-mode Klaus prefix also matches through the float32 C++ engine:
  20,000 instructions.

Remaining caveat:
- This solves hard-attention / hull fp32 high-address reads.  True softmax
  `--cache standard` still needs separate work for latest-write semantics,
  because softmax averages exact ties instead of applying the hull sequence
  tie-break.

## 2026-07-06 structural sparse-pruning research log

Tracked in jj:
- `umxxzzxr` — speed up fp32 engine with direct writes and active heads
- `rxoqortp` — fuse linear passthrough weights in the fp32 engine

What was tried:
- Counted structural work in the grouped model.  For nestest, the attention
  stack has 1,916 passthrough heads (3,809 V rows and 7,239 output nonzeros)
  versus only 124 lookup heads; the FFN stack has 3,240 linear passthrough
  neurons and 3,927 passthrough output nonzeros.
- Added `model.ffn_kinds` metadata and taught the C++ engine to fuse
  proven-linear passthrough/erase paths into direct sparse residual maps.
  This is structural pruning, not magnitude pruning: no learned weights are
  dropped, and the represented computation is algebraically identical.
- Result: exact graph/weights tests still pass, full nestest matches, and
  Klaus 20K still matches.  Payoff is ~1.2x single-thread speedup.  Useful,
  but not enough by itself for NES framerate.

Implication:
- There is still enough generic CSR work left that GPU CSR / custom Metal is
  worth investigating, but for a single strict instruction stream the port
  must fuse many layer operations per dispatch.  Plain per-matrix GPU CSR
  calls will likely lose to dispatch latency; batched NES instances or a
  custom fused kernel are the more credible GPU shape.

## 2026-07-06 GPU CSR feasibility probe

Tracked in jj:
- `posnqutp` — continue gpu csr performance research

What was checked:
- Local PyTorch has MPS (`torch.backends.mps.is_available() == True`), but
  MPS sparse CSR tensors are not implemented here (`new_compressed_tensor`
  fails on `mps:0`).  MLX is not installed.  So a GPU CSR backend is custom
  Metal/Objective-C++ work, not a device swap in the existing torch path.
- A temporary Metal CSR probe (`/private/tmp/metal_csr_ext.mm`) used a
  simple row-per-CSR-row kernel with resident buffers.  On representative
  model matrices:
  - `L22.ff_in` (2492x1554, nnz=2886): scipy CSR 2.75 us/mv; Metal with a
    wait after every dispatch 101 us/mv; 20K dispatches encoded into one
    command buffer 1.69 us/mv.
  - `L52.out_proj` (1554x1554, nnz=32): scipy CSR 1.64 us/mv; Metal
    wait-each 98 us/mv; one-command-buffer 1.95 us/mv.
- At the time of the probe, the fused CPU engine raw `_forward` on a
  synthetic STATE token was ~93.7 us/position (~10.7K positions/s), while
  full machine stepping was ~7.4-7.7K instr/s because Python still
  assembled/read back state and emitted writes.

Implication:
- Naive per-matrix GPU CSR is not viable for one NES stream: one
  synchronous Metal dispatch costs about as much as the entire current CPU
  transformer forward.
- GPU work only makes sense if either many machines are batched, or a custom
  backend fuses a large fraction of the layer stack into one/few command
  buffers and keeps state resident.  The hull lookup path is the hard part:
  if lookup queries stay on CPU, CPU/GPU synchronization will dominate.
- The CPU path still has headroom from generated/specialized sparse kernels
  (mostly +/-1 coefficients, many singleton rows) and from moving state
  readout/next-state assembly out of Python; those are likely better
  single-stream targets than unfused GPU CSR.

## 2026-07-06 engine readout/copy optimization

Tracked in jj:
- `svuxuxsq` — avoid fp32 engine readout copy

What was tried:
- The engine `_forward` path used to convert the whole fp32 residual vector
  back to float64 torch after every C++ forward, only so `machine_step`
  could index readout fields and run the tiny output head.
- The engine path now keeps the C++ result as a NumPy array, caches readout
  slots, and compares STATE/HALT using sparse output-head rows instead of
  `self.model.head(h)`.

Result:
- Weights tests pass.
- Full nestest remains exact and improves to ~1.1 s / ~8.5K instr/s.
- Klaus 20K remains exact and improves to ~2.4 s / ~8.4K instr/s.
- Raw `_forward` stays ~95 us including embedding; this was surrounding
  Python overhead, not a C++ matvec breakthrough.

## 2026-07-06 engine NumPy embedding optimization

Tracked in jj:
- `uvyvnukm` — use NumPy embeddings for engine tokens

What was tried:
- The previous engine path still built every START/STATE/full-WRITE token
  as a torch tensor, then took a NumPy view before entering C++.  A direct
  engine-only NumPy embedding path now fills the residual slots, applies
  the three position features, and calls the C++ engine without touching
  torch.  Dense/sparse diagnostic paths still use the original torch
  `_embed`/`_forward` flow.

Result:
- Weights tests pass.
- Full nestest remains exact and improves to ~0.8 s / ~10.9K instr/s.
- Klaus 20K remains exact and improves to ~1.9 s / ~10.6K instr/s.
- Synthetic `_forward_values` is ~75.8 us (~13.2K positions/s), much
  closer to direct C++ engine forward (~68.6 us in the earlier probe).

## 2026-07-06 hardening, weights checkpointing, forward-cost profile

Tracked in jj:
- `lxrykyso` — harden fp32 engine invariants; add weights-runner checkpointing
- `ynowvzrx` — scout optimization targets; land the generated fp32 engine

Hardening (from a review of the fusion/direct-write optimizations):
- `SparseWeightsEngine.insert_write` now refuses models without
  memory-tagged lookup heads (a direct write would otherwise be silently
  dropped); the runner gates the fast path on `supports_direct_writes`.
- The C++ engine throws at load if a filtered `out_proj` column survives
  at a non-lookup head slot — forward() no longer zeroes stale `head_out`
  entries, so the column structure is exactly what makes that safe.
- `build_model` exports `model.one_slot`, and the engine wrapper checks at
  load that every kind-2 FFN neuron's gate row is exactly the constant
  one.  The fusion drops the gate, so a mislabel must fail loudly.
- Known cross-language duplication to keep in mind: `insert_write_f32`
  hand-builds the WRITE key (`kx = 2·local`, `ky = −local² +
  LATEST_ALPHA·inv_log_pos`), baking in `KEY_OFFSET == 0`,
  `LATEST_ALPHA = 0.3`, and the 16×4 KiB grouping.  graph/core.py
  documents KEY_OFFSET as tunable — change any of these and the fast path
  desyncs.  The fp32 bit-exactness argument also dies if groups grow past
  4 KiB (8191² > 2²⁴).  Skipping WRITE tokens in non-memory hulls is
  equivalent only because every non-memory lookup head clear-gates them.

Checkpointing (the roadmap-#2 blocker, now done):
- `Cpu6502TransformerMachine` keeps a 64 KiB RAM mirror (updated by every
  `_insert_write`) and `checkpoint()` reseeds the hull logs from it,
  preserving architectural state, trace numbering, and pending NMI — the
  same shape as `KernelBus.checkpoint`.
- `checkpoint_tokens=N` rebuilds automatically once the token log outgrows
  N; `6502-klaus --checkpoint-tokens` (default 1M) wires it up, and the
  klaus weights lockstep now streams instead of materializing the trace.
  The NES `GraphCore` adapter delegates to the runtime's own checkpoint.
- Receipts: `test_checkpoint_rebuild_is_invisible` (nestest prefix with a
  rebuild every 8 tokens: lockstep-exact, trace numbering intact); Klaus
  100K prefix MATCHED with a rebuild every 20K tokens at ~10.4K instr/s —
  checkpointing costs approximately nothing.
- **Full Klaus through the weights: MATCHED.**  All 30,646,177
  instructions (decimal mode included, bcd=True) lockstep-exact against
  the reference through the fp32 engine in 3443 s (~57 min, ~8.9K
  instr/s sustained), 36 invisible checkpoints, flat memory.  Run with a
  1M-token window (~6 GB hull logs; the default is now 250K).

Forward-cost profile (where the ~11K instr/s ceiling lives):
- `machine_step` ≈ 84.6 µs: ~78 µs inside the C++ forward (92%), ~7 µs of
  Python glue.  Porting readout/state assembly to C++ caps out at ~1.1×.
- Per forward: 38,067 CSR nnz visited (ff_in 19.5K, attn_direct 7.2K,
  ff_out 5.6K, ffn_direct 3.9K, in_proj 1.5K, out_proj 0.2K), 5,299 live
  ReGLU neurons, 124 hull queries + 92 shared-hull inserts.
- Instrumented section shares: ff_in 39%, out_proj+attn_direct 21%,
  in_proj 17%, ff_out+ffn_direct 13%, hull 8%, activations 3%.
- Removing every mv() zero-fill is a wash (0.97×): the cost is the
  dependent scalar CSR gather (~1.5–2 ns/nnz), not memset.
- Of the kept in_proj/out_proj/ff_out coefficients, 6,253 of 7,373 are ±1;
  kept in_proj rows: 289 singletons, 455 multi-nnz.
- SMB via weights: ~0.90 s marginal per frame (~1.1 fps), ~11× better than
  the previous ~10 s receipt, still byte-identical to the kernel frame.
- Experiment gotcha: benchmarking a modified engine side by side requires
  renaming the C++ struct — two loaded extensions that both define
  `WeightsEngine` cross-resolve symbols and silently time the wrong code.

Implications, in payoff order:
1. **Generated per-layer sparse kernels** (the kernel.py trick applied to
   the weights engine): emit straight-line code per layer with column
   indices as immediates, ±1 coefficients as add/sub, singleton rows as
   plain moves.  Kills the indices/data loads that dominate today and
   frees the compiler to schedule independent loads.  Est. 2–4×
   single-thread, byte-exact by construction (same arithmetic, same
   order).
2. **Exact fixed-point integer path**: every residual/FFN value is a
   dyadic rational < 2²⁴, so int32 (int16 where provable) NEON arithmetic
   is semantically exact and vectorizes what fp32 scalar gather cannot.
   Research needed: per-slot scale factors, overflow proofs.  Pairs with
   codegen.
3. Together with a C++ machine_step (~1.1×) and MILP width compaction
   (≤1.8×), the realistic scalar ceiling is ~5–8× → ~60–90K instr/s →
   SMB at ~6–9 fps.  True 30+ fps needs 1+2 to land well, or the
   fused-Metal resident stack / batched-fleet throughput from the GPU
   probe.

## 2026-07-07 generated straight-line engine (codegen), new default

Tracked in jj:
- `ynowvzrx` — scout optimization targets; land the generated fp32 engine

What was built (`model/engine_gen.py`, scouting recommendation #1):
- A generator that emits one C++ translation unit specific to the built
  model: per-layer member functions with column indices as immediates,
  ±1 coefficients as add/sub, all other coefficients as float32 hexfloat
  literals, hull calls inlined with immediate hull ids, and the direct
  WRITE key construction generated from `graph/core.py`'s
  `LATEST_ALPHA`/`KEY_OFFSET` (it refuses `KEY_OFFSET != 0`), which
  retires that duplication hazard for the default path.
- The emission replicates the generic engine's arithmetic exactly: CSR
  column order, left-to-right accumulation, the same fusion composition
  (float64 merge, exact zeros dropped), the same residual-update order,
  the same hull insert/query sequence.  Digest-named source + struct
  (`Engine_<sha256[:16]>`) in `model/_engine_cache/`, compiled once per
  model (~5–10 s, cached), `py::module_local` — the kernel.py pattern,
  which also avoids the two-extensions-one-struct ODR trap.
- `matmul="codegen"` is the runner/CLI/console default; `engine` (generic
  CSR), `sparse`, `dense` remain diagnostics.  float64 stays on `engine`.

Receipts:
- `test_codegen_engine_matches_generic_engine`: nestest prefix traces are
  identical between codegen and the generic engine, checkpoint rebuilds
  included, and lockstep the reference.  All 6 weights tests + 47 fast
  tests pass with codegen as default.
- Full nestest MATCHED: **0.3 s, ~33–35K instr/s** (was ~0.8 s / 11K).
- Klaus 200K prefix MATCHED with decimal mode: **7.0 s, ~28.5K instr/s**
  (was 85.8 s at the 2026-07-04 receipt — 12× in three days).
- SMB frames through the weights: **~0.26 s/frame, ~3.8 fps** (was ~0.9 s),
  byte-identical to both the kernel frame and the generic-engine frame.
- Raw C++ forward: 76 µs → **15.7 µs (4.8×)**; whole `machine_step`
  26.1 µs (~38K steps/s synthetic).
- **Full Klaus re-run through codegen: MATCHED — 30,646,177 steps in
  1136 s (18.9 min, ~27K instr/s sustained)**, 146 checkpoints at the
  250K-token default, throughput flat for the entire run (the tighter
  window also removed the ~10% mid-run dip seen at 1M).

Implication for the next round:
- Python glue (embedding dict, per-field readout, TraceRow assembly) is
  now ~10 µs of the 26 µs step — **40%**.  Porting the machine_step loop
  into the generated C++ (read slots, extract writes, assemble next
  state, halt check; Python only for MMIO pauses, like the kernel) is
  promoted from ~1.1× to ~1.6×, and is the natural next step toward the
  kernel's architecture.  After that, the exact fixed-point int/NEON
  research item from the scouting log is what remains on the CPU path.

## 2026-07-07 machine_step in generated C++

Tracked in jj:
- `vysrktlq` — port machine_step into the generated engine

What was built:
- When the runner passes a `step_spec` (input/output dim slots, P-flag
  bit masks, STATE/HALT head rows), `engine_gen.py` additionally emits a
  `machine_step` member: STATE-token embedding with slots as immediates,
  position features, the whole layer stack, readout via `nearbyint`
  (Python `round()` is half-to-even — the kernel's old llround lesson),
  write extraction with direct hull inserts, and the halt-logit
  comparison, all in one pybind call that fills a caller-provided
  21-slot int64 buffer.
- Python's per-step work is now: 13 int args in, one `tolist()` out,
  TraceRow assembly, RAM-mirror/pos bookkeeping, and the checkpoint
  hook.  The machine protocol (GraphCore adapter, NMI, inject_write,
  checkpointing) is unchanged; non-codegen paths keep the Python step.

Receipts (all lockstep-exact, frames byte-identical to the kernel and to
every earlier engine):
- `machine_step`: 26.1 µs → **18.9 µs (53K steps/s synthetic)**.
- Full nestest MATCHED: **0.2 s, ~45K instr/s** (11K two days ago).
- Klaus 200K decimal prefix MATCHED: **5.6 s, ~36K instr/s**.
- SMB through the weights: **~0.20 s/frame (~5.1 fps)** — from ~10 s/frame
  (0.1 fps) at the start of this arc, a ~50× cumulative improvement.
- The codegen-vs-engine identity test now exercises the C++ step against
  the generic engine's Python step: traces identical, checkpoint
  rebuilds included.
- **Full Klaus re-validated: MATCHED — 30,646,177 steps in 920 s
  (15.3 min, ~33.3K instr/s sustained including the lockstep reference)**,
  146 checkpoints, flat throughput.  57 min → 18.9 min → 15.3 min across
  the three engine generations in two days.

Where the remaining time is:
- ~15.7 µs forward + ~3 µs Python (TraceRow/dict bookkeeping is now the
  floor of the per-step protocol).  The forward is again the target:
  the exact fixed-point int/NEON research item is next on the CPU path,
  and batching/Metal remain the throughput plays beyond that.

## 2026-07-07 exact fixed-point int32 forward + cleared-insert skip

Tracked in jj:
- `zozvwnqs` — exact integer forward for the generated engine

What was built (`engine_gen.py` grew `arith` modes; int32 is the default):
- **Certification pass first**: every residual-feeding coefficient is an
  integer ≤ 65536 except four ±0.5 direct-map coefficients; no
  residual-feeding row reads position slots 0–2 (only hull-bound Q/K
  rows do); all lookup heads are LATEST tie-break; STATE/HALT head rows
  are ±100000.  These invariants make an exact integer residual viable.
- **Fixed-point int32 residual at scale 2^F (F=1)**: embeds shift in,
  readouts shift out, the four ±0.5 coefficients become always-exact
  shifts, ReGLU products rescale with a guard.  Every guard throws
  instead of rounding, so an insufficient F fails loudly (and did:
  the diagnostic depth scan showed the only fractional values are
  halves — F=1 exactly covers the model).  Partial sums may wrap mod
  2^32 (`-fwrapv`); every final graph value fits int32, so results are
  exact.  Q/K/V and position features feed the hulls in double.
- **int64 variant measured and rejected**: 20.0 µs vs f32's 18.9 µs —
  the linear part is load-bound, so value density is what matters;
  int32 alone gave 16.5 µs.
- **Hull ablation probe**: with the linear part this fast, the hulls
  were ~60% of the step (ablated step 6.8 µs vs 16.5 µs).  Which led to:
- **Cleared-insert skip**: every forward inserted the current token into
  all 92 shared hulls, but most inserts are clear_key-gated
  (ky −= 1e30) and provably can never win a query — a hull that stays
  empty returns (0,0), exactly what a cleared winner returns.  The int
  emission now skips them (and their value-expression computation)
  behind a `ky > -1e29` test.  16.5 µs → **9.3 µs**.

Receipts (traces identical to the generic engine incl. checkpoint
rebuilds; frames byte-identical to the kernel and all prior engines):
- `machine_step`: 18.9 µs → **9.3 µs (107K steps/s synthetic)**.
- Full nestest MATCHED: **0.1 s, ~64K instr/s**.
- Klaus 200K decimal prefix MATCHED: **3.2 s, ~62K instr/s** (incl. the
  lockstep reference).
- SMB through the weights: **~0.12 s/frame (~8.3 fps)** — 0.1 fps at the
  start of this arc, so ~80× cumulative.
- **Full Klaus re-validated: MATCHED — 30,646,177 steps in 563 s
  (9.4 min, ~54.4K instr/s sustained including the lockstep reference)**,
  146 checkpoints, flat throughput.  The full-suite run has gone
  57 min → 18.9 → 15.3 → 9.4 across four engine generations in two days.
- Debugging war story for the record: the first int builds threw the
  fixed-point guards at F=1/2/4 — not deep fractional values but an
  unscaled `machine_step` embed (inputs entered 2^F too small).  The
  guards did their job: loud failure, no silent corruption.  The depth
  scan that "found" 53-bit fractions was measuring the position-feature
  slots (1/log2 − 1/log(pos+2)), which the int design already excludes.

Where the remaining 9.3 µs lives:
- ~3 µs Python protocol floor, ~6 µs C++ (roughly half linear/glue,
  half hull queries on populated memory hulls).  The next big CPU lever
  is replacing the memory-group hull queries with an exact
  nearest-written-address structure (sorted set + per-address latest
  array): the parabola scoring makes the winner the nearest written
  address with recency breaking equidistant ties, which a lower_bound
  probe reproduces exactly — same trick class as hull-vs-softmax, with
  the same receipt obligations.  Beyond that: batched machines.

## 2026-07-07 density limits + memory attention as an exact structure

Tracked in jj:
- `oypxuppy` — replace memory-group hulls with exact nearest-address structure

Density beyond int32 — answered with data, and a discovery:
- **int16 is dead**: 804 of 1551 residual slots exceed int16 range even on
  the tame verify program (pc, addresses, and match-gadget squares).
  Mixed per-slot width is possible in principle but the certification
  story is weak and the linear part is no longer dominant.
- **Discovery**: the value scan found residuals reaching 65535² ≈ 2³² —
  beyond int32 too, and beyond the "dyadic < 2²⁴" f32-exactness claim.
  These are `(address − found_addr)²` match gadgets when a memory group
  is *empty* (found_addr = 0): exactly the case where the graph
  multiplies the result by a zero value, masking the wrap (int32) or the
  rounding (f32 — which stays sign-true, so its step thresholds still
  land right).  So int32's correctness at these sites is
  masked-by-construction and receipt-certified (30.6M-step Klaus), not
  range-proven; the int64 mode remains the belt-and-suspenders variant.

Memory attention as an exact structure:
- All per-layer memory-group hulls held identical data per group (every
  write inserted into all of them), so **one structure per 4 KiB group
  replaces all 92**: `latest[4096]` value/seq arrays plus a 64-word
  written-bitset.  A query is a two-direction bit-scan for the nearest
  written address; the grouped parabola score −(q−a)² + q² + α·ilp(seq)
  is maximized by exactly that address, recency breaks equidistant ties
  toward the later write, and below the double ulp the hull's LATEST
  tie-break picks the same winner — the same equivalence class of
  argument as hull-vs-softmax.
- The queries turned out to be scaled by the hardmax temperature
  (q = [C·(addr−lo), C], C ≈ 1.414e10): a positive scalar multiple of
  every score, so the argmax is unchanged; the generator divides C out
  and computes the query in pure integer arithmetic (no doubles at all).
- A write (`insert_write`) is now three stores; **memory state no longer
  grows with the token log**.  Guards everywhere: a live forward insert
  into a memory hull throws (the direct-WRITE contract made loud), and
  hulls that would go unused stay empty.  12 non-memory lookup heads
  (constant-query latest-live-token lookups) remain on real hulls.

Receipts (traces identical to the generic engine, frames byte-identical
to the kernel and all engines, 6 weights + 47 fast tests green):
- Full nestest MATCHED: **0.1 s, ~93K instr/s** (incl. the reference).
- Klaus 200K decimal prefix MATCHED: **2.2 s, ~89K instr/s** — 1.44× over
  the pre-structure engine on real runs (the synthetic bench understated
  it: old memory hulls grew between checkpoints and queries slowed with
  them; the structure is O(1) regardless of history).
- SMB through the weights: **~0.09 s/frame (~11 fps)** — double digits;
  ~110× cumulative over the arc.
- **Full Klaus re-validated: MATCHED — 30,646,177 steps in 378 s
  (6.3 min, ~81K instr/s sustained including the lockstep reference)**,
  146 checkpoints, throughput flat-to-rising the whole run.  The
  full-suite run: 57 min → 18.9 → 15.3 → 9.4 → 6.3 across five engine
  generations in two days — a 9× on the flagship receipt.

Next, in payoff order:
1. Convert the 12 remaining constant-query hulls to running-argmax
   registers (insert-time comparison; query returns the register).  Ends
   *all* unbounded growth — long runs would no longer need checkpointing
   at all — and removes hull2d from the forward entirely.  Needs care at
   the double-ulp tie margin; receipts arbitrate.
2. Port the run loop into the generated C++ (KernelBus shape: RAM mirror
   already lives in the engine as mem_val; watch ranges pause back to
   Python for MMIO).  Removes the ~3 µs/step Python protocol floor — at
   the current ~6 µs C++ step that is worth ~1.5×.
3. Batched machines across cores after that.

## 2026-07-07 running-argmax registers — nothing grows anymore

Tracked in jj:
- `vnwtsory` — convert constant-query hulls to running-argmax registers

What was built:
- The 12 remaining non-memory lookup heads all have **constant queries**
  ([QX, QY], both rows a lone term on the one-slot — the hardmax
  temperature).  With a constant query, every token's attention score
  QX·kx + QY·ky is a fixed function of its key, so the historical argmax
  can be maintained **incrementally at insert time**: a register per head
  holds (best score, best values); a live insert (cleared tokens skipped
  as before) folds in with replace-on-≥, which reproduces the LATEST
  tie-break exactly (ilp is monotone in double, so a later equal-key
  token never compares below an earlier one).  A query is a register
  read; empty register returns (0,0), matching an empty hull.
- Generation is guarded like everything else: eligibility is checked per
  head (constant query, LATEST tie-break, integer value rows), with the
  real-hull path as fallback.  For this model all 12 convert, so **the
  generated forward contains zero hull2d calls** — grep the emitted
  source: no .query, no .insert.
- Consequence: with memory on the O(1) structure and these on registers,
  **no engine state grows with the token log**.  Snapshot-and-rebuild
  checkpointing is now optional for the weights machine (still supported,
  still used by the NES console; the growth motivation is gone).

Receipts:
- **Full Klaus with checkpointing DISABLED: MATCHED — 30,646,177 steps in
  373 s (6.2 min, ~82K instr/s including the lockstep reference), zero
  rebuilds, throughput flat from the 1M-step mark to the last** — the
  flat curve is itself the nothing-grows receipt.
- Full nestest MATCHED: 0.1 s, ~100K instr/s.  Klaus 200K decimal,
  checkpoint-free: 2.0 s, ~98K instr/s.
- SMB through the weights: ~0.09 s/frame (~11 fps), frames byte-identical
  to the kernel and all prior engines.  6 weights + 47 fast tests green;
  machine_step 8.6 µs synthetic.
- Two-day arc totals: full-suite Klaus 57 → 6.2 min; SMB 0.1 → 11 fps
  (~110×); the weights machine now runs the 6502 with O(1) state through
  genuine 71-layer transformer inference.

Next: the C++ run loop (KernelBus shape — mem_val already is the RAM
mirror) to remove the ~3 µs/step Python protocol floor, then batched
machines across cores.

## 2026-07-07 ceiling analysis: where single-stream CPU tops out

Real-time SMB needs ~590K instr/s (measured ~9,800 instructions/frame ×
60 fps).  The weights machine is at ~110-116K effective (~11 fps).  The
8.6 µs step splits ~3 µs Python protocol / ~4 µs linear math / ~1.6 µs
everything else — and the linear part executes ~36K indexed loads in
~4 µs, i.e. **~3 loads per cycle, the M-series load-port hardware limit**
(hull-ablation receipt).  The linear math is no longer unoptimized code;
it runs at the speed the memory subsystem can feed it.

What implementation still buys, honestly bounded:
- **C++ run loop** (KernelBus shape; `mem_val` already is the RAM
  mirror): removes the ~3 µs Python floor → ~180K instr/s ≈ ~18 fps.
- **Schedule/width compaction** (old #3): fewer live persists →
  somewhat less nnz; realistically 1.2–1.4× → low-20s fps.
- Nothing else: single-stream threading is dead (71 sequential layers,
  ~70 ns each), int16 is dead (804/1551 slots exceed it), the gather is
  at the port floor.  **The current construction tops out ~20-25 fps on
  one core.**  Batching still buys fleet throughput (state is O(1) per
  machine now), just not single-game latency.

The remaining ~3× to real-time is a *construction* gap, not an
engineering gap.  Candidates, in expected payoff order:
1. **Opcode-routed conditional computation (MoE-style)**: the graph
   evaluates every instruction family every step — all ALU families,
   all addressing modes, BCD — and one-hot decode gates discard ~95%
   of it.  Sparse routing where the fetched opcode selects the expert
   subgraph is worth an order of magnitude, not a factor of 1.3.  This
   is a genuine transformer-architecture change (token-routed experts),
   with its own equivalence/verification story to build.
2. **Multi-instruction tokens**: amortizes per-forward fixed costs;
   modest once the run loop lands (the fixed costs are what it removes).
3. **Depth compaction**: fewer than 71 layers via graph restructuring
   (gadget fusion, wider per-layer FFN); bounded by the critical path.

## 2026-07-08 opcode-routed experts (MoE) — the construction lever lands

What was built (`model/engine_gen.py`, `cpu6502/weights_runner.py`;
scouting recommendation #2 / the MoE probe): the generated engine now emits,
alongside the union `step_core`, a pruned `step_core__<op>` per hot opcode, and
the run loop / `machine_step` dispatch on `ram[st_pc]` (== the graph's fetched
opcode) — a hard router that hoists the decode one-hot the graph already
computes.  A pending NMI and any opcode without an expert fall back to the exact
union step, so correctness is workload-agnostic.

- **Approach: prune the union model in place** (not a rebuilt per-expert model).
  Per opcode, partial-evaluate the graph with `is_op_<byte>` pinned, then take
  liveness over the *unfolded* dependency graph with the zero-folded dims
  removed as edges: a residual dim is live iff it still reaches a readout root
  (incl. the STATE/HALT head dims) through non-zero contributions.  Dead work
  (memory reads of unused addressing modes, other operations' ALU neurons)
  drops out; the taken branch's `gate=1` multiplies fold to a live linear map
  and are kept (they must still run in the union arithmetic).  The emitter keeps
  only a row's live columns — dropping a zero/dead column is exact term
  elimination — so passthrough/erase slot bookkeeping, memory (updated
  unconditionally via `insert_write`), and the running-argmax registers are
  untouched.  Register pruning is safe because a step that reads a register also
  folds the current, highest-position token, so a step that pruned an earlier
  fold cannot leave a stale winner.
- The per-layer emission was refactored to a **plan-then-emit** shape: hull/
  register id assignment happens once (shared by every body); the union body is
  byte-for-byte what it was, and expert bodies are the same lines with dead
  columns/units elided.  Default expert set = the SMB gameplay top-16 opcodes
  (~94%: the frame-poll `LDA abs`/`AND imm`/`BEQ` + idle `JMP abs` spin, plus
  common ALU/branch/store ops); `expert_opcodes=()` disables.

Receipts (all byte-identical; every expert swept by nestest + Klaus):
- Full nestest **MATCHED** (~155K instr/s, was ~129K); Klaus 200K decimal
  **MATCHED** (~130K/s, was ~119K); the codegen-vs-generic identity test,
  7 weights + 8 console + 47 fast tests all green; **SMB frame 61
  byte-identical to the kernel**, cycles exact.
- **Full Klaus re-validated: MATCHED — 30,646,177 steps (decimal) in 229.8 s
  (3.83 min, ~133K instr/s incl. the lockstep reference, throughput flat)**,
  checkpoint-free.  Full-suite arc: 57 → 18.9 → 15.3 → 9.4 → 6.3 → 6.2 → 4.55 →
  **3.83 min** across the engine generations (decimal Klaus only partly hits the
  expert set, so the aggregate gain understates the hot-path 2–2.7×).
- **Hot-path win**: the JMP-to-self synthetic loop (pure expert path) went
  **7.80 → 2.90 µs/step (2.7×, 128K → 345K instr/s)**.  A real SMB *gameplay*
  CPU stream (mixed hot opcodes) went **7.15 → 3.28 µs/step (2.18×, 140K → 305K
  instr/s)**.  Mean kept 1522/5308 ReGLU neurons (0.29×) and 73/114 lookup heads
  (0.64×) — above the probe's 0.20× because the in-place approach keeps the
  `gate=1` folded-linear neurons the union computes (a rebuilt-expert model
  would reach 0.20× and is the thesis MoE artifact).
- **SMB gameplay fps nearly doubles: 14.65 → 28.24 fps (1.93×)** measured at
  the frame level (drive to gameplay on the kernel, hand off to weights, hold
  right).  A gameplay frame is **CPU-bound**: 9,870 instr/frame split ~33.5 ms
  native CPU / **0.7 ms PPU render** / **1.0 ms Python glue** (MMIO pause
  servicing, ~242 device reads/frame + discarded trace-array concat).  So the
  numpy renderer and the per-poll MMIO round-trips are together only ~5% of the
  frame — not the bottleneck; the CPU step is.  (An earlier note here claimed
  the frame was render/MMIO-bound and unchanged by routing; that was a
  benchmark bug — `expert_opcodes` is a def-time default, so monkeypatching the
  module constant left both configs routed.  Corrected 2026-07-08.)

Path to real-time (SMB needs ~592K instr/s = 9,870 × 60): routed gameplay is
~305K instr/s (~28 fps), so ~1.9× more CPU is the remaining gap.  Next, in
payoff order:
1. Rebuilt-expert weights model (inline the `gate=1` multiplies → the probe's
   0.20× vs the current 0.29×, ≈1.45×, and a genuine Switch-style top-1 MoE for
   the write-up) and/or wider/per-family expert coverage.
2. Width/schedule compaction and poll-loop expert micro-optimization for the
   last stretch to 60 fps.
3. Batched machines across cores for throughput (state is O(1) per machine) —
   after single-game latency, not before.
   (Render/MMIO are ~5% — a later cleanup, not a real-time lever.)

2026-07-08 follow-up investigation:
- A local generated-engine prototype tried to fold only ReGLU neurons whose
  gate becomes a positive constant under an opcode pin.  It was byte-identical
  on the codegen-vs-generic test, but it found only ~5 folded-linear neurons per
  hot expert.  That is not the missing 0.29× → 0.20× lever and the prototype was
  reverted.
- The missing work is broader partial evaluation through persist/mux chains:
  e.g. the hot opcodes have hundreds of current-only ReGLUs that disappear from
  the folded graph, while only a handful are locally constant-gate ReGLUs.  So
  the real implementation is a rebuilt expert graph/model, not a small
  post-build activation fold inside the union weights.
- SMB histogram confirmation: top 4 opcodes cover 86.7% of gameplay, top 16
  cover 93.7%, and the top-16 weighted folded-term ratio is ~0.195 of the union
  (top 24 is still ~0.195).  Per-opcode rebuilt experts for the hot set are the
  right granularity; widening coverage beyond the hot set mostly helps tail
  correctness/perf smoothness, not the frame average.
- Implementation shape to pursue: extract the partial evaluator from
  `experiments/moe_probe/moe_probe.py` into reusable code that produces a
  folded `ProgramGraph`/schedule per expert, then emit/build expert-specific
  weights from that graph.  Keep the union body as fallback for NMI and cold
  opcodes.  The hard part is preserving shared attention state across experts:
  memory already has the global 16×4 KiB structure, but any non-memory latest
  registers must keep the same insert/query semantics when a previous expert
  pruned a register update.
- Width/schedule compaction should be evaluated after rebuilt experts, not
  before.  The union model is still 71 layers / D≈1550, while the folded
  per-opcode probe is 9-22 layers and ~0.16-0.28 terms.  For the CPU codegen
  path, fewer indexed loads matter most; fewer layer calls and smaller local
  width are secondary but should combine with poll-loop-specific experts for
  the remaining ~1.3-1.5× to 60 fps.

## 2026-07-08 rebuilt expert models land — a genuine top-1 routed MoE

Follows stage 1 below in the same session.

What was built (`model/weights.py`, `model/engine_gen.py`,
`cpu6502/weights_runner.py`):
- `build_model` now records per-head lookup provenance
  (`head_lookup_ids`/`head_pairs`), so an engine hosting several models
  built from folds of one union graph can map every model's heads onto
  shared attention state.
- `generate_source` experts are now **full rebuilt models** (fold →
  asap_schedule → build_model per hot opcode), each with its own residual
  layout and `step_core__<op>`; the in-place keep-set pruning is retired.
  The per-layer planning is one `plan_model(mdl, label, union_lookup_of)`:
  the union pass allocates hull ids / memory insert targets / running-
  argmax registers (keyed by union lookup id + value pair) in head-scan
  order, and expert passes resolve onto that state through provenance,
  with hard errors for any classification disagreement (memory head not
  structure-eligible, register missing, head needing a real hull).  The
  st_* registers + RAM mirror + memory structure + registers are the
  cross-model interface; expert d_model/layout never has to match.
- **Union byte-identity receipt**: the refactored generator emits the
  byte-identical no-expert source (int32 and f32) as the committed
  pre-refactor generator, checked by diffing the two emissions on the same
  model in one process.
- The runner's `_build_experts` (keep-sets) became `_build_expert_models`;
  it refuses folds with insert-side hazards, builds per-expert step_specs
  from the fold's name map, and disables experts for f32 arith.  Default
  16 experts add ~6 s to machine build; the 17-model TU compiles in ~40 s
  (cached by digest).

Receipts (all lockstep-exact / byte-identical; 72 tests green, ruff clean):
- Full nestest MATCHED: **~176K instr/s** (was ~155K in-place).
- Klaus 200K decimal MATCHED: ~140K instr/s (was ~130K; Klaus barely hits
  the SMB-tuned expert set).
- **Full Klaus re-validated: MATCHED — 30,646,177 steps (decimal) in
  216.3 s (3.60 min, ~142K instr/s sustained incl. the lockstep reference,
  throughput flat), checkpoint-free.**  Full-suite arc: 57 → 18.9 → 15.3 →
  9.4 → 6.3 → 6.2 → 4.55 → 3.83 → **3.60 min** across the engine
  generations.
- SMB frame 61 byte-identical to the kernel, cycles exact (1,816,641).
- **JMP-to-self hot path: 7.76 → 1.75 µs/step (571K instr/s)** — 1.66×
  over the in-place experts' 2.90 µs, and at SMB's real-time budget.
- **Frame-poll loop (LDA abs / AND imm / BEQ): 8.29 → 2.15 µs/step**
  (465K instr/s).
- Union↔expert transition probe (NOP+JMP alternating): ~0.26 µs/step
  overhead — dispatch/I-cache cost is small.
- **SMB gameplay: 29.68 fps** (33.7 ms/frame; was 28.24 in-place),
  title-screen marginal frame ~27 ms (~37 fps, was ~73 ms).

Why gameplay didn't follow the hot path (measured, not guessed):
- Coverage: top-16 opcodes cover 93.7% of gameplay instructions; the 6.3%
  cold tail at ~7.8 µs costs ~0.5 µs/step on the frame average.  A top-48
  expert build (99.1% coverage, from the checked-in gameplay histogram)
  measured **33.30 fps** (30.0 ms/frame) — +12%, matching the tail model.
  Kept top-16 as the default (top-48 adds ~25 s of fold+build per machine
  construction; wrong tradeoff for tests) — widening is a one-line
  `expert_opcodes=` away.
- The rest of the frame is *busy-frame logic*: active gameplay frames are
  logic-dominated (the poll fraction shrinks exactly when the frame is
  expensive), and the broad logic mix runs at ~2.5-3.5 µs/step through the
  mid-size experts (JSR/RTS/STA abs,Y are 25-31 layers, d_model up to 540).
  Real-time (~1.7 µs/step average) therefore needs the *per-expert step
  cheaper*, not more coverage.

Next, in payoff order:
1. **Per-expert width/schedule compaction** — the exact-width MILP is
   plausibly tractable at expert scale (~1,700 ops / 17-31 layers vs the
   union's 6.4K ops / 71 layers where it blew up).  Fewer live slots →
   fewer passthrough heads/neurons → fewer indexed loads, on exactly the
   graphs that ship.
2. **Pin the fetched opcode byte** in the fold (the router predicate
   `ram[pc] == byte` is stronger than `is_op=1`): deletes one of the three
   fetch reads and the 256-way decode residue from every expert.
3. Poll-loop sequence expert (LDA/AND/BEQ as one folded step) for the
   frame-wait fraction; then batching for fleet throughput.

## 2026-07-08 rebuilt experts stage 1: the fold is real and receipted

Not yet in jj (this session's working copy).

What was built (`graph/fold.py`, `graph/eval.py`, `cpu6502/tests/test_fold.py`):
- **`GraphFolder.fold(pins)`** graduates the moe_probe partial evaluator into
  a graph transform that *materializes* a new `ProgramGraph`: pinned decode
  gates become constants, gate=1 ReGLUs inline as exact linear maps (the
  0.29x → 0.195x lever the in-place experts cannot reach), dead branches
  prune by liveness, and folded-away readouts (e.g. `is_store` for JMP) come
  back as named persists so step_specs always resolve by name.  Provenance
  is kept for the engine: union lookup ids per new LookUp, `memory_group`
  tags copied, and an **insert-side hazard check** (reachability from any
  lookup's key/value expressions to a pinned gate — flagged, not assumed
  safe; empty for all decode pins, which is the license to share the memory
  structure and running-argmax registers across expert models).
- **`ProgramEvaluator`** (`graph/eval.py`): a direct float64 evaluator for
  any ProgramGraph — insert-then-query hardmax with the LATEST tie-break —
  as an engine-free semantic oracle for graph transforms.

Receipts (all in the fast suite, ~1 s total; 58 fast tests green, ruff clean):
- Unpinned fold reproduces the union graph exactly: 71 layers, 29,136 terms,
  op counts identical to the probe baseline.
- Folded expert graphs shadow the union in lockstep: the union graph runs
  the verify program through ProgramEvaluator, checked against the reference
  oracle on the architectural fields every step; each expert (STA/LDA/BNE/
  JMP/JSR/ADC-imm) replays the identical token stream and must reproduce
  **all 22 runtime readouts exactly** on steps whose fetched opcode matches
  its pin — the engine's routed-dispatch contract, engine-free.
- Pipeline smoke: folded graphs go through `asap_schedule` + `build_model`
  **unmodified** — JMP abs becomes a real 17-layer d_model=348 transformer
  (union: 71 layers, d_model≈1550), LDA abs 22 layers / 350, memory heads
  intact.
- Correction to the probe: its per-opcode depths (9/13/22) were optimistic —
  it kept original expressions over pruned dim sets, so dependencies through
  inlined gate=1 folds were dropped as dangling edges.  True materialized
  depths run ~1.4-1.7x deeper (JMP abs 17 vs 10) but still ~0.2-0.4x of the
  union.  Terms (the load-bound cost) match the probe, which is what the CPU
  path cares about; the depth correction matters for the batched/GPU story.

Next (stage 2/3): schedule+build the 16 hot experts in the runner; rework
`engine_gen.generate_source` to emit one TU with the union model plus
per-expert *models* (each with its own slot layout and step_core — the st_*
scalars, ram mirror, memory structure, and registers are the interface),
mapping expert memory heads / registers onto the union's structures by
lookup provenance, with loud guards where an expert's head classification
disagrees with the union's.  Then the usual chain: codegen-vs-generic
identity, nestest, Klaus 200K, full Klaus, SMB frame + fps.

## 2026-07-07 C++ run loop (KernelBus shape) — Python floor removed

What was built (`model/engine_gen.py`, `cpu6502/weights_runner.py`,
`nes/console.py`; scouting recommendation #2 / the roadmap's next-step #2):
- The generated engine's per-instruction work was refactored into a
  `step_core()` that reads the persistent `st_*` registers, runs the whole
  machine step, and advances `st_*`.  `machine_step(args, out)` now just
  seeds `st_*` and calls it (byte-identical to before), and a new native
  `run(max_steps, stop_on_trap, max_cycles)` loops `step_core` entirely in
  C++ — a 64 KiB RAM mirror (updated in `insert_write_f32`), shadow-decode
  read-watch (pause *before* a device read), write-watch (pause after),
  base-cycle counting, and jump-to-self trap detection — returning the
  exact same result dict as the C++ kernel's `run()`.  The `READ_MODE`/
  `CYCLES` tables derive from `spec.OPCODES` (same source as kernel.py).
- Because the engine now speaks the kernel's machine API (`run`,
  `set_run_state`/`get_run_state`, `watch_read/write`, `set_nmi`,
  `inject_write`, `read_ram`, `dump_ram`), `Cpu6502TransformerMachine`
  exposes that API and **the existing `KernelBus` drives the weights engine
  unchanged** — the console's `== "kernel"` branches became an
  `_uses_kernel_bus` flag, and the weights backend now runs natively,
  pausing to Python only for MMIO.  The Python-stepped `Bus`/`GraphCore`
  path remains as the fallback for engines without a run loop.
- Klaus/nestest stream through the loop: `run_image_arrays` (single native
  call) + `iter_run_arrays` (fixed-size chunks, `w_row` rebased per chunk)
  reuse `lockstep_arrays_vs_reference`.  Engine state is O(1), so streaming
  needs no checkpoint rebuild.

Receipts (all lockstep-exact / byte-identical; 47 fast + 8 console + 7
weights tests green, ruff clean):
- New `test_run_loop_matches_per_step`: the native run loop's trace
  (states, writes, ir/ea/data) is identical to the verified per-step path,
  single-call and chunked, and it detects the JMP-to-self trap.
- New `test_weights_run_loop_console_matches_kernel`: the transformer
  driven by the run loop renders the MMIO test cart (OAM DMA, PPU
  registers, vblank NMI) byte- and cycle-identical to the kernel, plainly
  and with a checkpoint every frame (exercises dump_ram/load_image/
  set_state).
- Full nestest MATCHED: **0.1 s, ~132K instr/s** (was ~98K per-step).
- Klaus 200K decimal MATCHED: **1.7 s, ~119K instr/s** (was ~93K).
- Synthetic step: **9.69 µs → 6.92 µs (1.40×)** — the pure C++ loop; the
  removed cost was the ~1.2 µs Python glue plus the per-call pybind
  marshalling of 13 int args + `tolist()`.
- **SMB through the weights: ~73 ms/frame (~13.8 fps)** marginal, up from
  ~90 ms (~11 fps); frame 61 byte-identical to the kernel, cycles exact
  (1,816,641).
- **Full Klaus through the run loop: MATCHED — 30,646,177 steps
  (decimal mode) in 273 s (4.55 min, ~112K instr/s sustained incl. the
  lockstep reference), throughput flat 24M→30M** (the chunked streaming
  generator + `w_row` rebasing hold at scale; nothing grows).  Full-suite
  run: 57 → 18.9 → 15.3 → 9.4 → 6.3 → 6.2 → **4.55 min** across the engine
  generations.

Where the remaining ~6.9 µs lives: it is now essentially all C++ compute
(the load-bound linear gather at the M-series port floor + hull/mem +
readout), with only a thin per-call boundary left.  The single-stream CPU
construction is at its ceiling; the next real lever is the routed-MoE
construction from the scouting probe below (depth 71→~13, terms ~0.19×),
which the run loop is the natural host for — the expert dispatch reads
`mem_val[pc]`/`ram[pc]` and jumps to the expert `step_core` before the
forward.

## 2026-07-07 MoE routing scouting probe (measured, not guessed)

No repo code changes; probe lives in `experiments/moe_probe/`
(`moe_probe.py` = partial evaluation + liveness + depth, `smb_weighting.py`
= real-gameplay opcode histogram; results JSONs checked in alongside).

Method: partial-evaluate the built graph per candidate expert by pinning
the decode one-hot gates (`is_op_XX`, `is_nmi`) to constants — exactly the
information a hard router supplies.  Constant folding with linear inlining
for known-gate multiplies (so `mux(gate=1, a, b)` folds to `a` and dead
branches prune), backward liveness from `_RUNTIME_READ_DIMS`, then the
scheduler's own phase-parity critical path (`_min_layers`) on the pruned
graph.  **Self-validation: the unpinned baseline reproduces the union
graph at exactly 71 layers**, terms=29,136 (the measured engine visits
38,067 CSR nnz — the ~1.3× structural gap is fusion composition; ratios
transfer).

Per-expert results (ratios vs the 71-layer / 29,136-term union):
- Per-opcode experts: depth min/med/max = **9/13/22** layers, terms
  0.16–0.28×.  Family experts cost barely more than their worst member
  (in-family decode is cheap), *except* where modes diverge: family jmp
  is depth 28 (JMP ind's pointer read) while JMP abs alone is depth 10.
- Outliers: PLP/RTI at 15–17K terms — that's `_low_bit`'s 256-way match
  gadgets ×6 flags; an expert build replaces them with 8-bit extraction
  (deep-narrow is fine inside an 11-layer expert).  NMI expert: 4.8K
  terms, depth 9.
- Floor: even CLC is ~4.6K terms / 9 layers — the shared machinery
  (3-read instruction fetch across 16 memory groups, latest-state,
  write-channel key construction, trace readout).  Routing alone cannot
  go below ~0.16×; further reduction means attacking the fetch gadgets.

SMB workload (reference backend, scripted Start + hold-right + jumps,
Mario verifiably crosses page 1; 750 frames, ~10.7K instr/frame):
- **87% of executed instructions are 4 opcodes** — LDA abs / AND imm /
  BEQ rel (the frame-wait poll loop) + JMP abs (idle spin).  Top-16
  opcodes cover 93.7%.
- Gameplay-weighted per-opcode experts: **depth 12.8 (0.18×), terms
  0.194×, live lookups 64.6 of 113**.  Family-only experts are notably
  worse (0.25×, depth 0.42×) because hot families drag in cold modes —
  granularity should be per-opcode for the hot ~16, family for the rest.

Implications:
- On the CPU engine (straight-line codegen), depth is free; the win is
  loads: linear part ~4 µs × 0.194 ≈ 0.8 µs, lookup-structure work
  ~×0.57.  Stacked on the C++ run loop (kills the ~3 µs Python floor),
  the routed step lands ~1.6–2.0 µs ≈ **500–630K instr/s ≈ 45–60 fps**
  — SMB real-time needs ~640K (10.7K instr/frame × 60).  Width/schedule
  compaction or trimming the 4 poll-loop experts covers the remainder.
- The depth result (71 → ~13 median) is what matters for the batched/GPU
  path (fewer sequential dispatches) and for the write-up: the routed
  model is a genuine top-1 hard-routed MoE (Switch-style) whose router
  is the existing one-hot decode hoisted out of multiplicative gating.
- Engine-side dispatch is nearly free: the run loop reads `mem_val[pc]`
  (the RAM mirror the engine already maintains) and jumps to the expert
  step function before the forward.  Verification story: partial
  evaluation is algebraic (fold rules mirror the emission), receipts =
  the existing lockstep chain per expert; full Klaus + nestest sweep
  every opcode through every expert.
- Caveats recorded: probe terms undercount real nnz ~1.3× (structural);
  `operand_mem`/`data_out` stay live in every expert because the trace
  convention reports `data` — an expert build could trim further if the
  trace field is relaxed for non-memory ops; ~60–70 expert functions is
  fine for I-cache since the hot 4 stay resident.

Next, in payoff order:
1. **C++ run loop** (unchanged from the previous log; KernelBus shape,
   `mem_val` is already the RAM mirror).  Worth ~1.5× on its own and it
   is where the expert dispatch will live, so it lands first.
2. **Routed expert construction**: partial-eval expert emission in
   `engine_gen.py` (per-opcode for the hot ~16, per-family otherwise,
   union fallback for anything unproven), router = `mem_val[pc]` table;
   then the same construction as a real routed-MoE weights model for
   the thesis.  ~3–3.5× on top of the run loop.
3. Width compaction / poll-loop expert micro-optimization if the last
   ~1.2× to 60 fps needs it.

## Prioritized next steps

### 1. Key rescaling in the memory-lookup construction
The hard-attention half is now experimentally solved by grouped memory
lookups: `_memory_at` splits 64 KiB into 16 local 4 KiB parabolas, and the
C++ engine supports fp32 execution.  NES frame parity through the fp32
engine and weights-runner checkpointing are done (2026-07-06, see the
hardening/checkpointing log).  Remaining piece:
- separately solve true-softmax latest-write semantics if `--cache standard`
  remains a target for arbitrary high-address programs.

Payoff remains: the default float32 engine path makes GPU/batched-machine
work viable (MPS has no fp64).

### 2. Full Klaus through the transformer weights — DONE 2026-07-06
MATCHED: 30,646,177 steps exact through the fp32 engine in ~57 min with
36 invisible checkpoints (see the hardening/checkpointing log); re-run
through the generated engine 2026-07-07 in **18.9 min** at ~27K instr/s
sustained with 146 checkpoints.  Rerun:
`6502-klaus --runner weights --graph-steps 30646177` (checkpointing
defaults to every 250K tokens; every forward inserts into all 92 shared
hulls, so a 1M-token window costs ~6 GB of hull log — the rebuild itself
is milliseconds).

### 3. MILP formulation compaction (then, optionally, GNN warm starts)
Reformulate with mobility windows (variables only for ops with slack;
ASAP/ALAP fixes the rest) and an aggregate width model instead of
per-dim/per-boundary binaries — that's a ~100× smaller model HiGHS can
actually chew.  Only after that is a learned component (Gasse-style GNN
branching / Neural-Diving warm starts over the computation graph)
worth building; the exact-schedule prize is ≤ ~1.8× compute.

### 4. NES fidelity, in payoff order
- Branch-taken / page-cross penalty cycles (base cycles only today) —
  identical tables in `cpu6502/kernel.py` + `nes/bus.py`.
- Scanline- or split-aware rendering so SMB's sprite-0 status bar stops
  scrolling (frame-at-once renderer today; `nes/ppu.py`).
- RAM mirror device (`$0800-$1FFF`), sprite overflow flag, 8×16 sprite
  coverage tests.
- Mapper 1 (MMC1) to open the post-launch library; then audio.

### 5. GPU / batched fleet (after #1 only)
One stream stays CPU-bound by dispatch latency: the temporary Metal CSR
probe measured ~100 us for a synchronous dispatch, about the same as a full
fused CPU `_forward`.  GPUs pay off as **throughput across many machines**
or with a custom backend that fuses much of the layer stack into one/few
command buffers and ports/avoids CPU hull synchronization.

### 6. Write-up
The artifacts already exist: the verification-chain table, the numerics
rules (each learned from a real divergence), the fp32 step-0 receipt,
the MILP postmortem + balanced-scheduler numbers, SMB title/gameplay
frames (kernel) and the byte-identical transformer frame.  Outline
against `cpu6502/README.md`; the memory file has the session-by-session
history.

### 0. Housekeeping (cheap, do early)
- **Commit**: everything since "Initial commit" is one giant jj
  working-copy change ("starting work on 6502").  Split into reviewable
  commits (cleanup/purge → MMIO+NMI → NES console → checkpointing →
  weights transformer → scaling).
- `plan.yaml` / `allocation.yaml` are written to CWD on every weights
  build (gitignored, but noisy); consider routing to a build dir.
- CI's `slow` job now includes multi-minute weights tests — check the
  runtime budget.

## Command cheat sheet

```bash
uv run pytest -m "not slow"                    # 47 fast tests
uv run pytest                                  # + kernel/weights/console suites
uv run 6502-klaus --runner cpp-full            # full Klaus, ~48 s, exact step count
uv run 6502-nestest --runner both              # golden log: reference + graph + kernel
uv run 6502-nestest --runner weights           # full nestest through the generated int32 engine, ~0.1 s
uv run 6502-klaus --runner weights --graph-steps 30646177  # full weights receipt
uv run 6502-transformer --steps 30
uv run nes-run transformer_nes/nes/examples/smb.nes --frames 61 \
    --warmup-frames 60 --backend weights --out frame.ppm
```
