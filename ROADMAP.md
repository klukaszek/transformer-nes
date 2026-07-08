# Roadmap

Status snapshot and prioritized next steps.  Written 2026-07-06 as the
handoff point for fresh working sessions; the design details live in
[`transformer_vm/cpu6502/README.md`](transformer_vm/cpu6502/README.md).

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
  CPU stream (mixed hot opcodes, MMIO/render excluded) went **7.15 → 3.28
  µs/step (2.18×, 140K → 305K instr/s)**.  Mean kept 1522/5308 ReGLU neurons
  (0.29×) and 73/114 lookup heads (0.64×) — above the probe's 0.20× because the
  in-place approach keeps the `gate=1` folded-linear neurons the union computes
  (a rebuilt-expert model would reach 0.20× and is the thesis MoE artifact).
- **Honest caveat**: the SMB *frame* time is unchanged (~34 ms) — it is bound
  by the numpy PPU render plus the per-poll MMIO round-trips (each device read
  pauses the native loop back to Python), not the CPU step.  Routing speeds up
  CPU-bound streams (Klaus-hot, gameplay CPU); moving SMB *fps* now needs render
  / MMIO-batching work, not more CPU throughput.

Next, in payoff order:
1. Rebuilt-expert weights model (inline the `gate=1` multiplies → the probe's
   0.20× and a genuine Switch-style top-1 MoE for the write-up) and/or wider
   expert coverage (per-family for cold opcodes) if a CPU-bound target needs it.
2. The SMB frame is now render/MMIO-bound: batch device reads / speed the PPU
   renderer to convert the CPU headroom into fps.
3. Batched machines across cores (state is O(1) per machine).

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
uv run 6502-klaus --runner weights --graph-steps 200000
uv run 6502-transformer --steps 30 --cache standard --matmul sparse   # true softmax diagnostic
uv run nes-run transformer_vm/nes/examples/smb.nes --frames 61 \
    --warmup-frames 60 --backend weights --out frame.ppm
```
