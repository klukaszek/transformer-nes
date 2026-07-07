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
| `cpu6502/weights_runner.py` (real transformer) | ~62-107K after boot | default generated engine (codegen, exact fixed-point int32, C++ machine_step); full nestest MATCHED in ~0.1 s; **full Klaus MATCHED: 30,646,177 steps exact** (with invisible checkpointing) |

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
transformer weights** byte-identical to the kernel at ~0.12 s/frame
(~8.3 fps with the int32 generated engine, 2026-07-07;
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
