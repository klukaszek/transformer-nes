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
| `cpu6502/weights_runner.py` (real transformer) | ~7-8K after boot | default fp32 engine; full nestest MATCHED in ~1.2 s; Klaus 20K prefix MATCHED |

The weights transformer is genuine inference (d_model≈1550, 71 layers,
analytic weights, no training).  The default path is the float32 C++ engine
(`model/weights_engine.cpp`), which runs the whole layer stack per call;
dense/sparse Python paths remain diagnostics.

**NES layer** (`nes/`): MMIO bus with shadow-decode read injection,
in-graph NMI, frame-level PPU, OAM DMA, controllers, cycle-budget frame
pacing, invisible snapshot-and-rebuild checkpointing, and mid-game
backend handoff.  nestest boots and passes its suite interactively; SMB
plays at ~45 fps on the kernel; an SMB frame renders **through the
transformer weights** byte-identical to the kernel in ~10 s
(`nes-run --warmup-frames 60 --backend weights`).

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
- All residual/FFN values are dyadic rationals < 2²⁴, so the float32 engine
  is the default execution mode.
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
- Current fused CPU engine raw `_forward` on a synthetic STATE token is
  ~93.7 us/position (~10.7K positions/s), while full machine stepping is
  ~7.4-7.7K instr/s because Python still assembles/readbacks state and
  emits writes.

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

## Prioritized next steps

### 1. Key rescaling in the memory-lookup construction
The hard-attention half is now experimentally solved by grouped memory
lookups: `_memory_at` splits 64 KiB into 16 local 4 KiB parabolas, and the
C++ engine supports fp32 execution.  Next pieces:
- run NES frame parity through the fp32 engine after the direct-WRITE
  optimization;
- port checkpoint/snapshot support to the weights runner so long fp32 runs
  do not grow unbounded hull logs;
- separately solve true-softmax latest-write semantics if `--cache standard`
  remains a target for arbitrary high-address programs.

Payoff remains: the default float32 engine path makes GPU/batched-machine
work viable (MPS has no fp64).

### 2. Full Klaus through the transformer weights (30.6M instructions)
~3.7 h at current engine speed.  **Blocker first**: the weights machine
has no checkpointing, and 30M instructions ≈ 45M token positions will
blow up the 19 lookup-head hulls.  Port snapshot-and-rebuild to
`Cpu6502TransformerMachine` (needs a RAM mirror in the runner — track
writes like `nes/bus.py` does — then reseed like `KernelBus.checkpoint`).
Then run overnight: `6502-klaus --runner weights --graph-steps 30646177`.

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
uv run 6502-nestest --runner weights           # full nestest through default fp32 engine, ~1-2 s
uv run 6502-klaus --runner weights --graph-steps 200000
uv run 6502-transformer --steps 30 --cache standard --matmul sparse   # true softmax diagnostic
uv run nes-run transformer_vm/nes/examples/smb.nes --frames 61 \
    --warmup-frames 60 --backend weights --out frame.ppm
```
