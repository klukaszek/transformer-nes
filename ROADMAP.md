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
| `cpu6502/weights_runner.py` (real transformer) | ~800–2300 | full nestest MATCHED in 10.8 s; Klaus 200K prefix MATCHED |

The weights transformer is genuine inference (d_model=1468, 66 layers,
analytic weights, no training) with three byte-identical execution paths
(`--matmul dense|sparse|engine`); the C++ engine
(`model/weights_engine.cpp`) runs the whole layer stack per call.

**NES layer** (`nes/`): MMIO bus with shadow-decode read injection,
in-graph NMI, frame-level PPU, OAM DMA, controllers, cycle-budget frame
pacing, invisible snapshot-and-rebuild checkpointing, and mid-game
backend handoff.  nestest boots and passes its suite interactively; SMB
plays at ~45 fps on the kernel; an SMB frame renders **through the
transformer weights** byte-identical to the kernel in ~10 s
(`nes-run --warmup-frames 60 --backend weights`).

**Scheduling**: the optimal-width MILP does not scale (formulation
blowup, not the HiGHS engine); `scheduler/milp.py:asap_schedule` is a
deterministic load-balanced critical-path scheduler (66 layers, D=1468
vs liveness floor ~1100 → at most ~1.8× compute left on the table).

**Numerics facts** (each has a receipt in tests or the README):
- float64 is needed **only** for the K-side attention-score construction
  (`ky = -k²…` reaches 2³² with gap 1; fp32 diverges at nestest step 0,
  matches on low-address programs).
- All residual/FFN values are dyadic rationals < 2²⁴ → sums exact under
  any association (why dense/sparse/engine agree bytewise).
- Rewrites of high addresses tie below one double ulp; hull LATEST
  resolves, true softmax averages — keep rewrite-heavy verification in
  low memory until key rescaling (below) lands.

## Prioritized next steps

### 1. Key rescaling in the memory-lookup construction
The single change that unblocks the most: redesign `_memory_at`'s key
parabola in `cpu6502/interpreter.py` so scores stay < 2²⁴ (e.g. split
addresses hi/lo across the 2D key or across heads).  Payoff:
- full float32 end-to-end → **GPU becomes viable** (MPS has no fp64);
- kills the high-address rewrite tie hazard → true-softmax runs work on
  arbitrary programs (full Klaus through `--cache standard`).
Re-verify the whole chain after (evaluator, kernel, weights all consume
the same graph).

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
One stream stays CPU-bound by dispatch latency (~400 tiny ops/token,
strictly sequential): GPUs pay off as **throughput across many machines**
— a batch of NES instances, fp32 weights, fused layer stack (MLX or
torch-MPS), dense or block-sparse attention with checkpoint-bounded n.

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
uv run 6502-nestest --runner weights           # full nestest through the transformer, ~11 s
uv run 6502-klaus --runner weights --graph-steps 200000
uv run 6502-transformer --steps 30 --cache standard --matmul sparse   # true softmax
uv run nes-run transformer_vm/nes/examples/smb.nes --frames 61 \
    --warmup-frames 60 --backend weights --out frame.ppm
```
