# 6502 in the computation graph

A MOS 6502 CPU expressed as an analytic computation graph for the NES.  One
transformer step executes one 6502 instruction.

This package is the *pure* CPU: cores, conformance suites, and nothing
console-shaped.  The system layer (MMIO bus, devices, frame-level PPU,
cartridge loader) lives in `transformer_nes/nes/` and depends on this
package one-way.

## Components

| File | Role |
|------|------|
| `spec.py` | Single source of truth: 227-opcode table (151 official + stable unofficial), flags, addressing modes, tiny assembler |
| `interpreter.py` | The 6502 as a graph: ~6,500 dimensions built from ReGLU/persist/lookup primitives |
| `construction.py` | Construction boundary: builds the CPU graph and exposes handles used by folding, scheduling, and export |
| `trace.py` | Canonical trace format shared by the reference and weights backends |
| `backends/reference.py` | Plain-Python oracle core.  Trusted by construction; every other backend is verified against it |
| `backends/weights/run_protocol.py` | The weights engine's native run-loop protocol (tables + C++ blocks) |
| `backends/weights/runner.py` | The transformer machine: fixed production construction, engine-cache orchestration, and machine protocol |
| `backends/weights/experts.py` | Fold-and-build expert models (per-opcode, sequence, shape) with their dispatch licenses |
| `backends/weights/windows.py` | ROM-independent shape library, window scanning, and shape matching |
| `examples/klaus/harness.py` | Klaus Dormann functional-test harness (`6502-klaus`), next to its binary |

The nestest harness (`6502-nestest`, Nintendulator golden-log comparison)
lives in `examples/nestest.py`; its ROM and golden log sit with the other
NES assets in `transformer_nes/nes/examples/`, which the harness reads by
path.

## Construction and execution model

`interpreter.py` constructs one static append-only CPU graph. The graph is
folded into licensed experts, scheduled into transformer layers, and compiled
to the generated C++ weights engine. Its runtime state is represented by the
engine's residual slots and exact attention structures:

- **`start` token** — carries the initial PC.
- **WRITE tokens** — one per memory write, carrying `(addr, value)`.  The
  64 KiB image is loaded as one WRITE token per **nonzero** byte; reads of
  unwritten addresses are masked to zero by an exact-address check on the
  nearest-key attention hit.  Code and data share the log, so instruction
  fetch is just three memory reads at PC, PC+1, PC+2.
- **STATE tokens** — the full architectural state (PC, A/X/Y/S, eight flag
  bits) plus the previous instruction's fetch/EA/data fields.  The graph
  reads "current state" via a latest-STATE attention lookup.

The weights run loop reads the named result dimensions, emits up to three
WRITE records (JSR pushes two bytes, BRK three), builds the next STATE record,
and stops on HALT or a jump-to-self trap. The reference core provides the
independent execution oracle.

## Two backends, one semantics

| Backend | Speed (steps/s) | Role |
|---------|-----------------|------|
| `reference.py` (`run_traps`) | ~2.5M | Oracle; defines correct behavior |
| `weights/runner.py` (real transformer) | ~435K (full Klaus, incl. the lockstep reference) | The graph compiled to actual hardmax-attention + ReGLU transformer weights |

### The transformer itself

`backends/weights/` (`6502-transformer`) closes the loop on the project's
premise: the graph is scheduled onto layers and compiled by
`model/weights.py` into actual transformer weight matrices
(`WeightModel` — decimal-enabled d_model=624, 56 layers, 312 attention heads),
constructed analytically, no training.  Execution is genuine transformer
inference: embeddings, Q/K/V projections, hardmax attention, ReGLU FFN,
residual stream, output head.  There is **no position encoding** — token
order enters the model only through the LATEST tie-break (passthrough
heads tie every token with constant Q/K and take the latest, i.e.
themselves).  The runtime is the generated straight-line C++
engine (`model/engine_gen.py`), exact integer end-to-end: a fixed-point
int32 residual, exact nearest-address memory structures, and integer
running-argmax registers (`model/classify.py` holds the licenses; unsupported
lookup shapes are rejected during code generation).

Scheduling: `model/schedule.py:asap_schedule` takes the critical path and
load-balances ops across its slack with a reverse-topological leveling
pass (deterministic; ties go to the latest phase, which also shortens
lifetimes).  It is the only scheduler.  A persist-moving hygiene sweep
and an optimal-lifetime MILP were both excised after measurement — each
priced only part of the emitted cost.  The MILP went last (2026-07-22):
once experts got small enough to solve, it cut total dim lifetime 31.6%
yet produced an identical `d_model` and only 12.7% fewer passthroughs
(it trades `pt_h` down and `pt_g` up), for ~13 s of solve per expert —
about 1.7 h across a full build.  It can never beat ASAP on depth
either: it is handed `min_layers` as its layer budget, the same
critical-path floor ASAP already achieves.

Performance: the analytic weights are ~99.9% zeros, so inference runs the
same linear maps sparsely. `model/engine_gen.py` emits model-specific C++
with indices and coefficients embedded in expressions. The residual is exact
fixed-point int32; shifts are guarded and signed wrap is defined. Memory heads
use exact nearest-address structures and constant-query heads use integer
running-argmax registers. Expert
folds additionally rebuild each `memory_at` site as ONE fused 64K-span
lookup plus its exact-address mask (`graph/fold.py:masked_exact_read`) —
the 16-group form serves the union's f32 export score range, which the
exact-int expert engines never compute. Generated
artifacts are cached by a digest covering every source file that shapes
emission; a manifest hit skips folding and emission and dlopens the compiled
core directly.

The production core is workload-independent: all 227 documented opcodes use
compact delta/control/effect experts, alongside a canonical
**shape-expert library** folded once with pc and operands symbolic (any ROM,
homebrew included, zero per-ROM compilation). An explicit caller may attach a
per-ROM **sequence module** that outranks those tiers by pinning licensed
instruction windows as one forward (~160 KB TU, ~1 s cached). Interrupts,
unknown bytes retain the exact union path. Documented opcodes derive their full
trace receipt in the engine around compact execution instead of carrying a
second diagnostic transformer. Compact writes are
staged and routed through the same mapper and watch machinery as full
receipts before the instruction retires. Static control is engine metadata;
dynamic control returns one target slot, and only BRK returns a halt slot.

With the generated engine, **the full 8,991-instruction nestest suite runs
through the actual transformer weights in under 0.1 s** (`6502-nestest
--runner weights`) and the full 30.6M-instruction Klaus suite
lockstep-matches with decimal mode in ~70 s (435 K instructions/s); engine
state is O(1).  Gameplay
through the weights on the recorded input tapes runs at ~2,290 fps (SMB),
~1,300 fps (Zelda), ~930 fps (SMB3), ~920 fps (Mega Man 6) and ~850 fps
(Castlevania III), byte-identical frame-for-frame to the reference-driven
console under the same tapes.

Memory is split into 16 × 4 KiB groups in the union graph. Address scores
therefore remain in an exactly represented range, while LATEST ties are
resolved by insertion order. The standalone safetensors export remains
float32 by default; that storage choice is separate from the generated
runtime's int32 arithmetic (and from the expert folds' fused 64K reads,
which exist only in the engines' execution strategy, never the artifact).

### Compilation pipeline

1. The graph's dimensions form a static schedule in id (creation) order,
   which is topological by construction.
2. A backward liveness pass keeps only entries that transitively feed an
   externally read dim (`RUNTIME_READ_DIMS` + output-score terms).
3. Expressions are emitted as straight-line arithmetic over locals; lookup
   queries are inlined against the licensed exact attention state (nearest
   memory structure or running-argmax register).
4. Token inserts are compiled separately: each lookup's key/value
   expressions depend only on input dims through a small Persist/ReGLU
   chain, so inserting a token never evaluates the full graph.
5. Memory lookups with identical key/value expressions share one backing
   cache and receive one insert per WRITE token; the generated engines
   serve all 112 memory lookup sites from 16 exact per-4KiB-group
   structures (one insert per write, one O(1) query per read behind the
   group-dispatch check).

## Verification methodology

Chain of trust, each link checked instruction-by-instruction:

```
nestest golden log (Nintendulator)  ⇔  reference.py  ⇔  weights/runner.py
Klaus functional test (self-checking)  ⇑
```

- **Klaus** (`6502_functional_test.bin`): all 151 official opcodes,
  including decimal mode.  The reference and the weights transformer run
  the **full 30,646,177-instruction suite** to the success trap at
  `$3469`; both must agree on the exact step count.  Klaus reaches JSR/indirect addressing
  only after ~20K instructions, stack wrap at ~40K, shifts at ~46K, and
  decimal mode at ~26M — shallow prefix runs prove very little.
- **nestest**: all 8,991 golden-log lines compared field-by-field
  (PC/A/X/Y/P/SP), covering the unofficial opcodes (LAX, SAX, DCP, ISB,
  SLO, RLA, SRE, RRA, `SBC $EB`, NOP variants).  Runs in 2A03 mode
  (`bcd=False` / `bcd_enabled=False`) because the NES CPU has no decimal
  mode.

```bash
uv run 6502-klaus --runner reference          # full suite, ~13s
uv run 6502-klaus --runner weights --max-steps 30700000  # full suite through the WEIGHTS, ~90s
uv run 6502-nestest --runner both             # reference vs golden log + weights lockstep
uv run 6502-nestest --runner weights
```

## Numerics: rules that keep the graph exact

These were each learned the hard way; violating any of them produces bugs
that only surface millions of instructions into a run.

1. **Integer thresholds only.**  `stepglu(one, b)` equals `step(b >= 0)`
   only for integer `b`; at `b = -0.5` it returns 0.5.  The decimal-mode
   nibble carries originally used `x - 9.5` thresholds and were wrong
   exactly when a nibble sum hit 9.
2. **`wrap8`/`wrap16` have small domains.**  They subtract at most one
   modulus.  Splitting a 16-bit value into bytes must be built from values
   that are already bytes (see the JSR/BRK return-address construction).
3. **Round half-to-even on read-back.**  Python `round` and C++
   `nearbyint` (FE_TONEAREST) agree; `llround` does not.
4. **Ties are an architecture rule, not a numeric accident.**  Every
   coefficient in the construction is an integer or half-integer (the
   fractional recency term was retired once its receipts showed it
   vanishing below one double ulp at score scale), so equal scores are
   EXACT ties.  Every lookup declares `TieBreak::LATEST`, resolved by
   integer seq comparison / insertion order; the C++ default (AVERAGE)
   silently blends neighboring memory values.
5. **Decimal-mode V flag comes from the binary sum** on NMOS hardware;
   reference and weights both follow that convention.

## MMIO and NMI

Devices (PPU/APU/controllers) stay outside the graph, but they talk to the
CPU over the model's own bus, like real hardware.  The split is:
**transition logic in the graph, silicon outside, bus = tokens.**

- **NMI is graph logic.**  A STATE token carrying `in_nmi=1` makes that
  step execute the interrupt sequence instead of the fetched instruction:
  PCH/PCL/P (B clear, U set) are pushed through the same three write
  channels BRK uses, `S-3`, `I=1`, and the next PC is fetched through the
  shared vector-read attention slots, muxed to `$FFFA/$FFFB`.  The harness
  contribution is exactly one bit on one token; it never constructs CPU
  state by hand.
- **CPU→device**: every store already surfaces as a WRITE token; the bus
  watches the device's address range and forwards the value.
- **Device→CPU**: before an instruction that reads a device register
  executes, the bus shadow-decodes it from a RAM mirror and pre-injects
  the device's answer as a WRITE token.  The CPU then reads it through
  the same attention lookup as any other byte — a real 6502 also just
  reads whatever is on the bus.

The native bus (`nes/include/nes/bus.h`) implements this once. The reference
core reaches it through a thin adapter in `nes/backends.py`, and
`PythonEngineMachine` (`nes/bindings.cpp`) drives the generated weights
engine through one protocol
(`weights/run_protocol.py` emits it): `load_image()` / `run(max_steps)`
execute until a step budget, halt, jump-to-self trap, or a watched address
is touched (`watch_read` pauses *before* the reading instruction;
`watch_write` after the writing step), `inject_write()` / `set_nmi()` are
the device-side entry points, a read lease serves idempotent device
polls (the `$2002` vblank wait) natively without pausing, and a write
Device writes pause at the instruction boundary.  The shadow
decode uses a generated 256-entry table derived from `spec.READS_MEMORY`,
so every backend classifies device reads identically.

Verified in `nes/tests/native_hardware_test.cpp` (the C++ port replaced the
Python MMIO/console tests) plus `nes/tests/test_console_backends.py`, which
runs the reference and weights backends over SMB and Zelda and requires
identical frames, cycle counts and RAM.

## The NES on top of this

The console layer in `transformer_nes/nes/` (cartridge loader, the native
C++ frame-level PPU, clocked APU, controllers, mappers MMC1/MMC2/MMC3/MMC4/MMC5
alongside NROM, frame loop) runs ROMs against the reference or weights CPU
backend; frame pacing uses CPU-cycle accounting implemented identically in the
native engines and the Python bus.  nestest boots through its reset
vector and passes its full official-opcode suite interactively (controller
input, vblank NMI, OAM DMA); SMB gameplay replays at ~2,290 fps through
the transformer weights on its recorded tape, frames byte-identical across
backends.  See `transformer_nes/nes/`.

Remaining accuracy work is on the console side of that boundary:
- The PPU remains frame-at-once, with event and split-state modeling rather
  than per-dot rendering.
- DMC DMA stalls are charged at run boundaries; interleave them with CPU
  execution if cycle-exact CPU/audio interaction becomes necessary.
- Dynamic cycle classification is duplicated in the reference core, native
  bus, and generated run protocol; derive the native tables from `spec.py`.
