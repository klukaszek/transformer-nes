# 6502 in the computation graph

A MOS 6502 CPU expressed as a Percepta computation graph — the same
Append-Only Lookup Machine (ALM) construction as Percepta's original WASM
interpreter (which this repo previously housed), but for a real-world ISA
with the NES as the target application.  One transformer step executes one
6502 instruction.

This package is the *pure* CPU: cores, conformance suites, and nothing
console-shaped.  The system layer (MMIO bus, devices, frame-level PPU,
cartridge loader) lives in `transformer_nes/nes/` and depends on this
package one-way.

## Components

| File | Role |
|------|------|
| `spec.py` | Single source of truth: 227-opcode table (151 official + stable unofficial), flags, addressing modes, tiny assembler |
| `interpreter.py` | The 6502 as a graph: ~9,200 dimensions built from ReGLU/persist/lookup primitives |
| `trace.py` | Canonical trace format shared by every backend |
| `backends/reference.py` | Plain-Python oracle core.  Trusted by construction; every other backend is verified against it |
| `backends/evaluator.py` | Exact evaluator: compiles the graph's static schedule to one generated Python function |
| `backends/kernel.py` | C++ backend: emits the same schedule as one C++ translation unit (attention state, token inserts, run loop) |
| `backends/run_protocol.py` | The native run-loop protocol both generated engines emit (tables + shared C++ blocks) |
| `backends/weights/runner.py` | The transformer machine: construction, engine-cache orchestration, machine protocol, JIT window discovery |
| `backends/weights/experts.py` | Fold-and-build expert models (per-opcode, sequence, shape) with their dispatch licenses |
| `backends/weights/windows.py` | ROM-independent shape library, window scanning, and shape matching |
| `examples/klaus/harness.py` | Klaus Dormann functional-test harness (`6502-klaus`), next to its binary |

The nestest harness (`6502-nestest`, Nintendulator golden-log comparison)
lives in `transformer_nes/nes/examples/nestest.py` — nestest.nes is an NES
ROM, so it sits with its assets and depends downward on cpu6502.

## Execution model

The machine is an append-only token sequence; all mutable state lives in
attention:

- **`start` token** — carries the initial PC.
- **WRITE tokens** — one per memory write, carrying `(addr, value)`.  The
  64 KiB image is loaded as one WRITE token per **nonzero** byte; reads of
  unwritten addresses are masked to zero by an exact-address check on the
  nearest-key attention hit.  Code and data share the log, so instruction
  fetch is just three memory reads at PC, PC+1, PC+2.
- **STATE tokens** — the full architectural state (PC, A/X/Y/S, eight flag
  bits) plus the previous instruction's fetch/EA/data fields.  The graph
  reads "current state" via a latest-STATE attention lookup.

Each step evaluates the graph over the latest token: per-opcode one-hot
gates decode the fetched opcode, mode gates resolve the effective address
(indirect modes via pointer reads), ALU results are computed for every
operation family in parallel, and `_mux` chains select the new state.  The
run loop reads back ~22 named result dims, emits up to three WRITE tokens
(JSR pushes two bytes, BRK three), builds the next STATE token, and stops
when the HALT output token outscores STATE (or on a jump-to-self trap).

## Four backends, one semantics

| Backend | Speed (steps/s) | Role |
|---------|-----------------|------|
| `reference.py` (`run_traps`) | ~2.5M | Oracle; defines correct behavior |
| `evaluator.py` (generated Python) | ~6K | Exact graph semantics, debuggable |
| `kernel.py` (generated C++) | ~790K | Same semantics at NES-relevant speed |
| `weights/runner.py` (real transformer) | ~344K (full Klaus, incl. the lockstep reference) | The thesis: actual hardmax-attention + ReGLU transformer weights |

The evaluator and the C++ kernel are **bit-identical**: the kernel emits
the evaluator's schedule as the same IEEE double operations in the same
order with the same rounding on read-back.  Its attention state is served
by the exact O(1) structures where `model/classify.py` licenses them
(currently all of it) and by the same hull code (`hull2d_cht.h`) with the
same tie-break configuration otherwise — the classification is exact, so
either serving returns the identical winner.

### The transformer itself

`backends/weights/` (`6502-transformer`) closes the loop on the project's
premise: the graph is scheduled onto layers and compiled by
`model/weights.py` into actual transformer weight matrices
(`WeightModel` — d_model=1766, 74 layers, 883 attention heads),
constructed analytically, no training.  Execution is genuine transformer inference: per-position
embeddings, Q/K/V projections, hardmax attention, ReGLU FFN, residual
stream, output head.  The runtime is the generated straight-line C++
engine (`model/engine_gen.py`), exact integer end-to-end: a fixed-point
int32 residual, exact nearest-address memory structures, and integer
running-argmax registers (`model/classify.py` holds the licenses; anything
unlicensed keeps a real hull, so classification is a speed decision, never
a correctness one).

Scheduling: the optimal-width MILP does not scale at union size — the
blowup is the formulation (per-dim/per-boundary alive indicators; HiGHS
presolve alone exceeds an hour), not the solver.
`scheduler/milp.py:asap_schedule` instead takes the critical path,
load-balances ops across its slack with a reverse-topological leveling
pass (deterministic; ties go to the latest phase, which also shortens
lifetimes), then runs a greedy hygiene sweep that re-places persists
within their dependency windows to minimize passthrough/erase work —
capturing essentially the whole hygiene-MILP win in milliseconds.  The
MILP survives as an opt-in per-expert certificate
(`expert_schedule="milp"`, hygiene objective, width capped at the
balanced-ASAP width).

Performance: the analytic weights are ~99.9% zeros, so inference runs the
same linear maps sparsely. `model/engine_gen.py` emits model-specific C++
with indices and coefficients embedded in expressions. The residual is exact
fixed-point int32; shifts are guarded and signed wrap is defined. Memory heads
use exact nearest-address structures, constant-query heads use integer
running-argmax registers, and remaining heads use hull attention. Generated
artifacts are cached by a digest covering every source file that shapes
emission; a manifest hit skips folding and emission and dlopens the compiled
core directly.

Routing is a three-tier MoE, workload-independent by default: one rebuilt
expert model per opcode (227, no game-derived tuning set), a canonical
**shape-expert library** folded once with pc and operands symbolic (any ROM,
homebrew included, zero per-ROM compilation), and a per-ROM **sequence
module** that pins whole instruction windows as one forward (~160 KB TU,
~1 s cached; the console builds it automatically from a boot-time scan of
the cartridge image).  A profile-guided JIT counter discovers hot windows
at runtime — including code the program copies into RAM — and adds them to
the dispatch table mid-run.

With the generated engine, **the full 8,991-instruction nestest suite runs
through the actual transformer weights in ~0.1 s** (`6502-nestest --runner
weights`) and the full 30.6M-instruction Klaus suite lockstep-matches with
decimal mode in ~90 s, checkpointing disabled entirely (engine state is
O(1); the snapshot-and-rebuild machinery remains for the NES console).  SMB
gameplay through the weights runs at ~333 fps with its pinned module and
~155 fps from the shipped core alone, byte-identical to the C++ kernel's
frames and cycles.

Memory is split into 16 × 4 KiB groups. Address scores therefore remain in
an exactly represented range, while LATEST ties are resolved by insertion
order. The standalone safetensors export remains float32 by default; that
storage choice is separate from the generated runtime's int32 arithmetic.

### Compilation pipeline (evaluator and C++ kernel)

1. The graph's dimensions form a static schedule in id (creation) order,
   which is topological by construction.
2. A backward liveness pass keeps only entries that transitively feed an
   externally read dim (`RUNTIME_READ_DIMS` + output-score terms).
3. Expressions are emitted as straight-line arithmetic over locals; lookup
   queries are inlined against the classified attention state (exact
   memory structure, running-argmax register, or hull — `model/classify.py`
   decides).
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
nestest golden log (Nintendulator)  ⇔  reference.py  ⇔  evaluator.py  ⇔  kernel.py  ⇔  weights_runner.py
Klaus functional test (self-checking)  ⇑
```

- **Klaus** (`6502_functional_test.bin`): all 151 official opcodes,
  including decimal mode.  Reference, C++ kernel, and the weights
  transformer run the **full 30,646,177-instruction suite** to the success
  trap at `$3469`; all must agree on the exact step count.  Klaus reaches JSR/indirect addressing
  only after ~20K instructions, stack wrap at ~40K, shifts at ~46K, and
  decimal mode at ~26M — shallow prefix runs prove very little.
- **nestest**: all 8,991 golden-log lines compared field-by-field
  (PC/A/X/Y/P/SP), covering the unofficial opcodes (LAX, SAX, DCP, ISB,
  SLO, RLA, SRE, RRA, `SBC $EB`, NOP variants).  Runs in 2A03 mode
  (`bcd=False` / `bcd_enabled=False`) because the NES CPU has no decimal
  mode.

```bash
uv run 6502-klaus --runner reference          # full suite, ~13s
uv run 6502-klaus --runner cpp-full           # full suite through the graph, ~40s
uv run 6502-klaus --runner weights --graph-steps 30700000  # full suite through the WEIGHTS, ~90s
uv run 6502-klaus --runner graph|cpp --graph-steps N   # lockstep vs reference
uv run 6502-nestest --runner both             # reference vs golden log + graph lockstep
uv run 6502-nestest --runner cpp|weights
```

## Numerics: rules that keep the graph exact

These were each learned the hard way; violating any of them produces bugs
that only surface millions of instructions into a run.

1. **Integer thresholds only.**  `stepglu(one, b)` equals `step(b >= 0)`
   only for integer `b`; at `b = -0.5` it returns 0.5.  The decimal-mode
   nibble carries originally used `x - 9.5` thresholds and were wrong
   exactly when a nibble sum hit 9.
2. **`_wrap8`/`_wrap16` have small domains.**  They subtract at most one
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
   reference and graph both follow that convention.

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

`transformer_nes/nes/bus.py` implements this once over the two
Python-stepped cores (`Bus` + `ReferenceCore`/`GraphCore` adapters, part
of the NES system layer), and `KernelBus` drives either native engine —
the C++ kernel or the generated weights engine — through one protocol
(`run_protocol.py` emits it for both): `load_image()` / `run(max_steps)`
execute until a step budget, halt, jump-to-self trap, or a watched address
is touched (`watch_read` pauses *before* the reading instruction;
`watch_write` after the writing step), `inject_write()` / `set_nmi()` are
the device-side entry points, and a read lease serves idempotent device
polls (the `$2002` vblank wait) natively without pausing.  The shadow
decode uses a generated 256-entry table derived from `spec.READS_MEMORY`,
so every backend classifies device reads identically.

Verified in `nes/tests/test_mmio.py`: one device scenario (absolute and
(zp),Y device reads, device writes, NMIs at fixed instruction indices
including one nested into the handler) runs on all three backends and
must produce identical traces, memory writes, and device event logs.

## The NES on top of this

The console layer in `transformer_nes/nes/` (cartridge loader, frame-level
PPU, controllers, frame loop) runs NROM ROMs against any of the four CPU
backends; frame pacing uses base-cycle counting implemented identically in
the native engines and the Python bus.  nestest boots through its reset
vector and passes its full official-opcode suite interactively (controller
input, vblank NMI, OAM DMA); SMB plays at ~74 fps on the kernel and ~333
fps through the transformer weights, frames byte-identical across
backends.  See `transformer_nes/nes/`.

Long runs stay bounded via **snapshot-and-rebuild checkpointing**: past a
token threshold the console reseeds the machine's append-only log from
the RAM mirror at a frame boundary (`KernelBus.checkpoint()` /
`Bus.checkpoint()`).  This is semantically invisible -- unwritten and
zero bytes both read as 0 through the exact-address mask, so reseeding
only the nonzero RAM bytes preserves every future read; the invisibility
is asserted pixel-for-pixel and cycle-for-cycle in
`nes/tests/test_console.py`.

Still open on the CPU side of that boundary:
- Cycle accuracy: base cycles only (no branch-taken or page-cross
  penalties); revisit when a timing-sensitive game needs it.
- NES RAM mirroring (`$0800-$1FFF`) is not folded (PPU register mirrors
  through `$3FFF` are); add a mirror device if a ROM relies on it.
