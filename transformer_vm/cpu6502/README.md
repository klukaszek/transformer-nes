# 6502 in the computation graph

A MOS 6502 CPU expressed as a Percepta computation graph — the same
Append-Only Lookup Machine (ALM) construction as Percepta's original WASM
interpreter (which this repo previously housed), but for a real-world ISA
with the NES as the target application.  One transformer step executes one
6502 instruction.

This package is the *pure* CPU: cores, conformance suites, and nothing
console-shaped.  The system layer (MMIO bus, devices, and eventually the
PPU/cartridge) lives in `transformer_vm/nes/` and depends on this package
one-way.

## Components

| File | Role |
|------|------|
| `spec.py` | Single source of truth: 227-opcode table (151 official + stable unofficial), flags, addressing modes, tiny assembler |
| `reference.py` | Plain-Python oracle core.  Trusted by construction; every other backend is verified against it |
| `interpreter.py` | The 6502 as a graph: ~5,700 dimensions built from ReGLU/persist/lookup primitives |
| `evaluator.py` | Exact evaluator: compiles the graph's static schedule to one generated Python function |
| `kernel.py` | C++ backend: emits the same schedule as a single C++ translation unit (hull heads, token inserts, run loop) |
| `klaus.py` | Klaus Dormann functional-test harness (`6502-klaus`) |
| `nestest.py` | nestest harness with Nintendulator golden-log comparison (`6502-nestest`) |
| `trace.py` | Canonical trace format shared by every backend |

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
| `kernel.py` (generated C++) | ~450–600K | Same semantics at NES-relevant speed |
| `weights_runner.py` (real transformer) | ~10-11K after boot | The thesis: actual softmax-ReGLU transformer weights |

The Python and C++ backends are **bit-identical**: same IEEE double
operations in the same order, the same hull code (`hull2d_cht.h`), the same
tie-break configuration, and the same rounding on read-back.

### The transformer itself

`weights_runner.py` (`6502-transformer`) closes the loop on the project's
premise: the graph is scheduled onto layers and compiled by
`model/weights.py` into actual `VanillaTransformer` weights — d_model≈1550,
71 layers, ~777 attention heads, constructed analytically, no
training.  Execution is genuine transformer inference: per-position
embeddings, Q/K/V projections, attention (hull hardmax, or **true
softmax** via `--cache standard --matmul sparse`), ReGLU FFN, residual
stream, output head.  The default path is the generated straight-line C++
hull engine in float32; true softmax is still a diagnostic path and does
not implement latest-write tie-breaks.

Scheduling: the optimal-width MILP does not scale here — the blowup is
the formulation (per-dim/per-boundary alive indicators → ~12M rows, ~5M
binaries; HiGHS presolve alone exceeds an hour), not the solver (SciPy's
`milp` wraps the same HiGHS engine).  `scheduler/milp.py:asap_schedule`
instead takes the critical path and load-balances ops across its
slack with a reverse-topological leveling pass (deterministic; ties go to
the latest phase, which also shortens lifetimes).  Balancing is what
makes the width sane: plain ASAP piles 1,538 of the 6,439 ops into one
layer and d_model becomes head-bound at 1800; leveled, it is
liveness-bound (~1550) under the grouped-memory lookup.

Performance: the analytic weights are ~99.9% zeros, so inference runs the
identical linear maps sparsely.  The generated straight-line engine is the
production default; the generic CSR engine and the dense/scipy-sparse
paths remain useful diagnostics:

| `--matmul` | What runs | Speed (instr/s) |
|------------|-----------|-----------------|
| `dense` | dense torch matvecs, per-layer Python loop; float64 diagnostic | ~0.13 |
| `sparse` | scipy CSR matvecs, per-layer Python loop | ~30–95 |
| `engine` | `model/weights_engine.cpp`: generic CSR C++, whole stack per pybind call — float32 CSR matvecs, shared hull attention for lookup heads, direct WRITE-token insertion, structural fusion of passthrough/erase paths into direct CSR residual deltas | ~10-11K after boot |
| `codegen` | `model/engine_gen.py`: model-specific generated C++ with the same arithmetic — column indices as immediates, ±1 coefficients as add/sub, coefficients as float32 hexfloat literals; compiled once per model digest (~5-10 s, cached) | ~30-38K after boot |

With the generated engine, **the full 8,991-instruction nestest suite runs
through the actual transformer weights in ~0.3 s** (`6502-nestest --runner
weights`), a 200K-instruction Klaus prefix lockstep-matches with decimal
mode in ~7 s, and the full 30.6M-instruction Klaus suite matches with
snapshot-and-rebuild checkpointing keeping hull logs bounded.  The NES
command `nes-run --warmup-frames 60 --backend weights` renders SMB frames
through the weights at ~0.26 s/frame (~3.8 fps), byte-identical to the
C++ kernel's.

Precision: the original scalar 16-bit lookup needed float64 because
`ky = -k² + ...` reached ~2³² with an adjacent-address gap of 1.  `_memory_at`
now splits memory into 16 x 4 KiB lookup groups, so each local parabola is
12-bit and K-side address scores stay below 2²⁴.  The default
`dtype="float32"` engine matches high-address nestest end-to-end; hull
sequence tie-breaks still resolve same-address rewrites.  True softmax
remains separate because exact ties are averaged instead of resolved by
sequence.

Two departures from the WASM construction, both documented in the module:
STATE tokens carry ~20 wide-domain fields, so the harness builds each
position's input embedding directly instead of enumerating a vocabulary
(the layers are untouched); and the run loop reads the machine outputs
from the final layer's residual stream, kept alive by a synthetic READOUT
output token.  Known numerics limit: at 16-bit-address score magnitudes
(~2³²) the recency tie-break term is below one double ulp, so programs
that rewrite *high* addresses need the hull cache's sequence tie-break
(true softmax would average); rescaling the key construction is future
work.

### Compilation pipeline (shared by both backends)

1. The graph's dimensions form a static schedule in id (creation) order,
   which is topological by construction.
2. A backward liveness pass keeps only entries that transitively feed an
   externally read dim (`_RUNTIME_READ_DIMS` + output-score terms).
3. Expressions are emitted as straight-line arithmetic over locals; lookup
   queries are inlined against per-lookup hull caches.
4. Token inserts are compiled separately: each lookup's key/value
   expressions depend only on input dims through a small Persist/ReGLU
   chain, so inserting a token never evaluates the full graph.
5. Memory lookups with identical key/value expressions share one backing
   cache (7 lookup sites, 1 cache) and receive one insert per WRITE token.

## Verification methodology

Chain of trust, each link checked instruction-by-instruction:

```
nestest golden log (Nintendulator)  ⇔  reference.py  ⇔  evaluator.py  ⇔  kernel.py
Klaus functional test (self-checking)  ⇑
```

- **Klaus** (`6502_functional_test.bin`): all 151 official opcodes,
  including decimal mode.  Reference and C++ kernel run the **full
  30,646,177-instruction suite** to the success trap at `$3469`; both must
  agree on the exact step count.  Klaus reaches JSR/indirect addressing
  only after ~20K instructions, stack wrap at ~40K, shifts at ~46K, and
  decimal mode at ~26M — shallow prefix runs prove very little.
- **nestest**: all 8,991 golden-log lines compared field-by-field
  (PC/A/X/Y/P/SP), covering the unofficial opcodes (LAX, SAX, DCP, ISB,
  SLO, RLA, SRE, RRA, `SBC $EB`, NOP variants).  Runs in 2A03 mode
  (`bcd=False` / `bcd_enabled=False`) because the NES CPU has no decimal
  mode.

```bash
uv run 6502-klaus --runner reference          # full suite, ~13s
uv run 6502-klaus --runner cpp-full           # full suite through the graph, ~70s
uv run 6502-klaus --runner graph|cpp --graph-steps N   # lockstep vs reference
uv run 6502-nestest --runner both             # reference vs golden log + graph lockstep
uv run 6502-nestest --runner cpp
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
4. **Attention ties must be configured.**  The `0.3·inv_log_pos` recency
   term vanishes in double rounding once scores reach ~2³¹ (large
   addresses), producing exact ties.  Every hull head must be set to
   `TieBreak::LATEST`, whose integer seq comparison is exact; the C++
   default (AVERAGE) silently blends neighboring memory values.
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

`transformer_vm/nes/bus.py` implements this once over the two
Python-stepped cores (`Bus` + `ReferenceCore`/`GraphCore` adapters, part
of the NES system layer), and `KernelBus` drives the
C++ kernel, which implements the same protocol natively: `load_image()` /
`run(max_steps)` execute until a step budget, halt, jump-to-self trap, or
a watched address is touched (`watch_read` pauses *before* the reading
instruction; `watch_write` after the writing step), and `inject_write()` /
`set_nmi()` are the two device-side entry points.  The kernel's shadow
decode uses a generated 256-entry table derived from `spec.READS_MEMORY`,
so all three backends classify device reads identically.

Verified in `nes/tests/test_mmio.py`: one device scenario (absolute and
(zp),Y device reads, device writes, NMIs at fixed instruction indices
including one nested into the handler) runs on all three backends and
must produce identical traces, memory writes, and device event logs.

## The NES on top of this

The console layer in `transformer_vm/nes/` (cartridge loader, frame-level
PPU, controllers, frame loop) runs NROM ROMs against any of the three CPU
backends; frame pacing uses base-cycle counting implemented identically in
the C++ kernel and the Python bus.  nestest boots through its reset vector
and passes its full official-opcode suite interactively (controller input,
vblank NMI, OAM DMA) at ~45 fps on the kernel.  See `transformer_vm/nes/`.

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
