# Transformer VM

[![CI](https://github.com/Percepta-Core/transformer-vm/actions/workflows/ci.yml/badge.svg)](https://github.com/Percepta-Core/transformer-vm/actions/workflows/ci.yml)

A MOS 6502 CPU expressed as a transformer computation graph, with the NES
as the target application.  One transformer step executes one 6502
instruction; CPU state lives in the residual stream, memory is an
append-only log read through attention.

The construction follows Percepta's WASM-in-a-transformer experiment
([Can LLMs Be Computers?](https://www.percepta.ai/blog/can-llms-be-computers)),
whose WASM machine originally lived in this repo.  The 6502 machine is now
the sole implementation; the graph DSL, the O(log n) hull attention, and
the graph→weights transformer machinery are retained from that work.

## Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** package manager
- **C++17 compiler** — the hull-attention extension and the generated 6502
  kernel are built via torch's cpp_extension (`clang++` on macOS, `g++` on
  Linux)

## Quick Start

```bash
uv sync

# Klaus Dormann functional test suite (all 151 official opcodes + decimal
# mode), full 30.6M-instruction run through the graph semantics in C++:
uv run 6502-klaus --runner cpp-full

# nestest vs the Nintendulator golden log (unofficial opcodes, 2A03 mode):
uv run 6502-nestest --runner both

# Lockstep the exact Python graph evaluator against the reference core:
uv run 6502-klaus --runner graph --graph-steps 20000

# Run an NES ROM (NROM) with the frame-level PPU behind the MMIO bus:
uv run nes-run path/to/rom.nes --frames 120 --out frame.ppm
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `6502-reference` | Run a 6502 binary on the plain-Python reference core |
| `6502-eval` | Run a 6502 binary through the computation graph (exact arithmetic) |
| `6502-klaus` | Klaus Dormann functional test suite (reference / graph / C++ kernel) |
| `6502-nestest` | nestest suite vs the Nintendulator golden log (incl. unofficial opcodes) |
| `6502-transformer` | Run the 6502 as real transformer weights, lockstep vs the reference |
| `nes-run` | Run an NES ROM (NROM) on the console layer: PPU, NMI, DMA, controllers |

The 6502 machine — execution model, the four verified backends, the
compilation pipeline, the MMIO/NMI layer, and the numerics rules — is
documented in [`transformer_vm/cpu6502/README.md`](transformer_vm/cpu6502/README.md).
Current status and prioritized next steps live in [`ROADMAP.md`](ROADMAP.md).

## Layout

| Package | Role |
|---------|------|
| `transformer_vm/graph/` | Computation-graph DSL: Input/ReGLU/Persist/LookUp/CumSum dimensions, `reglu`/`stepglu` helpers |
| `transformer_vm/attention/` | O(log n) convex-hull KV cache (`hull2d_cht.h` + pybind11 ext) and the exact-evaluator attention runtime |
| `transformer_vm/cpu6502/` | The pure 6502: opcode spec, reference oracle, graph interpreter, exact Python evaluator, generated C++ kernel, Klaus/nestest harnesses |
| `transformer_vm/nes/` | NES system layer: MMIO bus, iNES cartridge loader, frame-level PPU, controllers, and the cycle-budgeted frame loop (`nes-run`) |
| `transformer_vm/model/` | Transformer machinery: analytical weight construction (`weights.py`), PyTorch model, standalone C++ inference engine |
| `transformer_vm/scheduler/` | MILP solver assigning graph gates to transformer layers |

The dependency direction is one-way: `nes/` uses `cpu6502/`, which uses
`graph/` + `attention/`; `cpu6502/` knows nothing about the NES.

## Key Concepts

### Computation Graph

The core abstraction (`transformer_vm/graph/core.py`) defines five primitive
types that compose into a DAG:

- **InputDimension** — token embedding values (set per-token)
- **ReGLUDimension** — `ReLU(b) * a`, the gated FFN unit
- **PersistDimension** — materializes an expression into a residual slot
- **LookUpDimension** — attention-based retrieval from token history
- **CumSumDimension** — cumulative sum via attention averaging

From these primitives, two helper functions build all conditional logic:

- `reglu(a, b)` = `ReLU(b) * a` (one FFN neuron)
- `stepglu(a, b)` = `a * step(b >= 0)` (two FFN neurons + persist)

### The 6502 Machine

`transformer_vm/cpu6502/interpreter.py` encodes the full official NMOS
opcode set plus the stable unofficial opcodes as one graph: per-opcode
one-hot gates decode the instruction, mode gates resolve the effective
address, ALU results are computed for every operation family in parallel,
and mux chains select the new state.  Memory is an append-only log of
WRITE tokens; reads (including instruction fetch) are attention lookups.
NMI is graph logic too: a state token flagged `in_nmi` executes the
interrupt sequence instead of the fetched instruction.

Three backends share this single semantics and are verified against each
other instruction-by-instruction: the plain-Python reference oracle
(~2.5M steps/s), the exact graph evaluator (~6K steps/s), and a generated
C++ kernel (~600K steps/s) that passes the full Klaus suite and nestest
bit-identically to the Python evaluator.

### O(log n) Hull KV Cache

Standard softmax attention is O(n) per step.  Since this model uses
hardmax attention (softmax with a very large temperature scaling,
effectively argmax), the winning key for each query is a vertex of the
2D convex hull of all keys.  The `transformer_vm/attention/` module maintains
an incremental convex hull per attention head, giving O(log n) insert and
query — critical for runs of millions of instructions.

### The transformer itself

`6502-transformer` runs the machine as **actual transformer weights**:
`scheduler/milp.py`'s load-balanced critical-path scheduler assigns the
gates to 71 layers, `model/weights.py` constructs `VanillaTransformer`
weights analytically (d_model≈1550, ~777 heads, no training), and
inference is genuine — Q/K/V projections, attention (hull hardmax or true
softmax), ReGLU FFN, residual stream — run by a sparse C++ engine at
~6K stepped instructions/s after boot in the default float32 mode.  **The
full 8,991-instruction nestest suite passes through the weights in ~1-2
seconds**, a Klaus decimal-mode prefix lockstep-matches, and the NES console
can hand a running game off to the weights backend mid-play
(`nes-run --warmup-frames 60 --backend weights` renders an SMB frame
byte-identical to the C++ kernel's, in ~10 s).  See
[`transformer_vm/cpu6502/README.md`](transformer_vm/cpu6502/README.md).

## Development

```bash
uv sync --extra dev

# Fast tests
uv run pytest -m "not slow"

# All tests (includes C++ kernel builds and full-suite runs)
uv run pytest

# Lint
uv run ruff check .
```
