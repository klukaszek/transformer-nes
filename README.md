# Transformer NES

A MOS 6502 CPU expressed as a transformer computation graph, with the NES as the target application.  One transformer step executes one 6502 instruction; CPU state lives in the residual stream, memory is an append-only log read through attention.

The construction follows Percepta's WASM-in-a-transformer experiment ([Can LLMs Be Computers?](https://www.percepta.ai/blog/can-llms-be-computers)), whose WASM machine originally lived in this repo.  The 6502 machine is now the sole implementation; the graph DSL, the O(log n) hull attention, and the graph→weights transformer machinery are retained from that work.

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
| `6502-klaus` | Klaus Dormann functional test suite (reference / graph / C++ kernel / weights) |
| `6502-nestest` | nestest suite vs the Nintendulator golden log (incl. unofficial opcodes) |
| `6502-transformer` | Run the 6502 as real transformer weights, lockstep vs the reference |
| `nes-run` | Run an NES ROM (NROM) on the console layer: PPU, NMI, DMA, controllers |

The 6502 machine — execution model, the four verified backends, the compilation pipeline, the MMIO/NMI layer, and the numerics rules — is documented in [`transformer_nes/cpu6502/README.md`](transformer_nes/cpu6502/README.md).

Current status and prioritized next steps live in [`ROADMAP.md`](ROADMAP.md).

## Layout

| Package | Role |
|---------|------|
| `transformer_nes/graph/` | Computation-graph DSL: Input/ReGLU/Persist/LookUp dimensions, `reglu`/`stepglu` helpers, partial evaluation (`fold.py`) |
| `transformer_nes/attention/` | O(log n) convex-hull KV cache (`hull2d_cht.h` + pybind11 ext) and the exact-evaluator attention runtime |
| `transformer_nes/cpu6502/` | The pure 6502: opcode spec, reference oracle, graph interpreter, exact Python evaluator, generated C++ kernel, the transformer machine (`weights_runner.py` + expert builders + window policy), Klaus harness |
| `transformer_nes/nes/` | NES system layer: MMIO bus, iNES cartridge loader, frame-level PPU, controllers, and the cycle-budgeted frame loop (`nes-run`) |
| `transformer_nes/model/` | Transformer machinery: analytical weight construction (`weights.py`), attention-serving classification (`classify.py`), the generated C++ engine (`engine_gen.py`), safetensors export (`export.py`) |
| `transformer_nes/scheduler/` | Layer scheduler: load-balanced critical-path default plus the opt-in exact MILP |

The dependency direction is one-way: `nes/` uses `cpu6502/`, which uses `graph/` + `attention/`; `cpu6502/` knows nothing about the NES.

## Key Concepts

### Computation Graph

The core abstraction (`transformer_nes/graph/core.py`) defines four primitive types that compose into a DAG:

- **InputDimension** — token embedding values (set per-token)
- **ReGLUDimension** — `ReLU(b) * a`, the gated FFN unit
- **PersistDimension** — materializes an expression into a residual slot
- **LookUpDimension** — attention-based retrieval from token history

From these primitives, two helper functions build all conditional logic:

- `reglu(a, b)` = `ReLU(b) * a` (one FFN neuron)
- `stepglu(a, b)` = `a * step(b >= 0)` (two FFN neurons + persist)

### The 6502 Machine

`transformer_nes/cpu6502/interpreter.py` encodes the full official NMOS opcode set plus the stable unofficial opcodes as one graph: per-opcode one-hot gates decode the instruction, mode gates resolve the effective address, ALU results are computed for every operation family in parallel, and mux chains select the new state. Memory is an append-only log of WRITE tokens; reads (including instruction fetch) are attention lookups. NMI is graph logic too: a state token flagged `in_nmi` executes the interrupt sequence instead of the fetched instruction.

Four backends share this single semantics and are verified against each other instruction-by-instruction: the plain-Python reference oracle
(~2.5M steps/s), the exact graph evaluator (~6K steps/s), a generated C++ kernel (~790K steps/s) that passes the full Klaus suite and nestest bit-identically to the Python evaluator, and the transformer weights themselves (next sections).

### Sub-linear attention state

Standard softmax attention is O(n) per step.  Since this model uses hardmax attention (effectively argmax), the winning key for each query is a vertex of the 2D convex hull of all keys; `transformer_nes/attention/` maintains an incremental convex hull per head, giving O(log n) insert and query.  That is the exact evaluator's mechanism.  The generated engines go further: heads whose queries pass the licenses in `model/classify.py` are served by O(1) state instead — an exact nearest-written-address structure per 4 KiB memory group, and integer running-argmax registers for constant-query heads — with the hull kept as the fallback for anything unlicensed.

### The transformer itself

`6502-transformer` runs the machine as **actual transformer weights**:
`scheduler/milp.py`'s load-balanced critical-path scheduler assigns the gates to 74 layers, `model/weights.py` constructs the transformer's weight matrices analytically (`WeightModel`: d_model=1766, 883 heads, no training), and inference is genuine — Q/K/V projections, hardmax attention, ReGLU FFN, residual stream — run by generated straight-line C++ with a native run loop, exact integer arithmetic end-to-end, O(1) attention state, and three dispatch tiers (per-opcode experts, canonical shape experts serving any ROM with zero per-ROM compilation, and per-ROM pinned sequence modules).
**The full 8,991-instruction nestest suite passes through the weights in
~0.1 seconds**, the full 30.6M-instruction Klaus suite lockstep-matches
with decimal mode in ~90 s, and SMB gameplay runs at ~333 fps
(155 fps with zero ROM-specific compilation), byte-identical to the
C++ kernel.  See
[`transformer_nes/cpu6502/README.md`](transformer_nes/cpu6502/README.md).

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
