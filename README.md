# Transformer NES

A MOS 6502 CPU expressed as a transformer computation graph, with the NES as the target application. One transformer step executes one 6502 instruction; CPU state lives in the residual stream, and memory is an append-only log read through attention.

The project was inspired by Percepta's [WASM-in-a-transformer experiment](https://www.percepta.ai/blog/can-llms-be-computers). The 6502 machine is built from the local graph DSL and executed by a plain-Python reference backend or an analytically generated transformer-weights backend.

## Emulator Gallery

These screenshots were captured from the NES emulator running on the compiled transformer-weights backend.

<table>
  <tr>
    <td width="50%"><img src="public/legendofzelda.png" alt="The Legend of Zelda running in the NES emulator" width="100%"></td>
    <td width="50%"><img src="public/smb3.png" alt="Super Mario Bros. 3 running in the NES emulator" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><em>The Legend of Zelda</em></td>
    <td align="center"><em>Super Mario Bros. 3</em></td>
  </tr>
  <tr>
    <td width="50%"><img src="public/megaman6.png" alt="Mega Man 6 running in the NES emulator" width="100%"></td>
    <td width="50%"><img src="public/castlevania3.png" alt="Castlevania III running in the NES emulator" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><em>Mega Man 6</em></td>
    <td align="center"><em>Castlevania III</em></td>
  </tr>
</table>

## Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** package manager
- **C++26 compiler** for the native NES target; the generated 6502 engine is compiled as a C++17 PyTorch extension

## Quick Start

```bash
uv sync

# Klaus Dormann functional test suite (all 151 official opcodes + decimal
# mode), full 30.6M-instruction run through the transformer weights:
uv run 6502-klaus --runner weights --max-steps 30700000

# nestest vs the Nintendulator golden log (unofficial opcodes, 2A03 mode):
uv run 6502-nestest --runner both

# Run an NES ROM with the native C++ console hardware:
uv run nes-run path/to/rom.nes

# Replay a captured controller tape headlessly and benchmark it uncapped:
uv run nes-run path/to/rom.nes --tape tapes/game_tape
```

ROMs and tape captures are local inputs and are not included in the repository. The `6502-nestest` command expects `nestest.nes` at `transformer_nes/nes/examples/nestest.nes`; `nes-run` accepts a local ROM path and tape path directly.

## CLI Commands

| Command | Description |
|---------|-------------|
| `6502-reference` | Run a 6502 binary on the plain-Python reference core |
| `6502-klaus` | Klaus Dormann functional test suite (reference / weights) |
| `6502-nestest` | nestest suite vs the Nintendulator golden log (incl. unofficial opcodes) |
| `6502-transformer` | Run the 6502 as real transformer weights, lockstep vs the reference |
| `nes-run` | Run an NES ROM or benchmark a captured input tape on the native console |

The [`transformer_nes/cpu6502/README.md`](transformer_nes/cpu6502/README.md) documents the 6502 machine, including its graph construction, execution backends, compilation pipeline, MMIO/NMI layer, and numerics rules.

## Layout

| Package | Role |
|---------|------|
| `transformer_nes/graph/` | Computation-graph DSL: Input/ReGLU/Persist/LookUp dimensions, `reglu`/`stepglu` helpers, partial evaluation (`fold.py`) |
| `transformer_nes/cpu6502/` | The pure 6502: opcode spec, graph construction, trace format, `backends/reference.py`, `backends/weights/`, and `examples/klaus/` (harness + binary) |
| `transformer_nes/nes/` | Native C++ NES hardware: bus, cartridge, mapper/MMC headers, PPU, APU, console, and tape replay; `app.py` only builds it, adapts a CPU backend, and launches it |
| `transformer_nes/model/` | Transformer machinery: layer scheduling (`schedule.py` for load-balanced critical-path leveling), analytical weight construction (`weights.py`), attention-serving classification (`classify.py`), the generated C++ engine (`engine_gen.py`), safetensors export (`export.py`) |

The dependency direction is one-way: `nes/` depends on the NES-independent CPU package; the CPU construction depends on `graph/`, while its weights backend also uses `model/` for scheduling and code generation.

## Key Concepts

### Computation Graph

The core abstraction (`transformer_nes/graph/core.py`) defines four primary graph dimension types that compose into a DAG. Engine-served memory reads are modeled separately for the CPU's 64 KiB address space.

- **InputDimension:** token embedding values set per token
- **ReGLUDimension:** `ReLU(b) * a`, the gated FFN unit
- **PersistDimension:** materializes an expression into a residual slot
- **LookUpDimension:** attention-based retrieval from token history

From these primitives, two helper functions build all conditional logic:

- `reglu(a, b)` = `ReLU(b) * a` (one FFN neuron)
- `stepglu(a, b)` = `a * step(b >= 0)` (two FFN neurons + persist)

### The 6502 Machine

`transformer_nes/cpu6502/interpreter.py` encodes the full official NMOS opcode set plus the stable unofficial opcodes as one graph: per-opcode one-hot gates decode the instruction, mode gates resolve the effective address, ALU results are computed for every operation family in parallel, and disjoint family gates sum in parallel to select the new state. Memory is an append-only log of WRITE tokens; reads (including instruction fetch) are attention lookups. NMI is graph logic too: a state token flagged `in_nmi` executes the interrupt sequence instead of the fetched instruction.

Two execution backends share this construction: the plain-Python reference oracle
(~2.5M steps/s on the benchmark machine) and the transformer weights, which pass the full Klaus suite and lockstep against the nestest oracle.

### Bounded attention state

The generated engine serves the graph's hardmax lookups with exact nearest-written-address structures for memory and integer running-argmax registers for constant-query state. These are the only attention structures accepted by code generation; unsupported lookup shapes fail the build instead of carrying a second runtime.

### The transformer itself

`6502-transformer` runs the machine as **actual transformer weights**:
`model/schedule.py`'s load-balanced critical-path scheduler assigns the decimal-enabled graph to 56 layers, while `model/weights.py` constructs the ragged attention stack analytically (`WeightModel`: d_model=624, no training). Inference performs genuine Q/K/V projections, hardmax attention, ReGLU FFN, and residual updates without position encoding; token order enters through the LATEST tie-break. The generated straight-line C++ engine uses an exact int32 runtime, a native run loop, O(1) attention state, and four routed execution tiers: optional pinned sequence modules, canonical shape experts, compact opcode execution, and the exact union proof model for interrupts and unknown bytes.

Recorded benchmark results are hardware-dependent. The full 8,991-instruction nestest suite passes through the weights in under 0.1 seconds. The full 30.6M-instruction Klaus suite lockstep-matches with decimal mode in roughly 70 seconds (435K instructions/s). Gameplay replays on the recorded input tapes run at about 2,290 fps (SMB), 1,300 fps (Zelda), 930 fps (SMB3), 920 fps (Mega Man 6), and 850 fps (Castlevania III). That is roughly 14 to 38 times real time, with byte-identical output compared with the reference-driven console. See
[`transformer_nes/cpu6502/README.md`](transformer_nes/cpu6502/README.md).

## Development

```bash
uv sync --extra dev

# Fast tests
uv run pytest -m "not slow"

# All tests (includes generated-engine builds and full-suite runs)
uv run pytest

# Lint
uv run ruff check .
```
