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

## Interactive Emulator

Run a ROM without `--tape` to open the interactive emulator window:

```bash
uv run nes-run path/to/rom.nes
```

The default CPU backend is `weights`. Use `--backend reference` to run the plain-Python CPU instead. Add `--ppu weights` to run the PPU's register file and background rendering as transformer weights too, and `--xray` to open the overlay on the transformer X-ray.

| Key | NES control |
|-----|-------------|
| `X` | A |
| `Z` | B |
| `Tab` | Select |
| `Enter` | Start |
| Arrow keys | D-pad |
| `Esc` | Quit the emulator |
| Grave accent / tilde | Open or close the debug overlay |

### Debug overlay

Press the grave-accent/tilde key to open the Dear ImGui debug window. Press `Esc` while it is open to close the overlay; keyboard input is returned to the game afterward.

The game keeps running and taking input while the overlay is open. It provides:

- **Stats:** display FPS, backend name, frame number, CPU cycles, and emulation time per frame.
- **Options:** toggle overscan cropping, pause emulation, remove the PPU's eight-sprites-per-scanline limit, or mute audio.
- **Step frame:** pause the emulator and advance exactly one frame.
- **Transformer X-ray** (`nes/xray.py`), read live from the running engines:
  - *CPU paths:* how each frame's instructions were retired -- whole loops in one forward, folded instruction shapes, per-opcode experts, interrupt experts, or the full model.
  - *CPU memory:* a 256x256 heatmap of the addresses the CPU's attention heads read.
  - *Instruction dissection:* the next instruction through the full 56-layer model on a copy of the engine, every named value at the layer that computes it (memory reads with their addresses).
  - *PPU experts:* which folded render expert drew each 8-pixel slot of the frame, optionally tinted over the game picture (the tint stays on with the overlay closed).
  - *Pixel dissection:* click any pixel to trace its tile slot through the full render graph -- the nametable, attribute, pattern and palette reads that produced its color.
  - *Ablation:* knock out any value in that dissection and the render transformer computes without it, in every slot, live.
- **Memory:** inspect the full CPU address space from `$0000` through `$FFFF`. Enter a hexadecimal address in the `addr` field to jump to it.

The CPU views need the weights CPU, the pixel views `--ppu weights`. The `--debug` command-line flag enables verbose logging. Battery-backed saves are loaded from and written to a `.sav` file next to the ROM; tape replays intentionally ignore saves.

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
| `transformer_nes/graph/` | Computation-graph DSL: Input/ReGLU/Persist/LookUp dimensions, `reglu`/`stepglu` helpers, arithmetic and memory idioms (`ops.py`), direct evaluation (`evaluate.py`), partial evaluation (`fold.py`) |
| `transformer_nes/cpu6502/` | The pure 6502: opcode spec, graph construction, trace format, `backends/reference.py`, `backends/weights/`, and `examples/klaus/` (harness + binary) |
| `transformer_nes/nes/` | Native C++ NES hardware: bus, cartridge, mapper/MMC headers, PPU, APU, console, tape replay, and the window with its debug overlay; `app.py` builds it, adapts a CPU backend, and launches it; `xray.py` reads the engines for the overlay |
| `transformer_nes/ppu2c02/` | The PPU as computation graphs: its memory layout (`space.py`), the register file (`registers.py`), background rendering and compositing with its folded experts (`render.py`), and the harness that compiles them and checks them against native traces (`machine.py`) |
| `transformer_nes/model/` | Transformer machinery: layer scheduling (`schedule.py` for load-balanced critical-path leveling), analytical weight construction (`weights.py`), attention-serving classification (`classify.py`), the generated C++ engines (`engine_gen.py` for the 6502, `graph_engine.py` for any graph and its folded experts), safetensors export (`export.py`), and an exact reference forward for any constructed model (`inference.py`) |

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
`model/schedule.py`'s load-balanced critical-path scheduler assigns the decimal-enabled graph to 56 layers, while `model/weights.py` constructs the ragged attention stack analytically (`WeightModel`: d_model=626, no training). Inference performs genuine Q/K/V projections, hardmax attention, ReGLU FFN, and residual updates without position encoding; token order enters through the LATEST tie-break. The generated straight-line C++ engine uses an exact int32 runtime, a native run loop, O(1) attention state, and four routed execution tiers: optional pinned sequence modules, canonical shape experts, compact opcode execution, and the exact union proof model for interrupts and unknown bytes.

Time is an output of the forward pass too. Each instruction's cycle count is computed by the graph and read out like a register: the opcode's base cost, +1 for a taken branch and +1 more when it lands on another page, and +1 when an indexed read carries into the high address byte. The console clock, the PPU catch-up, and the cycle budgets all advance by what the model says, and the emulator keeps no cycle table of its own. The experts carry the same output: where the pins decide an instruction's cost it folds to a constant, otherwise it stays a model output. A window that ends in a branch can therefore dispatch whole, with the model timing the branch.

Recorded benchmark results are hardware-dependent. The full 8,991-instruction nestest suite passes through the weights in under 0.1 seconds. The full 30.6M-instruction Klaus suite lockstep-matches with decimal mode in roughly 70 seconds (435K instructions/s). The console calls the generated engine through a native ABI (no Python per run or device access). Gameplay replays on the recorded input tapes run at about:

| Tape | CPU as weights | CPU and PPU as weights |
|------|---------------:|-----------------------:|
| Super Mario Bros. | 4,100 fps | 2,300 fps |
| The Legend of Zelda | 3,700 fps | 1,900 fps |
| Super Mario Bros. 3 | 2,750 fps | 1,750 fps |
| Mega Man 6 | 2,300 fps | 1,500 fps |
| Castlevania III | 1,600 fps | 1,150 fps |

That is 19 to 68 times real time, with byte-identical output compared with the reference-driven console. See
[`transformer_nes/cpu6502/README.md`](transformer_nes/cpu6502/README.md).

### The PPU

The native PPU (`transformer_nes/nes/ppu.cpp`) is a catch-up machine on an absolute dot clock. It runs lazily: every register access, mapper write, and console event first catches it up to the dot of that access, so CPU writes are ordered against rendering by time instead of being reconstructed after the frame. Rendering follows the hardware's loopy `v`/`t`/`x`/`w` scroll registers at tile-slot (8-dot) resolution, which covers mid-frame scroll, `$2006` reloads, CHR bank switches, and rendering toggles without per-game split handling. Sprite-0 hits and sprite overflow are timed to the dot, MMC3 counts filtered A12 rises from the actual fetch pattern, MMC5 counts scanline starts, and MMC2/MMC4 latches see fetches in hardware order.

The console loop runs the CPU only up to the next event that must interrupt it: vblank, the frame's end, or a mapper IRQ predicted by a dry run of the PPU. The same dry run tells the engine how long a `$2002` poll result stays valid, so polling loops remain native.

With the transformer backend, the PPU passes blargg's `ppu_vbl_nmi` (10/10), `vbl_nmi_timing` (7/7), `sprite_hit_tests` (11/11), `sprite_overflow_tests` (5/5), and `mmc3_test_2` (all except the alternate chip revision). See [`transformer_nes/nes/examples/testroms/`](transformer_nes/nes/examples/testroms/README.md).

### The PPU as weights

`transformer_nes/ppu2c02/` builds the PPU's own work as two graphs over one memory: an append-only log of WRITE tokens read through the same grouped exact-address attention the 6502 uses, laid out in `space.py` (the CHR windows, the nametable source pages including MMC5's ExRAM and fill mode, palette, OAM).

- **The register file** (`registers.py`, 12 layers): one step per CPU access to `$2000-$2007`, carrying the access and loopy `t`/`v` as their fields (coarse x, coarse y, nametable, fine y). It computes the register file after the access, the value a read returns, and the memory write it makes.
- **Background rendering and compositing** (`render.py`, 19 layers): one step per 8-pixel tile slot. It reads the nametable, attribute, pattern and palette bytes, fetches the next tile, steps `v`, composites the sprite line by priority, finds sprite-0 hits, and drives the MMC2/MMC4 latches; MMC5's extended attributes read the whole CHR ROM.

The render graph is also folded (`graph/fold.py`) into 42 smaller exact experts of 7 to 9 layers by pinning what a slot's state fixes: no sprite pixels, the fine-x scroll, the pattern table, the edge slots, the latch logic off. The native PPU picks an expert per slot whose pins hold and falls back to the full graph otherwise. What depends on the dot rather than the data stays native: sprite evaluation, the status bits and their races, A12 edges for the mappers, and the IRQ dry runs.

```bash
uv run nes-run path/to/rom.nes --ppu weights
```

Frames stay pixel-identical to the native PPU on every mapper the console supports (NROM, MMC1, MMC3, MMC2/MMC4, MMC5), checked over game captures and synthetic carts that sweep every register path. `model/inference.py` runs either graph exactly in float64 as the reference the compiled engines match.

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
