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

# nestest vs the Nintendulator golden log (unofficial opcodes, 2A03 mode,
# registers and cycle counts):
uv run 6502-nestest --runner both

# Export the weights, then run them in a plain PyTorch transformer that
# imports nothing from this repository (about two minutes for nestest):
uv run 6502-export checkpoint/
uv run 6502-checkpoint checkpoint/ --nestest transformer_nes/nes/examples/nestest.nes \
    transformer_nes/nes/examples/nestest.log

# The folded per-opcode experts, routed by opcode in PyTorch (about a second):
uv run 6502-export experts/ --experts
uv run 6502-experts experts/ --nestest transformer_nes/nes/examples/nestest.nes \
    transformer_nes/nes/examples/nestest.log

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

The default CPU backend is `weights`. Use `--backend reference` to run the plain-Python CPU instead. Add `--ppu weights` to run the PPU's register file, background rendering and sprite unit as transformer weights too, and `--xray` to open the overlay on the transformer X-ray.

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
| `6502-export` | Write the 6502 transformer as a standalone checkpoint (`weights.safetensors` + `machine.json`) |
| `6502-checkpoint` | Run an exported checkpoint in plain PyTorch against the nestest golden log |
| `6502-experts` | Run the exported per-opcode experts as a routed mixture in PyTorch against the nestest golden log |
| `nes-run` | Run an NES ROM or benchmark a captured input tape on the native console |

The [`transformer_nes/cpu6502/README.md`](transformer_nes/cpu6502/README.md) documents the 6502 machine, including its graph construction, execution backends, compilation pipeline, MMIO/NMI layer, and numerics rules.

## Layout

| Package | Role |
|---------|------|
| `transformer_nes/graph/` | Computation-graph DSL: Input/ReGLU/Persist/LookUp dimensions, `reglu`/`stepglu` helpers, arithmetic and memory idioms (`ops.py`), direct evaluation (`evaluate.py`), partial evaluation (`fold.py`) |
| `transformer_nes/cpu6502/` | The pure 6502: opcode spec, graph construction, trace format, `backends/reference.py`, `backends/weights/`, and `examples/klaus/` (harness + binary) |
| `transformer_nes/nes/` | Native C++ NES hardware: bus, cartridge, mapper/MMC headers, PPU, APU, console, tape replay, and the window with its debug overlay; `app.py` builds it, adapts a CPU backend, and launches it; `xray.py` reads the engines for the overlay |
| `transformer_nes/ppu2c02/` | The PPU as computation graphs: its memory layout (`space.py`), the register file (`registers.py`), background rendering and compositing with its folded experts (`render.py`), and the harness that compiles them and checks them against native traces (`machine.py`) |
| `transformer_nes/model/` | Transformer machinery: layer scheduling (`schedule.py` for load-balanced critical-path leveling), analytical weight construction (`weights.py`), attention-serving classification (`classify.py`), the generated C++ engines (`engine_gen.py` for the 6502, `graph_engine.py` for any graph and its folded experts), safetensors export (`export.py`) and standalone PyTorch runners for the union and the routed experts (`checkpoint_runner.py`, `expert_runner.py`), and an exact reference forward for any constructed model (`inference.py`) |

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

Time is an output of the forward pass too. Each instruction's cycle count is computed by the graph and read out like a register: the opcode's base cost, +1 for a taken branch and +1 more when it lands on another page, and +1 when an indexed read carries into the high address byte. The console clock, the PPU catch-up, and the cycle budgets all advance by what the model says, and the emulator keeps no cycle table of its own. The experts carry the same output: where the pins decide an instruction's cost it folds to a constant, otherwise it stays a model output. A window that ends in a branch can therefore dispatch whole, with the model timing the branch. A loop window goes further: a poll of a value nothing in the loop can change, or a register counted down by DEX/BNE. It runs one iteration as a forward pass, and every iteration after costs exactly what the model computed for that one. So the run loop retires the rest at once, stopping where single steps would, and the final, falling-through iteration runs as its own forward pass. Waiting for vblank then costs one dispatch, not thousands.

Each game also gets windows of its own code, learned by profile. A weights run counts which code blocks the CPU enters, meaning the instructions it reaches other than by falling through. The hottest become per-ROM windows: up to 32 instructions folded into one forward pass. A window runs through `JMP` and `JSR`, holds pushes and pulls, and ends at a branch or a return, whose target is the model's. The fold licenses a window only when it can prove it exact:
- its stores stay in RAM, below `$2000`;
- its later reads cannot reach a device;
- a later read either cannot touch any earlier step's store, mirrors included, or takes that store's value directly (store-to-load forwarding).

A window is cut just before the first step a license refuses. Its budget guard uses an upper bound on its cycles, so a window dispatches only when every instruction in it is one single steps would run.

The profile also counts which way each branch goes. When a block ends at a branch that goes one way at least 90% of the time, a superblock runs on through it into the predicted block, and further through the next biased branch, up to 32 instructions. The fold pins each interior branch's direction. The flag that branch tests becomes a guard output: the model's flag after the previous step, or the state's own flag when the window starts at the branch. The run loop checks every guard before the window commits anything. When a prediction is wrong, the window declines and the shorter window at the same head runs instead. Dispatch always tries the longest window first.

The windows are saved per ROM in `windows/` (learned data, kept apart from the engine cache, which can be cleared at will) and attach at boot as a compiled module. Each window's fold is cached on its own as soon as it is made, under a digest of just the code that folds (`model/sources.py`: the modules that build it and what they import), so a build cut short by quitting keeps its work, and an edit elsewhere keeps every fold. A game learns them while it plays: a background process folds the blocks the profile found hot, a couple of seconds after boot on a ROM's first run, then every half minute, and they attach between frames. A window only changes which forward retires an instruction, never what it computes, so windows can come and go mid-game. A `--tape` run learns after the tape instead. With them, a frame of these games takes about 1,200 to 3,000 dispatches instead of about 9,500 single-instruction forward passes.

### The checkpoint

`6502-export` writes the transformer as an ordinary checkpoint: `weights.safetensors` (exact int32 matrices, about 0.8 MB) and `machine.json` (the token contract and execution rules). `model/checkpoint_runner.py` runs it as a plain PyTorch module and imports nothing from this repository, only torch and safetensors. Each layer has Q/K/V projections, hardmax attention with the latest token winning ties, and a ReGLU feed-forward block over an integer residual stream. Running the 6502 is autoregressive decoding: WRITE tokens lay down memory, and each STATE token's forward pass reads out the next registers, the instruction's cycle count and its memory writes, which the runner appends as the next tokens. It passes all 8,991 nestest instructions against the golden log, registers and cycle counts. Two exact shortcuts keep its KV cache small: a head whose key never varies always attends to the current token, and each head keeps one entry per distinct key, since an older entry with the same key can never win a tie.

`6502-export --experts` writes what the generated engine actually runs: one folded model per opcode and per interrupt, a few layers and a few hundred weights each, instead of the 63-layer union. `model/expert_runner.py`, also standalone, runs them as a mixture of experts routed by the byte at PC. Their memory reads are structural: a head's query names one address, so it reads that byte from a flat memory, and a write is an assignment. It passes all of nestest, registers and cycles, in about a second.

The runners measure where a PyTorch program lands against real time, which is about 540K instructions/s. The union in eager PyTorch runs about 250 instructions/s, and the routed experts eagerly about 11K: each instruction is dozens of small dependent ops, and launching each one costs more than its arithmetic.

`nes-run --cpu-forwards torch` runs a real game with every CPU forward pass in PyTorch. The native run loop keeps dispatch, budgets and device pauses, and an engine built with a forward hook hands each expert's and window's residual stream to PyTorch. Every layer of every expert is the same computation with different weights, so it is one PyTorch function (`model/expert_runner.py:layer`, in the engine's own int32 arithmetic) compiled once by AOTInductor with every size dynamic, into five shared libraries cached by the layer's source and the torch version and used by every ROM: one each running 1, 2, 4 or 8 layers per call, and one for a layer with half coefficients. An expert is then only data, its layers' weight arrays, made from the cached folds, so a new ROM or a newly learned window needs no compiling. A native hook built against libtorch makes an expert's calls through the libraries' minimal ArrayRef interface on the engine's own residual and RAM, with no Python per forward. A game starts at once: the libraries build (once, about 20 seconds) and the experts load in the background, and until an expert is in the engine runs its forward itself. A `--tape` run waits for everything, so its numbers are PyTorch's alone. On the SMB tape that is about 900 forward passes a frame at about 4.8 µs each, all in: about 240 fps, with the cycle count matching the generated engine's; Castlevania III, the heaviest, takes about 10 ms a frame once its windows are in.

The formulation is what makes this work. The first expert runner gathered from sparse weight lists and scattered into the residual with runtime index tensors; compiled, that cost about 21 µs a call. Dense matrices in the engine's int32 arithmetic (whole coefficients, the few halves as one floor shift, each matrix-vector product written as a multiply and a row sum so it stays in the kernel) compile to tight loops. Each layer keeps only what it uses: no keys (a passthrough head's output is its own value, a memory head's query names its byte), and of each projection only the residual slots it reads, gathered, and writes, added back. Nothing in an expert needs 64 bits: its memory reads are structural, so there are no attention scores. Only the union checkpoint's scores, an address times a head's temperature against another, accumulate in int64, as in the native union. What a forward costs is AOTInductor's runtime, not the arithmetic: with dynamic sizes, every call allocates its intermediates and wraps each in a tensor, and copies its outputs out of thread-local arrays. A layer costs about a microsecond of that and a call about as much again, while the kernel, one fused loop nest per layer, takes a fraction; hence several layers to a call, and each layer's projections merged so it makes fewer intermediates. Compiling each expert with its weights baked in avoided that (2.3 µs a forward) but made a library per expert, about a thousand per game, compiled per ROM; torch 2.14's runtime gives every library thread-local storage, and macOS allows a process 512 such keys, so that design cannot run a game at all now. AOTInductor also once miscompiled a scatter into a fresh buffer for an expert with no passthrough heads, silently zeroing RTI's reads; the AOT-against-native comparison over nestest caught it, and heads now write through the output projection with no such scatter. Before all this, the experts were stacked into one gathered-weight model, as MoE layers are served, and blocks of instructions were unrolled into one `torch.compile`d graph with masks for traps, halts, the cycle budget and device accesses; that was exact, but ran at about 150 µs an instruction.

Checking the cycle column this way found a bug that this repository's own cores shared: the unofficial absolute,X NOPs take an extra cycle when the address crosses a page, like other indexed reads. `6502-nestest` now checks that column for both cores too.

Recorded benchmark results are hardware-dependent. The full 8,991-instruction nestest suite passes through the weights in under 0.1 seconds. The full 30.6M-instruction Klaus suite lockstep-matches with decimal mode in roughly 70 seconds (435K instructions/s). The console calls the generated engine through a native ABI (no Python per run or device access). Gameplay replays on the recorded input tapes run at about:

| Tape | CPU as weights | CPU and PPU as weights |
|------|---------------:|-----------------------:|
| Super Mario Bros. | 4,750 fps | 2,100 fps |
| The Legend of Zelda | 3,950 fps | 1,800 fps |
| Super Mario Bros. 3 | 3,550 fps | 1,650 fps |
| Mega Man 6 | 2,900 fps | 1,450 fps |
| Castlevania III | 1,900 fps | 1,100 fps |

That is 18 to 79 times real time, with byte-identical output compared with the reference-driven console. See
[`transformer_nes/cpu6502/README.md`](transformer_nes/cpu6502/README.md).

### The PPU

The native PPU (`transformer_nes/nes/ppu.cpp`) is a catch-up machine on an absolute dot clock. It runs lazily: every register access, mapper write, and console event first catches it up to the dot of that access, so CPU writes are ordered against rendering by time instead of being reconstructed after the frame. Rendering follows the hardware's loopy `v`/`t`/`x`/`w` scroll registers at tile-slot (8-dot) resolution, which covers mid-frame scroll, `$2006` reloads, CHR bank switches, and rendering toggles without per-game split handling. Sprite-0 hits and sprite overflow are timed to the dot, MMC3 counts filtered A12 rises from the actual fetch pattern, MMC5 counts scanline starts, and MMC2/MMC4 latches see fetches in hardware order.

The console loop runs the CPU only up to the next event that must interrupt it: vblank, the frame's end, or a mapper IRQ predicted by a dry run of the PPU. The same dry run tells the engine how long a `$2002` poll result stays valid, so polling loops remain native.

With the transformer backend, the PPU passes blargg's `ppu_vbl_nmi` (10/10), `vbl_nmi_timing` (7/7), `sprite_hit_tests` (11/11), `sprite_overflow_tests` (5/5), and `mmc3_test_2` (all except the alternate chip revision). See [`transformer_nes/nes/examples/testroms/`](transformer_nes/nes/examples/testroms/README.md).

### The PPU as weights

`transformer_nes/ppu2c02/` builds the PPU as one transformer: a token type per kind of work, all over one memory: an append-only log of WRITE tokens read through the same grouped exact-address attention the 6502 uses, laid out in `space.py` (the CHR windows, the nametable source pages including MMC5's ExRAM and fill mode, palette, OAM).

- **The register file** (`registers.py`, 12 layers): one token per CPU access to `$2000-$2007`, carrying the access and loopy `t`/`v` as their fields (coarse x, coarse y, nametable, fine y). It computes the register file after the access, the value a read returns, and the memory write it makes; `$2007` reads see CHR through the MMC2/MMC4 latches or MMC5's banks.
- **Background rendering and compositing** (`render.py`, 19 layers): one step per 8-pixel tile slot. It reads the nametable, attribute, pattern and palette bytes, fetches the next tile, steps `v`, composites the sprite line by priority, finds sprite-0 hits, and drives the MMC2/MMC4 latches; MMC5's extended attributes read the whole CHR ROM.
- **The sprite unit** (`sprites.py`), in the hardware's two phases per line. *Evaluation* reads all 64 OAM Y bytes through attention and picks the first eight sprites covering the line in OAM order without sorting: with P_i the covering sprites among 0..i, the k-th pick is the number of i with P_i <= k. *Fetch* reads the picked sprites' OAM bytes and pattern rows (vertical and horizontal flips, 8x16 tiles, the MMC2/MMC4 latches each fetch flips in order, MMC5's sprite banks). It returns each row as 8 screen-ordered pixels for the render graph to composite, and the pattern table of every fetch for the mapper's A12 edges.

The four graphs are token types of one engine sharing one memory (`model/graph_engine.py`), and each is folded (`graph/fold.py`) into smaller exact experts by pinning what a token's state fixes. For a register access that is the register and direction (16 experts, most a single layer), so the full graph never runs in play. For a tile slot it is no sprite pixels, the fine-x scroll, the pattern table, the edge slots, or the latch logic off (42 experts of 7 to 9 layers). For the sprite unit it is the sprite size and, for fetch, the number of sprites on the line, so an empty line's fetch folds to almost nothing. The native PPU picks an expert per token whose pins hold and falls back to the full graph otherwise. What depends on the dot rather than the data stays native: the sprite-overflow search with its diagonal-scan bug, the status bits and their races, A12 edges for the mappers, and the IRQ dry runs.

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
