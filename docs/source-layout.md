# Reading and organizing the source

The source follows a machine description through graph construction, partial
evaluation, matrix construction and native execution.

```
transformer_nes/          Python only: nothing generated, no console C++, no ROMs
  graph/                  graph DSL
  model/                  the transformer model and the machinery around it
  cpu6502/                the 6502 (construction, its chip, machine API, oracle)
  ppu2c02/                the PPU
  apu2a03/                the APU
  nes/                    console-level circuits (controller, mappers) and CPU adapters
  frontend/               nes-run, observation, X-ray
  build.py                paths, source identities, caches, native loading
native/                   the C++ console, bindings, window and test oracle
fixtures/                 ROMs, golden logs and test binaries
.cache/                   the saved model, its library and reusable kernels
                          (untracked, disposable)
```

## Four layers

| Layer | Owns | Files |
|---|---|---|
| Construct | circuits to weights | the graph DSL, `model/construct.py` (token types, folding into specializations), `model/schedule.py`, `model/weights.py` |
| Model | the NES as PyTorch modules: sparse exact weights, eager forward | `model/transformer.py` (imports only torch), `nes/model.py` (`NES`, saved as `nes.safetensors` and `nes.json`) |
| Compile | one native library for the whole model | `model/backend.py`, `model/compiler.py`, `model/ssa.py`, `model/native/` |
| Run | the console: clock, devices, I/O | `native/`, handed its compiled chips by Python |

The model holds weights only; memory is the runtime's state, passed to each
forward. A specialization is its own smaller circuit, valid where its pins
hold; specializations that fold to the same circuit store it once. A view is a
token type read off another's circuits at fewer outputs (the PPU's status and
bus previews): it stores no weights, and its kernels keep only the work its
outputs need. The library is keyed by the model's weights and the backend's sources,
built on first use and published by rename. Every chip, the CPU included,
is on this path.

## A reading order

1. `cpu6502/graph.py`: the CPU transition circuit and `Cpu6502Graph` contract.
   `spec.py` supplies opcode metadata; `trace.py` describes observable state.
2. `cpu6502/experts.py`: fold candidates (`folds.py`), check dispatch
   conditions, project execution effects (`execution.py`) and build the fixed
   expert bank. `shapes.py` defines reusable instruction patterns.
3. `model/schedule.py` and `model/weights.py`: assign layers and residual
   slots, then construct the attention and ReGLU matrices.
4. `model/compiler.py`: project the live matrix rows and expand their actual
   coefficients into FX arithmetic. `model/ssa.py` lowers that arithmetic,
   shares fragments and emits scalar kernels.
5. `cpu6502/chip.py`: the executions as the CPU chip's specializations, with
   one named interface, and the run loop's dispatch tables;
   `native/console/cpu.cpp` is the run loop; `cpu6502/runner.py` the machine
   API over it; `cpu6502/eager.py` runs the same chip in eager PyTorch.

For a smaller circuit, read `nes/controller.py`: `chip()` builds its
`model.transformer.Chip` through `model/construct.py`. `model/transformer.py`
is the PyTorch model it becomes, and `model/backend.py` compiles every chip of
`nes/model.py:NES` into one library.

## Boundaries

| Area | Rule |
|---|---|
| Dependency direction | `graph/` ← `model/` ← chip packages ← `nes/` ← `frontend/`. `model/` imports without any CPU code; a test enforces it. |
| Chip packages | `cpu6502/`, `ppu2c02/` and `apu2a03/` each describe one chip's circuits. Device circuits that are not a chip (controller, mapper boards) live in `nes/`. |
| Eager check | `cpu6502/eager.py` runs the saved model's CPU importing only Torch, safetensors and `model/transformer.py`. Its independence is the verification contract; a test enforces it. |
| Oracles | `cpu6502/reference.py` and `native/reference/` are conventional implementations used only to check the weights. The PPU oracle builds only into `nes_hardware_reference.so`. |
| Native code | `native/include/nes/` holds device contracts, `native/console/` their implementations. `bindings.cpp` is the only translation unit that sees Python. `frontend/` links the window libraries, which hardware tests never need. |
| Generated output | Everything generated goes to `.cache/` (the model and its library in `nes/`, reusable kernels in `compiler/`) or `bin/`. Cache keys hash module names, paths and contents, so moving or editing a construction module causes one rebuild. Generated code embeds no paths. |
| Fixtures | Code finds them through `build.FIXTURES_DIR`. Only the nestest ROM/log and the Klaus binary are tracked; game ROMs, saves and downloaded test ROMs stay local. |
| Tests | Tests sit beside the package they test, in a `tests/` directory. |
