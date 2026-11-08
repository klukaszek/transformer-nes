# Reading and organizing the source

The source follows a machine description through graph construction, partial
evaluation, matrix construction and native execution.

```
transformer_nes/          Python only: nothing generated, no console C++, no ROMs
  graph/                  graph DSL
  model/                  generic scheduler, weights, compiler, token engine
  cpu6502/                the 6502 (construction, runtime, oracle, checkpoints)
  ppu2c02/                the PPU
  apu2a03/                the APU
  nes/                    console-level circuits (controller, mappers) and CPU adapters
  frontend/               nes-run, observation, X-ray
  build.py                paths, source identities, caches, native loading
native/                   the C++ console, bindings, window and test oracle
fixtures/                 ROMs, golden logs and test binaries
.cache/engine/            generated kernels and banks (untracked, disposable)
```

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
5. `cpu6502/runner.py`: the public machine API. `bank.py` compiles and owns its
   neural forward context; `engine.py` emits the native scheduling shell, whose
   runtime logic lives in `cpu6502/native/engine.cpp`.

For a smaller circuit, read `nes/controller.py`, then follow its `TokenGraph`
through `model/graph_engine.py`. It uses the same scheduler, weights and
compiler.

## Boundaries

| Area | Rule |
|---|---|
| Dependency direction | `graph/` ← `model/` ← chip packages ← `nes/` ← `frontend/`. `model/` imports without any CPU code; a test enforces it. |
| Chip packages | `cpu6502/`, `ppu2c02/` and `apu2a03/` each describe one chip's circuits. Device circuits that are not a chip (controller, mapper boards) live in `nes/`. |
| Standalone checkpoints | `cpu6502/checkpoint/` runners import only Torch, safetensors and the standard library. Their independence is the verification contract; a test enforces it. |
| Oracles | `cpu6502/reference.py` and `native/reference/` are conventional implementations used only to check the weights. The PPU oracle builds only into `nes_hardware_reference.so`. |
| Native code | `native/include/nes/` holds device contracts, `native/console/` their implementations. `bindings.cpp` is the only translation unit that sees Python. `frontend/` links the window libraries, which hardware tests never need. The CPU machine ABI (`cpu6502/native/machine_abi.h`) is shared with the generated engine and stays with it. |
| Generated output | Everything generated goes to `.cache/engine/` (`build.CACHE_DIR`) or `bin/`. Cache keys hash module names, paths and contents, so moving or editing a construction module causes one rebuild. Generated code embeds no paths. |
| Fixtures | Code finds them through `build.FIXTURES_DIR`. Only the nestest ROM/log and the Klaus binary are tracked; game ROMs, saves and downloaded test ROMs stay local. |
| Tests | Tests sit beside the package they test, in a `tests/` directory. |
