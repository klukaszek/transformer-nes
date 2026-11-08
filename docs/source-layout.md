# Reading and organizing the source

The source follows a machine description through graph construction, partial
evaluation, matrix construction and native execution. Start with one path:

1. `cpu6502/graph.py`: the CPU transition circuit and `Cpu6502Graph` contract.
   `spec.py` supplies opcode metadata; `trace.py` describes observable state.
2. `cpu6502/backends/weights/experts.py`: fold candidates, check dispatch
   conditions, project execution effects and build the fixed expert bank.
3. `model/schedule.py` and `model/weights.py`: assign layers and residual slots,
   then construct the attention and ReGLU matrices.
4. `model/compiler.py`: project the live matrix rows and expand their actual
   coefficients into FX arithmetic. `model/ssa.py` lowers that arithmetic,
   shares fragments and emits scalar kernels.
5. `cpu6502/backends/weights/runner.py`: the public machine API. `bank.py`
   compiles and owns its neural forward context; `engine.py` emits the native
   scheduling shell. The shell's runtime logic lives in `native/engine.cpp`.

For a smaller circuit, read `nes/controller.py`, then follow its `TokenGraph`
through `model/graph_engine.py`. It uses the same scheduler, weights and compiler.

## Consolidation

The source audit counted 49 production Python modules, excluding tests, examples
and package initializers. Consolidation reduces that to 39. The compiler's
`model/` package goes from 12 modules to six. Imports use the new owners directly;
there are no forwarding modules retaining the old layout.

| Previously separate files | Current owner | Reason |
|---|---|---|
| `files.py`, `model/build.py`, `model/sources.py`, `model/kernel_cache.py`, `nes/build.py` | `build.py` | One place for source identities, package/repository paths, atomic publication, process locks, cache eviction, native loading and Makefile builds. |
| `cpu6502/interpreter.py`, `cpu6502/construction.py` | `cpu6502/graph.py` | The graph's public contract belongs with the circuit that implements it. |
| `model/weights.py`, `model/artifacts.py` | `model/weights.py` | Matrix construction and exact serialization share the same coefficient representation. |
| `model/torch_layout.py`, `model/torch_codegen.py` | `model/compiler.py` | Live matrix projection is the first stage of the FX compiler. |
| `model/scalar_ir.py`, `model/outline.py` | `model/ssa.py` | Fragment sharing operates on the SSA representation and uses its emitter. |
| CPU `weights/folds.py`, `weights/proofs.py` | CPU `weights/folds.py` | Sequence composition and the proofs that license its execution are read together. |
| CPU `weights/native/__init__.py`, `weights/engine.py` | CPU `weights/engine.py` | Sharding is part of native source emission; the source directory needs no Python package behavior. |
| `ppu2c02/clock.py`, `ppu2c02/control.py` | `ppu2c02/control.py` | Scanline event descriptors belong with the PPU's clock and bus effects. |

The former CPU `torch_forward.py` is named `bank.py`: it compiles and loads a
native forward bank. Its compiler now constructs union weights directly instead
of allocating an uninitialized machine with `__new__`. The runner explicitly
owns `self.forwards`; bank construction does not install itself as a side effect.

Other cleanup removes the duplicate `mux`/`select` and `match_byte`/`equals`
entry points, the unused SSA `from_weights` wrapper, the cache-lock wrapper and
the two-step NES make/import helpers. Cache location is read from the shared
build module, avoiding copies of mutable path globals. Diagnostic compilation
and graph layout methods have descriptive names used consistently by their
callers. Helpers that encode real operations, such as atomic writes, bit
extraction and fragment matching, remain named operations.

## Boundaries retained

| Area | Files and responsibilities |
|---|---|
| Graph language | `core.py` owns nodes, graph contexts and captured graphs; `ops.py` expresses bounded arithmetic idioms; `evaluate.py` evaluates known facts; `fold.py` rebuilds a specialized graph. These have different state and correctness contracts. |
| Generic model | `schedule.py` owns phases/liveness; `weights.py` owns matrices/serialization; `compiler.py` and `ssa.py` own successive lowering representations; `graph_engine.py` supplies the token engine; `inference.py` is an independent numerical reference. The package imports without the CPU package. |
| CPU backend | `runner.py` owns machine lifetime; `bank.py` owns neural kernels; `engine.py` owns the native shell; `experts.py` orchestrates construction; `folds.py` specializes/proves sequences; `execution.py` defines architectural effects and their slot ABI; `shapes.py` defines reusable instruction patterns. |
| Standalone checkpoints | `export.py` writes the artifact. `checkpoint_runner.py` and `expert_runner.py` each run it with only Torch, safetensors and the standard library. Their independence is part of the verification contract; sharing repository helpers would break that contract. |
| PPU circuits | `space.py` defines the shared memory address space; `registers.py`, `render.py` and `sprites.py` describe separate hardware work; `control.py` describes timing/bus effects; `machine.py` assembles their token bank and dispatch tables. |
| Other NES circuits | `controller.py`, `mappers.py` and `audio.py` describe distinct devices. `oscillators.py` contains the audio timer/noise transition mathematics, keeping that derivation separate from register decoding and mixing. |
| Frontend and observation | `nes/app.py` owns CLI/frontend lifecycle; `backends.py` adapts the CPU interface; `debug.py` coordinates observations; `xray.py` decodes traces and visual summaries. They do not participate in circuit compilation. |
| Native console | `bus`, `ppu`, `apu` and `console` have corresponding headers/implementations. Headers expose device contracts; implementation files contain their transitions and scheduling. `cartridge.h`, `chr.h`, mapper headers and palette constants supply their own data/transport contracts. |
| Native adapters | `nes/bindings.cpp` translates Python objects and native device interfaces. Compiler templates under `model/native/` and CPU `weights/native/` adapt the different kernel callers and the CPU machine ABI. |
| Native UI | `nes/app.cpp` owns the window and callbacks; `ui/debug_ui` owns the overlay; `frame_clock.h` owns pacing. `sokol_impl.m` is a separate platform-language translation unit. These link against frontend dependencies that hardware tests do not need. |
| Tests and examples | Tests stay beside their owning package. The conventional reference CPU/PPU and ROM harnesses remain separate from production neural construction. Synthetic carts are shared test fixtures. |
| Build and project metadata | The root Makefile owns native translation units; `pyproject.toml` owns Python packaging/commands; CI invokes those same build and test paths. Vendor code stays outside first-party source cleanup. |

Graph construction, compiler artifacts and emulation state have different
lifetimes. Keeping those owners explicit makes it possible to follow one
instruction or token without also reading frontend code, filesystem utilities
or a reference emulator.
