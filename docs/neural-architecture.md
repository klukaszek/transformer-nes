# From machine semantics to neural execution

This project expresses a programmable 6502 CPU and the NES PPU's data-path
operations as analytically constructed transformers. Attention retrieves
state and memory; linear projections and ReGLU neurons compute transitions.
The weights are derived from declared machine semantics. There is no parameter
training or learned instruction-window discovery.

For a first source read, follow the compilation stages below, then the worked
instruction. The [CPU guide](../transformer_nes/cpu6502/README.md) explains the
6502 construction in more detail.

## The compilation path

The CPU is the worked path below. PPU token graphs share graph folding,
scheduling, weight construction and Torch lowering; their host consumes named
token readouts rather than the CPU execution-effect IR.

```mermaid
flowchart LR
    semantics[Machine semantics] --> graph[Computation graph]
    graph --> fold[Partial evaluation]
    fold --> proof[Dispatch proofs]
    proof --> effects[Execution effects]
    effects --> schedule[Layer scheduling]
    schedule --> weights[Transformer matrices]
    weights --> abi[Residual-slot ABI]
    abi --> torch[Torch compilation]
    torch --> runtime[Native scheduling and devices]
```

| Stage | Implementation | Contract |
|---|---|---|
| Machine semantics | [`cpu6502/interpreter.py`](../transformer_nes/cpu6502/interpreter.py), [`ppu2c02/`](../transformer_nes/ppu2c02/) | Construct expressions for a machine transition or PPU token. `interpreter.py` builds a graph; it is not the runtime CPU emulator. |
| Graph language | [`graph/core.py`](../transformer_nes/graph/core.py), [`graph/ops.py`](../transformer_nes/graph/ops.py) | Inputs, attention lookups, ReGLUs, linear persists and captured graphs. A ReGLU computes `a * max(0, b)`. |
| Partial evaluation | [`graph/evaluate.py`](../transformer_nes/graph/evaluate.py), [`graph/fold.py`](../transformer_nes/graph/fold.py) | Substitute licensed constants, remove dead work and rebuild a graph. Each construction has its own node identities and interning state. |
| CPU composition | [`weights/folds.py`](../transformer_nes/cpu6502/backends/weights/folds.py) | Chain instruction graphs with register feedback and symbolic PC offsets. |
| CPU proofs | [`weights/proofs.py`](../transformer_nes/cpu6502/backends/weights/proofs.py) | Bound reads, identify constants and exhaustively check symbolic PC expressions where their dependency domain permits it. |
| CPU effects | [`weights/execution.py`](../transformer_nes/cpu6502/backends/weights/execution.py) | Project a licensed graph to final state, ordered writes, control and cycles. Lower graph dimensions to residual slots after scheduling. |
| Expert construction | [`weights/experts.py`](../transformer_nes/cpu6502/backends/weights/experts.py) | Orchestrate folding, validation, projection and construction in stable dispatch order. |
| Layers and weights | [`model/schedule.py`](../transformer_nes/model/schedule.py), [`model/weights.py`](../transformer_nes/model/weights.py) | Schedule attention/linear/FFN phases and construct their actual matrices. |
| Torch lowering | [`model/torch_layout.py`](../transformer_nes/model/torch_layout.py), [`model/torch_codegen.py`](../transformer_nes/model/torch_codegen.py) | Expand the matrices into exact integer FX operations and compile fixed kernels with Inductor. |
| CPU runtime | [`weights/native/engine.cpp`](../transformer_nes/cpu6502/backends/weights/native/engine.cpp), [`weights/engine.py`](../transformer_nes/cpu6502/backends/weights/engine.py) | Embed live inputs, dispatch kernels and commit licensed effects within device and cycle budgets. |

The graph and model packages are independent of the CPU package. The neural
construction and runtime paths also import without loading the reference CPU
emulator. Both dependency boundaries are checked in tests. Opcode metadata and
flag definitions live in `cpu6502/spec.py`; the compiler does not obtain those
constants from the reference executor.

## One instruction: `LDA #$42`

Assume PC is `$8000` and memory contains `$A9 $42`. The opcode expert is compiled
with the decoder pinned to `$A9`; dispatch verifies that byte at the live PC.
The PC and operand value remain runtime data. This is one ROM-independent
expert, not a model compiled for this instruction address or game.

1. The folded graph reads the operand at `PC + 1` through attention. Its ALU
   and flag expressions compute the accumulator value and zero/negative flags.
2. The compiler validates the dispatch facts before removing proof-only
   readouts. Projection produces typed architectural effects:

   | Effect | Source |
   |---|---|
   | Accumulator, zero flag, negative flag | Model outputs |
   | X, Y, stack pointer and other mutable status flags | Preserve incoming values |
   | Next PC | Instruction head plus two |
   | Cycles | Constant two |
   | Writes | None |

3. Scheduling assigns residual slots to the live dimensions. `lower_execution`
   converts those dimensions to the slot-based specification consumed by the
   runtime and standalone checkpoint runner. Slot indices are layout details;
   the architectural effects above are the stable meaning.
4. The runtime embeds PC, runs the expert's matrices against the current byte
   image, and commits A=`$42`, Z=0 and N=0. It advances PC by two and retires two
   cycles using the licensed constant effects.

`ArchitecturalOutput` represents a model dimension, a constant or preservation
directly. Its source is derived from those fields. There is no intermediate
classification dictionary of dimension names to decode again. The final slot
specification remains a dictionary because it is also the checkpoint wire format.

## What the compiler specializes

The CPU's full graph computes instruction transitions. Partial evaluation
produces opcode, interrupt and fixed polling/counter experts. The fixed shape
library, recurrence checks and ROM matcher live together in
[`weights/shapes.py`](../transformer_nes/cpu6502/backends/weights/shapes.py).

A dispatch proof is a build-time condition, not a confidence estimate. For
example, fixed shapes must not hide writes from device arbitration; later reads
must stay in licensed memory regions; and interior control must equal the
expected symbolic PC progression. The complete proof graph remains available
until these checks pass. Projection then removes receipts needed only to justify
execution.

Both ordinary and linearly inlined candidates are constructed. The builder
accepts inlining only when its layer count and sparse matrix work do not
increase. These are compiler decisions over the same machine semantics.

The scheduler recycles dead residual slots. Erase projections subtract the old
value before reuse. The constant-one channel remains live because the linear
passthrough and erase gates introduced during weight construction depend on it.

The residual scale follows the matrices: integral projections use whole
integers; fractional projections retain half-unit precision. The PPU uses whole
integers. This removes unnecessary rescaling and lets kernels write directly to
the host's integer output buffers.

## What executes as neural work

CPU and PPU share the Torch compiler. It expands their actual attention and
ReGLU matrices; it is not supplied handwritten opcode or rendering formulas.
Inductor specializes constant coefficients, removes dead residual slots and
fuses arithmetic. Large graphs may use several kernels connected through stack
scratch. Equivalent FX graphs share a kernel within a bank.

The graph fold licenses indexed reads over the latest-written byte image before
scheduling. This replaces the declared memory attention's address-selection
fabric, rather than arbitrary attention. Execution needs no growing token
history, nearest-address bitmap or argmax-register cache. The CPU weight bank is
independent of the ROM; PPU banks account for their token and memory layouts.

The host owns scheduling, mapper and device interactions, the APU, controllers,
display and audio delivery. It also commits state preservation and licensed
control effects, including flag tests and operand-relative targets. Some circuits reduce entirely to
constant or preserved effects and need no neural kernel.

A polling or byte-counter recurrence starts with a transformer forward. Further
iterations may be skipped only when that forward confirms the state change and
closing branch, and device, instruction and cycle budgets permit them. The
counter's exit iteration still runs through the transformer. A verified
jump-to-self spin has no changing architectural state and can retire directly
within the budget. These host reductions are part of the compiled system, not
evidence that every retired instruction performs a neural forward.

## PPU tokens and conditional effects

The PPU has token graphs for register access, rendering, sprite evaluation,
fetch and overflow. A render token returns palette pixels, sprite hits and
conditional tile-fetch effects. Payloads are zero when `fetched` is clear, and
the host consumes them only when it is set. Folding can therefore remove the
inactive circuitry.

A blank-state expert pins both rendering enables off while retaining palette
addressing and greyscale behavior. Other experts pin verified control states.
Timing previews project graphs to bus/status readouts before scheduling, so
unused pixel work disappears. Previews write no framebuffer and restore live
state afterwards.

Sprite overflow follows the diagonal OAM scan, including non-Y bytes. Unlimited
sprites repeat eight-index selection and fetch; extra rows cannot alter CHR
latches or mapper bus effects. Conventional PPU formulas are compiled only into
the independent test oracle under `nes/tests/ppu_reference.*`.

## Runtime ownership and observation

The complete execution bank is attached before play, with no native neural
forward fallback or background learner. Cached starts load compiled extensions;
initial compilation can take several minutes. Artifact locks cover writing,
loading and publication, and pruning skips active builds. Generic build/cache
lifecycle lives in `model/build.py`; CPU source sharding lives beside its runtime
template. `native/machine_abi.h` defines the console interface.

Dense models and debug graphs are built on demand. CPU/PPU X-rays label actual
residual live ranges, and ablation intervenes on a computed residual value.
Diagnostic kernels compile lazily. These facilities make transitions observable
without maintaining a second production neural backend.

The frontend advances emulation from monotonic wall time, preserving fractional
NES time when dropping overdue frames. It reserves presentation time before
starting catch-up work. Diagnostic snapshots are reused between emulated frames.

## Evidence and research scope

`uv run pytest` checks graph isolation, folding licenses, exact expert forwards,
reference/checkpoint agreement, device timing, rendering and diagnostic
interventions. The reference CPU shares declarative opcode metadata with the
compiler; nestest's external golden log and conformance ROMs provide additional
checks beyond that shared definition. Standalone union and routed-expert runners
retain independent PyTorch inference implementations, including softmax attention.

The current evidence supports a constructive neural implementation and compiler
for this supported machine. It does not demonstrate learned execution,
statistical generalization to new instruction sets, universal replacement of
attention by indexed reads, or an entirely neural NES. The finite 6502 memory
model also does not establish an unbounded-computation result. A broader
neural-computing claim needs additional machines or tasks and explicit evaluation.

For performance experiments, replay identical controller tapes and exclude
compilation/loading from steady-state timing. Compare pixels, audio,
architectural state, cycles and final RAM. Measure the actual frontend too:
uncapped replay averages do not establish live frame latency or display cadence.
Keep local timing dumps and benchmark JSON outside the source tree. Published
results should state hardware, software versions, input tapes, warmup policy,
compiler settings and the contribution of host reductions.
