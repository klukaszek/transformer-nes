# 6502 CPU as a transformer computation graph

This package models a MOS 6502 as an analytic computation graph. The graph
can be compiled into transformer weights, with one semantic machine step per
6502 instruction. The generated runtime may fold a licensed instruction
window into one native call, but it preserves the same instruction-level
state and trace contract.

The package contains the CPU and its verification tools. NES-specific
hardware—such as the bus, PPU, APU, controllers, cartridges, and mappers—lives
in [`transformer_nes/nes/`](../nes/).

## What is here

| File | Purpose |
|------|---------|
| `spec.py` | Shared opcode, addressing-mode, flag, memory-read, and timing metadata. It contains 227 opcode entries: 151 official NMOS opcodes plus the stable unofficial opcodes exercised by the tests. |
| `graph.py` | Builds the 6502 graph and exposes the named dimensions consumed by scheduling and code generation. |
| `trace.py` | Instruction-level state and memory-write records shared by both backends. |
| `reference.py` | Straightforward Python implementation used as the behavioral oracle. |
| `chip.py` | The CPU as a chip of the NES model: the union circuit, every execution as a specialization with the same named outputs, read requests, and the run loop's dispatch tables. |
| `runner.py` | The machine API over the native run loop (`native/console/cpu.cpp`). |
| `eager.py` | Runs the saved model's CPU chip in eager PyTorch, importing only Torch, safetensors and `model/transformer.py`. |
| `experts.py` | Orchestrates folding, dispatch proofs and effect projection. |
| `folds.py` | Opcode pins, sequence feedback/composition, range proofs and symbolic-PC/read licenses. |
| `execution.py` | Typed architectural effects, control/cycles and graph projection. |
| `shapes.py` | Fixed shape templates, recurrence proofs and matching against live cartridge bytes. |
| `conformance/klaus.py` | Harness for the Klaus Dormann functional test (binary in `fixtures/klaus/`). |
| `conformance/nestest.py` | Harness for the nestest ROM and its Nintendulator golden log (in `fixtures/roms/`). |

## How the graph represents a CPU

The graph uses an append-only history of tokens:

- A `start` token supplies the initial program counter.
- `WRITE` tokens record memory writes as `(address, value)`. The initial
  64 KiB image is seeded as nonzero writes, and stores and stack operations
  append more writes. Reads select the latest write for an exact address;
  an unwritten address reads as zero. Code and data therefore share one
  memory model, and instruction fetch is just a memory read.
- `STATE` tokens carry the architectural state—PC, A, X, Y, S, and all eight
  status flags—along with the previous instruction's fetch and effective
  address fields. The graph retrieves the latest state token for the next
  step.

For each step, the graph fetches the opcode and operand bytes, resolves the
addressing mode, computes the operation and flags, and emits the next state
plus any memory writes. The graph exposes three write channels so operations
such as JSR, BRK, and interrupt entry can report their stack writes in one
step.

Memory lookups are split into sixteen 4 KiB groups. Each group uses an exact
nearest-address attention structure and an exact-address mask, so a nearby
write cannot be mistaken for the requested address. Equal attention scores
use the `LATEST` tie-break, which is part of the machine's semantics rather
than a floating-point convenience.

## Execution backends

Both backends consume the same opcode specification and are checked against
the same trace contract.

### Reference backend

`reference.py` is intentionally ordinary Python. It uses a dense
64 KiB memory array, implements the full instruction set represented by
`spec.py`, and provides both traced execution (`step()` and `run()`) and a
fast trap-oriented loop (`run_traps()`). It is the independent behavioral
oracle for the weights backend.

The reference core can model either of the relevant CPU variants:

- `bcd_enabled=True` models an NMOS 6502 with decimal ADC/SBC.
- `bcd_enabled=False` models the NES 2A03, whose ADC/SBC ignore the decimal
  flag.

The transformer's decimal circuitry assumes valid packed BCD operands;
invalid decimal digits can disagree with the reference. The NES variant
always uses binary arithmetic, including when its decimal flag is set.

### Transformer-weights backend

`runner.py` builds the graph, schedules it with
`model/schedule.py:asap_schedule`, constructs the weights analytically, and
emits a generated C++ engine. There is no training step. Its attention and
FFN widths vary by layer according to liveness and the schedule; current
sizes and paired measurements are in
[the architecture notes](../../docs/neural-architecture.md).

The model uses hardmax attention, ReGLU feed-forward units, and a residual
stream. It has no position encoding: ordering is represented by the `LATEST`
tie-break. In the generated engine, the residual arithmetic is exact
fixed-point int32, and attention is served by licensed integer structures:

- licensed structural memory queries index the latest-written byte image;
- full and folded execution models share the Torch weight compiler; and
- unsupported lookup shapes are rejected during code generation.

The engine routes execution through several equivalent paths:

1. A ROM-independent shape library recognizes polling and counter loops.
   Shapes fold once with symbolic PC and operand bytes; dispatch rechecks their
   instruction bytes and budgets against live memory. Four bounded counter
   experts compute additional iterations, X/Y, flags and elapsed cycles in weights.
2. Per-opcode and interrupt experts compute compact architectural receipts:
   registers, flags, ordered writes, next PC and exact cycles.
3. The union model supplies the complete fallback and diagnostic trace.

The executions are the CPU chip's circuits, compiled with every other chip into
the NES model's library. FX expands sparse matrix coefficients into integer
operations; scalar SSA shares surviving values and folds integer affine routing
while preserving half-unit rounding. Kernels read the chip's memory and return
the step token's named outputs. The chip is independent of the ROM and loads
before the first frame. There is no runtime learner, mutable window program,
weight tensor pool or background attachment queue. `eager.py` runs the same
weights in eager PyTorch, a separate check of ordinary transformer inference.

The CPU chip is saved and compiled with the rest of the NES model
(`nes/model.py`): a cache hit loads it without repeating construction.

Graph construction exploits its integer domains before scheduling. A Boolean
product needs one ReGLU even when the payload is signed; selecting a constant
payload needs only an affine expression. Opcode equality uses unit tents whose
neighboring comparisons share ReLU hinges. Pulled status bytes use the serial
14-unit bit extractor within the existing critical path. The final bit is the
remaining 0/1 value, so it needs no extra threshold.

The CPU chip has 55,587 nonzero weights: its union circuit 11,009, its 220
executions 41,865 and its 15 read requests 1,887 (executions that fold alike
share one circuit). The saved model packs each tensor into the smallest exact
signed integer type; values, shapes and fixed-point units are unchanged by
packing.

## Compilation pipeline

The weights backend follows this sequence:

1. `Cpu6502Graph` builds the graph and identifies the dimensions needed by the
   machine protocol.
2. `asap_schedule` computes the critical path and load-balances operations
   across their legal phase windows. It is the production scheduler.
3. `model/weights.py` assigns live dimensions to residual slots and constructs
   the analytic attention and ReGLU matrices.
4. `graph/fold.py` partially evaluates the union graph to build licensed
   opcode, interrupt, shape, and sequence models.
5. `model/compiler.py` expands the matrices into FX; `model/ssa.py` lowers and
   shares scalar arithmetic, shared with the PPU. The graph fold exposes
   licensed indexed reads before scheduling. `chip.py` gives the executions
   one interface; the native run loop (`native/console/cpu.cpp`) embeds the
   state, dispatches and commits.

The run loop can execute for a step or cycle budget, stop at a HALT or
jump-to-self trap, stream bounded trace arrays, and pause on watched device
reads or writes. Full-suite verification uses chunked streaming, so it does
not build a 30-million-row trace in memory.

## Interrupts and NES device access

NMI and IRQ delivery are part of the graph. The surrounding machine only
asserts a pending line; on the next step, the graph performs the interrupt
sequence instead of executing the instruction at PC. It pushes PCH, PCL, and
the status byte (with B clear and U set), updates S and I, and reads the
appropriate vector. NMI takes priority over IRQ, and IRQ is delivered only
when the I flag is clear.

The CPU remains unaware of PPU, APU, controller, and mapper implementations.
The native NES bus connects those devices to the CPU protocol:

- CPU stores are returned as write records and dispatched to mapped devices.
- A neural read-request expert emits the operand address and timing before
  committing the instruction. The bus serves a watched device read, and a
  retained neural continuation consumes the reply and commits once. The
  reference backend uses opcode metadata to preview the equivalent read.
- Idempotent device reads may provide a read lease, allowing the native run
  loop to serve repeated polls without returning to Python on every read.
- Mapper bank changes can replace mapped ROM bytes in the engine's address
  structures without pretending that the CPU executed a write for every byte.

Python boots the weights machine through `load_image` and installs device
watches. During play the bus calls the native CPU (`native/include/nes/cpu.h`)
directly for runs, writes, cartridge mapping, read leases and interrupts. The
Python `run` and injection methods support verification harnesses.

## Verification

The two main CPU suites cover complementary behavior:

- **Klaus Dormann functional test:** exercises the 151 official opcodes,
  including decimal mode. A passing run reaches the jump-to-self trap at
  `$3469` after 30,646,177 instructions. The weights runner is checked in
  lockstep against the reference core.
- **nestest:** compares all 8,991 entries in the Nintendulator golden log
  field by field (`PC`, `A`, `X`, `Y`, `P`, and `SP`). It runs in NES 2A03
  mode and includes the stable unofficial opcodes used by the ROM.

Run the suites from the repository root:

```bash
# Reference implementation: full decimal-mode Klaus suite
uv run 6502-klaus --runner reference

# Full Klaus suite through the generated transformer engine
uv run 6502-klaus --runner weights --max-steps 30700000

# Reference vs. the nestest golden log, then weights vs. the reference
uv run 6502-nestest --runner both

# Weights-only nestest run
uv run 6502-nestest --runner weights

# CPU package tests
uv run pytest transformer_nes/cpu6502/tests
```

The nestest harness reads `nestest.nes` and `nestest.log` from
`fixtures/roms/`. The NES console tests in
`transformer_nes/nes/tests/` additionally compare reference and weights
backends at the hardware boundary, including frames, cycles, RAM, device
reads, and mapper writes.
