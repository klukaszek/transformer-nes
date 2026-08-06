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
| `spec.py` | Shared opcode, addressing-mode, flag, memory-read, and timing metadata. It contains 227 opcode entries: 151 official NMOS opcodes plus the stable unofficial opcodes exercised by the tests. It also includes a small assembler for examples. |
| `interpreter.py` | Builds the 6502 graph from the graph DSL. |
| `construction.py` | Defines the construction boundary and exposes the named dimensions consumed by scheduling and code generation. |
| `trace.py` | Instruction-level state and memory-write records shared by both backends. |
| `backends/reference.py` | Straightforward Python implementation used as the behavioral oracle. |
| `backends/weights/runner.py` | Builds, caches, and runs the transformer-weights backend. |
| `backends/weights/experts.py` | Builds folded opcode, interrupt, shape, and sequence execution models. |
| `backends/weights/windows.py` | Finds concrete instruction windows and matches the ROM-independent shape library. |
| `backends/weights/run_protocol.py` | Generates the native run-loop protocol used by the weights backend and the NES bus. |
| `examples/klaus/` | Klaus Dormann functional-test binary and harness. |
| `examples/nestest.py` | Harness for the nestest ROM and its Nintendulator golden log. |

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

`backends/reference.py` is intentionally ordinary Python. It uses a dense
64 KiB memory array, implements the full instruction set represented by
`spec.py`, and provides both traced execution (`step()` and `run()`) and a
fast trap-oriented loop (`run_traps()`). It is the independent behavioral
oracle for the weights backend.

The reference core can model either of the relevant CPU variants:

- `bcd_enabled=True` models an NMOS 6502 with decimal ADC/SBC.
- `bcd_enabled=False` models the NES 2A03, whose ADC/SBC ignore the decimal
  flag.

### Transformer-weights backend

`backends/weights/runner.py` builds the graph, schedules it with
`model/schedule.py:asap_schedule`, constructs the weights analytically, and
emits a generated C++ engine. There is no training step. The current
production build with decimal mode enabled is 624 dimensions wide and 56
layers deep; its attention and FFN widths vary by layer according to liveness
and the schedule.

The model uses hardmax attention, ReGLU feed-forward units, and a residual
stream. It has no position encoding: ordering is represented by the `LATEST`
tie-break. In the generated engine, the residual arithmetic is exact
fixed-point int32, and attention is served by licensed integer structures:

- memory queries use exact nearest-written-address state;
- constant-query `LATEST` lookups use running argmax registers; and
- unsupported lookup shapes are rejected during code generation.

The engine routes execution through several equivalent paths:

1. Optional per-ROM sequence modules handle explicitly attached, statically
   licensed instruction windows.
2. A ROM-independent shape library recognizes common windows such as counter
   loops and polling branches. Shape models are folded once with the PC and
   operand bytes left symbolic, then matched against the live image.
3. Folded opcode experts cover all 227 entries in the opcode table. Their
   compact execution path stages writes and routes them through the same
   memory, mapper, and watch machinery as the general path.
4. The union graph remains the exact fallback for unsupported dispatch cases
   and other paths that cannot use a specialized model. Interrupts have
   dedicated folded experts and still retain the union implementation behind
   them.

The generated engine is cached by a digest of the source and construction
configuration. A cache hit can load the compiled core without repeating the
folding and emission work. The optional standalone safetensors export is a
self-contained `wide_bits=False` union model; the generated runtime uses a
`wide_bits=True` build to reduce depth while its harness supplies the extra
byte-bit inputs.

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
5. `model/engine_gen.py` emits straight-line C++ and the native run loop.

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
- Before a device read, the bus uses the shared opcode metadata to identify
  the memory-reading instruction, watches its target, and injects the device's
  value into the CPU memory model. The CPU then performs an ordinary memory
  read.
- Idempotent device reads may provide a read lease, allowing the native run
  loop to serve repeated polls without returning to Python on every read.
- Mapper bank changes can replace mapped ROM bytes in the engine's address
  structures without pretending that the CPU executed a write for every byte.

The public weights-machine methods used by the bus include `load_image`,
`run`, `inject_write`, `set_nmi`, `set_irq`, `watch_read`, and `watch_write`.

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
`transformer_nes/nes/examples/`. The NES console tests in
`transformer_nes/nes/tests/` additionally compare reference and weights
backends at the hardware boundary, including frames, cycles, RAM, device
reads, and mapper writes.
