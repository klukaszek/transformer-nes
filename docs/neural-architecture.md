# From machine semantics to neural execution

This project expresses a programmable 6502 CPU, NES PPU data paths and cartridge
control as analytically constructed transformers. Attention retrieves
state and memory; linear projections and ReGLU neurons compute transitions.
The weights are derived from declared machine semantics. There is no parameter
training or learned instruction-window discovery.

For a first source read, follow the compilation stages below, then the worked
instruction. The [CPU guide](../transformer_nes/cpu6502/README.md) explains the
6502 construction in more detail.

## The compilation path

The CPU is the worked path below. PPU, mapper and controller tokens share graph folding,
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
| Torch lowering | [`model/torch_layout.py`](../transformer_nes/model/torch_layout.py), [`model/torch_codegen.py`](../transformer_nes/model/torch_codegen.py) | Expand the matrices into integer or float32 FX operations and compile fixed kernels with Inductor. |
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
   | Next PC | Model output: instruction head plus two |
   | Cycles | Constant two |
   | Writes | None |

3. Scheduling assigns residual slots to the live dimensions. `lower_execution`
   converts those dimensions to the slot-based specification consumed by the
   runtime and standalone checkpoint runner. Slot indices are layout details;
   the architectural effects above are the stable meaning.
4. The runtime embeds PC, runs the expert's matrices against the current byte
   image, and commits A=`$42`, Z=0, N=0 and the model's next PC. It retires two
   cycles using the licensed constant effect.

`ArchitecturalOutput` represents a model dimension, a constant or preservation
directly. Its source is derived from those fields. There is no intermediate
classification dictionary of dimension names to decode again. The final slot
specification remains a dictionary because it is also the checkpoint wire format.

## What the compiler specializes

The CPU's full graph computes instruction transitions. Partial evaluation
produces opcode continuations, read requests, interrupts and fixed polling/counter experts. The fixed shape
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

Memory-free real-valued circuits declare `dtype="float32"` on their graph
bank. Their coefficients and residuals retain float32 precision, and ReGLU
executes as multiplication by a rectified gate. They use the same matrix
lowering, compilation, tracing and ablation machinery, with a typed float
input/output ABI. Memory lookup heads are explicitly rejected in this mode;
CPU and discrete device banks keep their existing integer contract.

## What executes as neural work

CPU, PPU and mapper graphs share the Torch compiler. It expands their actual attention and
ReGLU matrices; it is not supplied handwritten opcode or rendering formulas.
Inductor specializes constant coefficients, removes dead residual slots and
fuses arithmetic. Large graphs may use several kernels connected through stack
scratch. Equivalent FX graphs share a kernel within a bank.

CPU kernels return only their live architectural readouts, packed contiguously.
The CPU caller decodes that packet directly, using a layout derived from its
effect specification during compilation. Kernels write directly into the
caller's separate output buffer; there is no copy or expansion back to a
residual buffer. Inputs are embedded afresh for each call; unused output slots
are not initialized. Full layer traces retain their complete
residual layout. PPU kernels already return their named token outputs directly.
The generic weight compiler consumes declared input/output slots; the CPU's
effect schema and output ordering stay in its backend.

The graph fold licenses indexed reads over the latest-written byte image before
scheduling. This replaces the declared memory attention's address-selection
fabric, rather than arbitrary attention. Execution needs no growing token
history, nearest-address bitmap or argmax-register cache. The CPU weight bank is
independent of the ROM; PPU banks account for their token and memory layouts.

Opcode, interrupt and instruction-shape experts expose their next PC as an architectural output,
alongside registers and flags. Its weights compute fallthrough, branch tests,
signed relative targets, jumps and returns. The runtime commits the returned PC
as a 16-bit field; it does not reread control operands or select branch targets.
The exported expert checkpoint uses the same state-readout contract (execution
ABI 3); its standalone runner also has no branch or jump implementation.

The host owns scheduling, opcode/shape routing, interrupt eligibility and
delivery of neural device effects. CPU, PPU, APU, controller and mapper
state transitions run through weights, including analog mixing and filtering.
The host also
commits state preservation and constant effects. A fully pinned circuit can
reduce entirely to constant or preserved effects and need no neural kernel;
an ordinary NOP still runs a forward to compute its next PC.

A polling or byte-counter recurrence starts with an instruction-shape forward.
Further iterations require that forward to confirm the state change and closing
branch, with device, instruction and cycle budgets still enforced by the host.
Four counter-transition experts then compute the bounded iteration count,
updated X or Y, zero/negative flags and elapsed cycles through ReGLU weights.
For decrement, the counter saturates at one; for increment, at 255. A single
ReGLU implements that saturation after the budgeted update. The difference
from the incoming counter gives the retired iterations, and another ReGLU
multiplies it by the first iteration's neural cycle count. There is no outcome
table or host counter/flag update.
The exit iteration still runs through the instruction graph with its different
branch timing. Tests compare these weights with literal stepping for every
licensed byte counter and byte-sized limit, as well as compiled execution under
cycle budgets and page crossings.

Polling retains its confirmed state only while its read lease holds. A verified
jump-to-self spin has no changing architectural state and can retire directly
within the budget. These reductions are part of the compiled system, not
evidence that every retired instruction performs a neural forward.

## Read requests and controller packets

Before a watched read, the CPU emits an address and access-cycle packet through
weights projected from the same instruction graph. The projection keeps only
`ea` and `cycles`: instruction-byte and pointer reads remain where needed, while
operand-value and ALU work disappear. The runtime uses the packet to service a
device or a valid read lease. It commits no CPU state until execution proceeds
with the reply.

The 151 memory-reading opcode routes share 15 request circuits. Sharing requires
identical matrices, memory-read layouts and input/output contracts. A request
returns a sixteen-byte packet: effective address, cycles and the two instruction
operand bytes, each held as an integer word. Its opcode routing identity is also
retained. CPU registers stay unchanged while the request is pending.

The compiler binds the prepared address and variable cycle count as inputs when
folding each opcode's continuation. Constant cycles remain licensed constant
effects. A compiler check rejects any continuation whose memory queries depend
on instruction or pointer reads instead of the retained address. After the
reply, the continuation reads the operand and computes the ALU, flags, writes
and next PC. It does not recompute the effective address or page-cross timing.

A device reply may change mapped memory or raise an interrupt. The instruction
uses its retained context and completes before that interrupt is taken. A fixed
instruction window runs only if its first instruction still matches the saved
opcode and operands. Checkpoints retain complete instruction graphs for an
independent runner; diagnostic union execution uses the same continuations at
device boundaries. Production memory-reading opcode experts always use the
split construction.

This models the operand-read boundary at the instruction's access cycle. It does
not introduce a complete cycle-by-cycle 6502 bus model.

[`nes/controller.py`](../transformer_nes/nes/controller.py) defines write, read,
lease and consume tokens. They return serial state, the direct reply and up to
eight queued reply bytes. The bus serves the packet and reports its consumed
length to another transition. C++ retains input delivery and packet transport;
it has no serial-index or strobe-update formula. Tests preserve the existing
input-sampling and lease behavior.

## PPU tokens and conditional effects

The PPU has token graphs for register access, event/bus control, memory-view
descriptors, rendering, sprite evaluation, fetch and overflow. A render token returns palette pixels, sprite hits and
conditional tile-fetch effects. Payloads are zero when `fetched` is clear, and
the host consumes them only when it is set. Folding can therefore remove the
inactive circuitry.

[`ppu2c02/clock.py`](../transformer_nes/ppu2c02/clock.py) constructs the bounded
scanline clock. Given a line and event index, its weights return the next line,
frame-wrap effect, timeline identity, event count, dot, action, tile slot and
contiguous run length. It preserves ordering at tied dots: sprite evaluation
precedes the dot-65 tile, and pre-render clearing precedes the dot-1 tile.
Run lengths stop at intervening actions, so IRQ previews retain their batching
boundary. Tests enumerate every line and event and also compare compiled
execution with the constructed matrices.

The construction uses seven layers and 38 residual slots; its layer matrices
contain 507 nonzero coefficients.

Clock descriptors depend only on fixed geometry. At engine attachment the
host evaluates the circuit for 262 line transitions and the five shared
timelines: 333 forwards in total, with a 3,144-byte transition table and 75
event packets. During play both live execution and previews consume these
immutable readouts, with no extra neural forward per tile or dot. The literal
timeline and rollover formulas exist only in the test oracle. Absolute clock
addition and storage remain host operations.

A blank-state expert pins both rendering enables off while retaining palette
addressing and greyscale behavior. Other experts pin verified control states.
Timing previews project graphs to bus/status readouts before scheduling, so
unused pixel work disappears. Previews write no framebuffer and restore live
state afterwards.

Register tokens return vblank-clear and NMI-enable/disable effects as well as
register state and memory writes. These effects feed the access event in
[`ppu2c02/control.py`](../transformer_nes/ppu2c02/control.py). Its weights compute
vblank suppression, cancellation of a recent NMI edge, and delayed NMI enables,
including the last-dot exclusion. Register conditions and timing rules are not
decoded again by the host. Memory writes commit directly from the register
token's write packet.

Five further event experts compute horizontal and vertical scroll copies,
sprite-line OAMADDR reset, odd-frame shortening, vblank entry and pre-render
status clearing. The same odd-frame expert predicts the next frame deadline;
there is no separate native prediction formula. State remains in the PPU's
preview snapshot, so live execution and dry runs use identical circuits.

The event token carries frame-relative phase and a bounded flag age: -1 before
the vblank flag has been set, 0 through 2 within its race window, and 3 for older
settings. Absolute clocks stay outside the integer model. Neural readouts
request timestamp set/clear and status-change effects; the host delivers them
at the scheduled dot. This keeps fixed-size model state independent of the
number of elapsed frames. Event experts are checked against the full matrices
and conventional rules across all status/effect bit combinations at the race
boundaries, then through the PPU/MMC3 conformance ROMs.

Bus tokens qualify MMC3 A12 edges. A direct pin event returns the next level,
a low-timestamp effect and a qualified rise. Background and sprite events
combine their ordered low/high pin pair into one forward, with the second pin
at offset four. The low-age context saturates to [-1, 11]. A rise requires at least eleven
low dots; paired fetches test at offset four. A pair starting high first resets the low timestamp and cannot
qualify its four-dot rise. A pair starting low can qualify at its second pin.
The host applies the returned timestamp and emits the edge at its returned
offset. Live rendering and IRQ previews use identical circuits and snapshots.

Register-side effect experts compute A12 drive requests, MMC2/MMC4 post-read
FD/FE latch updates, CHR invalidation and mapper-schedule invalidation. They
preserve MMC2's exact low-table addresses, MMC4's eight-row ranges and 15-bit
scroll-address aliases. A projection of the same graph returns the PPUMASK
write's one-dot catch-up delay before register mutation; it prunes the state
and bus readouts. The native register condition chain and edge filter are gone.

MMC5 view tokens return ExRAM visibility, extended-attribute mode, fill bytes
and span boundary, and the background CHR bank-set selection. The host applies
this descriptor through page views and bulk fill delivery. The same ExRAM
storage remains shared with CPU accesses. The weights compute mode decoding, fill-color expansion and sprite-size bank
selection. Independent CHR/latch and nametable formulas live in the test oracle.

Sprite overflow follows the diagonal OAM scan, including non-Y bytes. Unlimited
sprites repeat eight-index selection and fetch; extra rows cannot alter CHR
latches or mapper bus effects.

An eight-lane composition token in `control.py` updates the sprite line in OAM order. Its
bounded state packet contains the existing pixel window and the two occupied
tile flags it can touch. ReGLU selection preserves the first opaque pixel,
including its palette, behind-background and sprite-zero metadata. The circuit
clips the right edge and updates occupancy; a padded line and a sentinel tile
let the host apply every packet with the same fixed-size copies. Status previews
use this transition too, and snapshot the line state with the other render
state. Unlimited sprites reuse it across fetch batches, preserving priority.
No extra attention cache or frame history is introduced. This is an explicit
recurrent state transition, rather than a new attention-memory construction.
It shares the memory-free control bank, leaving the CHR attention variants
unchanged.

Conventional PPU implementations live under `nes/tests/ppu_reference.*` as
independent oracles. The host stores neural line readouts and schedules dots.

## Typed CPU write effects

The CPU graph emits each write's address, result and original operand for
memory read-modify-write instructions. Ordinary and stack writes carry no
original operand. The discriminator is constructed from the instruction's
neural decode, with interrupts and accumulator shifts excluded. It never
comes from a native opcode test.

Execution projection keeps the original byte only for live RMW effects.
The exported execution ABI 3 uses named address/value/original slots, with
`null` for an ordinary write. The full model emits a signed `-1` sentinel
instead. Both paths deliver the same typed receipt through machine ABI 2;
traced execution retains it too. The operand is the byte consumed by the
forward, including a serviced device reply, before the final write commits.

MMC1 consumes this original-byte effect directly. Its adapter no longer
rereads mapped ROM after the CPU has written, and neither CPU backend's bus
decodes an opcode to identify an RMW. The reference adapter now transports
only writes; its unused PC accessor and register snapshots are removed.

This preserves the existing instruction-granular device callback. Separately
scheduled intermediate writes and per-cycle arbitration remain ordered-bus
extensions; the receipt now contains the operand those extensions need.

## DMA transfer descriptors

The PPU control bank constructs OAM DMA as a bounded bulk-effect packet. Its
request projection returns the CPU source page address and parity-dependent
stall. Its transfer projection returns two destination spans and the rendering
schedule invalidation. Both project the same `dma_graph`; no source-page table
or transfer history is stored.

These projections preserve the ordering of the existing bus contract. The
source request precedes delivery, while OAMADDR and control are sampled after
PPU catch-up: crossing dot 257 can reset OAMADDR before the bytes arrive. The
host reads the selected CPU page and applies two copies from the returned
descriptor. The native page-selection, stall-length, wrapped-destination loop
and sprite-size invalidation rules have been removed from production. The
independent PPU test oracle retains its conventional byte loop.

DMA continues to use instruction-granular source-image delivery. Per-byte
MMIO reads and DMC/OAM bus arbitration remain ordered-bus work. The two-span
descriptor supplies neural control without one forward per transferred byte.

## Audio transitions and recurrent sample construction

[`nes/audio.py`](../transformer_nes/nes/audio.py) constructs seven integer token graphs
for the 2A03 and a separate float32 sample bank. Quarter, half and combined modulation experts advance all three
envelopes, four length counters, the triangle linear counter and both pulse
sweeps together. The combined event preserves quarter-then-half
order, including immediate clocks from a five-step sequencer write. Pulse 1's
negative sweep subtracts the extra unit; target overflow and the shift-zero
case retain their distinct update and mute rules.

The sample construction prunes state updates and computes pulse duty gates,
envelope/constant volume selection, triangle amplitude and channel muting
from current state. A diagnostic projection exposes its four digital DAC levels. Binary decomposition
and gated shifts implement sweep arithmetic; thresholds express the periodic
waveforms without state/outcome caches or native duty/triangle tables.

Frame-event experts handle sequencer clocks, control writes, status-read IRQ
acknowledgement and observation. They return the next phase, frame IRQ,
quarter/half clock effects and the next cycle deadline. The host advances to
this deadline, with no duplicate native sequencer table or phase-decision rule.
Only a due event runs a frame forward; empty oscillator spans still skip
arithmetically. All CPU backends require these audio weights.

Register-access tokens decode all APU ports, including channel enable/length
clears, timer programming, status reads and control effects. The chip's fixed
length and period ROMs become decoder weights. Frame-control and DMC
write effects route to the corresponding event experts, preserving their
ordering after the channel register updates.

The program token's DMC-write effect also ends the CPU chunk and re-plans
the next fetch without another forward.

DMC tokens implement register writes, sample-buffer replies, bit clocks and
initial observation. They compute bounded DAC steps, byte/bit reloads, silence,
reader wrap from $FFFF to $8000, sample restart/looping and IRQ transitions.
The fourteen state words use one fixed int32 packet. Fetch requests, four-cycle
stall effects and the next DMA horizon are neural readouts. The host fulfills
a requested memory read before the bit clock and returns its byte in a reply
token. It decrements the returned timer budget without rerunning the circuit
on each empty cycle; period changes leave the current budget intact.

[`nes/oscillators.py`](../transformer_nes/nes/oscillators.py) constructs clock
transitions for all four oscillators. The construction bound is
`S = min(429, max(1, ceil(cpu_frequency / sample_rate)))`: the DMC and sample
deadlines bound every span before a forward runs. It cannot cross a frame event
or audio sample. The sample rate is immutable console geometry, so the compiler
specializes the clock and noise circuits once at setup.
Bounded restoring division returns exact timer expiries and remaining counts,
including when a period write leaves the current counter above its new reload.
The same token computes CPU/APU parity and pulse/triangle phase advances.
A one-clock construction removes general division.

Noise carries fifteen binary residual lanes between forwards. Its transition
is linear over GF(2): construction squares the tap-1 and tap-6 matrices to
obtain at most eight conditional powers, or five at 44.1 kHz. Balanced XOR reductions use
`a + b - 2 * ReGLU(a, b)`; count bits select their composition. Up to 215 noise
steps execute without a native LFSR loop or state/outcome table. The clock token
emits the noise-step count; the host dispatches the recurrence token only when
that count is nonzero. A single-step expert folds away unused recurrence powers.
This avoids evaluating the entire noise circuit on spans without a noise expiry.
Sample tokens consume bit zero directly, avoiding repeated scalar unpacking.

The host batches through each boundary and delivers DMC/frame effects on its
final cycle. DMC memory reads do not observe oscillator state, so the oscillator
span can commit together without changing their transport order. The original
span-then-clock sample-accumulator additions remain unchanged. Literal clock
oracles check all noise states in both modes across reachable counts, randomized
timer contexts and composition of spans. Native tests also compare batching
against single-cycle advancement at six sample rates.

The same sample forward computes both nonlinear DAC branches, the recurrent
DC blocker and clipping to the playback range. Each rational DAC branch is
written as a linear numerator times a reciprocal. Four Newton steps,
`r = r * (2 - x * r)`, use ReGLU products with positive factors and start at
`r = 2/3`. Both normalized denominators lie in `[1, 2]` over the full channel
domain. Each step squares the reciprocal's relative error, yielding an ideal
bound of `3^-16` before float32 rounding. There is no amplitude lookup table,
runtime division or native mixer formula.

The DC blocker subtracts a small signed `0.0005` leak from its previous
output, rather than storing the near-one pole coefficient `0.9995` in
float32. Signed ReGLU pairs also stage the current-minus-previous mixed
level before adding feedback. Both changes preserve the real-arithmetic
filter and reduce coefficient bias and cancellation in float32. Long
constant-level decay and recurrent random inputs check the resulting error.

A control-rate token derives four channel gains after register or frame-sequencer
updates. The host retains these 16 bytes until the next control update; waveform
phases, noise and DMC state stay live. Sweep/envelope eligibility work scales
with control updates; waveform and mixer work scales with PCM samples.

The sample token receives 13 values: gains, duties, phases, the noise bit, DMC
level and two feedback values. Its three float32 outputs are the new mixed level,
filtered level and clipped PCM sample. C++ retains the feedback and delivers PCM
for playback. Level and DAC projections share the same constructions for
inspection and numerical verification.

The host retains scheduling, DMA transport and sample delivery. Independent
conventional register, DMC, modulation, frame and DAC oracles live in the native
hardware test. Matrix tests cover routed experts, immediate clocks, IRQ resets
and filter histories. Exhaustive checks of all 1,015,808 DAC combinations bound
absolute error at `2.67e-7`; a million-sample recurrent check stays below one
16-bit quantization step (`1/32768`). Audio equivalence uses numerical error
bounds as well as replay hashes.

## Runtime ownership and observation

The complete execution bank is attached before play, with no native neural
forward fallback or background learner. Cached starts load compiled extensions;
initial compilation can take several minutes. Artifact locks cover writing,
loading and publication, and pruning skips active builds. Generic build/cache
lifecycle lives in `model/build.py`; CPU source sharding lives beside its runtime
template. `native/machine_abi.h` defines the console interface.

After a cold CPU build, construction graphs are released and their cycles are
collected before gameplay. Evaluators share work while their source graph is
live; the compiler retains no global execution-folder cache owning that graph.

Dense models and debug graphs are built on demand. CPU/PPU X-rays label actual
residual live ranges, and ablation intervenes on a computed residual value.
Diagnostic kernels compile lazily. These facilities make transitions observable
without maintaining a second production neural backend.

The frontend advances emulation from monotonic wall time, preserving fractional
NES time when dropping overdue frames. It reserves presentation time before
starting catch-up work.

Memory-read instrumentation and heatmap collection run only while the X-ray
or its game tint is visible. Python and native diagnostic snapshots are reused
between emulated frames, and unchanged heatmap textures are not uploaded again.
Palette conversion uses packed RGBA lookups at the display boundary.

## Mapper tokens and effect horizons

[`nes/mappers.py`](../transformer_nes/nes/mappers.py) constructs banking transitions for MMC1,
MMC3, MMC2/MMC4, MMC5, their CPU access protocols and IRQ transitions for
MMC3/MMC5 as weights.

Banking uses two stages. A register-event token returns the next registers and
a dirty mask for PRG, CHR and mirroring. Only a dirty transition runs the layout
projection, which returns the selected banks and mirroring mode. Nonterminal
MMC1 serial writes and MMC3 selector-only writes therefore avoid layout work.
MMC1 read-modify-write tokens combine the original-byte write and the qualified
final reset into one transition, preserving their order.

The layout weights specialize immutable cartridge geometry, while all register
values remain runtime inputs. ROM contents never enter construction; cartridges
with the same chip and bank counts share a model. Constant-modulus division uses
bounded binary subtraction circuits, including for non-power-of-two capacities,
without a state/outcome cache.

[`BankMapper`](../transformer_nes/nes/include/nes/mappers/banked.h) holds opaque
register state and applies the returned maps. It copies changed
PRG windows and delivers CHR maps to the PPU. It contains no banking formulas
or serial-register implementation.

The CPU image adapter caches licensed instruction-shape locations for ROM
by immutable cartridge slice and CPU destination. Cache hits neither hash
nor retain copies of the ROM bytes. Writable RAM keeps content validation:
remapping code after a write must match the new contents even when the physical
allocation is unchanged. An MMC5 regression exercises that transition with
executable RAM.

MMC5 shares that transition/layout interface. Its layout returns physical ROM
or RAM banks, separate sprite/background CHR maps, RAM write permission and RAM
bank selection, nametable/fill configuration and multiplier bytes. The MMC5 adapter owns storage and applies these packets. CHR construction reduces each register by the period implied
by cartridge geometry before expanding a bank, discarding irrelevant upper bits
and registers before compilation.

Access tokens receive byte-split CPU addresses, write direction, an optional
original RMW byte, and the live neural bank map. They return a register endpoint,
bank event, IRQ event, scheduling effect, storage handle, physical byte index,
literal reply, permitted write and CPU-image repair request. Weights compute
MMC1/MMC3/MMC2/MMC4 port aliases, MMC3 IRQ register selection, MMC5 IRQ reads and
writes, multiplier read selection, RAM/ROM/ExRAM selection and write protection.

Read/write experts and address projections share each chip's access graph.
After a PRG-changing write, the address projection consumes the **new** neural
bank map before repairing the CPU's already-committed byte. It cannot reuse the
pre-write physical address. Other accesses need one access forward. The mapper
adapters dispatch returned endpoint/storage handles, copy changed windows and
apply effects; they contain no request-address classification or physical-index formula.
The control bank is shared with IRQ tokens, independent of ROM contents and
bank counts, and uses a fixed sixteen-word input packet with no address table.

For IRQ tokens, inputs are counter, latch/target, phase,
pending and enable state, plus an event and its data. Phase means a pending
reload on MMC3 and in-frame on MMC5. Event experts pin only the event tag;
register values and state remain live inputs. Both chips share one small,
ROM-independent control bank with no outcome table or growing history.

Outputs include next state, IRQ level and a conditional IRQ-line write,
MMC5 status-read data, prediction eligibility and the qualified-edge distance
to the next IRQ. The host applies these effects; it has no production counter
or status-acknowledgement implementation. IRQ writes remain explicit effects
because reasserting an existing level and acknowledging an inactive level must
preserve the device protocol, including the reference CPU adapter.

The effect horizon avoids replaying mapper state transitions during PPU
prediction. For an enabled MMC3, the first edge reaching zero is `latch + 1`
when reload is pending or the counter is zero, and `counter` otherwise. The
weights emit this distance after every transition. An MMC5 horizon accounts
for whether its first scanline has arrived, whether its target has passed,
and whether an IRQ is already pending. A passed target waits for an idle event;
that event runs the same weights on private preview state to begin a new epoch.

[`MapperIrqProbe`](../transformer_nes/nes/include/nes/mappers/irq.h) counts
qualified arrivals against this neural deadline and reports their absolute dot.
Its countdown is a scheduling budget, not a second mapper counter. Preview
never commits mapper state or IRQ writes. Mapper register accesses catch the
PPU up first and invalidate predictions when they change the IRQ schedule.
PPU bus tokens qualify A12 edges and return timestamped effects. The host
schedules absolute dots and delivers those effects to the mapper/probe.

The standalone native hardware test supplies an independent conventional
oracle. Production consoles require the compiled weights for both CPU backends.
Boundary tests compare full and routed weight execution, including zero-latch
reloads, control-byte thresholds, pending acknowledgements and frame epochs.
MMC3 conformance ROMs and identical controller tapes check integrated timing.

## Candidates for further neural execution

The remaining opportunities extend the existing transition and effect contracts:

| Candidate | Current boundary | Proposed construction and constraint |
|---|---|---|
| Cross-device event arbitration | The PPU's local timeline comes from weights, but `Console::run_until` and `Apu::advance_to` still coordinate CPU budgets, DMC stalls, frame-sequencer deadlines and PCM sampling. | A bounded deadline token emits the next span and ordered effects from relative clock distances. Keep absolute timestamps on the host and preserve access ordering and PCM phase at boundary ties. |
| Ordered bus effects | Operand-read continuations retain addresses and timing; OAM DMA emits neural bulk-transfer descriptors. Instruction, pointer, stack and vector reads still use the neural memory image directly. | Extend the effect contract to ordered bus requests, including intermediate RMW writes, DMA arbitration and precise access cycles. Preserve the single-commit continuation contract. |
| Banked memory attention | Mapper accesses select physical ROM/RAM addresses through weights. Instruction fetches still use copied PRG windows; [`sync_graph_memory`](../transformer_nes/nes/ppu.cpp) delivers mapped CHR views. | Extend physical bank addressing to instruction/fetch attention while storing ROM once. Measure copying first: adding translation to every read can cost more than copying on a switch. |

Ordered bus effects and banked attention offer the larger compiler/memory
contribution. DMA bytes,
framebuffer delivery and audio delivery remain storage/transport operations;
neural control need not turn each byte copy into a separate forward.

## Runtime cost model

The November 2026 audit replayed all 3,334 frames of the SMB3 gameplay tape,
including 120 tail frames, on an Apple M4 Pro with Python 3.13.7 and PyTorch
2.14.1. Tape SHA-256:
`bce3b91b5c1536b65510728aa5f4fdabd6089bbffe8e66ed18ec003f225e2889`.
Native ABI wrappers counted forwards; uninstrumented window/audio runs measured
latency. Per-call timers perturb short kernels and are excluded from those
latency measurements.

| Path | Mean forwards/frame, before → after | Work bound |
| --- | ---: | --- |
| PPU bus | 25,945 → 15,746 | Individual pin/fetch requests |
| Background spans | 0 → 954 | O(H) runs per IRQ lookahead over H lines |
| PPU render | 8,183 | Eight pixels per tile slot |
| PPU status projection | 1,600 | Tile work needed for sprite-zero timing |
| Sprite evaluation | 599 → 236 | O(UNK + QK) per OAM/expert version |
| CPU | 5,620 | Fixed expert work, including read continuations |
| APU clock | 2,012 | Four O(log S) bounded timer divisions |
| APU noise | 832 | O(15² log S) XOR/select work per update |
| Audio sample | 734 | Fixed channel gates, DAC refinement and filtering |

The NES sizes are fixed; these parameters describe construction families.
Total forwards fell from 48,114 to 38,504 per frame. The structural changes are:

- **Background runs:** only the first low/high pair can qualify a rise.
  A constant-size neural circuit returns final pin/stamp state, the first edge
  and the first-pair stamp for early IRQ stops. Precomputed event-run lengths
  let the scheduler clip at the requested horizon and skip completed runs.
  Background traversal changes from O(HT) to O(H) per lookahead, with T tiles
  per line. Live rendering and status prediction still process tiles.
- **Shared selection:** a 240-row table retains the first neural selection
  batch across rendering and previews. OAM generation, evaluation expert and
  engine replacement govern invalidation. For N=64 sprites, K=8 indices,
  Q requests and U≤240 distinct queried lines, work changes from O(QNK) to
  O(UNK + QK). Overflow timing and extra debug batches still run their tokens.
- **Address watches:** two 65,536-bit sets replace O(R) range scans with O(1)
  membership tests. Registering A addresses costs O(1 + A/64) word operations.
- **Read leases:** four inline entries match the CPU's four slots; replies
  contain at most eight bytes. Lookup, flush and invalidation inspect at most
  four entries. The shared expiry barrier and served-reply order are preserved.

Full window/audio runs averaged 5.22–5.24 ms/frame before and 4.40–4.48 ms after;
p99 changed from 7.29–7.52 to 6.56 ms. Uncapped replay changed from 0.834 to
0.703 ms/frame. These are workload-specific measurements, and replay throughput
does not establish display latency. Pixels, audio, CPU state, cycles and final
RAM matched over 13,129 frames across SMB3, CV3, Mega Man 6 and Zelda.

Warm-cache window timing also depends strongly on processor operating state.
A subsequent Mega Man 6 startup check reproduced roughly 3 ms/frame on both
the pre-cleanup revision (`7452b0a4`) and the neural-clock revision (`8b373c78`).
Repeated 600-frame runs measured 2.3–2.6 ms on the former and 2.4–2.6 ms on the
latter. These runs do not reproduce a sixfold cost increase from the clock.

In a single unmodified neural-clock process, alternating a bounded second CPU
workload produced the following windowed emulation times. Each phase lasted
300 frames; averages exclude its first 30 frames. No scripted input was sent.

| Phase | Additional CPU workload | Mean emulation time |
| --- | --- | ---: |
| Frames 30–299 | Off | 3.025 ms |
| Frames 330–599 | On | 0.448 ms |
| Frames 630–899 | Off | 2.790 ms |
| Frames 930–1199 | On | 0.454 ms |

Reversing the order measured 0.457 / 2.501 / 1.141 / 2.742 ms for on/off/on/off.
This supports processor frequency or core scheduling as a cause of the large
timing swings; the test does not identify the exact macOS mechanism. Audio
filter probes found no subnormal inputs or outputs during a slow run. Activity
and thread-priority hints did not reliably restore the faster timing.

The extra load was a temporary diagnostic, not part of the runtime. Compare
revisions in interleaved runs at matched load and report processor operating
conditions alongside wall time. Cache warmup, compiler activity and per-kernel
profiling can change those conditions; a warm cache alone does not normalize
live performance. Retired-instruction and cycle counts help distinguish added
work from a change in execution throughput.

| Storage | Size |
| --- | ---: |
| CPU inline state, including watch bits | 414,912 bytes |
| Watch bits within CPU state | 16 KiB |
| Diagnostic read counters within CPU state | 256 KiB |
| CPU byte image / shape index | 64 KiB each |
| CPU residual and readout buffers | 2,296 bytes |
| Host lease table | 64 bytes |
| PPU inline state, including neural clock metadata | 158,616 bytes |
| Clock line transitions / materialized event readouts | 3,144 bytes inline / 1,200 bytes in vectors |
| Sprite-row table within PPU state | 2,208 bytes |
| PPU synchronization metadata within PPU state | 8,968 bytes plus CHR dirty bits |
| Ordinary PPU graph byte image | 28 KiB; MMC5 variants grow with CHR geometry |

PPU state includes two 60 KiB frames and two 7.5 KiB expert maps. The PPU owns
CHR storage after mapper geometry setup; SMB3 avoids a duplicate 128 KiB payload.
Audio delivery recycles two vectors, about 8 KiB at 44.1 kHz. The completed frame
is copied once at the Python boundary. Graph-memory delivery averaged 15,325
byte writes/frame; render ABI packets represented 2.31 MiB/frame before kernel
liveness prunes unused loads. These interface volumes are not DRAM traffic.

Report owned state, compiled artifacts and process RSS separately. A cache-loaded
process measured about 193 MiB RSS after importing Torch and 214 MiB after
creating the console; RSS includes mapped libraries and allocator effects.
Dense construction matrices and generated source size do not measure runtime
working-set size.

Remaining targets are fusion of bus effects into live render readouts, folded
input/readout packing, on-demand diagnostic storage, and compiler-free cache-hit
loading. Measure bank-copy traffic before replacing mapped windows with physical
attention: translation on every read may cost more than occasional copies.

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
