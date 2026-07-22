# Optimization playbook

Levers that took the 6502 weights transformer from 0.13 instr/s to
~750K instr/s (full Klaus incl. lockstep; SMB gameplay tape ~1,700 fps),
stated architecture-agnostically so they
transfer to future targets (other CPUs, WASM, batched fleets).  Each
entry: the lever, the receipt here, and what it needs to transfer.
Session-by-session history lives in [ROADMAP.md](ROADMAP.md); design
details in [transformer_nes/cpu6502/README.md](transformer_nes/cpu6502/README.md).

## Preconditions

**A lockstep verification chain is the license for everything below.**
Golden logs ⇔ reference oracle ⇔ exact graph ⇔ weights, compared
field-for-field at every step.  Every lever landed
with a byte-identity receipt; several were *rejected* by one.  Without
the chain, aggressive semantic optimizations (exact structures, routing,
leases) are unverifiable.

**Measure before choosing.**  Three plans this project skipped because a
measurement overturned them: per-expert artifact caching (the 25 s
premise was stale), the width-objective MILP (the engine is nnz-bound,
not width-bound), unconditional linear inlining (symbolic bytes can
explode one row even when depth falls).

## Current construction receipts

Sparse terms below count the four emitted forward matrices (attention
in/out and FFN in/out), not zeroes or the unused dense shape.

| Model set | Previous construction | Current construction | Delta |
| --- | --- | --- | --- |
| 227 documented-opcode experts | 3,719 layers; max `d_model` 294; 265,934 sparse terms | 825 layers; max `d_model` 106; 126,706 sparse terms | −77.8% layers; −63.9% max width; −52.4% sparse work |
| Production full-receipt experts | 227 experts; 825 layers; max `d_model` 106; 126,706 sparse terms | none | −100% experts, layers and sparse work |
| 227 compact opcode experts | 825 full-receipt layers; max `d_model` 106; 126,706 sparse terms | 1,089 delta/control/effect layers; max `d_model` 78; 51,940 sparse terms | +32.0% layers; −26.4% max width; −59.0% sparse work |
| 134 untraced opcode experts | 438 full-receipt layers; max `d_model` 70; 66,379 sparse terms | 510 delta-only layers; max `d_model` 34; 21,473 sparse terms | +16.4% layers; −51.4% max width; −67.6% sparse work |
| 83 store/RMW opcode experts | 365 full-receipt layers; max `d_model` 106; 56,331 sparse terms | 560 delta/effect layers; max `d_model` 78; 29,439 sparse terms | +53.4% layers; −26.4% max width; −47.7% sparse work |
| 15 workload-independent shape experts | 257 layers; max `d_model` 60; 17,463 sparse terms | 64 layers; max `d_model` 34; 2,488 sparse terms | −75.1% layers; −43.3% max width; −85.8% sparse work |

The deterministic 24-instruction self-JMP is the clearest construction
trace.  Its full proof receipt built as 38 layers / `d_model` 596 / 6,889
sparse terms.  Projecting to one final execution receipt first made that
38 / 50 / 2,616; structural serving plus guarded inlining made it
1 / 44 / 188; classified state updates made it 0 / 32 / 0; deriving the
live-in ABI finally made it 0 / 2 / 0.  The proof graph still exists long
enough to license the optimization; none of it survives merely to explain
an already-proven execution.

For the generic LDA-absolute / AND / BEQ polling shape, typed lowering takes
the 141-dimension proof fold to a 20-dimension execution graph with five live
state fields, one `MemoryRead`, zero token `LookUp`s and none of
`byte_value`/`addr_lo`/`addr_hi`/`is_memory_write`.  Its emitted receipt is
3 layers / `d_model` 12 / 92 sparse terms.

## The levers

1. **Straight-line codegen specialization** (~5x).  Emit one C++ TU per
   model: column indices as immediates, ±1 coefficients as add/sub,
   coefficients as hexfloat literals, replicating the generic engine's
   arithmetic order exactly.  Transfers to any sparse analytic-weights
   model; the win is killing the indices/data loads of CSR gather.

2. **Exact fixed-point integer residual** (~2x).  Certify invariants
   first (integer coefficients, bounded values),
   then run the residual in int32 at scale 2^F with every rescaling shift
   guarded (throw, never round).  int64 lost (load-bound: value density
   wins); int16 died (range).  Transfers wherever the construction is
   dyadic-rational by design.

3. **Replace attention with exact structures where semantics prove it**
   (~1.5x + O(1) state).  Memory lookups whose argmax is provably the
   nearest written address became one latest-value array + bitset per
   group; constant-query heads became running-argmax registers updated at
   insert.  Result: nothing grows with the token log, checkpointing
   becomes optional.  Needs a tie-break equivalence argument per
   structure (same class as hull-vs-softmax).

4. **Move the protocol loop into the engine** (~1.4x).  The per-step
   Python floor (embedding dicts, readout, trace assembly) becomes a
   generated machine_step + native run loop; Python only services device
   pauses.  Do this *before* conditional computation — the run loop is
   where the router lives.

5. **Route conditional computation by construction, not workload**
   (2-3x hot paths).  Partial-evaluate the union graph per decode
   one-hot (the router is the decode the model already computes,
   hoisted), rebuild each folded graph as its own model, share attention
   state via provenance, fall back to the exact union for anything
   unrouted.  Route the *entire ISA*: the tuning-knob version (top-N of
   a game histogram) was retired the moment build cost was measured —
   workload-derived expert sets fall apart on the next workload.

6. **Strengthen the router predicate to strengthen the fold.**
   Dispatching on `ram[pc] == byte` (not just "this opcode's gate is 1")
   let the fold delete the opcode fetch itself.  Aligning trace-context
   conventions with the reference killed operand reads the experts never
   used.  General form: every fact the router checks is a pin the fold
   can spend.

7. **Profile the system boundary, not just the model** (2.6x here).
   The mature engine lost more time crossing the host boundary (MMIO
   pauses at ~6-8.5 µs each, 1,420/frame) than computing forwards.  The
   fix (read lease: device grants a value frozen until a known event;
   engine serves it natively) generalizes to any emulated device whose
   register semantics are piecewise-constant between events the host
   already schedules.

8. **Content-addressed build caching + distributable artifacts**
   (30 s → 1 s construction, 11x test suite).  Cache key = digest of
   every source file that shapes the emission + the build config —
   computable without doing the work; manifest maps key → compiled .so,
   dlopened directly.  Over-invalidation on any source edit is the
   safety argument.  Ship the cache directory: same platform loads with
   zero compiles, others rebuild from the shipped source.

9. **Hoist runtime one-hots into hard branches at every scale**
   (~2x step time here).  The opcode router (lever 5) is one instance of
   a general move: wherever the model computes an exact 0/1 selector and
   gates parallel branches with it, the generated code can evaluate the
   selector natively and run only the selected branch — substituting the
   provably-masked branches' already-exercised "empty" value.  Applied
   per memory read (16 grouped lookups → 1, since the group one-hot is
   `addr >> 12`), it took the hot loops from 0.87 to 0.37 µs/step.  The
   soundness argument is always the same: the gate is exactly zero for
   unselected branches, and the substitute equals a state the arithmetic
   already handles for real.

10. **Separate the proof graph from the execution graph before building
   weights.**  Fold the complete K-step composition under router pins and
   retain every intermediate readout until control, memory, write, halt,
   watch and self-modification licenses pass.  Then lower it to typed state
   leaves, external memory reads, one final architectural update and its
   statically-live ordered writes; traced
   execution uses the per-instruction path.  Generic shapes keep symbolic
   pc/operands, while concrete hot windows spend stronger byte pins.  The
   router re-verifies those facts live, and budget/speculation guards keep
   retirement step-for-step equivalent.  Proof state is compiler evidence,
   not a runtime ABI.

11. **Attribute emitted cost to construction gadgets before touching the
   construction** (found the 4x).  Wrap the graph's dim constructors to
   record each dim's creation site, replay the fold's liveness pass, and
   histogram live dims per helper-function line.  Here it showed ~80% of
   a median expert was ONE gadget (the grouped memory read) and RTI was
   ~78% bit-extraction ladders — nothing else was worth touching.
   Transfers to any generated-from-graph system: the emitted artifact
   tells you *how much*, only creation-site attribution tells you *what*.

12. **Serve formulation constraints only to the consumer that has them.**
   The union model splits memory into 16 lookup groups so exported f32
   attention scores stay distinguishable.  The exact engine does not
   compute those scores: a structural fold embeds current state directly
   and rebuilds each live RAM read as one first-class `MemoryRead(address)`.
   The read exposes value, exact address and only the bits requested by its
   consumers; its exact-address test lives in the primitive, so the graph
   carries no group gates, exact mask, key rows or WRITE-token value rows.
   General form: preserve the formulation required by the portable
   artifact, but erase it from a consumer with a stronger exact primitive.

13. **Share one monotone threshold ladder across parallel decodes of the
   same value** (union d_model 1762 → 702).  k separate bit/range
   extractions of one bounded integer each cost a full equality ladder
   (256 match gates per bit); one shared ge-threshold ladder (255
   parallel stepglus, depth one) makes every extraction a FREE linear
   telescoping run-sum over it.  Transfers to any construction that
   decodes fields from packed integers: build the cumulative ladder
   once, take differences linearly, persist only the named results.

14. **Pay the hazard convention only where a hazard exists** (−26% total
   emitted lines; Klaus +3.4%).  Phase-structured in-place updates
   (capture all reads, then apply all writes) cost two statements per
   value moved or erased.  A linear scan in apply order proves most
   rows' operands are untouched by the phase's earlier writes, so the
   pair collapses to one statement — and an erase whose target slot is
   unwritten becomes plain assignment.  Transfers to any generated
   engine with a read-then-write phase discipline: the convention is
   for the general case; the emitter can prove the special cases.

15. **A partial-cost objective is worse than none: the solver dumps cost
   into whatever you did not price.**  The scheduling MILP's "hygiene"
   objective priced attention-side passthroughs only and measured as a
   wash — it reduced the priced pool 28% while growing the unpriced
   (3x larger) FFN pool.  Total dim lifetime — Σ(death − birth), linear
   in variables the formulation already had — prices every carried slot
   uniformly and wins 10-20% per expert.  Generalizes to any greedy
   sweep too: check what the objective does NOT see before trusting a
   win on what it does.

16. **Inline linear intermediates only when the emitted model proves the
   trade is favorable.**  Inlining deletes materialization depth but can
   duplicate a symbolic value into thousands of downstream terms.  Build
   the structural materialized and inlined candidates, count their actual
   sparse forward work, and accept the inlined model only when it uses no
   more terms and no more layers.  This policy applies uniformly to all
   227 opcode experts, generic shapes and concrete sequences; it is not a
   ROM knob.  Together with structural serving it cut the opcode census
   from 3,719 to 1,574 layers and 265,934 to 140,764 sparse terms.  Recheck
   ambiguous choices in the generated engine: for LDA/AND/BEQ, the selected
   11-layer / 442-term model was slightly faster over interleaved five-million-
   instruction runs than the 5-layer / 476-term alternative.

17. **Make architectural deltas and dependency closure the model ABI.**
   Classify every final field—including each P bit independently—as preserved,
   compile-time constant, or model-computed; only computed fields remain output
   roots.  Do not carry five unchanged flags merely to repack a changed Z/N
   pair into a byte.  Trace context is not architectural state: gameplay
   models omit opcode, operands, effective address and data entirely, while
   traced execution uses the exact one-step path.  Non-halting
   is already a construction license, so sequence models carry no halt
   classifier.  The lowering produces a typed execution IR of live
   `StateField` leaves, `MemoryRead` leaves, ordered `MemoryWrite` effects and
   classified architectural outputs; the proof fold remains intact.  Embed
   only fields reachable from computed outputs or writes.  This is what lets
   a deterministic sequence become a genuine zero-layer model with only the
   constant-one residual slot (`d_model=2`).  Splitting P after the initial
   state-delta pass reduced max `d_model` 44 → 38 and sparse terms 4,013 →
   3,747.  Removing trace-only outputs then reached width 34 and 2,528 terms;
   DEX/BNE became 2 layers / width 10 / 90 terms with X as its only live-in.

18. **Return control decisions, not reconstructed program counters.**  A
   licensed region already knows its fallthrough and deterministic targets;
   a generic relative branch also exposes its signed operand byte to the
   engine.  Keep only the branch predicate as a model output and represent
   the successor as typed metadata (absolute, head-relative, or signed
   operand-relative).  This removed PC arithmetic, wrapping and mux chains
   from the shape library: 104 → 91 layers, max `d_model` 52 → 48 and
   6,178 → 5,657 sparse terms, with no ROM-specific assumptions.  A closing
   branch's final status bit is already an architectural output, so the engine
   now resolves control from P instead of retaining a duplicate `branch_taken`
   slot; the current shape census drops another 40 terms to 2,488.

19. **Expose bits by dependency, not by representation.**  The exact engine
   may serve byte bits directly, but carrying eight bit coordinates through
   every model simply moves the extraction cost into width.  Build the
   served-bit proof graph, then let execution projection retain only consumed
   state bits and let each `MemoryRead` enumerate only its requested bit
   positions (an ADC operand read, for example, exposes `bit7`, not eight
   bits).  Scalar loops keep their scalar ABI.  This cut the opcode census
   from 1,574 → 825 layers and 140,764 → 126,706 sparse terms; shapes fell
   from 91 → 59 layers and 5,657 → 4,013 terms.

20. **Summarize recurrences from ISA semantics, with the ordinary execution
   graph as fallback.**  Identity/self-JMP regions retire with no transformer
   forward.  DEX/DEY/INX/INY followed by BNE use an affine byte recurrence
   and terminal zero predicate.  LDA/BIT, optional AND, and a closing branch
   form an invariant polling recurrence after one ordinary forward; finite
   read-lease queues explicitly disable bulk retirement.  Every family is
   applied only when the live signed branch operand closes to the region
   head.  Step/cycle caps price the prefix before the final instruction, so
   partial regions fall back instruction-by-instruction; traced execution
   always does.  Fifty million untraced instructions fell from 1.256 s to
   0.0073 s on reset + DEX/BNE (~172×), and from 1.741 s to 19 µs on a stable
   LDA/AND/BEQ poll (>90,000×), with identical final state.

21. **Build proof facts on the dependency closure, not the union graph.**
   Interior sequence steps materialize only feedback plus the facts consumed
   by licenses: control, halt, effective address and ordered writes.  The
   final step alone materializes the architectural ABI, and structural
   `MemoryRead` depends only on its address rather than fictional token-side
   values.  Partial evaluation walks the static ancestor closure of those
   roots; the full receipt remains available for trace rollback.  A 24-step
   proof fell from 796 → 543 dimensions and 1,472 → 1,104 graph terms, with
   construction time 0.547 → 0.264 s.  The 15-shape library built in
   1.674 → 0.820 s while emitting the identical 59 layers / 4,013 sparse
   terms.

22. **Lower typed execution primitives directly in the generated engine.**
   A first-class `MemoryRead(address)` is an exact RAM load, not a reason to
   reconstruct the old nearest-write attention query over the engine's bitset
   index.  Evaluate its address and load RAM once, then derive every requested
   value/bit lane from that local; an eight-bit consumer went from five indexed
   queries/loads to one load.  Likewise, apply architectural deltas straight to
   persistent state: preserved fields emit no code, constants are immediates,
   computed fields read their residual slot, and control metadata assigns PC.
   Sequence and shape forwards no longer allocate or populate the legacy
   672-slot receipt buffer.

23. **Give untraced single instructions the same delta ABI as regions.**
   A shape miss used to force gameplay through a full 21-field tracing expert.
   Build a second one-instruction family only for opcodes whose current read has
   already passed watch arbitration and whose instruction has no write, pull,
   interrupt or unresolved jump effect.  The engine uses it only with
   `trace=false`; every excluded effect and every interrupt falls through to the
   original exact expert.  This admits 134 documented opcodes, including 39
   zero-layer experts, at 21,473 sparse terms instead of 66,379 for the same full
   receipts.  On a shape-disabled LDX/DEX/BNE loop, two million instructions fell
   from 49.1 ms to 9.7–10.4 ms (~4.9×).  Zero-layer emission advances the model
   position and applies metadata/deltas without clearing an unused residual.

24. **Turn measured construction policy into code, not constructor knobs.**
   The production machine now has one semantic option, decimal mode. It always
   builds the entire ISA, compact untraced experts and the canonical shape
   library; every fold uses structural reads, compares materialized and inlined
   forms by emitted layers/terms, and uses deterministic balanced-ASAP
   scheduling. Removed alternatives included expert subsets, shape-library
   overrides, per-expert MILP timing, forced inlining/materialization, token-read
   folds, runtime heat thresholds, speculative shape rollback and guarded shape
   stores. The last two were especially misleading: their metadata had no
   generated consumer. Removing the unused trace-window construction and native
   proposal/profiling path removed over 700 net lines without changing the model
   census. Concrete store windows remain an explicit, statically licensed
   sequence-module operation rather than a tuning mode.

25. **Do not build a diagnostic receipt beside an equivalent execution
   model.** The 134 compact opcodes already own untraced execution, so their
   full 21-field experts served tracing alone. Traced steps now use the exact
   union fallback for those bytes, leaving full receipts only for the 93
   opcodes not yet compacted. The production core fell from 376 to 242 experts;
   its full-receipt family fell from 825 to 387 layers and 126,706 to 60,327
   sparse terms. Gameplay routing is unchanged, while tracing remains exact.

26. **Stage effects; do not preserve a receipt merely because an instruction
   writes.** A one-step execution model now returns architectural deltas plus
   at most three ordered `(address, value)` effects. The run loop commits those
   effects through the existing mapper, journal and write-watch arbitration,
   preserving self-modification order, `pause_ir`, RMW routing and token
   positions without carrying 21 trace fields. This admits all 83 store/RMW
   opcodes: their sparse work falls 56,331 → 29,439 and max width 106 → 78.
   Across the production opcode/shape core, sparse terms fall 84,288 → 57,396
   while max width falls 106 → 78. One million shape-disabled STA-heavy
   instructions fell 27.3 → 12.5 ms (~2.18×); INC-heavy instructions fell
   30.6 → 17.2 ms (~1.78×).

27. **Control is an effect, not a reason to keep the trace receipt.** Absolute
   JMP and JSR targets are instruction-byte metadata; stack pushes and pulls use
   the existing ordered write and structural-read primitives; indirect JMP,
   RTS and RTI return one 16-bit successor; BRK alone returns an additional halt
   bit because a zero vector halts this machine. This admits the final ten
   stack/control opcodes without reconstructing `next_pc` for static control or
   retaining trace context. Their construction falls from 22 layers / max
   `d_model` 60 / 3,996 sparse terms to 19 / 26 / 1,028; absolute JMP is zero
   layers. The complete opcode+shape core falls 57,396 → 54,428 sparse terms,
   with no full-receipt experts left in production.

## Dead ends (kept so they stay dead)

- GPU per-matrix sparse matvec for a single stream: one dispatch costs a
  whole CPU forward; only batched fleets or a fused resident stack.
- Single-stream threading across layers (sequential dependency),
  zero-fill elimination (cost is the gather, not memset).
- Width-minimizing schedules: ~7% step time for optimal width — realign
  the objective to the engine's cost model (nnz + slot hygiene) instead.

## Bench traps that produced wrong conclusions here

- Def-time default arguments: monkeypatching a module constant left both
  A/B configs identical.  Construct with explicit arguments.
- Two torch extensions defining the same struct name cross-resolve
  symbols and silently time the wrong code; digest-name every struct.
- Same-window comparisons only: fps in a sprite-0-polling window is not
  comparable to a title-screen window (this produced a stale "MMIO is
  5%" conclusion that survived three sessions).
- The artifact digest must include EVERY source that shapes the model:
  the graph construction file was missing from the engine cache key, so
  a construction edit silently benchmarked (and nearly shipped) stale
  engines.  Caught only because d_model changed enough to crash.
- Paced slices are not throughput: a frame loop that runs 1.3 ms of
  emulation then idles 15 ms to the next display callback measures
  4.3 ms/frame for the SAME work (DVFS drops the clocks and cools the
  caches between slices), and the penalty compresses real workload
  differences into near-identical numbers.  Extrapolating "uncapped fps"
  from paced slice times severely understates throughput; measure sustained
  regimes (tape replay) for throughput claims.
