# Optimization playbook

Levers that took the 6502 weights transformer from 0.13 instr/s to
~335K instr/s (SMB at 123 fps), stated architecture-agnostically so they
transfer to future targets (other CPUs, WASM, batched fleets).  Each
entry: the lever, the receipt here, and what it needs to transfer.
Session-by-session history lives in [ROADMAP.md](ROADMAP.md); design
details in [transformer_nes/cpu6502/README.md](transformer_nes/cpu6502/README.md).

## Preconditions

**A lockstep verification chain is the license for everything below.**
Golden logs ⇔ reference oracle ⇔ exact graph ⇔ generated kernel ⇔
weights, compared field-for-field at every step.  Every lever landed
with a byte-identity receipt; several were *rejected* by one.  Without
the chain, aggressive semantic optimizations (exact structures, routing,
leases) are unverifiable.

**Measure before choosing.**  Three plans this project skipped because a
measurement overturned them: per-expert artifact caching (the 25 s
premise was stale), the width-objective MILP (the engine is nnz-bound,
not width-bound), the poll-loop sequence expert as the 60 fps lever
(the frame was pause-bound, not forward-bound).

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

10. **Multi-instruction sequence experts** (~2x frame time here; the
   trio runs at 0.07 µs/instr, 5.7x its per-instruction cost).  Fold a
   K-step composition of the union graph under router pins (pc + every
   window byte); state chains through a feedback map, and everything
   the router verifies becomes a pin (lever 6 again).  License checked
   per window at generation: write-free, non-halting, straight-line,
   only the first instruction reads memory.  Find windows by ISA
   pattern (closed poll loops, self-jump spins), never by workload —
   and re-verify the window bytes at dispatch so self-modifying code
   only loses the fast path.  Budget guards keep dispatch step-for-step
   equivalent to the per-instruction path.

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
