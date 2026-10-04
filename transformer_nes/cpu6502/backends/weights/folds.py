"""6502 folding: opcode pins, chained execution graphs and cached experts."""

from __future__ import annotations

import os
import pickle
import zlib
from dataclasses import dataclass

from transformer_nes.files import write_atomic
from transformer_nes.graph import core as graph
from transformer_nes.graph.core import (
    Dimension,
    Expression,
    InputDimension,
    PersistDimension,
    construction,
    stepglu,
)
from transformer_nes.graph.evaluate import Evaluation
from transformer_nes.graph.fold import (
    FoldResult,
    GraphFolder,
    GraphMaterializer,
    const_of,
    scaled_sum,
    walk_live,
)
from transformer_nes.model.build import CACHE_DIR, artifact_lock, manifest_put
from transformer_nes.model.sources import source_digest

EXECUTION_INTERIOR_ROOTS = frozenset({
    "ea", "next_pc", "new_halted", "cycles",
    "is_store", "write_addr_out", "write_value_out",
    "is_store2", "write_addr2_out", "write_value2_out",
    "is_store3", "write_addr3_out", "write_value3_out",
})


PC_NAMES = ("current_pc_lo", "current_pc_hi")


@dataclass
class SequenceStep:
    """One step of a sequence fold, evaluated: its pins, the inputs frozen
    (bound to the previous step's outputs) and left unbound, its
    Evaluation, the insert-side hazards its pins reach, and the
    non-constant outputs the next step binds."""

    frozen: set
    unbound: set
    evaluation: Evaluation
    hazards: set
    fb_source: dict


def opcode_pins(byte: int, opcodes, *, pin_fetched_byte: bool = True) -> dict[str, float]:
    """Pin every decode gate for a single-opcode expert: ``is_op_<byte>=1``,
    every other ``is_op_*=0``, ``is_nmi=0`` (an NMI overrides the fetched
    instruction, so a routed step must not claim it).

    With ``pin_fetched_byte`` (the default), the ``fetched_opcode`` persist
    itself is pinned to the byte.  The router dispatches on
    ``ram[pc] == byte``, which is strictly stronger than ``is_op = 1``, so
    the memory read at PC provably returns the byte -- pinning it folds
    that read (one of the three grouped fetch reads) and any remaining
    decode residue out of the expert graph."""
    pins: dict[str, float] = {"is_nmi": 0.0, "is_irq": 0.0}
    for op in opcodes.values():
        pins[f"is_op_{op.byte:02X}"] = 1.0 if op.byte == byte else 0.0
    if pin_fetched_byte:
        pins["fetched_opcode"] = float(byte)
    return pins


# State-input persist -> previous step's output, for fold_execution_sequence over the
# 6502 graph.  pc and the trace-context bytes are pinned per step instead
# (sequence_pins knows them); ea feeds back as a byte split, so it must fold
# constant -- true for the write-free poll/spin windows this is built for.
SEQUENCE_FEEDBACK = {
    "current_a": "new_a", "current_x": "new_x", "current_y": "new_y",
    "current_s": "new_s", "current_halted": "new_halted",
    "current_p_c": "new_p_c_final", "current_p_z": "new_p_z_final",
    "current_p_i": "new_p_i_final", "current_p_d": "new_p_d_final",
    "current_p_b": "new_p_b_final", "current_p_u": 1.0,  # U reads 1 always
    "current_p_v": "new_p_v_final", "current_p_n": "new_p_n_final",
    "current_ir": "fetched_opcode",
    "current_oplo": "fetched_operand_lo",
    "current_ophi": "fetched_operand_hi",
    "current_ea_lo": ("ea", "lo"), "current_ea_hi": ("ea", "hi"),
    "current_data": "data_out",
}


def sequence_shape_pins(code: bytes, opcodes) -> tuple[list[dict[str, float]], list[int]]:
    """Per-step pins for a canonical window SHAPE: opcode one-hots and the
    fetched opcode byte only -- pc and operand bytes stay symbolic, so one
    fold serves every (head, operands) instantiation.  The router verifies
    the opcode byte at each returned offset; operand bytes need no check
    (the expert reads them through the live memory structure).  Returns
    (pin_list, pc_offsets) for fold_execution_sequence(..., pc_offsets=...)."""
    pin_list: list[dict[str, float]] = []
    offsets: list[int] = []
    off = 0
    while off < len(code):
        op = opcodes.get(code[off])
        if op is None:
            raise ValueError(f"byte {code[off]:02X} at +{off} is not an opcode")
        pins = opcode_pins(op.byte, opcodes)
        pins["current_halted"] = 0.0
        pin_list.append(pins)
        offsets.append(off)
        off += op.length
    if off != len(code):
        raise ValueError("window does not end on an instruction boundary")
    return pin_list, offsets


def sequence_pins(head_pc: int, code: bytes, opcodes) -> list[dict[str, float]]:
    """Per-step pins for a K-instruction window: the bytes at head_pc, in
    execution order (an absolute JMP or JSR before the last step continues
    the window at its target). Interior conditional branches are rejected.

    Each step gets its opcode pins plus its pc and every in-window operand
    byte pinned (the router verifies pc == head_pc and the whole window's
    bytes, so all of it is router-guaranteed).  current_halted is pinned 0:
    the run loop never dispatches a halted machine, and the runner checks
    each step's folded new_halted is constant 0.
    """
    pin_list = []
    pc = head_pc
    off = 0
    while off < len(code):
        op = opcodes.get(code[off])
        if op is None:
            raise ValueError(f"byte {code[off]:02X} at +{off} is not an opcode")
        pins = opcode_pins(op.byte, opcodes)
        pins["current_pc_lo"] = float(pc & 0xFF)
        pins["current_pc_hi"] = float((pc >> 8) & 0xFF)
        pins["current_halted"] = 0.0
        if off + 1 < len(code):
            pins["fetched_operand_lo_all"] = float(code[off + 1])
        if off + 2 < len(code):
            pins["fetched_operand_hi_all"] = float(code[off + 2])
        pin_list.append(pins)
        interior = off + op.length < len(code)
        if interior and op.operation in ("jmp", "jsr") and op.mode == "abs":
            pc = code[off + 1] | (code[off + 2] << 8)
        elif interior and op.mode == "rel":
            raise ValueError(f"interior branch at +{off} is not licensed")
        else:
            pc += op.length
        pc &= 0xFFFF
        off += op.length
    if off != len(code):
        raise ValueError("window does not end on an instruction boundary")
    return pin_list


class SequenceFolder(GraphFolder):
    """Compose 6502 steps with register feedback and symbolic PC offsets."""

    def fold_execution_sequence(self, pin_list, feedback, label: str = "",
                                pc_offsets=None, *, inline_linear: bool = False,
                                bit_feedback_until: int | None = None) -> FoldResult:
        """Fold a K-step composition of the union graph into ONE graph: it
        evaluates every step, then materializes them all.

        pin_list[k] pins step k (the router supplies them: pc, instruction
        bytes, is_nmi=0).  feedback maps a step's state-input persist name to
        the previous step's output name -- either "out_name" (bound as an
        expression, or as a pin when the previous fold made it constant) or
        ("out_name", "lo"|"hi") (byte split; a non-constant source leaves
        the input UNBOUND -- fine when the fold proves it dead, a loud
        KeyError when anything live still reads it).
        Interior steps retain the facts needed to license execution; the
        final step retains architectural outputs. Roots carry a ``__s<k>``
        suffix and stay live through READOUT. Step 0's inputs keep their
        canonical names (the composed graph embeds one state token).

        pc_offsets, if given, folds a canonical *shape*: step k's pc is not
        pinned but derived as head + pc_offsets[k], where head is step 0's
        symbolic (input-bound) pc.  The byte split is one stepglu carry per
        step -- exact for integer bytes -- so a single fold serves every
        head the router matches.  License (checked by the caller): the
        window must not wrap $FFFF, and intermediate control flow must
        actually be pc + offset (verify folded next_pc__sk).

        """
        n_steps = len(pin_list)
        feedback = dict(feedback)
        # Bit-state feedback (wide_bits only): A's bits chain like A itself.
        # ``bit_feedback_until`` prunes the chain past the last step whose
        # opcode consumes A bits, so unseeded steps drop the arith ladder
        # rather than keep it live; wrong pruning fails loudly in tr().
        bit_ins, bit_outs = set(), set()
        for b in range(8):
            if f"new_a_bit{b}" in self.by_name:
                feedback.setdefault(f"current_a_bit{b}", f"new_a_bit{b}")
                bit_ins.add(f"current_a_bit{b}")
                bit_outs.add(f"new_a_bit{b}")
        for out in feedback.values():
            if isinstance(out, (int, float)):
                continue  # constant feedback (e.g. the always-1 U flag)
            out_name = out[0] if isinstance(out, tuple) else out
            if out_name not in self.by_name:
                raise KeyError(f"feedback output {out_name!r} not in the graph")

        steps = []
        fb_const: dict[str, float] = {}
        fb_live: set[str] = set()  # non-constant feedback the next step binds
        for k in range(n_steps):
            pins = dict(pin_list[k])
            frozen: set[str] = set()
            unbound: set[str] = set()
            if pc_offsets is not None and k > 0:
                frozen |= set(PC_NAMES)
            if k > 0:
                for in_name, out in feedback.items():
                    if in_name in pins:  # explicit pins beat feedback
                        continue
                    if isinstance(out, (int, float)):
                        pins[in_name] = float(out)
                        continue
                    if isinstance(out, tuple):
                        out_name, part = out
                        c = fb_const.get(out_name)
                        if c is None:
                            # Non-constant byte-split source: leave the
                            # input unbound.  Dead under the fold -> fine;
                            # a live reference fails loudly in tr().
                            frozen.add(in_name)
                            unbound.add(in_name)
                            continue
                        v = int(round(c))
                        pins[in_name] = float(v & 0xFF if part == "lo" else (v >> 8) & 0xFF)
                    elif (c := fb_const.get(out)) is not None:
                        pins[in_name] = c
                    else:
                        frozen.add(in_name)
                        if in_name in bit_ins and out not in fb_live:
                            # Pruned bit feedback: leave unbound; a live
                            # reader fails loudly in tr().
                            unbound.add(in_name)

            evaluation = self.evaluate(
                pins, frozenset(frozen), inline_linear=inline_linear,
                preserve=frozenset(self.by_name[n] for n in PC_NAMES)
                if pc_offsets is not None and k == 0 else frozenset(),
            )
            val = evaluation.val
            hazards = {lu.name or f"lookup_{lu.id}" for lu in self.insert_hazards(frozenset(pins))}

            fb_const, fb_live, fb_source = {}, set(), {}
            if k + 1 < n_steps:
                # The last step feeds nothing; non-root feedback outputs
                # (bit state) are not kept live there, so tr would fail.
                seed_bits = bit_feedback_until is None or k < bit_feedback_until
                for out in set(feedback.values()):
                    if isinstance(out, (int, float)):
                        continue
                    out_name = out[0] if isinstance(out, tuple) else out
                    if out_name in bit_outs and not seed_bits:
                        continue
                    d = self.by_name[out_name]
                    src = val.get(d, Expression({d: 1}))
                    c = const_of(src, self.pg.one)
                    if c is not None:
                        fb_const[out_name] = c
                    else:
                        fb_live.add(out_name)
                        fb_source[out_name] = src
            steps.append(SequenceStep(frozen, unbound, evaluation, hazards, fb_source))
        return self.materialize_sequence(
            steps, feedback, label=label, pc_offsets=pc_offsets)

    @construction
    def materialize_sequence(self, steps, feedback, *, label, pc_offsets) -> FoldResult:
        """The evaluated steps materialized as one graph, step by step."""
        pg = self.pg
        n_steps = len(steps)
        root_dims = [self.by_name[n] for n in self.root_names]
        base_sub: dict = {pg.one: Expression({graph.current_graph().one: 1})}
        for d in self.inputs:
            base_sub.setdefault(d, Expression({InputDimension(d.name): 1}))

        mat = GraphMaterializer(  # provenance accumulates per step
            structural_reads=self.structural_reads,
        )
        keep_alive = Expression()
        fb_expr: dict[str, Expression] = {}  # prev step's outputs, new-graph space
        new_inputs = None
        pc_exprs = None  # step-0 (head) pc byte expressions, new-graph space

        for k, step in enumerate(steps):
            val, folded, lookup_exprs = step.evaluation
            step_root_dims = (
                root_dims
                if k + 1 == n_steps
                else [
                    self.by_name[name] for name in self.root_names
                    if name in EXECUTION_INTERIOR_ROOTS
                ]
            )
            fb_dims = [self.by_name[name] for name in step.fb_source]
            # Liveness seeds: execution roots and the next step's feedback.
            seeds: set = set()
            for d in step_root_dims + fb_dims:
                seeds |= set(val[d].terms) if d in val else {d}

            def is_leaf(d, *, frozen=step.frozen):
                return (
                    isinstance(d, InputDimension)
                    or (isinstance(d, PersistDimension) and d.name in frozen)
                )

            def bit_leaf(d):
                rb = getattr(d, "read_bit", None)
                return rb[0] if rb is not None and self.structural_reads else None

            live = walk_live(seeds, val, folded, lookup_exprs, is_leaf,
                             bit_leaf=bit_leaf,
                             structural_reads=self.structural_reads)

            # Materialize this step (fresh sub: the same union dim holds a
            # different value per step); frozen persists resolve to the
            # previous step's already-materialized output expressions.
            sub = dict(base_sub)
            for in_name in step.frozen:
                if in_name in step.unbound or in_name in PC_NAMES:
                    continue
                sub[self.by_name[in_name]] = fb_expr[feedback[in_name]]
            if pc_offsets is not None and k > 0:
                # Shape fold: pc(k) = head + offset, byte-split with one
                # stepglu carry (exact on integer bytes).  No $FFFF wrap by
                # the caller's license.
                off = int(pc_offsets[k])
                assert pc_exprs is not None  # captured at k == 0
                lo0, hi0 = pc_exprs
                if off == 0:
                    lo_e, hi_e = lo0, hi0
                else:
                    carry = stepglu(1, lo0 + Expression({graph.current_graph().one: off - 256}))
                    lo_e = lo0 + Expression({graph.current_graph().one: float(off)}) + carry * -256.0
                    hi_e = hi0 + carry
                sub[self.by_name["current_pc_lo"]] = lo_e
                sub[self.by_name["current_pc_hi"]] = hi_e

            def tr(expr: Expression, *, sub=sub, val=val, k=k) -> Expression:
                def term(d):
                    s = sub.get(d)
                    if s is None:
                        if d not in val:
                            raise KeyError(
                                f"fold_execution_sequence({label}) s{k}: {d!r} neither "
                                "live nor folded"
                            )
                        s = sub[d] = tr(val[d])
                    return s
                return scaled_sum((term(d), c) for d, c in expr.terms.items())

            def sname(name, *, k=k):
                return f"{name}__s{k}" if name else None

            mat.materialize(self.in_order(live), live, val, folded, lookup_exprs,
                            sub, tr, name_of=sname, is_leaf=is_leaf)

            # Roots: every step's readout set, suffixed; folded-away roots
            # come back as named persists (same contract as fold()).
            for d in step_root_dims:
                if d in val:
                    nd = PersistDimension(tr(val[d]), name=sname(d.name))
                    sub[d] = Expression({nd: 1})
                keep_alive = keep_alive + 1e-6 * sub[d]

            if pc_offsets is not None and k == 0 and n_steps > 1:
                pc_exprs = tuple(
                    tr(Expression({self.by_name[n]: 1})) for n in PC_NAMES
                )
            fb_expr = {name: tr(src) for name, src in step.fb_source.items()}

            if k + 1 == n_steps:
                new_inputs = {
                    name: tr(expr) if isinstance(expr, Expression) else expr
                    for name, expr in pg.input_tokens.items()
                }
        assert new_inputs is not None
        new_outputs = {"READOUT": keep_alive}
        new_pg = graph.current_graph().capture(new_inputs, new_outputs)

        dim_of_name: dict[str, Dimension] = {}
        for d in new_pg.all_dims:
            dim_of_name.setdefault(d.name, d)

        return FoldResult(
            label=label,
            program_graph=new_pg,
            dim_of_name=dim_of_name,
            union_lookup_of=mat.union_lookup_of,
            insert_side_hazards=tuple(sorted(set().union(*(step.hazards for step in steps)))),
            n_steps=n_steps,
        )


def cached_fold(name: str, build):
    code = source_digest(
        "transformer_nes.cpu6502.backends.weights.experts", "transformer_nes.cpu6502.construction"
    )
    digest = code[:16]
    path = os.path.join(CACHE_DIR, f"build_{digest}_{name}.pklz")
    with artifact_lock(digest):
        if not os.path.exists(path):
            write_atomic(path, zlib.compress(pickle.dumps(build())))
        with open(path, "rb") as source:
            result = pickle.loads(zlib.decompress(source.read()))
        manifest_put(f"fold:{name}", {"kind": "fold", "digest": digest}, code=code)
        return result
