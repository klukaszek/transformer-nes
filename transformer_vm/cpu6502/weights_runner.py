"""Run the 6502 through the real weights-based transformer.

This is the end of the verification chain: the same computation graph that
the exact evaluator and the C++ kernel execute is scheduled onto transformer
layers and compiled into actual transformer weights (analytically -- no
training).  Execution is then genuine transformer inference: per-position
embeddings, Q/K/V projections, hardmax (hull) attention, ReGLU FFN,
residual stream, and
an output head that decides STATE vs HALT.

One departure from the WASM construction: WASM tokens formed a finite
vocabulary (each token type's values were baked into its embedding row).
A 6502 STATE token carries ~20 wide-domain fields, so token *values*
cannot be enumerated; instead the harness builds each position's input
embedding directly -- the value of every input dim placed at its residual
slot, exactly what an enumerated vocabulary row would have contained.
The transformer layers themselves are untouched.

The run loop is the same protocol as the exact evaluator: the machine's
result dims are read back from the residual stream after the final layer
(kept alive to the end via a synthetic READOUT output token), writes are
appended as WRITE-token positions, and the next STATE embedding is built
from the read-back state.

Numerics caveat: the engine path is float32 (codegen) and uses grouped
4 KiB memory lookups so high-address reads survive Q/K rounding; hull
hardmax tie-breaks resolve same-address rewrites by latest write.
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from transformer_vm.cpu6502 import reference
from transformer_vm.cpu6502.evaluator import _RUNTIME_READ_DIMS, Cpu6502Runtime
from transformer_vm.cpu6502.spec import OPCODES
from transformer_vm.cpu6502.trace import Cpu6502State, MemWrite, Trace, TraceRow
from transformer_vm.graph.core import (
    Expression,
    InputDimension,
    LookUpDimension,
    PersistDimension,
    ProgramGraph,
    ReGLUDimension,
)
from transformer_vm.model.weights import add_position_encoding, build_model
from transformer_vm.scheduler.milp import asap_schedule

logger = logging.getLogger(__name__)

# Opcode-routed experts, on by default for the generated int engine.  These
# are the hot opcodes of the SMB gameplay histogram (experiments/moe_probe):
# the frame-wait poll loop (LDA abs / AND imm / BEQ) plus the idle JMP spin
# dominate (~87%), and the top ~16 cover ~94%.  Every other opcode (and every
# NMI) falls back to the exact union step, so correctness is workload-agnostic;
# the set only decides which opcodes get the faster pruned step.
DEFAULT_EXPERT_OPCODES = (
    0xF0, 0xAD, 0x29, 0x4C, 0xC8, 0xD0, 0x85, 0xC9,
    0x4A, 0x99, 0xA5, 0xBD, 0x20, 0x60, 0x90, 0x18,
)


class Cpu6502TransformerMachine:
    """The 6502 as actual transformer weights, stepped position by position."""

    def __init__(self, *, bcd: bool = False,
                 codegen_arith: str = "int32",
                 checkpoint_tokens: int | None = None,
                 expert_opcodes=DEFAULT_EXPERT_OPCODES):
        # Build the graph (and keep the evaluator's dim handles; its Python
        # step function is not used here).
        self.rt = Cpu6502Runtime(bcd=bcd)
        dim_by_id = {d.id: d for d in self.rt.all_dims}
        self._dim_by_name = {
            name: dim_by_id[dim_id] for name, dim_id in self.rt.dim_id_by_name.items()
        }

        # The run loop reads the machine outputs from the residual stream,
        # so they must survive to the final layer: a synthetic output token
        # marks them as outputs for the scheduler.  Its head row is ignored
        # at prediction time (only STATE vs HALT compete).
        output_tokens = dict(self.rt.output_tokens)
        readout = 0
        for name in _RUNTIME_READ_DIMS:
            readout = readout + 1e-6 * self._dim_by_name[name]
        output_tokens["READOUT"] = readout

        pg = ProgramGraph(self.rt.input_tokens, output_tokens)
        build_start = time.perf_counter()
        # ASAP schedule: the optimal-width MILP does not solve at this graph
        # size (~5M integer variables); critical-path scheduling costs some
        # d_model but no solver time.
        sched = asap_schedule(pg.input_tokens, pg.output_tokens, program_graph=pg)
        self.model, all_tokens, tok_to_idx, shared = build_model(
            program_graph=pg, schedule=sched,
        )
        self.build_seconds = time.perf_counter() - build_start
        self.n_layers = shared[3]
        self._std_layers = shared[4]
        self.slot_of = shared[6]
        self.d_model = shared[9]
        self._state_row = tok_to_idx["STATE"]
        self._halt_row = tok_to_idx["HALT"]
        self._read_slots = {
            name: self.slot_of[self._dim_by_name[name]]
            for name in _RUNTIME_READ_DIMS
        }
        self._one_slot = self.slot_of[self.rt._one]

        # The whole layer stack -- hull attention plus the structural
        # passthrough fusion -- runs inside the generated C++ engine
        # (model/engine_gen.py): model-specific straight-line code with an
        # exact fixed-point int32 residual.  The grouped memory lookup keeps
        # K-side address scores below 2^24, so float32 Q/K rounding still
        # distinguishes adjacent high addresses.
        self.matmul = "codegen"
        self.dtype = "float32"
        self.np_dtype = np.float32
        self._state_head_cols, self._state_head_vals = self._head_row(self._state_row)
        self._halt_head_cols, self._halt_head_vals = self._head_row(self._halt_row)

        from transformer_vm.model.engine_gen import GeneratedWeightsEngine

        experts = self._build_experts(expert_opcodes)
        self._engine = GeneratedWeightsEngine(
            self.model, step_spec=self._step_spec(),
            arith=codegen_arith, experts=experts,
        )
        self._direct_writes = self._engine.supports_direct_writes
        self._cpp_step = getattr(self._engine, "has_machine_step", False)
        if self._cpp_step:
            self._step_out = np.empty(21, dtype=np.int64)

        self.pos = 0
        self._trace_len = 0
        # RAM mirror for snapshot-and-rebuild checkpointing (KernelBus
        # keeps the same mirror for the C++ kernel).  checkpoint_tokens, if
        # set, rebuilds automatically once the token log grows past it, so
        # long runs do not grow unbounded hull logs.
        self._ram = bytearray(0x10000)
        self._checkpoint_tokens = checkpoint_tokens
        self._seed_pos = 0
        self.checkpoints = 0

    # ── Transformer plumbing ────────────────────────────────────────

    def _embed_array(self, values: dict) -> np.ndarray:
        """Engine input embedding: dim values at their residual slots.

        Equivalent to a vocabulary embedding row for a token type that
        enumerated exactly these values; the three position features (slots
        0-2) are added by _forward_values.
        """
        x = np.zeros(self.d_model, dtype=self.np_dtype)
        x[self._one_slot] = 1.0
        for dim, value in values.items():
            x[self.slot_of[dim]] = float(value)
        return x

    def _forward_values(self, values: dict) -> np.ndarray:
        """One genuine transformer position: embed, add position features,
        and run the whole layer stack (hull attention + ReGLU) in C++."""
        x = self._embed_array(values)
        add_position_encoding(x, self.pos)
        self.pos += 1
        self._engine.forward(x)  # in place, whole stack
        return x

    def _build_experts(self, opcodes) -> list[dict]:
        """Per-opcode keep-sets for the opcode-routed engine.

        For each expert opcode, partial-evaluate the graph with the decode
        one-hot gates pinned (is_op_<byte>=1, every other is_op_*=0, is_nmi=0)
        -- exactly what a hard router supplies.  A ReGLU whose gate resolves to
        <=0 (or whose value resolves to 0) is the zero function under this
        opcode: a non-taken instruction / addressing-mode branch.

        The scheduled model is the union of every opcode, so pruning is done as
        liveness over its *unfolded* dependency graph with the zero-folded dims
        removed as edges: a residual dim is live iff it still reaches a readout
        root through non-zero contributions.  Dead dims (e.g. the memory reads
        of addressing modes this opcode never uses, and the ALU neurons of
        other operations) drop out; the taken branch's gate=1 neurons stay live
        (they fold to a real linear map and must still run).  A live ff_out /
        out_proj row keeps only its live columns -- dropping a zero/dead column
        is exact term elimination -- so passthrough/erase slot bookkeeping,
        memory, and the registers are all untouched.

        Correctness rests on ram[pc] == the graph's fetched opcode (the run
        loop dispatches on the same byte the memory read returns) and on the
        pruning-safety invariants in engine_gen.generate_source.  The full
        lockstep chain (every opcode swept through its expert by nestest and
        Klaus) is the receipt.
        """
        opcodes = [int(b) for b in (opcodes or []) if int(b) in OPCODES]
        if not opcodes:
            return []

        all_dims = list(self.rt.all_dims)
        one = self.rt._one

        # Readout roots plus everything the STATE/HALT output heads read: the
        # step reads those logits, so their inputs must survive pruning too.
        roots = [self._dim_by_name[n] for n in _RUNTIME_READ_DIMS]
        for tok in ("STATE", "HALT"):
            expr = self.rt.output_tokens.get(tok)
            if isinstance(expr, Expression):
                roots.extend(d for d in expr.terms if not isinstance(d, InputDimension))

        def const_of(e):
            for d in e.terms:
                if d is not one:
                    return None
            return e.terms.get(one, 0.0)

        def subst(expr, val):
            out = Expression()
            for d, c in expr.terms.items():
                v = val.get(d)
                out = out + (Expression({d: c}) if v is None else v * c)
            return out

        def keep_sets(pins):
            # 1) Fold: find the dims that are provably zero under these pins.
            val = {one: Expression({one: 1.0})}
            zero = set()
            for d in all_dims:
                if isinstance(d, PersistDimension) and d.name in pins:
                    val[d] = Expression({one: float(pins[d.name])})
                    continue
                if isinstance(d, ReGLUDimension):
                    a = subst(d.a_expr, val)
                    b = subst(d.b_expr, val)
                    bc, ac = const_of(b), const_of(a)
                    if (bc is not None and bc <= 0) or (ac is not None and ac == 0):
                        val[d] = Expression()
                        zero.add(d)
                    elif bc is not None and ac is not None:
                        val[d] = Expression({one: ac * bc})
                        if ac * bc == 0:
                            zero.add(d)
                    elif bc is not None:
                        val[d] = a * bc
                elif isinstance(d, PersistDimension):
                    e = subst(d.expr, val)
                    c = const_of(e)
                    if c is not None:
                        val[d] = Expression({one: c})
                        if c == 0:
                            zero.add(d)
            # 2) Liveness over the UNFOLDED graph, stopping at zero dims: a dim
            #    is live iff it reaches a root through non-zero contributions.
            live = set()
            stack = [r for r in roots]
            while stack:
                d = stack.pop()
                if (d in live or d in zero or isinstance(d, InputDimension)):
                    continue
                live.add(d)
                deps = set()
                if isinstance(d, ReGLUDimension):
                    deps |= set(d.a_expr.terms) | set(d.b_expr.terms)
                elif isinstance(d, PersistDimension):
                    deps |= set(d.expr.terms)
                elif isinstance(d, LookUpDimension):
                    lu = d.lookup
                    for e in (lu.query_exprs_2d + lu.key_exprs_2d + lu.value_exprs):
                        deps |= set(e.terms)
                    live.update(lu.dims)  # a live lookup produces all its values
                stack.extend(deps)
            return live

        experts = []
        for byte in opcodes:
            pins = {"is_nmi": 0.0}
            for op in OPCODES.values():
                pins[f"is_op_{op.byte:02X}"] = 1.0 if op.byte == byte else 0.0
            live = keep_sets(pins)

            keep_heads: dict[int, set] = {}
            keep_neurons: dict[int, set] = {}
            for li, (attn, _p1, ffn, _p2) in enumerate(self._std_layers):
                # Head numbering mirrors build_model: lookup heads first, in
                # LookUp order, two value components per head.
                kh = set()
                h = 0
                for lu in attn:
                    nv = len(lu.value_exprs)
                    for p in range((nv + 1) // 2):
                        served = [lu.dims[2 * p]]
                        if 2 * p + 1 < nv:
                            served.append(lu.dims[2 * p + 1])
                        if any(d in live for d in served):
                            kh.add(h)
                        h += 1
                keep_heads[li] = kh
                keep_neurons[li] = {j for j, rg in enumerate(ffn) if rg in live}

            experts.append({
                "name": f"{byte:02X}", "byte": byte,
                "keep_heads": keep_heads, "keep_neurons": keep_neurons,
                "_nk": sum(len(v) for v in keep_neurons.values()),
                "_hk": sum(len(v) for v in keep_heads.values()),
            })
        n_neurons = sum(len(ffn) for _a, _p, ffn, _q in self._std_layers)
        n_heads = sum(len(a) for a, _p, _f, _q in self._std_layers)
        logger.info(
            "routed experts: %d opcodes; mean kept %d/%d ReGLU neurons, "
            "%d/%d lookup heads (e.g. %s)",
            len(experts),
            sum(e["_nk"] for e in experts) // max(len(experts), 1), n_neurons,
            sum(e["_hk"] for e in experts) // max(len(experts), 1), n_heads,
            experts[0]["name"],
        )
        return experts

    def _step_spec(self) -> dict:
        """Slot map for the generated C++ machine_step (engine_gen.py)."""
        rt = self.rt
        s = lambda dim: self.slot_of[dim]  # noqa: E731
        return {
            "inputs": {
                "is_state": s(rt.is_state),
                "pc_lo": s(rt.in_pc_lo), "pc_hi": s(rt.in_pc_hi),
                "a": s(rt.in_a), "x": s(rt.in_x), "y": s(rt.in_y), "s": s(rt.in_s),
                "ir": s(rt.in_ir), "oplo": s(rt.in_oplo), "ophi": s(rt.in_ophi),
                "ea_lo": s(rt.in_ea_lo), "ea_hi": s(rt.in_ea_hi),
                "data": s(rt.in_data), "halted": s(rt.in_halted),
                "nmi": s(rt.in_nmi),
            },
            "p_flags": [
                (s(rt.in_p_c), reference.FLAG_C), (s(rt.in_p_z), reference.FLAG_Z),
                (s(rt.in_p_i), reference.FLAG_I), (s(rt.in_p_d), reference.FLAG_D),
                (s(rt.in_p_b), reference.FLAG_B), (s(rt.in_p_u), reference.FLAG_U),
                (s(rt.in_p_v), reference.FLAG_V), (s(rt.in_p_n), reference.FLAG_N),
            ],
            "outputs": dict(self._read_slots),
            "halt_head": (self._halt_head_cols.tolist(),
                          self._halt_head_vals.tolist()),
            "state_head": (self._state_head_cols.tolist(),
                           self._state_head_vals.tolist()),
        }

    def _head_row(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        w = self.model.head.weight[row]
        cols = np.nonzero(w)[0].astype(np.int32)
        return cols, w[cols].astype(self.np_dtype, copy=False)

    def _head_logit(self, h: np.ndarray, cols: np.ndarray, vals: np.ndarray) -> float:
        return float(np.dot(h[cols], vals))

    def _read(self, h: np.ndarray, name: str) -> float:
        slot = self._read_slots.get(name)
        if slot is None:
            slot = self.slot_of[self._dim_by_name[name]]
            self._read_slots[name] = slot
        return float(h[slot])

    # ── Machine protocol (mirrors evaluator.Cpu6502Runtime, so the NES
    # bus's GraphCore adapter can drive this machine unchanged) ──────

    def boot_image(self, memory_image, *, start_pc: int, state: dict | None = None):
        # Fresh token log: clear the engine's attention state so a machine
        # instance can be reused for another program.
        self._engine.reset()
        self.pos = 0
        self._trace_len = 0
        ram = bytearray(0x10000)
        ram[: len(memory_image)] = bytes(memory_image)
        self._ram = ram
        rt = self.rt
        self._forward_values({
            rt.start_pc_lo: start_pc & 0xFF,
            rt.start_pc_hi: (start_pc >> 8) & 0xFF,
            rt.is_start: 1,
        })
        for addr, byte in enumerate(ram):
            if byte:
                self._insert_write(addr, byte)
        self._seed_pos = self.pos
        self._mstate = {
            "pc": start_pc & 0xFFFF, "a": 0, "x": 0, "y": 0, "s": 0xFD,
            "p": reference.FLAG_U | reference.FLAG_I,
            "ir": 0, "oplo": 0, "ophi": 0, "ea": 0, "data": 0, "halted": 0,
        }
        if state is not None:
            self._mstate.update({k: int(state[k]) for k in self._mstate if k in state})
        self._nmi_pending = False
        self.machine_halted = False

    @property
    def machine_state(self) -> dict:
        return dict(self._mstate)

    def set_nmi(self):
        """The interrupt sequence itself then runs inside the weights."""
        self._nmi_pending = True
        # In the native run loop the pending-NMI flag lives in the engine
        # (the per-step path passes it as an argument instead).
        if self._cpp_step and getattr(self._engine, "has_run_loop", False):
            self._engine.set_nmi()

    def inject_write(self, addr: int, value: int):
        self._insert_write(addr & 0xFFFF, value & 0xFF)

    def _insert_write(self, addr: int, value: int):
        addr &= 0xFFFF
        value &= 0xFF
        self._ram[addr] = value
        if self._direct_writes:
            self._engine.insert_write(addr, value, self.pos)
            self.pos += 1
            return
        rt = self.rt
        self._forward_values({
            rt.byte_value: value & 0xFF,
            rt.is_memory_write: 1,
            rt.addr_lo: addr & 0xFF,
            rt.addr_hi: (addr >> 8) & 0xFF,
        })

    def _machine_step_cpp(self) -> TraceRow:
        """One instruction entirely in generated C++ (embed, layers,
        readout, direct WRITE inserts, halt logits); Python only unpacks
        the 21-slot result and keeps the mirrors in sync."""
        st = self._mstate
        nmi = 1 if self._nmi_pending else 0
        self._nmi_pending = False
        self._engine.machine_step(
            st["pc"], st["a"], st["x"], st["y"], st["s"], st["p"],
            st["ir"], st["oplo"], st["ophi"], st["ea"], st["data"],
            st["halted"], nmi, self._step_out,
        )
        o = self._step_out.tolist()  # one C call instead of 21 scalar reads
        n_writes = o[14]
        writes = []
        for i in range(n_writes):
            addr, value = o[15 + 2 * i], o[16 + 2 * i]
            self._ram[addr] = value
            writes.append(MemWrite(self._trace_len, addr, value))
        self.pos += 1 + n_writes

        halted = o[11] != 0
        row = TraceRow(
            step=self._trace_len,
            state=Cpu6502State(
                pc=o[0], a=o[1], x=o[2], y=o[3], s=o[4], p=o[5], ir=o[6],
                operand_lo=o[7], operand_hi=o[8],
                ea=o[9], data=o[10], halted=halted,
            ),
            writes=tuple(writes),
        )
        self._trace_len += 1
        st.update(
            pc=o[12], a=o[1], x=o[2], y=o[3], s=o[4], p=o[5], ir=o[6],
            oplo=o[7], ophi=o[8], ea=o[9], data=o[10],
            halted=1 if halted else 0,
        )
        if halted or o[13]:
            self.machine_halted = True
        return row

    def checkpoint(self) -> bool:
        """Snapshot-and-rebuild the token log from the RAM mirror.

        Mirrors KernelBus.checkpoint: captures the architectural state,
        reseeds the attention state (hull logs) with a fresh WRITE-token
        prefix from the mirror, and restores the state.  Semantically
        invisible: unwritten and zero bytes both read as 0 through the
        exact-address mask, so dropping the history and reseeding only the
        nonzero RAM bytes preserves every future read.
        """
        if self.machine_halted:
            return False
        state = self.machine_state
        trace_len = self._trace_len
        nmi = self._nmi_pending
        self.boot_image(self._ram, start_pc=state["pc"], state=state)
        self._trace_len = trace_len
        self._nmi_pending = nmi
        self.checkpoints += 1
        return True

    def machine_step(self) -> TraceRow:
        if (
            self._checkpoint_tokens
            and self.pos - self._seed_pos >= self._checkpoint_tokens
        ):
            self.checkpoint()
        if self._cpp_step:
            return self._machine_step_cpp()
        rt = self.rt
        st = self._mstate
        p = st["p"]
        nmi = 1 if self._nmi_pending else 0
        self._nmi_pending = False
        flag = Cpu6502Runtime._flag_bit
        h = self._forward_values({
            rt.is_state: 1,
            rt.in_pc_lo: st["pc"] & 0xFF,
            rt.in_pc_hi: (st["pc"] >> 8) & 0xFF,
            rt.in_a: st["a"], rt.in_x: st["x"], rt.in_y: st["y"], rt.in_s: st["s"],
            rt.in_p_c: flag(p, reference.FLAG_C), rt.in_p_z: flag(p, reference.FLAG_Z),
            rt.in_p_i: flag(p, reference.FLAG_I), rt.in_p_d: flag(p, reference.FLAG_D),
            rt.in_p_b: flag(p, reference.FLAG_B), rt.in_p_u: flag(p, reference.FLAG_U),
            rt.in_p_v: flag(p, reference.FLAG_V), rt.in_p_n: flag(p, reference.FLAG_N),
            rt.in_ir: st["ir"], rt.in_oplo: st["oplo"], rt.in_ophi: st["ophi"],
            rt.in_ea_lo: st["ea"] & 0xFF, rt.in_ea_hi: (st["ea"] >> 8) & 0xFF,
            rt.in_data: st["data"], rt.in_halted: st["halted"],
            rt.in_nmi: nmi,
        })

        def read(name, mask):
            return int(round(self._read(h, name))) & mask

        writes = []
        for store, addr, value in (
            ("is_store", "write_addr_out", "write_value_out"),
            ("is_store2", "write_addr2_out", "write_value2_out"),
            ("is_store3", "write_addr3_out", "write_value3_out"),
        ):
            if read(store, 0xFF):
                writes.append(MemWrite(self._trace_len, read(addr, 0xFFFF), read(value, 0xFF)))

        halted = read("new_halted", 0xFF) != 0
        row = TraceRow(
            step=self._trace_len,
            state=Cpu6502State(
                pc=read("current_pc", 0xFFFF),
                a=read("new_a", 0xFF), x=read("new_x", 0xFF), y=read("new_y", 0xFF),
                s=read("new_s", 0xFF), p=read("new_p", 0xFF),
                ir=read("fetched_opcode", 0xFF),
                operand_lo=read("fetched_operand_lo", 0xFF),
                operand_hi=read("fetched_operand_hi", 0xFF),
                ea=read("ea", 0xFFFF), data=read("data_out", 0xFF),
                halted=halted,
            ),
            writes=tuple(writes),
        )
        self._trace_len += 1

        for w in row.writes:
            self._insert_write(w.addr, w.value)

        s = row.state
        st.update(
            pc=read("next_pc", 0xFFFF), a=s.a, x=s.x, y=s.y, s=s.s, p=s.p,
            ir=s.ir, oplo=s.operand_lo, ophi=s.operand_hi,
            ea=s.ea, data=s.data, halted=1 if s.halted else 0,
        )
        halt_logit = self._head_logit(h, self._halt_head_cols, self._halt_head_vals)
        state_logit = self._head_logit(h, self._state_head_cols, self._state_head_vals)
        if halted or halt_logit > state_logit:
            self.machine_halted = True
        return row

    def run_image(self, memory_image, *, start_pc: int, max_steps: int) -> Trace:
        self.boot_image(list(memory_image), start_pc=start_pc)
        trace = Trace()
        for _ in range(max_steps):
            trace.append(self.machine_step())
            if self.machine_halted:
                break
        return trace

    # ── C++ run loop (KernelBus shape) ──────────────────────────────
    # The generated int/f32 engine can run the whole instruction stream
    # natively, pausing back to Python only on halt / jump-to-self trap /
    # watched device access / cycle budget (removes the per-step Python
    # protocol floor).  These helpers expose it for the streaming Klaus /
    # nestest harnesses; the NES console drives it through WeightsKernelBus.

    @property
    def supports_run_loop(self) -> bool:
        return self._cpp_step and getattr(self._engine, "has_run_loop", False)

    def _seed_run_state(self) -> None:
        st = self._mstate
        self._engine.set_run_state(
            st["pc"], st["a"], st["x"], st["y"], st["s"], st["p"],
            st["ir"], st["oplo"], st["ophi"], st["ea"], st["data"], st["halted"],
        )

    def _pull_run_state(self) -> None:
        """Copy the engine's architectural state back into the Python mirror
        (called after a native run so machine protocol / checkpointing see it)."""
        s = self._engine.get_run_state()
        self._mstate.update(
            pc=int(s["pc"]), a=int(s["a"]), x=int(s["x"]), y=int(s["y"]),
            s=int(s["s"]), p=int(s["p"]), ir=int(s["ir"]), oplo=int(s["oplo"]),
            ophi=int(s["ophi"]), ea=int(s["ea"]), data=int(s["data"]),
            halted=1 if s["halted"] else 0,
        )
        self.machine_halted = bool(s["halted"])
        self.pos = int(s["pos"])

    def run_image_arrays(self, memory_image, *, start_pc: int, max_steps: int,
                         stop_on_trap: bool = False) -> dict:
        """Boot and run up to max_steps entirely in the C++ run loop.

        Returns the same result dict as the C++ kernel's run(): per-step
        trace arrays (pc/a/x/y/s/p/ir/oplo/ophi/ea/data/halted), write arrays
        (w_row 0-based within this call, w_addr, w_value), and the run's
        "reason"/"cycles".  Materializes every step's arrays, so it is for
        bounded runs (nestest, a Klaus prefix); use iter_run_arrays to stream
        the full suite.
        """
        if not self.supports_run_loop:
            raise RuntimeError("this engine has no C++ run loop")
        self.boot_image(list(memory_image), start_pc=start_pc)
        self._seed_run_state()
        out = self._engine.run(max_steps, stop_on_trap, 0)
        self._pull_run_state()
        return out

    def iter_run_arrays(self, memory_image, *, start_pc: int, max_steps: int,
                        chunk: int = 200_000, stop_on_trap: bool = False):
        """Stream the C++ run loop in fixed-size chunks (w_row rebased to 0
        per chunk), so the full Klaus suite verifies without materializing a
        30M-row trace.  Yields the kernel-shape result dict for each chunk.

        Engine state is O(1) in the token log (memory on the exact
        nearest-address structure, constant-query heads on running-argmax
        registers), so streaming needs no checkpoint rebuild between chunks.
        """
        if not self.supports_run_loop:
            raise RuntimeError("this engine has no C++ run loop")
        self.boot_image(list(memory_image), start_pc=start_pc)
        self._seed_run_state()
        done = 0
        while done < max_steps:
            base = int(self._engine.get_run_state()["steps"])
            out = self._engine.run(min(chunk, max_steps - done), stop_on_trap, 0)
            n = len(out["pc"])
            if n == 0:
                break
            # w_row is the engine's global step_index; rebase to this chunk.
            out["w_row"] = out["w_row"] - base
            done += n
            yield out
            if out["reason"] in ("halt", "trap"):
                break
        self._pull_run_state()

    # ── Kernel-shape machine API (lets KernelBus drive the run loop) ─
    # These mirror cpu6502.kernel.Cpu6502CppRuntime so the NES bus's
    # KernelBus adapter drives the weights engine natively, pausing back to
    # Python only for MMIO -- the same protocol as the C++ kernel.

    def load_image(self, memory_image, *, start_pc: int) -> None:
        self.boot_image(list(memory_image), start_pc=start_pc)
        self._seed_run_state()

    def run(self, max_steps: int, *, stop_on_trap: bool = False,
            max_cycles: int = 0) -> dict:
        """Resume the native run loop from the engine's persistent state
        (no reboot), pausing on halt/trap/watch/cycle budget."""
        out = self._engine.run(max_steps, stop_on_trap, max_cycles)
        self._pull_run_state()
        return out

    def watch_read(self, lo: int, hi: int) -> None:
        self._engine.watch_read(lo, hi)

    def watch_write(self, lo: int, hi: int) -> None:
        self._engine.watch_write(lo, hi)

    def read_ram(self, addr: int) -> int:
        return self._engine.read_ram(addr)

    def dump_ram(self) -> bytes:
        return self._engine.dump_ram()

    def get_state(self) -> dict:
        return dict(self._engine.get_run_state())

    def set_state(self, state: dict) -> None:
        """Restore architectural state after a snapshot rebuild (mirrors the
        C++ kernel's set_state; the trace context fields carry over)."""
        st = dict(self._mstate)
        for k in st:
            if k in state:
                st[k] = int(state[k])
        self._engine.set_run_state(
            st["pc"], st["a"], st["x"], st["y"], st["s"], st["p"],
            st["ir"], st["oplo"], st["ophi"], st["ea"], st["data"],
            1 if state.get("halted") else 0,
        )
        self._mstate.update(st)
        self.machine_halted = bool(state.get("halted", self.machine_halted))


# ── Verification program (low addresses; see numerics caveat above) ──

VERIFY_START = 0x0400


def verify_program() -> bytes:
    """Loop + rewrites + stack + subroutine + arithmetic, all below $0500."""
    image = bytearray(0x10000)
    code = bytes([
        0xA9, 0x05,        # LDA #$05
        0x85, 0x10,        # STA $10
        0xA2, 0x03,        # LDX #$03
        0xE6, 0x10,        # loop: INC $10   (rewrites the same address)
        0xCA,              # DEX
        0xD0, 0xFB,        # BNE loop
        0x20, 0x20, 0x04,  # JSR $0420
        0xA5, 0x10,        # LDA $10
        0x38,              # SEC
        0xE9, 0x02,        # SBC #$02
        0x85, 0x11,        # STA $11
        0x69, 0x10,        # ADC #$10
        0x48,              # PHA
        0x68,              # PLA
        0x85, 0x12,        # STA $12
        0x4C, 0x1B, 0x04,  # trap: JMP trap ($041B)
    ])
    image[VERIFY_START:VERIFY_START + len(code)] = code
    image[0x0420:0x0426] = bytes([
        0xA0, 0x07,        # LDY #$07
        0x84, 0x13,        # STY $13
        0x60,              # RTS
    ])
    return bytes(image)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Verify the 6502 running as real transformer weights."
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--codegen-arith", choices=["int32", "int64", "f32"], default="int32",
        help="arithmetic mode for the generated engine (default: int32)",
    )
    parser.add_argument(
        "--no-experts", action="store_true",
        help="disable opcode-routed generated experts and run the union model",
    )
    args = parser.parse_args()

    machine = Cpu6502TransformerMachine(
        bcd=False, codegen_arith=args.codegen_arith,
        expert_opcodes=() if args.no_experts else DEFAULT_EXPERT_OPCODES,
    )
    logger.info(
        "transformer: d_model=%d n_layers=%d arith=%s (%.1fs build)",
        machine.d_model,
        machine.n_layers,
        args.codegen_arith,
        machine.build_seconds,
    )

    image = verify_program()
    run_start = time.perf_counter()
    trace = machine.run_image(image, start_pc=VERIFY_START, max_steps=args.steps)
    run_elapsed = time.perf_counter() - run_start

    ref = reference.Cpu6502(memory=bytearray(image), bcd_enabled=False)
    ref.pc = VERIFY_START
    failures = 0
    for i, row in enumerate(trace):
        ref_row = ref.step()
        a, b = ref_row.state, row.state
        ok = (a.pc, a.a, a.x, a.y, a.s, a.p) == (b.pc, b.a, b.x, b.y, b.s, b.p)
        ok = ok and {(w.addr, w.value) for w in ref_row.writes} == {
            (w.addr, w.value) for w in row.writes
        }
        if not ok:
            failures += 1
            logger.error("step %d DIVERGED:\nREF:         %s\nTRANSFORMER: %s", i, a, b)
    status = "MATCHED" if failures == 0 else f"{failures} DIVERGENCES"
    logger.info(
        "transformer: %s  %d steps in %.1fs (%.2f steps/s)",
        status, len(trace.rows), run_elapsed, len(trace.rows) / run_elapsed,
    )
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
