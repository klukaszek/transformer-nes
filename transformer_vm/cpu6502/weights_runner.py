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
from transformer_vm.cpu6502.spec import OPCODES, READS_MEMORY
from transformer_vm.cpu6502.trace import Cpu6502State, MemWrite, Trace, TraceRow
from transformer_vm.graph.core import ProgramGraph
from transformer_vm.graph.fold import GraphFolder, opcode_pins
from transformer_vm.model.weights import add_position_encoding, build_model
from transformer_vm.scheduler.milp import asap_schedule

logger = logging.getLogger(__name__)

# Opcode-routed experts, on by default: one expert per documented opcode, so
# routing is workload-independent (no game-derived tuning set).  NMI steps and
# any opcode outside a narrowed set fall back to the exact union step.
DEFAULT_EXPERT_OPCODES = tuple(sorted(OPCODES))


def scan_sequence_windows(image, *, spin_repeat: int = 4, limit: int = 16):
    """Find sequence-expert windows by ISA pattern, not by workload: closed
    LDA abs / AND imm / BEQ -7 poll loops and JMP-to-self spins.  The router
    re-verifies every window byte at dispatch, so a stale match only costs
    the fast path, never correctness."""
    data = bytes(image)
    wins = []
    for addr in range(len(data) - 6):
        if (data[addr] == 0xAD and data[addr + 3] == 0x29
                and data[addr + 5] == 0xF0 and data[addr + 6] == 0xF9):
            wins.append((addr, data[addr:addr + 7], 1))
    for addr in range(len(data) - 2):
        if data[addr] == 0x4C and data[addr + 1] | (data[addr + 2] << 8) == addr:
            wins.append((addr, data[addr:addr + 3], spin_repeat))
    return tuple(wins[:limit])


class Cpu6502TransformerMachine:
    """The 6502 as actual transformer weights, stepped position by position."""

    def __init__(self, *, bcd: bool = False,
                 codegen_arith: str = "int32",
                 checkpoint_tokens: int | None = None,
                 expert_opcodes=DEFAULT_EXPERT_OPCODES,
                 sequence_windows=(),
                 expert_schedule: str = "asap",
                 expert_milp_seconds: int = 900):
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
        self._pg = pg  # kept for the expert folds (graph/fold.py)
        build_start = time.perf_counter()
        # ASAP schedule: the optimal-width MILP does not solve at this graph
        # size (~5M integer variables); critical-path scheduling costs some
        # d_model but no solver time.
        sched = asap_schedule(pg.input_tokens, pg.output_tokens, program_graph=pg)
        self.model, _tokens, tok_to_idx, info = build_model(
            program_graph=pg, schedule=sched,
        )
        self.build_seconds = time.perf_counter() - build_start
        self.n_layers = info.n_layers
        self.slot_of = info.slot_of
        self.d_model = info.d_model
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

        from transformer_vm.model.engine_gen import (
            GeneratedWeightsEngine,
            engine_config_key,
        )

        if expert_schedule not in ("asap", "milp"):
            raise ValueError(f"unknown expert_schedule {expert_schedule!r}")
        self._expert_schedule = expert_schedule
        self._expert_milp_seconds = expert_milp_seconds
        # The engine is fully determined by (source code, bcd, arith, expert
        # set), so a manifest hit skips the expert folds and the emission and
        # loads the compiled artifact directly.  MILP schedules are excluded:
        # their incumbents depend on solve time.
        windows = [
            (int(head) & 0xFFFF, bytes(code), int(repeat))
            for head, code, repeat in (sequence_windows or ())
        ] if codegen_arith != "f32" else []
        cache_key = None
        self._engine = None
        if expert_schedule == "asap":
            expert_bytes = [
                int(b) for b in (expert_opcodes or ()) if int(b) in OPCODES
            ] if codegen_arith != "f32" else []
            seq_tag = ",".join(
                f"{head:04X}:{code.hex()}x{repeat}" for head, code, repeat in windows
            )
            cache_key = engine_config_key(
                bcd=bcd, arith=codegen_arith, expert_bytes=expert_bytes,
                extra=seq_tag,
            )
            self._engine = GeneratedWeightsEngine.from_cache(cache_key)
            if self._engine is not None:
                logger.info("engine cache hit (%d experts)", self._engine.n_experts)
        if self._engine is None:
            experts = []
            if codegen_arith != "f32":
                experts = self._build_expert_models(expert_opcodes)
                experts += [self._build_sequence_expert(*w) for w in windows]
            elif expert_opcodes:
                logger.info(
                    "opcode-routed experts disabled: f32 arith keeps every lookup "
                    "head on a real hull, which cannot be shared across models"
                )
            self._engine = GeneratedWeightsEngine(
                self.model, step_spec=self._step_spec(),
                arith=codegen_arith, experts=experts, cache_key=cache_key,
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

    def _build_expert_models(self, opcodes) -> list[dict]:
        """Rebuilt per-opcode expert models for the opcode-routed engine.

        For each expert opcode, the union graph is partially evaluated with
        the decode one-hot gates pinned (is_op_<byte>=1, every other
        is_op_*=0, is_nmi=0 -- exactly what the hard router supplies), and
        the *folded* graph (graph/fold.py) is scheduled and weight-compiled
        into its own transformer.  Unlike the earlier in-place pruning of
        the union model, a mux whose gate folds to 1 really disappears here
        -- its linear map inlines into consumers -- which is the measured
        0.29x -> ~0.2x terms gap.

        Safety is checked, not assumed: the fold refuses lookups whose
        insert-side expressions can reach a pin (shared attention state must
        be pin-independent), and engine_gen refuses any expert head whose
        classification (memory structure / running-argmax register, resolved
        by lookup provenance) disagrees with the union's.  Correctness rests
        on ram[pc] == the graph's fetched opcode; the lockstep chain (every
        opcode swept through its expert by nestest and Klaus) is the receipt.
        """
        opcodes = [int(b) for b in (opcodes or []) if int(b) in OPCODES]
        if not opcodes:
            return []

        folder = GraphFolder(self._pg, root_names=_RUNTIME_READ_DIMS)
        experts = []
        for byte in opcodes:
            r = folder.fold(opcode_pins(byte, OPCODES), label=f"{byte:02X}")
            if r.insert_side_hazards:
                raise RuntimeError(
                    f"expert {byte:02X}: decode pins reach insert-side "
                    f"expressions of {r.insert_side_hazards}; shared attention "
                    "state would diverge"
                )
            fp = r.program_graph
            if self._expert_schedule == "milp":
                # Exact-width schedule: tractable at expert scale (the union
                # formulation blowup was size-driven); at the time limit the
                # solver's best incumbent is used -- feasible, just unproven.
                from transformer_vm.scheduler.milp import milp_schedule

                t0 = time.perf_counter()
                # The engine is nnz-bound, so the objective is the schedule's
                # hygiene cost (passthrough persistence), capped at the
                # balanced-ASAP width the default path would have used.
                asap = asap_schedule(fp.input_tokens, fp.output_tokens,
                                     program_graph=fp, log=logger.debug)
                _m, _t, _ti, ainfo = build_model(program_graph=fp, schedule=asap)
                sched = milp_schedule(fp.input_tokens, fp.output_tokens,
                                      program_graph=fp, log=logger.debug,
                                      time_limit=self._expert_milp_seconds,
                                      objective="hygiene",
                                      width_bound=ainfo.d_model)
                logger.info("expert %02X: hygiene MILP in %.0fs",
                            byte, time.perf_counter() - t0)
            else:
                sched = asap_schedule(fp.input_tokens, fp.output_tokens,
                                      program_graph=fp, log=logger.debug)
            model, _tokens, tok_to_idx, info = build_model(
                program_graph=fp, schedule=sched,
            )
            experts.append({
                "name": f"{byte:02X}", "byte": byte, "model": model,
                "step_spec": self._expert_step_spec(r, model, tok_to_idx, info),
                "union_lookup_of": r.union_lookup_of,
                "_layers": info.n_layers, "_d": info.d_model,
            })
        logger.info(
            "routed experts: %d rebuilt models, %d-%d layers, d_model %d-%d "
            "(union: %d layers, d_model %d)",
            len(experts),
            min(e["_layers"] for e in experts), max(e["_layers"] for e in experts),
            min(e["_d"] for e in experts), max(e["_d"] for e in experts),
            self.n_layers, self.d_model,
        )
        return experts

    def _expert_step_spec(self, fold_result, model, tok_to_idx, info) -> dict:
        """The generated-step slot map for one expert model: same shape as
        _step_spec, resolved through the fold's name map and the expert's
        own schedule/slot assignment."""

        def s(name):
            return info.slot_of[fold_result.dim_of_name[name]]

        def head_row(row):
            w = model.head.weight[row]
            cols = np.nonzero(w)[0]
            return ([int(c) for c in cols], [float(w[c]) for c in cols])

        return {
            "inputs": {
                "is_state": s("is_state"),
                "pc_lo": s("in_pc_lo"), "pc_hi": s("in_pc_hi"),
                "a": s("in_a"), "x": s("in_x"), "y": s("in_y"), "s": s("in_s"),
                "ir": s("in_ir"), "oplo": s("in_oplo"), "ophi": s("in_ophi"),
                "ea_lo": s("in_ea_lo"), "ea_hi": s("in_ea_hi"),
                "data": s("in_data"), "halted": s("in_halted"),
                "nmi": s("in_nmi"),
            },
            "p_flags": [
                (s("in_p_c"), reference.FLAG_C), (s("in_p_z"), reference.FLAG_Z),
                (s("in_p_i"), reference.FLAG_I), (s("in_p_d"), reference.FLAG_D),
                (s("in_p_b"), reference.FLAG_B), (s("in_p_u"), reference.FLAG_U),
                (s("in_p_v"), reference.FLAG_V), (s("in_p_n"), reference.FLAG_N),
            ],
            "outputs": {name: s(name) for name in _RUNTIME_READ_DIMS},
            "halt_head": head_row(tok_to_idx["HALT"]),
            "state_head": head_row(tok_to_idx["STATE"]),
        }

    def _build_sequence_expert(self, head: int, code: bytes, repeat: int) -> dict:
        """One sequence expert: a K-instruction window folded as ONE forward.

        The router guarantees pc == head, the window's bytes, no pending NMI,
        and a remaining step/cycle budget covering the whole window, so the
        fold pins all of it.  The license is checked, not assumed: every step
        must be write-free and non-halting (folded constants), control flow
        inside the window must be straight-line (folded next_pc == the next
        step's pinned pc), and only the first instruction may read memory
        (the run loop's shadow-read watch check covers just that one).
        """
        from transformer_vm.graph.fold import SEQUENCE_FEEDBACK, sequence_pins

        pin_list = sequence_pins(head, code, OPCODES) * repeat
        n_steps = len(pin_list)
        ops, off = [], 0
        while off < len(code):
            op = OPCODES[code[off]]
            ops.append(op)
            off += op.length
        ops = ops * repeat
        for op in ops[1:]:
            if op.operation in READS_MEMORY and op.mode not in ("imp", "acc", "imm"):
                raise RuntimeError(
                    f"sequence @{head:04X}: {op.mnemonic} reads memory past "
                    "the shadow-watch check (only the first instruction may)"
                )

        folder = GraphFolder(self._pg, root_names=_RUNTIME_READ_DIMS)
        label = f"seq_{head:04X}"
        r = folder.fold_sequence(pin_list, SEQUENCE_FEEDBACK, label=label)
        if r.insert_side_hazards:
            raise RuntimeError(
                f"sequence @{head:04X}: pins reach insert-side expressions "
                f"of {r.insert_side_hazards}"
            )

        one = r.program_graph.one

        def const_root(name, k):
            d = r.dim_of_name[f"{name}__s{k}"]
            expr = getattr(d, "expr", None)
            if expr is None or any(t is not one for t in expr.terms):
                raise RuntimeError(
                    f"sequence @{head:04X} step {k}: {name} did not fold "
                    "to a constant"
                )
            return int(round(expr.terms.get(one, 0.0)))

        for k in range(n_steps):
            for store in ("is_store", "is_store2", "is_store3"):
                if const_root(store, k) != 0:
                    raise RuntimeError(
                        f"sequence @{head:04X} step {k}: not write-free"
                    )
            if const_root("new_halted", k) != 0:
                raise RuntimeError(f"sequence @{head:04X} step {k}: may halt")
            if k + 1 < n_steps:
                nxt = int(pin_list[k + 1]["current_pc_lo"]) | (
                    int(pin_list[k + 1]["current_pc_hi"]) << 8
                )
                if const_root("next_pc", k) != nxt:
                    raise RuntimeError(
                        f"sequence @{head:04X} step {k}: control flow leaves "
                        "the window (next_pc != the next step's pc)"
                    )

        fp = r.program_graph
        sched = asap_schedule(fp.input_tokens, fp.output_tokens,
                              program_graph=fp, log=logger.debug)
        model, _tokens, tok_to_idx, info = build_model(
            program_graph=fp, schedule=sched,
        )
        cycles = sum(op.cycles for op in ops)
        return {
            "kind": "seq", "name": label, "head": head, "window": bytes(code),
            "n_steps": n_steps, "cycles": cycles,
            "cycles_prefix": cycles - ops[-1].cycles, "model": model,
            "step_spec": self._sequence_step_spec(r, model, tok_to_idx, info),
            "union_lookup_of": r.union_lookup_of,
            "_layers": info.n_layers, "_d": info.d_model,
        }

    def _sequence_step_spec(self, fold_result, model, tok_to_idx, info) -> dict:
        """The _expert_step_spec shape, with one outputs map per step."""

        def s(name):
            return info.slot_of[fold_result.dim_of_name[name]]

        def head_row(row):
            w = model.head.weight[row]
            cols = np.nonzero(w)[0]
            return ([int(c) for c in cols], [float(w[c]) for c in cols])

        return {
            "inputs": {
                "is_state": s("is_state"),
                "pc_lo": s("in_pc_lo"), "pc_hi": s("in_pc_hi"),
                "a": s("in_a"), "x": s("in_x"), "y": s("in_y"), "s": s("in_s"),
                "ir": s("in_ir"), "oplo": s("in_oplo"), "ophi": s("in_ophi"),
                "ea_lo": s("in_ea_lo"), "ea_hi": s("in_ea_hi"),
                "data": s("in_data"), "halted": s("in_halted"),
                "nmi": s("in_nmi"),
            },
            "p_flags": [
                (s("in_p_c"), reference.FLAG_C), (s("in_p_z"), reference.FLAG_Z),
                (s("in_p_i"), reference.FLAG_I), (s("in_p_d"), reference.FLAG_D),
                (s("in_p_b"), reference.FLAG_B), (s("in_p_u"), reference.FLAG_U),
                (s("in_p_v"), reference.FLAG_V), (s("in_p_n"), reference.FLAG_N),
            ],
            "step_outputs": [
                {name: s(f"{name}__s{k}") for name in _RUNTIME_READ_DIMS}
                for k in range(fold_result.n_steps)
            ],
            "halt_head": head_row(tok_to_idx["HALT"]),
            "state_head": head_row(tok_to_idx["STATE"]),
        }

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

    def set_read_lease(self, addr: int, value: int) -> None:
        self._engine.set_read_lease(addr, value)

    def clear_read_lease(self) -> None:
        self._engine.clear_read_lease()

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
