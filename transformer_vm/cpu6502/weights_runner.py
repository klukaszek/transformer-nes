"""Run the 6502 through the real weights-based transformer.

This is the end of the verification chain: the same computation graph that
the exact evaluator and the C++ kernel execute is MILP-scheduled onto
transformer layers and compiled into actual VanillaTransformer weights
(analytically -- no training).  Execution is then genuine transformer
inference: per-position embeddings, Q/K/V projections, hardmax attention
(hull cache, or a true softmax KV cache), ReGLU FFN, residual stream, and
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

Numerics caveat: the default hull/engine path is float32 and uses grouped
4 KiB memory lookups so high-address reads survive Q/K rounding.  True
softmax (`cache="standard"`) still averages exact ties instead of applying
the hull sequence tie-break, so arbitrary high-address rewrite-heavy
programs remain a separate standard-cache problem.
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import scipy.sparse as sp
import torch

from transformer_vm.attention import HullKVCache, StandardKVCache
from transformer_vm.cpu6502 import reference
from transformer_vm.cpu6502.evaluator import _RUNTIME_READ_DIMS, Cpu6502Runtime
from transformer_vm.cpu6502.trace import Cpu6502State, MemWrite, Trace, TraceRow
from transformer_vm.graph.core import ProgramGraph
from transformer_vm.model.transformer import add_position_encoding
from transformer_vm.model.weights import build_model
from transformer_vm.scheduler.milp import asap_schedule

logger = logging.getLogger(__name__)


class Cpu6502TransformerMachine:
    """The 6502 as actual transformer weights, stepped position by position."""

    def __init__(self, *, bcd: bool = False, cache: str = "hull",
                 matmul: str = "codegen", dtype: str = "float32",
                 codegen_arith: str = "int32",
                 checkpoint_tokens: int | None = None):
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
        self.model.requires_grad_(False)  # inference only
        self.build_seconds = time.perf_counter() - build_start
        self.n_layers = shared[3]
        self.slot_of = shared[6]
        self.d_model = shared[9]
        self._state_row = tok_to_idx["STATE"]
        self._halt_row = tok_to_idx["HALT"]
        self._read_slots = {
            name: self.slot_of[self._dim_by_name[name]]
            for name in _RUNTIME_READ_DIMS
        }
        self._one_slot = self.slot_of[self.rt._one]

        n_heads = self.model.attn[0].num_heads
        # "engine" (generic CSR C++) and "codegen" (generated straight-line
        # C++) share every runner code path; only the engine class differs.
        self._cpp_stack = matmul in ("engine", "codegen")
        if self._cpp_stack:
            self.cache = None  # attention state lives inside the C++ engine
        elif cache == "hull":
            self.cache = HullKVCache(self.n_layers, n_heads)
            for layer_idx in range(self.n_layers):
                for h in range(n_heads):
                    if self.model.head_tiebreak[layer_idx][h]:
                        self.cache.set_tiebreak(layer_idx, h, True)
        elif cache == "standard":
            # True softmax over HARD_K-scaled scores; exact ties average,
            # see the module docstring.
            self.cache = StandardKVCache(self.n_layers, n_heads)
        else:
            raise ValueError(f"unknown cache {cache!r}")

        # Precision experiment knob.  The grouped memory lookup keeps K-side
        # address scores below 2^24, so float32 Q/K rounding can distinguish
        # adjacent high addresses; hull tie-breaks still resolve rewrites.
        if dtype not in ("float64", "float32"):
            raise ValueError(f"unknown dtype {dtype!r}")
        self.dtype = dtype
        self.np_dtype = np.float32 if dtype == "float32" else np.float64
        self.torch_dtype = torch.float32 if dtype == "float32" else torch.float64
        self._state_head_cols, self._state_head_vals = self._head_row(self._state_row)
        self._halt_head_cols, self._halt_head_vals = self._head_row(self._halt_row)

        self.matmul = matmul
        self._direct_writes = False
        self._cpp_step = False
        if self._cpp_stack:
            # Whole layer stack per pybind call; hull attention and
            # structural passthrough fusion live in C++.  "engine" loads
            # the generic CSR engine (model/weights_engine.cpp); "codegen"
            # compiles model-specific straight-line code
            # (model/engine_gen.py).
            if cache != "hull":
                raise ValueError(f"matmul={matmul!r} has hull attention built in")
            if matmul == "codegen":
                from transformer_vm.model.engine_gen import GeneratedWeightsEngine

                self._engine = GeneratedWeightsEngine(
                    self.model, dtype=dtype, step_spec=self._step_spec(),
                    arith=codegen_arith,
                )
            else:
                from transformer_vm.model.engine import SparseWeightsEngine

                self._engine = SparseWeightsEngine(self.model, dtype=dtype)
            self._direct_writes = self._engine.supports_direct_writes
            self._cpp_step = getattr(self._engine, "has_machine_step", False)
            if self._cpp_step:
                self._step_out = np.empty(21, dtype=np.int64)
        elif matmul == "sparse":
            # The analytic weights are ~99.9% zeros; CSR matvecs run the
            # identical linear maps orders of magnitude faster.  This path
            # remains useful for storage experiments and true-softmax checks;
            # the C++ engine is the default production path.
            self._layers = []
            for layer_idx in range(self.n_layers):
                ip = sp.csr_matrix(self.model.attn[layer_idx].in_proj_weight.numpy().astype(self.np_dtype))
                opw = sp.csr_matrix(self.model.attn[layer_idx].out_proj.weight.numpy().astype(self.np_dtype))
                fi = sp.csr_matrix(self.model.ff_in[layer_idx].weight.numpy().astype(self.np_dtype))
                fo = sp.csr_matrix(self.model.ff_out[layer_idx].weight.numpy().astype(self.np_dtype))
                self._layers.append((ip, opw, fi, fo))
        elif matmul == "dense":
            if dtype != "float64":
                raise ValueError("matmul='dense' is a float64 diagnostic path")
        else:
            raise ValueError(f"unknown matmul {matmul!r}")

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

    @torch.no_grad()
    def _forward(self, x: torch.Tensor) -> torch.Tensor | np.ndarray:
        """One genuine transformer position: attention + ReGLU per layer."""
        add_position_encoding(x, self.pos)
        self.pos += 1
        if self._cpp_stack:
            xv = x.numpy().astype(self.np_dtype, copy=False)
            self._engine.forward(xv)  # in place, whole stack
            return xv
        if self.matmul == "sparse":
            return self._forward_sparse(x)
        model = self.model
        for layer_idx, (attn, ff_in, ff_out) in enumerate(
            zip(model.attn, model.ff_in, model.ff_out, strict=True)
        ):
            q, k, v = (attn.in_proj_weight @ x).chunk(3, dim=-1)
            out = self.cache.layer_step(layer_idx, k, q, v)
            x = x + attn.out_proj(out)
            gate, val = ff_in(x).chunk(2, dim=-1)
            x = x + ff_out(torch.nn.functional.relu(gate) * val)
        return x

    def _forward_sparse(self, x: torch.Tensor) -> torch.Tensor:
        """The same layer stack through CSR matvecs."""
        xv = x.numpy().astype(self.np_dtype, copy=False)
        d = self.d_model
        for layer_idx, (ip, opw, fi, fo) in enumerate(self._layers):
            qkv = ip @ xv
            # Scores are always computed in double inside the caches; in
            # float32 mode the rounding damage to K/Q happened in the
            # matvec above, which is exactly what the experiment measures.
            out = self.cache.layer_step(
                layer_idx,
                torch.from_numpy(qkv[d:2 * d].astype(np.float64, copy=False)),
                torch.from_numpy(qkv[:d].astype(np.float64, copy=False)),
                torch.from_numpy(qkv[2 * d:].astype(np.float64, copy=False)),
            )
            xv = xv + opw @ out.numpy().astype(self.np_dtype, copy=False)
            gv = fi @ xv
            half = gv.shape[0] // 2
            xv = xv + fo @ (np.maximum(gv[:half], 0.0) * gv[half:])
        return torch.from_numpy(xv.astype(np.float64, copy=False))

    def _embed_array(self, values: dict) -> np.ndarray:
        """Engine input embedding without constructing a torch tensor."""
        x = np.zeros(self.d_model, dtype=self.np_dtype)
        x[self._one_slot] = 1.0
        for dim, value in values.items():
            x[self.slot_of[dim]] = float(value)
        return x

    def _forward_values(self, values: dict) -> torch.Tensor | np.ndarray:
        if not self._cpp_stack:
            return self._forward(self._embed(values))
        x = self._embed_array(values)
        add_position_encoding(x, self.pos)
        self.pos += 1
        self._engine.forward(x)
        return x

    def _embed(self, values: dict) -> torch.Tensor:
        """Input embedding for one token: dim values at their slots.

        Equivalent to a vocabulary embedding row for a token type that
        enumerated exactly these values; position features are added by
        _forward (slots 0-2 are reserved for them).
        """
        x = torch.zeros(self.d_model, dtype=self.torch_dtype)
        x[self.slot_of[self.rt._one]] = 1.0
        for dim, value in values.items():
            x[self.slot_of[dim]] = float(value)
        return x

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
        w = self.model.head.weight[row].detach().numpy()
        cols = np.nonzero(w)[0].astype(np.int32)
        return cols, w[cols].astype(self.np_dtype, copy=False)

    def _head_logit(self, h: np.ndarray, cols: np.ndarray, vals: np.ndarray) -> float:
        return float(np.dot(h[cols], vals))

    def _read(self, h: torch.Tensor | np.ndarray, name: str) -> float:
        slot = self._read_slots.get(name)
        if slot is None:
            slot = self.slot_of[self._dim_by_name[name]]
            self._read_slots[name] = slot
        return float(h[slot])

    # ── Machine protocol (mirrors evaluator.Cpu6502Runtime, so the NES
    # bus's GraphCore adapter can drive this machine unchanged) ──────

    def boot_image(self, memory_image, *, start_pc: int, state: dict | None = None):
        # Fresh token log: clear attention state so a machine can be reused.
        if self._cpp_stack:
            self._engine.reset()
        else:
            self.cache.clear()
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
        if self._cpp_stack:
            halt_logit = self._head_logit(h, self._halt_head_cols, self._halt_head_vals)
            state_logit = self._head_logit(h, self._state_head_cols, self._state_head_vals)
        else:
            logits = self.model.head(h)
            halt_logit = float(logits[self._halt_row])
            state_logit = float(logits[self._state_row])
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
    parser.add_argument("--cache", choices=["hull", "standard"], default="hull")
    parser.add_argument("--matmul", choices=["engine", "sparse", "dense"], default="engine")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = parser.parse_args()

    machine = Cpu6502TransformerMachine(
        bcd=False, cache=args.cache, matmul=args.matmul, dtype=args.dtype
    )
    logger.info(
        "transformer: d_model=%d n_layers=%d matmul=%s dtype=%s (%.1fs build)",
        machine.d_model,
        machine.n_layers,
        machine.matmul,
        machine.dtype,
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
        "transformer(%s): %s  %d steps in %.1fs (%.2f steps/s)",
        args.cache, status, len(trace.rows), run_elapsed, len(trace.rows) / run_elapsed,
    )
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
