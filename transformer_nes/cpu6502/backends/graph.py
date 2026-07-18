"""Evaluate 6502 programs through the computation graph with exact arithmetic.

This runs the graph DSL directly -- no transformer weights needed.  It is
the graph's definitional semantics: the weights backend builds its model
FROM this runtime's graph and dim handles, and locksteps against it in the
fold receipts.

Execution protocol (one transformer step = one 6502 instruction):
  1. A ``start`` token carries the initial PC, and one WRITE token per nonzero
     image byte seeds the append-only memory log.  Code and data share the
     log, so instruction fetch is just a memory read at PC.
  2. A STATE token carries the full architectural state (PC, registers, flag
     bits, and the previous instruction's fetch/EA/data fields).
  3. ``evaluate_current`` runs the graph over the latest token: it executes
     one instruction and exposes the named result dims (RUNTIME_READ_DIMS).
  4. The run loop appends the instruction's memory writes as WRITE tokens,
     builds the next STATE token from the results, and repeats until the
     graph predicts HALT.

The graph's static schedule is compiled once into a single generated Python
function: expressions become straight-line arithmetic over locals, attention
queries are inlined against per-lookup hull caches, and a backward
liveness pass drops dims that do not feed an externally read result.

Exactness rules shared with interpreter.py and the generated engine:
  - stepglu/match gates are exact only for integer-valued arguments, so all
    thresholds in the graph must be integers.
  - Read-back uses round-half-to-even (Python round; C++ nearbyint).
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from transformer_nes.attention.runtime import HullAttention, load_hull_ext
from transformer_nes.cpu6502.backends import reference
from transformer_nes.cpu6502.interpreter import build as build_6502_graph
from transformer_nes.cpu6502.trace import Cpu6502State, MemWrite, Trace, TraceRow
from transformer_nes.graph import core as graph_core
from transformer_nes.graph.core import (
    InputDimension,
    LookUpDimension,
    PersistDimension,
    ReGLUDimension,
    reset_graph,
)

logger = logging.getLogger(__name__)

OP_SKIP = 0
OP_REGLU = 2
OP_PERSIST = 3
OP_LOOKUP = 4

# Dimensions the run loop reads back from evaluate_current() via read_dim.
# The compiled step function only materializes these (and the output-token
# score terms); everything else stays in locals and is dead-code-eliminated
# when it does not feed them.
RUNTIME_READ_DIMS = (
    "current_pc", "new_a", "new_x", "new_y", "new_s", "new_p", "new_halted",
    "fetched_opcode", "fetched_operand_lo", "fetched_operand_hi",
    "ea", "data_out", "next_pc",
    "is_store", "write_addr_out", "write_value_out",
    "is_store2", "write_addr2_out", "write_value2_out",
    "is_store3", "write_addr3_out", "write_value3_out",
)


class Cpu6502Runtime:
    """Graph evaluator for the 6502 CPU graph."""

    def __init__(self, *, bcd: bool = True, wide_bits: bool = False):
        reset_graph()
        self.input_tokens, self.output_tokens = build_6502_graph(
            bcd=bcd, wide_bits=wide_bits)
        self.all_dims = list(graph_all_dims())
        self.all_lookups = list(graph_all_lookups())
        self.dim_id_by_name: dict[str, int] = {}
        for dim in self.all_dims:
            if dim.name not in self.dim_id_by_name:
                self.dim_id_by_name[dim.name] = dim.id

        self.one = find_builtin_dim("one")

        self.byte_value = find_input_dim("byte_value")
        self.is_memory_write = find_input_dim("is_memory_write")
        self.addr_lo = find_input_dim("addr_lo")
        self.addr_hi = find_input_dim("addr_hi")
        self.is_state = find_input_dim("is_state")
        self.is_start = find_input_dim("is_start")
        self.start_pc_lo = find_input_dim("start_pc_lo")
        self.start_pc_hi = find_input_dim("start_pc_hi")

        self.in_pc_lo = find_input_dim("in_pc_lo")
        self.in_pc_hi = find_input_dim("in_pc_hi")
        self.in_a = find_input_dim("in_a")
        self.in_x = find_input_dim("in_x")
        self.in_y = find_input_dim("in_y")
        self.in_s = find_input_dim("in_s")
        self.in_p_c = find_input_dim("in_p_c")
        self.in_p_z = find_input_dim("in_p_z")
        self.in_p_i = find_input_dim("in_p_i")
        self.in_p_d = find_input_dim("in_p_d")
        self.in_p_b = find_input_dim("in_p_b")
        self.in_p_u = find_input_dim("in_p_u")
        self.in_p_v = find_input_dim("in_p_v")
        self.in_p_n = find_input_dim("in_p_n")
        self.in_ir = find_input_dim("in_ir")
        self.in_oplo = find_input_dim("in_oplo")
        self.in_ophi = find_input_dim("in_ophi")
        self.in_ea_lo = find_input_dim("in_ea_lo")
        self.in_ea_hi = find_input_dim("in_ea_hi")
        self.in_data = find_input_dim("in_data")
        self.in_halted = find_input_dim("in_halted")
        self.in_nmi = find_input_dim("in_nmi")
        self.in_irq = find_input_dim("in_irq")

        self.hull_ext = load_hull_ext()
        if self.hull_ext is None:
            raise RuntimeError(
                "Hull attention extension could not be loaded; it is required by the 6502 runtime"
            )

        self.attention = {lu.id: HullAttention(lu, self.hull_ext) for lu in self.all_lookups}
        self.lookup_by_id = {lu.id: lu for lu in self.all_lookups}

        # Classify the lookups so each token type inserts only into the
        # caches that store it: WRITE -> memory reads, start -> start lookup,
        # STATE -> state lookup.
        self.memory_read_lus = []
        start_lu = state_lu = None
        for lu in self.all_lookups:
            key_terms = set(lu.key_exprs_2d[0].terms.keys()) | set(lu.key_exprs_2d[1].terms.keys())
            value_first_dims = set(lu.value_exprs[0].terms.keys()) if lu.value_exprs else set()
            if self.byte_value in value_first_dims:
                self.memory_read_lus.append(lu)
            elif self.is_start in key_terms:
                start_lu = lu
            elif self.is_state in key_terms:
                state_lu = lu
        assert start_lu is not None and state_lu is not None
        self.start_lu = start_lu
        self.state_lu = state_lu

        self.share_identical_memory_caches()
        self.build_indexed_schedule()
        self.reset()

    def build_indexed_schedule(self):
        self.num_dims = len(self.all_dims)

        self.indexed_dim_ops = [OP_SKIP] * self.num_dims
        self.indexed_dim_args = [None] * self.num_dims
        self.lookup_dim_ids = {
            lu.id: [dim.id for dim in lu.dims]
            for lu in self.all_lookups
        }
        self.lookup_query_exprs = {
            lu.id: (
                self.compile_expr(lu.query_exprs_2d[0]),
                self.compile_expr(lu.query_exprs_2d[1]),
            )
            for lu in self.all_lookups
        }

        for dim in self.all_dims:
            if isinstance(dim, ReGLUDimension):
                self.indexed_dim_ops[dim.id] = OP_REGLU
                self.indexed_dim_args[dim.id] = (
                    self.compile_expr(dim.a_expr),
                    self.compile_expr(dim.b_expr),
                )
            elif isinstance(dim, PersistDimension):
                self.indexed_dim_ops[dim.id] = OP_PERSIST
                self.indexed_dim_args[dim.id] = self.compile_expr(dim.expr)
            elif isinstance(dim, LookUpDimension):
                self.indexed_dim_ops[dim.id] = OP_LOOKUP
                self.indexed_dim_args[dim.id] = dim.lookup.id

        self.indexed_schedule = [
            (dim_id, op, arg)
            for dim_id, (op, arg) in enumerate(zip(self.indexed_dim_ops, self.indexed_dim_args, strict=True))
            if op != OP_SKIP
        ]
        self.indexed_outputs = [
            (name, self.compile_expr(expr))
            for name, expr in self.output_tokens.items()
        ]

        # Compile the static schedule to straight-line Python once: the
        # interpreted loop spends most of its time in eval_indexed_expr
        # (~9K calls per instruction), which codegen eliminates entirely.
        self.step_fn = self.compile_step_fn()
        self.insert_kv_fns = {lu.id: self.compile_insert_kv_fn(lu) for lu in self.all_lookups}

    @staticmethod
    def expr_src(arg, fmt="v[{}]"):
        """Render a compiled (n, ids, coeffs) expression as Python source.

        Terms are emitted in storage order so float addition order (and
        therefore exact results) matches eval_indexed_expr.
        """
        n, ids, coeffs = arg
        if n == 0:
            return "0.0"
        parts = []
        for dim_id, coeff in zip(ids, coeffs, strict=True):
            ref = fmt.format(dim_id)
            if coeff == 1:
                parts.append(ref)
            elif coeff == -1:
                parts.append(f"-{ref}")
            else:
                parts.append(f"{coeff!r}*{ref}")
        return " + ".join(parts)

    def compile_step_fn(self):
        """Generate one function that evaluates the whole indexed schedule.

        Every intermediate dimension becomes a Python local (STORE_FAST is
        much cheaper than a list item store), lookup queries are inlined
        with per-lookup reusable buffers, and only the dimensions read
        outside the step -- named dims for the trace row and output-token
        score terms -- are written back to the values list.
        """
        fmt = "d{}"
        computed = {dim_id for dim_id, _op, _arg in self.indexed_schedule}
        external = {
            self.dim_id_by_name[name]
            for name in RUNTIME_READ_DIMS
            if name in self.dim_id_by_name
        }
        for _name, arg in self.indexed_outputs:
            external.update(arg[1])

        # Backward pass: keep only schedule entries that (transitively) feed
        # an externally read dimension.  Dimension ids are topologically
        # ordered, so a single reverse sweep finds the closure.
        needed = set(external)
        for dim_id, op, arg in reversed(self.indexed_schedule):
            if op == OP_LOOKUP:
                lu_dim_ids = self.lookup_dim_ids[arg]
                if any(d in needed for d in lu_dim_ids):
                    needed.update(lu_dim_ids)
                    qx_expr, qy_expr = self.lookup_query_exprs[arg]
                    needed.update(qx_expr[1])
                    needed.update(qy_expr[1])
            elif dim_id in needed:
                if op == OP_REGLU:
                    needed.update(arg[0][1])
                    needed.update(arg[1][1])
                else:
                    needed.update(arg[1])
        live_schedule = [
            (dim_id, op, arg)
            for dim_id, op, arg in self.indexed_schedule
            if dim_id in needed
        ]

        # Exposed for alternate consumers of the compiled schedule.
        self.live_schedule = live_schedule
        self.step_input_ids = sorted(needed - computed)
        self.external_store_ids = sorted(external & computed)

        namespace = {}
        lines = ["def step(g, v):"]
        for dim_id in self.step_input_ids:
            lines.append(f"    d{dim_id} = g.get({dim_id}, 0.0)")
        emitted_lookups = set()
        for dim_id, op, arg in live_schedule:
            if op == OP_PERSIST:
                lines.append(f"    d{dim_id} = {self.expr_src(arg, fmt)}")
            elif op == OP_REGLU:
                a_expr, b_expr = arg
                lines.append(f"    _b = {self.expr_src(b_expr, fmt)}")
                lines.append(f"    d{dim_id} = ({self.expr_src(a_expr, fmt)}) * _b if _b > 0.0 else 0.0")
            elif op == OP_LOOKUP and arg not in emitted_lookups:
                emitted_lookups.add(arg)
                lines.extend(self.lookup_query_lines(arg, namespace, fmt))
        for dim_id in self.external_store_ids:
            lines.append(f"    v[{dim_id}] = d{dim_id}")
        logger.debug(
            "compiled step: %d/%d schedule entries live",
            len(live_schedule), len(self.indexed_schedule),
        )
        exec("\n".join(lines), namespace)  # noqa: S102
        return namespace["step"]

    def lookup_query_lines(self, lu_id, namespace, fmt):
        """Emit inline query code for one lookup into the step function."""
        attn = self.attention[lu_id]
        qx_expr, qy_expr = self.lookup_query_exprs[lu_id]
        buf = f"_q{lu_id}buf"
        cache = f"_q{lu_id}cache"
        namespace[buf] = np.empty((attn.num_value_pairs, 2))
        namespace[cache] = attn.cache
        lines = [
            f"    {buf}[:, 0] = {self.expr_src(qx_expr, fmt)}",
            f"    {buf}[:, 1] = {self.expr_src(qy_expr, fmt)}",
            f"    o = {cache}.query(0, {buf}).ravel().tolist()",
        ]
        for i, dim_id in enumerate(self.lookup_dim_ids[lu_id]):
            lines.append(f"    d{dim_id} = o[{i}]")
        return lines

    def insert_kv_plan(self, lu):
        """Topologically order the derived dims feeding a lookup's key/values.

        Returns (order, input_ids) where order lists Persist/ReGLU dims in
        dependency order, or None if the chain touches attention state (a
        key/value that depends on another lookup cannot be computed from the
        token alone).
        """
        order = []
        seen = set()
        supported = True

        def visit(dim):
            nonlocal supported
            if not supported or dim.id in seen:
                return
            seen.add(dim.id)
            if isinstance(dim, PersistDimension):
                for d in dim.expr.terms:
                    visit(d)
                order.append(dim)
            elif isinstance(dim, ReGLUDimension):
                for d in dim.a_expr.terms:
                    visit(d)
                for d in dim.b_expr.terms:
                    visit(d)
                order.append(dim)
            elif not isinstance(dim, InputDimension):
                supported = False

        for expr in [*lu.key_exprs_2d, *lu.value_exprs]:
            for dim in expr.terms:
                visit(dim)
        if not supported:
            return None
        input_ids = sorted(
            dim_id for dim_id in seen
            if isinstance(self.all_dims[dim_id], InputDimension)
        )
        return order, input_ids

    def compile_insert_kv_fn(self, lu):
        """Generate a direct (kx, ky, values) evaluator for one lookup.

        Token insertion only needs the lookup's key and value expressions,
        whose derived dimensions form a small arithmetic chain over the
        input dims.
        """
        plan = self.insert_kv_plan(lu)
        if plan is None:
            raise NotImplementedError(
                f"lookup {lu.id} key/values depend on attention state; "
                "tokens for it cannot be inserted directly"
            )
        order, input_ids = plan

        fmt = "d{}"
        lines = ["def kv(g):"]
        for dim_id in input_ids:
            lines.append(f"    d{dim_id} = g.get({dim_id}, 0.0)")
        for dim in order:
            if isinstance(dim, PersistDimension):
                src = self.expr_src(self.compile_expr(dim.expr), fmt)
                lines.append(f"    d{dim.id} = {src}")
            else:
                a_src = self.expr_src(self.compile_expr(dim.a_expr), fmt)
                b_src = self.expr_src(self.compile_expr(dim.b_expr), fmt)
                lines.append(f"    _b = {b_src}")
                lines.append(f"    d{dim.id} = ({a_src}) * _b if _b > 0.0 else 0.0")
        kx_src = self.expr_src(self.compile_expr(lu.key_exprs_2d[0]), fmt)
        ky_src = self.expr_src(self.compile_expr(lu.key_exprs_2d[1]), fmt)
        val_srcs = ", ".join(
            self.expr_src(self.compile_expr(expr), fmt) for expr in lu.value_exprs
        )
        lines.append(f"    return {kx_src}, {ky_src}, ({val_srcs},)")
        namespace = {}
        exec("\n".join(lines), namespace)  # noqa: S102
        return namespace["kv"]

    @staticmethod
    def compile_expr(expr):
        ids = []
        coeffs = []
        for dim, coeff in expr.terms.items():
            ids.append(dim.id)
            coeffs.append(coeff)
        return len(ids), tuple(ids), tuple(coeffs)

    def share_identical_memory_caches(self):
        """Use one KV cache for memory lookups with identical keys and values.

        The graph still has multiple memory lookup sites because it needs
        different query addresses in the same instruction.  Their stored WRITE
        keys and values are identical, so the exact evaluator can share one
        backing cache and query it through each lookup's own query expression.
        """
        if not self.memory_read_lus:
            return
        first = self.attention[self.memory_read_lus[0].id]
        for lu in self.memory_read_lus[1:]:
            attn = self.attention[lu.id]
            if (
                attn.num_value_pairs != first.num_value_pairs
                or attn.nv != first.nv
                or not same_exprs(lu.value_exprs, self.memory_read_lus[0].value_exprs)
                or not same_exprs(lu.key_exprs_2d, self.memory_read_lus[0].key_exprs_2d)
            ):
                continue
            attn.cache = first.cache

    def reset(self):
        self.pos = 0
        for attn in self.attention.values():
            attn.clear()
        self.last_vals: dict | None = None
        self.trace_len = 0

    def set_token(self, token_name: str, overrides: dict | None = None):
        vals = {self.one: 1.0}
        embedding = self.input_tokens[token_name]
        for dim, coeff in embedding.terms.items():
            vals[dim] = coeff
        if overrides:
            vals.update(overrides)
        return vals

    def direct_insert(self, token_name: str, overrides: dict | None = None, *, lookup_ids: set[int]):
        """Insert a token into the requested attention caches.

        Each token type only participates in specific lookups (WRITE ->
        memory reads, start -> start lookup, STATE -> state lookup); shared
        backing caches receive one insert per token.
        """
        vals = self.set_token(token_name, overrides)
        g = {dim.id: value for dim, value in vals.items()}
        inserted_caches: set[int] = set()
        for lid in sorted(lookup_ids):
            attn = self.attention[lid]
            cache_id = id(attn.cache)
            if cache_id in inserted_caches:
                continue
            inserted_caches.add(cache_id)
            kx, ky, raw_vals = self.insert_kv_fns[lid](g)
            attn.insert_raw(kx, ky, raw_vals)
        self.pos += 1
        self.last_vals = vals
        return vals

    def evaluate_current(self):
        """Evaluate the graph over the latest token without inserting a new one."""
        if self.last_vals is None:
            raise RuntimeError("no prior token to evaluate from")
        g = {}
        values = [0.0] * self.num_dims
        for dim, value in self.last_vals.items():
            value = float(value)
            g[dim.id] = value
            values[dim.id] = value
        self.step_fn(g, values)
        return values

    def predict_next(self, values: list[float]) -> str:
        best_score = -1e300
        best_name = None
        for name, score_expr in self.indexed_outputs:
            score = eval_indexed_expr(score_expr, values)
            if score > best_score:
                best_score = score
                best_name = name
        assert best_name is not None  # STATE/HALT always compete
        return best_name

    def run(self, program: list[int], *, start_pc: int = 0x8000, max_steps: int = 100):
        # Tiny programs are assumed to run with mostly-zero memory.  Seed a single
        # default WRITE at address 0 so reads of uninitialized memory return 0.  Any
        # writes performed by the program itself add their own WRITE tokens.
        memory_image = [0] * 0x10000
        return self.run_with_image(memory_image, program, start_pc=start_pc, max_steps=max_steps)

    def run_image(self, memory_image: bytes | bytearray, *, start_pc: int = 0x0400, max_steps: int = 100_000):
        if len(memory_image) != 0x10000:
            raise ValueError("memory_image must be exactly 64 KiB")
        image = list(memory_image)
        return self.run_with_image(image, None, start_pc=start_pc, max_steps=max_steps)

    def run_with_image(
        self,
        memory_image: list[int],
        program_bytes: list[int] | None,
        *,
        start_pc: int,
        max_steps: int,
    ) -> Trace:
        if program_bytes is not None:
            for offset, byte in enumerate(program_bytes):
                memory_image[(start_pc + offset) & 0xFFFF] = byte & 0xFF

        self.boot_image(memory_image, start_pc=start_pc)

        trace = Trace()
        for _ in range(max_steps):
            row = self.machine_step()
            trace.append(row)
            if self.machine_halted:
                break
        return trace

    # ── Machine interface ───────────────────────────────────────────
    # The stateful step-at-a-time protocol used both by the run loop above
    # and by the MMIO bus (transformer_nes/cpu6502/bus.py): boot_image() seeds
    # the token log, machine_step() executes one instruction (or a pending
    # NMI), and inject_write()/set_nmi() are the two device-side entry
    # points.  The first STATE token is inserted lazily by machine_step(),
    # so WRITE tokens injected between steps are always visible to the next
    # instruction -- the bus is part of the model, only the device is not.

    def boot_image(self, memory_image, *, start_pc: int, state: dict | None = None):
        """Reset and seed the token log with the start token and memory image.

        state, if given, restores a previously captured machine_state (used
        by snapshot-and-rebuild checkpointing: the log is reseeded from a
        RAM snapshot and execution resumes mid-program).
        """
        self.reset()
        prefix_start = time.perf_counter()
        self.memory_ids = {lu.id for lu in self.memory_read_lus}

        # Feed start token with initial PC.
        self.direct_insert(
            "start",
            {
                self.start_pc_lo: start_pc & 0xFF,
                self.start_pc_hi: (start_pc >> 8) & 0xFF,
                self.is_start: 1,
            },
            lookup_ids={self.start_lu.id},
        )

        # Feed the memory image as WRITE tokens for memory reads.  The graph's
        # memory lookup masks non-exact nearest-address hits to zero, so only
        # nonzero initial bytes need explicit tokens.
        for addr, byte in enumerate(memory_image):
            if byte == 0:
                continue
            self.insert_memory_write(addr, byte, memory_ids=self.memory_ids)

        self.mstate = {
            "pc": start_pc & 0xFFFF, "a": 0, "x": 0, "y": 0, "s": 0xFD,
            "p": reference.FLAG_U | reference.FLAG_I,
            "ir": 0, "oplo": 0, "ophi": 0, "ea": 0, "data": 0, "halted": 0,
        }
        if state is not None:
            self.mstate.update({k: state[k] for k in self.mstate if k in state})
        self.nmi_pending = False
        self.irq_pending = False
        self.machine_halted = False
        self.last_prefix_seconds = time.perf_counter() - prefix_start

    @property
    def machine_state(self) -> dict:
        """Copy of the architectural state the next machine_step starts from."""
        return dict(self.mstate)

    def set_nmi(self):
        """Request an NMI: the next machine_step executes the interrupt sequence."""
        self.nmi_pending = True

    def set_irq(self):
        """Request an IRQ: delivered at the next step with the I flag clear."""
        self.irq_pending = True

    def clear_irq(self):
        self.irq_pending = False

    def inject_write(self, addr: int, value: int):
        """Append a device-originated WRITE token (a value appearing on the bus)."""
        self.insert_memory_write(addr & 0xFFFF, value & 0xFF, memory_ids=self.memory_ids)

    def machine_step(self) -> TraceRow:
        """Insert the next STATE token, evaluate one step, append its writes."""
        st = self.mstate
        nmi = 1 if self.nmi_pending else 0
        self.nmi_pending = False
        p = st["p"]
        irq = 1 if reference.irq_deliverable(p, self.irq_pending, nmi) else 0
        if irq:
            self.irq_pending = False
        self.direct_insert(
            "STATE",
            {
                self.is_state: 1,
                self.in_pc_lo: st["pc"] & 0xFF,
                self.in_pc_hi: (st["pc"] >> 8) & 0xFF,
                self.in_a: st["a"],
                self.in_x: st["x"],
                self.in_y: st["y"],
                self.in_s: st["s"],
                self.in_p_c: self.flag_bit(p, reference.FLAG_C),
                self.in_p_z: self.flag_bit(p, reference.FLAG_Z),
                self.in_p_i: self.flag_bit(p, reference.FLAG_I),
                self.in_p_d: self.flag_bit(p, reference.FLAG_D),
                self.in_p_b: self.flag_bit(p, reference.FLAG_B),
                self.in_p_u: self.flag_bit(p, reference.FLAG_U),
                self.in_p_v: self.flag_bit(p, reference.FLAG_V),
                self.in_p_n: self.flag_bit(p, reference.FLAG_N),
                self.in_ir: st["ir"],
                self.in_oplo: st["oplo"],
                self.in_ophi: st["ophi"],
                self.in_ea_lo: st["ea"] & 0xFF,
                self.in_ea_hi: (st["ea"] >> 8) & 0xFF,
                self.in_data: st["data"],
                self.in_halted: st["halted"],
                self.in_nmi: nmi,
                self.in_irq: irq,
            },
            lookup_ids={self.state_lu.id},
        )

        vals = self.evaluate_current()
        row = self.make_trace_row(vals)
        self.trace_len += 1
        for write in row.writes:
            self.insert_memory_write(write.addr, write.value, memory_ids=self.memory_ids)

        s = row.state
        st.update(
            pc=int(round(self.read_dim("next_pc", vals))) & 0xFFFF,
            a=s.a, x=s.x, y=s.y, s=s.s, p=s.p,
            ir=s.ir, oplo=s.operand_lo, ophi=s.operand_hi,
            ea=s.ea, data=s.data, halted=1 if s.halted else 0,
        )
        if s.halted or self.predict_next(vals) == "HALT":
            self.machine_halted = True
        return row

    def insert_memory_write(self, addr: int, value: int, *, memory_ids: set[int]):
        self.direct_insert(
            "WRITE",
            {
                self.byte_value: value & 0xFF,
                self.is_memory_write: 1,
                self.addr_lo: addr & 0xFF,
                self.addr_hi: (addr >> 8) & 0xFF,
            },
            lookup_ids=memory_ids,
        )

    def make_trace_row(self, vals):
        # The trace row describes the instruction that just executed, so PC is
        # the *current* PC; registers/memory reflect the post-execution state.
        pc = int(round(self.read_dim("current_pc", vals))) & 0xFFFF
        a = int(round(self.read_dim("new_a", vals))) & 0xFF
        x = int(round(self.read_dim("new_x", vals))) & 0xFF
        y = int(round(self.read_dim("new_y", vals))) & 0xFF
        s = int(round(self.read_dim("new_s", vals))) & 0xFF
        p = int(round(self.read_dim("new_p", vals))) & 0xFF
        halted = int(round(self.read_dim("new_halted", vals))) != 0
        ir = int(round(self.read_dim("fetched_opcode", vals))) & 0xFF
        oplo = int(round(self.read_dim("fetched_operand_lo", vals))) & 0xFF
        ophi = int(round(self.read_dim("fetched_operand_hi", vals))) & 0xFF
        ea = int(round(self.read_dim("ea", vals))) & 0xFFFF
        data = int(round(self.read_dim("data_out", vals))) & 0xFF

        writes = []
        for store_dim, addr_dim, value_dim in (
            ("is_store", "write_addr_out", "write_value_out"),
            ("is_store2", "write_addr2_out", "write_value2_out"),
            ("is_store3", "write_addr3_out", "write_value3_out"),
        ):
            if int(round(self.read_dim(store_dim, vals))):
                write_addr = int(round(self.read_dim(addr_dim, vals))) & 0xFFFF
                write_value = int(round(self.read_dim(value_dim, vals))) & 0xFF
                writes.append(MemWrite(self.trace_len, write_addr, write_value))
        writes = tuple(writes)

        return TraceRow(
            step=self.trace_len,
            state=Cpu6502State(
                pc=pc,
                a=a,
                x=x,
                y=y,
                s=s,
                p=p,
                ir=ir,
                operand_lo=oplo,
                operand_hi=ophi,
                ea=ea,
                data=data,
                halted=halted,
            ),
            writes=writes,
        )

    @staticmethod
    def flag_bit(p: int, mask: int) -> int:
        return 1 if (p & mask) != 0 else 0

    def read_dim(self, name: str, values: list[float]) -> float:
        dim_id = self.dim_id_by_name.get(name)
        return values[dim_id] if dim_id is not None else 0.0


def graph_all_dims():
    return graph_core.graph_dims


def graph_all_lookups():
    return graph_core.graph_lookups


def find_builtin_dim(name: str):
    # Module-attribute access on purpose: ``one`` rebinds at reset_graph.
    return getattr(graph_core, name)


def find_input_dim(name: str):
    for d in graph_all_dims():
        if isinstance(d, InputDimension) and d.name == name:
            return d
    raise KeyError(name)


def same_exprs(left, right) -> bool:
    if len(left) != len(right):
        return False
    return all(a.terms == b.terms for a, b in zip(left, right, strict=True))


def eval_indexed_expr(expr, values: list[float]) -> float:
    _n, ids, coeffs = expr
    total = 0.0
    for dim_id, coeff in zip(ids, coeffs, strict=True):
        total += coeff * values[dim_id]
    return total


def run_program(
    program_file: str,
    *,
    start_pc: int = 0x8000,
    max_steps: int = 100,
) -> Trace:
    with open(program_file, "rb") as f:
        program = list(f.read())
    rt = Cpu6502Runtime()
    return rt.run(program, start_pc=start_pc, max_steps=max_steps)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Evaluate 6502 programs through the computation graph (exact arithmetic)."
    )
    parser.add_argument("program", help="Path to a .bin file")
    parser.add_argument("--start-pc", type=lambda x: int(x, 0), default=0x8000)
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()

    trace = run_program(args.program, start_pc=args.start_pc, max_steps=args.max_steps)
    print(trace.format())


if __name__ == "__main__":
    main()
