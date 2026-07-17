"""Construct and run the 6502 analytic transformer weights backend.

The graph is scheduled, converted to weights, and executed by generated C++.
Python owns construction, cache orchestration, and the public machine API;
the native engine owns instruction execution and attention state.
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from transformer_nes.cpu6502.backends import reference
from transformer_nes.cpu6502.backends.graph import RUNTIME_READ_DIMS, Cpu6502Runtime
from transformer_nes.cpu6502.backends.weights.experts import (
    build_opcode_experts,
    build_sequence_expert,
    build_shape_experts,
    head_row,
    step_spec_base,
)
from transformer_nes.cpu6502.backends.weights.windows import (
    SHAPE_LIBRARY,
    match_shape_at,
    match_shape_windows,
    propose_window,
)
from transformer_nes.cpu6502.spec import OPCODES
from transformer_nes.cpu6502.trace import Cpu6502State, MemWrite, Trace, TraceRow
from transformer_nes.graph.core import ProgramGraph
from transformer_nes.model.engine_gen import (
    GeneratedWeightsEngine,
    SequenceModule,
    engine_config_key,
    seqmod_config_key,
)
from transformer_nes.model.weights import build_model
from transformer_nes.scheduler.milp import asap_schedule

logger = logging.getLogger(__name__)

# Opcode-routed experts, on by default: one expert per documented opcode, so
# routing is workload-independent (no game-derived tuning set).  NMI steps and
# any opcode outside a narrowed set fall back to the exact union step.
DEFAULT_EXPERT_OPCODES = tuple(sorted(OPCODES))

# A pc must report hot this many times (re-earning jit_min_count between
# reports) before it earns a pinned-window proposal; the budget bounds the
# next boot's fold cost.
JIT_PROPOSAL_HEAT = 3
JIT_PROPOSAL_CAP = 96


class Cpu6502TransformerMachine:
    """The 6502 as actual transformer weights, stepped position by position."""

    def __init__(self, *, bcd: bool = False,
                 expert_opcodes=DEFAULT_EXPERT_OPCODES,
                 sequence_windows=(),
                 shape_library=SHAPE_LIBRARY,
                 jit_interval: int = 256,
                 jit_min_count: int = 512,
                 expert_schedule: str = "asap",
                 expert_milp_seconds: int = 900):
        # Build the graph (and keep the evaluator's dim handles; its Python
        # step function is not used here).
        self.rt = Cpu6502Runtime(bcd=bcd)
        dim_by_id = {d.id: d for d in self.rt.all_dims}
        self.dim_by_name = {
            name: dim_by_id[dim_id] for name, dim_id in self.rt.dim_id_by_name.items()
        }

        # The run loop reads the machine outputs from the residual stream,
        # so they must survive to the final layer: a synthetic output token
        # marks them as outputs for the scheduler.  Its head row is ignored
        # at prediction time (only STATE vs HALT compete).
        output_tokens = dict(self.rt.output_tokens)
        readout = 0
        for name in RUNTIME_READ_DIMS:
            readout = readout + 1e-6 * self.dim_by_name[name]
        output_tokens["READOUT"] = readout

        pg = ProgramGraph(self.rt.input_tokens, output_tokens)
        self.pg = pg  # kept for the expert folds (graph/fold.py)
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
        self.state_row = tok_to_idx["STATE"]
        self.halt_row = tok_to_idx["HALT"]
        self.read_slots = {
            name: self.slot_of[self.dim_by_name[name]]
            for name in RUNTIME_READ_DIMS
        }
        self.one_slot = self.slot_of[self.rt.one]

        # The whole layer stack -- hull attention plus the structural
        # passthrough fusion -- runs inside the generated C++ engine
        # (model/engine_gen.py): model-specific straight-line code with an
        # exact fixed-point int32 residual.  The grouped memory lookup keeps
        # K-side address scores below 2^24, so float32 Q/K rounding still
        # distinguishes adjacent high addresses.
        if expert_schedule not in ("asap", "milp"):
            raise ValueError(f"unknown expert_schedule {expert_schedule!r}")
        self.expert_schedule = expert_schedule
        self.expert_milp_seconds = expert_milp_seconds
        # The core engine is determined by source, bcd, and expert set—not
        # the ROM's sequence windows—so it
        # compiles once per code digest and every ROM shares it; a manifest
        # hit skips the expert folds and the emission and loads the compiled
        # artifact directly.  MILP schedules are excluded: their incumbents
        # depend on solve time.
        windows = [
            (int(head) & 0xFFFF, bytes(code), int(repeat))
            for head, code, repeat in (sequence_windows or ())
        ]
        shapes = tuple(shape_library or ())
        cache_key = None
        engine = None
        if expert_schedule == "asap":
            expert_bytes = [
                int(b) for b in (expert_opcodes or ()) if int(b) in OPCODES
            ]
            shape_tag = ",".join(
                f"{name}:{bytes(t).hex()}x{r}" for name, t, r in shapes
            )
            cache_key = engine_config_key(
                bcd=bcd, expert_bytes=expert_bytes,
                extra=f"shapes={shape_tag}",
            )
            engine = GeneratedWeightsEngine.from_cache(cache_key)
            if engine is not None:
                logger.info("engine cache hit (%d experts)", engine.n_experts)
        if engine is None:
            shapes_meta: list[dict] = []
            experts = build_opcode_experts(
                self.pg, expert_opcodes,
                schedule=self.expert_schedule,
                milp_seconds=self.expert_milp_seconds,
            )
            if shapes:
                shape_experts, shapes_meta = build_shape_experts(self.pg, shapes)
                experts.extend(shape_experts)
            engine = GeneratedWeightsEngine(
                self.model, step_spec=self.step_spec(),
                experts=experts, cache_key=cache_key,
                shapes_meta=shapes_meta,
            )
        self.engine = engine
        # The ROM's windows land in a tiny per-ROM module attached through
        # the core's function-pointer hook (no core recompile per ROM).
        self.seq_module = None
        if not self.engine.has_machine_step:
            raise RuntimeError("generated 6502 engine is missing machine_step")
        if not self.engine.supports_direct_writes:
            raise RuntimeError("generated 6502 engine is missing direct memory writes")
        if windows:
            self.attach_sequence_module(windows, bcd=bcd)
        self.step_out = np.empty(21, dtype=np.int64)

        # Profile-guided JIT: every jit_interval native run() returns, pcs
        # with >= jit_min_count fallback steps since the last poll are
        # matched against the shape library at their live RAM bytes --
        # discovering windows the boot scan could not see (code copied to
        # RAM at runtime, hot straight-line idioms).
        self.jit_interval = int(jit_interval)
        self.jit_min_count = int(jit_min_count)
        self.jit_calls = 0
        # Per-pc re-report count: a pc re-enters jit_hot only after
        # re-accumulating min_count steps, so heat measures SUSTAINED
        # execution, not one burst (a level-load memcpy reports once).
        self.jit_heat: dict[int, int] = {}
        # Sustained-hot pcs become pinned-window PROPOSALS for the next boot.
        # This includes canonical shape hits: shapes are the zero-compile
        # baseline, while a hot concrete instance deserves a smaller model
        # with its operands folded.  Mid-run compilation would stall frames.
        self.jit_proposals: dict[int, tuple[int, bytes, int]] = {}
        # Shape matches per injected (base, bank content): a bank switch
        # re-adds its windows from this memo instead of rescanning.
        self.region_windows: dict[tuple[int, bytes], tuple] = {}

        self.pos = 0
        self.trace_len = 0
        # RAM mirror: shape-window scans and the machine-protocol paths
        # read it (the native run loop keeps its own copy).
        self.ram = bytearray(0x10000)

    # ── Transformer plumbing ────────────────────────────────────────

    def embed_array(self, values: dict) -> np.ndarray:
        """Engine input embedding: dim values at their residual slots.

        Equivalent to a vocabulary embedding row for a token type that
        enumerated exactly these values.  There is no position encoding:
        order enters only through the LATEST tie-break.
        """
        x = np.zeros(self.d_model, dtype=np.float32)
        x[self.one_slot] = 1.0
        for dim, value in values.items():
            x[self.slot_of[dim]] = float(value)
        return x

    def forward_values(self, values: dict) -> np.ndarray:
        """One genuine transformer position: embed and run the whole layer
        stack (attention + ReGLU) in C++."""
        x = self.embed_array(values)
        self.pos += 1
        self.engine.forward(x)  # in place, whole stack
        return x

    def attach_sequence_module(self, windows, *, bcd: bool) -> None:
        """Compile-or-load the per-ROM sequence module and attach its window
        dispatch to the core engine.  A new ROM's window set costs one small
        TU (cached by digest); the 43 MB core is never re-emitted for it."""
        key = seqmod_config_key(
            core_digest=self.engine.digest, bcd=bcd, windows=windows
        )
        mod = SequenceModule.from_cache(key, core_digest=self.engine.digest)
        if mod is None:
            experts = []
            for w in windows:
                # The scanner proposes shapes; the fold's license checks
                # arbitrate.  A rejected window only loses its fast path.
                try:
                    experts.append(build_sequence_expert(self.pg, *w))
                except RuntimeError as e:
                    logger.info("sequence window skipped: %s", e)
            if not experts:
                return
            mod = SequenceModule(
                self.model, self.step_spec(), experts,
                core_digest=self.engine.digest, cache_key=key,
            )
        else:
            logger.info("sequence module cache hit (max window %d steps)", mod.max_k)
        self.seq_module = mod
        heads = b"".join(
            head.to_bytes(2, "little")
            for head in sorted({head for head, _code, _repeat in windows})
        )
        self.engine.attach_seq(mod.fn_ptr, mod.ctx_ptr, heads)

    def populate_shape_windows(self) -> None:
        """Fill the engine's runtime window table from the current RAM image
        (called at every boot/reseed; dispatch re-verifies bytes live, so a
        stale entry is only a missed fast path)."""
        self.jit_calls = 0
        self.jit_heat.clear()
        meta = getattr(self.engine, "shapes_meta", None) or []
        if not (meta and self.engine.has_run_loop):
            return
        self.engine.clear_windows()
        for sid, head in match_shape_windows(self.ram, meta):
            self.engine.add_window(sid, head)

    def jit_tick(self) -> None:
        """Periodic JIT poll after a native run: table-add any hot fallback
        pc whose bytes match a library shape, read from the ENGINE's live RAM
        (the native run loop's writes never reach the Python mirror -- a
        runtime-copied loop exists only engine-side)."""
        meta = getattr(self.engine, "shapes_meta", None) or []
        if not meta:
            return
        self.jit_calls += 1
        if self.jit_calls % self.jit_interval:
            return
        for pc in self.engine.jit_hot(self.jit_min_count):
            pc = int(pc)
            heat = self.jit_heat.get(pc, 0) + 1
            self.jit_heat[pc] = heat
            if heat > 1:
                continue  # shapes were checked at first heat; proposals below
            sid = match_shape_at(self.engine.read_ram, pc, meta)
            if sid is not None:
                self.engine.add_window(sid, pc)
                logger.info("jit window: %s at %04X", meta[sid]["name"], pc)
        # Proposals demand sustained heat, a bounded budget, and no overlap
        # with an already-proposed window (suffix pcs of one hot run would
        # each propose their own tail otherwise).
        for pc, heat in self.jit_heat.items():
            if heat != JIT_PROPOSAL_HEAT or len(self.jit_proposals) >= JIT_PROPOSAL_CAP:
                continue
            if pc in self.jit_proposals or any(
                head <= pc < head + len(code)
                for head, code, _r in self.jit_proposals.values()
            ):
                continue
            proposal = propose_window(self.engine.read_ram, pc, max_len=6)
            if proposal is not None:
                self.jit_proposals[pc] = proposal
                logger.info("jit proposal: %d bytes at %04X (next boot)",
                            len(proposal[1]), pc)

    def step_spec(self) -> dict:
        """Slot map for the generated C++ machine_step (engine_gen.py)."""
        spec = step_spec_base(
            lambda name: self.slot_of[self.dim_by_name[name]]
        )
        spec["outputs"] = dict(self.read_slots)
        spec["halt_head"] = head_row(self.model, self.halt_row)
        spec["state_head"] = head_row(self.model, self.state_row)
        return spec

    # ── Machine protocol (mirrors evaluator.Cpu6502Runtime, so the NES
    # bus's GraphCore adapter can drive this machine unchanged) ──────

    def boot_image(self, memory_image, *, start_pc: int, state: dict | None = None):
        # Fresh token log: clear the engine's attention state so a machine
        # instance can be reused for another program.
        self.engine.reset()
        self.pos = 0
        self.trace_len = 0
        ram = bytearray(0x10000)
        ram[: len(memory_image)] = bytes(memory_image)
        self.ram = ram
        rt = self.rt
        self.forward_values({
            rt.start_pc_lo: start_pc & 0xFF,
            rt.start_pc_hi: (start_pc >> 8) & 0xFF,
            rt.is_start: 1,
        })
        # Native seed: one call for the image instead of one per byte.
        self.pos += int(self.engine.seed_image(bytes(ram)))
        self.mstate = {
            "pc": start_pc & 0xFFFF, "a": 0, "x": 0, "y": 0, "s": 0xFD,
            "p": reference.FLAG_U | reference.FLAG_I,
            "ir": 0, "oplo": 0, "ophi": 0, "ea": 0, "data": 0, "halted": 0,
        }
        if state is not None:
            self.mstate.update({k: int(state[k]) for k in self.mstate if k in state})
        self.nmi_pending = False
        self.irq_pending = False
        self.machine_halted = False
        self.populate_shape_windows()

    @property
    def machine_state(self) -> dict:
        return dict(self.mstate)

    def set_nmi(self):
        """The interrupt sequence itself then runs inside the weights."""
        self.nmi_pending = True
        # In the native run loop the pending-NMI flag lives in the engine
        # (the per-step path passes it as an argument instead).
        if self.engine.has_run_loop:
            self.engine.set_nmi()

    def set_irq(self):
        """Delivered at the next step boundary with the I flag clear."""
        self.irq_pending = True
        if self.engine.has_run_loop:
            self.engine.set_irq()

    def clear_irq(self):
        self.irq_pending = False
        if self.engine.has_run_loop:
            self.engine.clear_irq()

    def inject_write(self, addr: int, value: int):
        self.insert_write(addr & 0xFFFF, value & 0xFF)

    def inject_region(self, base: int, data: bytes):
        """Bulk injection (mapper bank switches): one WRITE token per byte
        engine-side, so the position mirror advances by len(data)."""
        base &= 0xFFFF
        data = bytes(data)
        self.ram[base:base + len(data)] = data
        self.engine.inject_region(base, data)
        self.pos += len(data)

    def map_region(self, base: int, data: bytes):
        """Replace a mapped cartridge region without creating WRITE tokens
        (the engine's map_write stamps state in place; no sequence advance)."""
        self.engine.map_region(base, data)
        # Newly banked-in code gets its shape windows immediately instead
        # of waiting out JIT warmup; matches are memoized per bank content.
        meta = getattr(self.engine, "shapes_meta", None) or []
        if meta and self.engine.has_run_loop:
            key = (base, data)
            hits = self.region_windows.get(key)
            if hits is None:
                hits = self.region_windows[key] = tuple(
                    match_shape_windows(data, meta, base=base)
                )
            for sid, head in hits:
                self.engine.add_window(sid, head)

    def insert_write(self, addr: int, value: int):
        addr &= 0xFFFF
        value &= 0xFF
        self.ram[addr] = value
        self.engine.insert_write(addr, value, self.pos)
        self.pos += 1

    def machine_step(self) -> TraceRow:
        """One instruction entirely in generated C++ (embed, layers,
        readout, direct WRITE inserts, halt logits); Python only unpacks
        the 21-slot result and keeps the mirrors in sync."""
        st = self.mstate
        nmi = 1 if self.nmi_pending else 0
        self.nmi_pending = False
        irq = 1 if reference.irq_deliverable(st["p"], self.irq_pending, nmi) else 0
        if irq:
            self.irq_pending = False
        self.engine.machine_step(
            st["pc"], st["a"], st["x"], st["y"], st["s"], st["p"],
            st["ir"], st["oplo"], st["ophi"], st["ea"], st["data"],
            st["halted"], nmi, irq, self.step_out,
        )
        o = self.step_out.tolist()  # one C call instead of 21 scalar reads
        n_writes = o[14]
        writes = []
        for i in range(n_writes):
            addr, value = o[15 + 2 * i], o[16 + 2 * i]
            self.ram[addr] = value
            writes.append(MemWrite(self.trace_len, addr, value))
        self.pos += 1 + n_writes

        halted = o[11] != 0
        row = TraceRow(
            step=self.trace_len,
            state=Cpu6502State(
                pc=o[0], a=o[1], x=o[2], y=o[3], s=o[4], p=o[5], ir=o[6],
                operand_lo=o[7], operand_hi=o[8],
                ea=o[9], data=o[10], halted=halted,
            ),
            writes=tuple(writes),
        )
        self.trace_len += 1
        st.update(
            pc=o[12], a=o[1], x=o[2], y=o[3], s=o[4], p=o[5], ir=o[6],
            oplo=o[7], ophi=o[8], ea=o[9], data=o[10],
            halted=1 if halted else 0,
        )
        if halted or o[13]:
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

    # ── C++ run loop ────────────────────────────────────────────────
    # Native streaming API used by Klaus, nestest, and the NES console.

    @property
    def supports_run_loop(self) -> bool:
        return self.engine.has_run_loop

    def seed_run_state(self) -> None:
        st = self.mstate
        self.engine.set_run_state(
            st["pc"], st["a"], st["x"], st["y"], st["s"], st["p"],
            st["ir"], st["oplo"], st["ophi"], st["ea"], st["data"], st["halted"],
        )

    def pull_run_state(self) -> None:
        """Copy the engine's architectural state back into the Python mirror
        (called after a native run so the machine protocol sees it)."""
        s = self.engine.get_run_state()
        self.mstate.update(
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

        Returns the run-protocol result dict: per-step
        trace arrays (pc/a/x/y/s/p/ir/oplo/ophi/ea/data/halted), write arrays
        (w_row 0-based within this call, w_addr, w_value), and the run's
        "reason"/"cycles".  Materializes every step's arrays, so it is for
        bounded runs (nestest, a Klaus prefix); use iter_run_arrays to stream
        the full suite.
        """
        if not self.supports_run_loop:
            raise RuntimeError("this engine has no C++ run loop")
        self.boot_image(list(memory_image), start_pc=start_pc)
        self.seed_run_state()
        out = self.engine.run(max_steps, stop_on_trap, 0)
        self.pull_run_state()
        return out

    def iter_run_arrays(self, memory_image, *, start_pc: int, max_steps: int,
                        chunk: int = 200_000, stop_on_trap: bool = False):
        """Stream the C++ run loop in fixed-size chunks (w_row rebased to 0
        per chunk), so the full Klaus suite verifies without materializing a
        30M-row trace.  Yields the run-protocol result dict for each chunk.

        Engine state is O(1) in the token log (memory on the exact
        nearest-address structure, constant-query heads on running-argmax
        registers), so streaming needs no checkpoint rebuild between chunks.
        """
        if not self.supports_run_loop:
            raise RuntimeError("this engine has no C++ run loop")
        self.boot_image(list(memory_image), start_pc=start_pc)
        self.seed_run_state()
        done = 0
        while done < max_steps:
            base = int(self.engine.get_run_state()["steps"])
            out = self.engine.run(min(chunk, max_steps - done), stop_on_trap, 0)
            n = len(out["pc"])
            if n == 0:
                break
            # w_row is the engine's global step_index; rebase to this chunk.
            out["w_row"] = out["w_row"] - base
            done += n
            yield out
            if out["reason"] in ("halt", "trap"):
                break
        self.pull_run_state()

    # ── Machine API (lets the NES console drive the run loop) ────────
    # The run-protocol machine surface: the native console's
    # PythonEngineMachine (nes/bindings.cpp) drives the weights engine,
    # pausing back to Python only for MMIO (run_protocol.py).

    def load_image(self, memory_image, *, start_pc: int) -> None:
        self.boot_image(list(memory_image), start_pc=start_pc)
        self.seed_run_state()

    def run(self, max_steps: int, *, stop_on_trap: bool = False,
            max_cycles: int = 0, trace: bool = True) -> dict:
        """Resume the native run loop from the engine's persistent state
        (no reboot), pausing on halt/trap/watch/cycle budget."""
        out = self.engine.run(max_steps, stop_on_trap, max_cycles, trace)
        self.pull_run_state()
        # A watch_write pause ends on the step that wrote, so this is the
        # writing opcode: the bus routes RMW writes through write_rmw.
        out["pause_ir"] = self.mstate["ir"]
        self.jit_tick()
        return out

    def watch_read(self, lo: int, hi: int) -> None:
        self.engine.watch_read(lo, hi)

    def watch_write(self, lo: int, hi: int) -> None:
        self.engine.watch_write(lo, hi)

    def set_write_journal(self, groups: bytes) -> None:
        self.engine.set_write_journal(bytes(groups))

    def set_read_lease(self, addr: int, value: int, group: int = 0) -> int:
        return int(self.engine.set_read_lease(addr, value, group))

    def set_read_lease_seq(self, addr: int, values: bytes, group: int = 0) -> int:
        return int(self.engine.set_read_lease_seq(addr, bytes(values), group))

    def take_lease_served(self, addr: int) -> int:
        return int(self.engine.take_lease_served(addr))

    def clear_read_lease(self) -> None:
        self.engine.clear_read_lease()

    def clear_read_lease_addr(self, addr: int) -> None:
        self.engine.clear_read_lease_addr(addr)

    def set_mapper_mmc3(self, prg: bytes, bank_select: int, regs: bytes,
                        cur: bytes) -> None:
        self.engine.set_mapper_mmc3(bytes(prg), bank_select, bytes(regs),
                                    bytes(cur))

    def set_mapper_mmc1(self, prg: bytes, shift: int, control: int,
                        prg_reg: int, lo: int, hi: int) -> None:
        self.engine.set_mapper_mmc1(bytes(prg), shift, control, prg_reg, lo, hi)

    def clear_mapper(self) -> None:
        self.engine.clear_mapper()

    def read_ram(self, addr: int) -> int:
        return self.engine.read_ram(addr)

    def dump_ram(self) -> bytes:
        return self.engine.dump_ram()

    def get_state(self) -> dict:
        return dict(self.engine.get_run_state())

    def set_state(self, state: dict) -> None:
        """Restore architectural state after a snapshot rebuild (mirrors the
        run protocol's set_state; the trace context fields carry over)."""
        st = dict(self.mstate)
        for k in st:
            if k in state:
                st[k] = int(state[k])
        self.engine.set_run_state(
            st["pc"], st["a"], st["x"], st["y"], st["s"], st["p"],
            st["ir"], st["oplo"], st["ophi"], st["ea"], st["data"],
            1 if state.get("halted") else 0,
        )
        self.mstate.update(st)
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
        "--no-experts", action="store_true",
        help="disable opcode-routed generated experts and run the union model",
    )
    args = parser.parse_args()

    machine = Cpu6502TransformerMachine(
        bcd=False, expert_opcodes=() if args.no_experts else DEFAULT_EXPERT_OPCODES,
    )
    logger.info(
        "transformer: d_model=%d n_layers=%d exact-int32 (%.1fs build)",
        machine.d_model,
        machine.n_layers,
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
