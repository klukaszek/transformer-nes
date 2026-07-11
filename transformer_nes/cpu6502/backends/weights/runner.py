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
from transformer_nes.cpu6502.backends.evaluator import RUNTIME_READ_DIMS, Cpu6502Runtime
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
)
from transformer_nes.cpu6502.spec import OPCODES
from transformer_nes.cpu6502.trace import Cpu6502State, MemWrite, Trace, TraceRow
from transformer_nes.graph.core import ProgramGraph
from transformer_nes.model.weights import add_position_encoding, build_model
from transformer_nes.scheduler.milp import asap_schedule

logger = logging.getLogger(__name__)

# Opcode-routed experts, on by default: one expert per documented opcode, so
# routing is workload-independent (no game-derived tuning set).  NMI steps and
# any opcode outside a narrowed set fall back to the exact union step.
DEFAULT_EXPERT_OPCODES = tuple(sorted(OPCODES))


class Cpu6502TransformerMachine:
    """The 6502 as actual transformer weights, stepped position by position."""

    def __init__(self, *, bcd: bool = False,
                 checkpoint_tokens: int | None = None,
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
        self._dim_by_name = {
            name: dim_by_id[dim_id] for name, dim_id in self.rt.dim_id_by_name.items()
        }

        # The run loop reads the machine outputs from the residual stream,
        # so they must survive to the final layer: a synthetic output token
        # marks them as outputs for the scheduler.  Its head row is ignored
        # at prediction time (only STATE vs HALT compete).
        output_tokens = dict(self.rt.output_tokens)
        readout = 0
        for name in RUNTIME_READ_DIMS:
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
            for name in RUNTIME_READ_DIMS
        }
        self._one_slot = self.slot_of[self.rt.one]

        # The whole layer stack -- hull attention plus the structural
        # passthrough fusion -- runs inside the generated C++ engine
        # (model/engine_gen.py): model-specific straight-line code with an
        # exact fixed-point int32 residual.  The grouped memory lookup keeps
        # K-side address scores below 2^24, so float32 Q/K rounding still
        # distinguishes adjacent high addresses.
        from transformer_nes.model.engine_gen import (
            GeneratedWeightsEngine,
            engine_config_key,
        )

        if expert_schedule not in ("asap", "milp"):
            raise ValueError(f"unknown expert_schedule {expert_schedule!r}")
        self._expert_schedule = expert_schedule
        self._expert_milp_seconds = expert_milp_seconds
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
        self._engine = None
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
            self._engine = GeneratedWeightsEngine.from_cache(cache_key)
            if self._engine is not None:
                logger.info("engine cache hit (%d experts)", self._engine.n_experts)
        if self._engine is None:
            shapes_meta: list[dict] = []
            experts = build_opcode_experts(
                self._pg, expert_opcodes,
                schedule=self._expert_schedule,
                milp_seconds=self._expert_milp_seconds,
            )
            if shapes:
                shape_experts, shapes_meta = build_shape_experts(self._pg, shapes)
                experts.extend(shape_experts)
            self._engine = GeneratedWeightsEngine(
                self.model, step_spec=self._step_spec(),
                experts=experts, cache_key=cache_key,
                shapes_meta=shapes_meta,
            )
        # The ROM's windows land in a tiny per-ROM module attached through
        # the core's function-pointer hook (no core recompile per ROM).
        self._seq_module = None
        if not self._engine.has_machine_step:
            raise RuntimeError("generated 6502 engine is missing machine_step")
        if not self._engine.supports_direct_writes:
            raise RuntimeError("generated 6502 engine is missing direct memory writes")
        if windows:
            self._attach_sequence_module(windows, bcd=bcd)
        self._step_out = np.empty(21, dtype=np.int64)

        # Profile-guided JIT: every jit_interval native run() returns, pcs
        # with >= jit_min_count fallback steps since the last poll are
        # matched against the shape library at their live RAM bytes --
        # discovering windows the boot scan could not see (code copied to
        # RAM at runtime, hot straight-line idioms).
        self._jit_interval = int(jit_interval)
        self._jit_min_count = int(jit_min_count)
        self._jit_calls = 0
        self._jit_rejected: set[int] = set()

        self.pos = 0
        self._trace_len = 0
        # RAM mirror for optional snapshot/reseed checkpoints.
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
        x = np.zeros(self.d_model, dtype=np.float32)
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

    def _attach_sequence_module(self, windows, *, bcd: bool) -> None:
        """Compile-or-load the per-ROM sequence module and attach its window
        dispatch to the core engine.  A new ROM's window set costs one small
        TU (cached by digest); the 43 MB core is never re-emitted for it."""
        from transformer_nes.model.engine_gen import (
            SequenceModule,
            seqmod_config_key,
        )

        key = seqmod_config_key(
            core_digest=self._engine.digest, bcd=bcd, windows=windows
        )
        mod = SequenceModule.from_cache(key, core_digest=self._engine.digest)
        if mod is None:
            experts = []
            for w in windows:
                # The scanner proposes shapes; the fold's license checks
                # arbitrate.  A rejected window only loses its fast path.
                try:
                    experts.append(build_sequence_expert(self._pg, *w))
                except RuntimeError as e:
                    logger.info("sequence window skipped: %s", e)
            if not experts:
                return
            mod = SequenceModule(
                self.model, self._step_spec(), experts,
                core_digest=self._engine.digest, cache_key=key,
            )
        else:
            logger.info("sequence module cache hit (max window %d steps)", mod.max_k)
        self._seq_module = mod
        self._engine.attach_seq(mod.fn_ptr, mod.ctx_ptr)

    def _populate_shape_windows(self) -> None:
        """Fill the engine's runtime window table from the current RAM image
        (called at every boot/reseed; dispatch re-verifies bytes live, so a
        stale entry is only a missed fast path)."""
        self._jit_calls = 0
        self._jit_rejected.clear()
        meta = getattr(self._engine, "shapes_meta", None) or []
        if not (meta and self._engine.has_run_loop):
            return
        self._engine.clear_windows()
        for sid, head in match_shape_windows(self._ram, meta):
            self._engine.add_window(sid, head)

    def _jit_tick(self) -> None:
        """Periodic JIT poll after a native run: table-add any hot fallback
        pc whose bytes match a library shape, read from the ENGINE's live RAM
        (the native run loop's writes never reach the Python mirror -- a
        runtime-copied loop exists only engine-side)."""
        meta = getattr(self._engine, "shapes_meta", None) or []
        if not meta:
            return
        self._jit_calls += 1
        if self._jit_calls % self._jit_interval:
            return
        for pc in self._engine.jit_hot(self._jit_min_count):
            pc = int(pc)
            if pc in self._jit_rejected:
                continue
            sid = match_shape_at(self._engine.read_ram, pc, meta)
            if sid is None:
                self._jit_rejected.add(pc)
            else:
                self._engine.add_window(sid, pc)
                logger.info("jit window: %s at %04X", meta[sid]["name"], pc)

    def _step_spec(self) -> dict:
        """Slot map for the generated C++ machine_step (engine_gen.py)."""
        spec = step_spec_base(
            lambda name: self.slot_of[self._dim_by_name[name]]
        )
        spec["outputs"] = dict(self._read_slots)
        spec["halt_head"] = head_row(self.model, self._halt_row)
        spec["state_head"] = head_row(self.model, self._state_row)
        return spec

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
        self.mstate = {
            "pc": start_pc & 0xFFFF, "a": 0, "x": 0, "y": 0, "s": 0xFD,
            "p": reference.FLAG_U | reference.FLAG_I,
            "ir": 0, "oplo": 0, "ophi": 0, "ea": 0, "data": 0, "halted": 0,
        }
        if state is not None:
            self.mstate.update({k: int(state[k]) for k in self.mstate if k in state})
        self.nmi_pending = False
        self.machine_halted = False
        self._populate_shape_windows()

    @property
    def machine_state(self) -> dict:
        return dict(self.mstate)

    def set_nmi(self):
        """The interrupt sequence itself then runs inside the weights."""
        self.nmi_pending = True
        # In the native run loop the pending-NMI flag lives in the engine
        # (the per-step path passes it as an argument instead).
        if self._engine.has_run_loop:
            self._engine.set_nmi()

    def inject_write(self, addr: int, value: int):
        self._insert_write(addr & 0xFFFF, value & 0xFF)

    def _insert_write(self, addr: int, value: int):
        addr &= 0xFFFF
        value &= 0xFF
        self._ram[addr] = value
        self._engine.insert_write(addr, value, self.pos)
        self.pos += 1

    def _machine_step_cpp(self) -> TraceRow:
        """One instruction entirely in generated C++ (embed, layers,
        readout, direct WRITE inserts, halt logits); Python only unpacks
        the 21-slot result and keeps the mirrors in sync."""
        st = self.mstate
        nmi = 1 if self.nmi_pending else 0
        self.nmi_pending = False
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
        """Snapshot and reseed attention state from the RAM mirror.

        Mirrors KernelBus.checkpoint: captures the architectural state,
        reseeds attention with a fresh WRITE-token prefix from the mirror,
        and restores the state. Semantically
        invisible: unwritten and zero bytes both read as 0 through the
        exact-address mask, so dropping the history and reseeding only the
        nonzero RAM bytes preserves every future read.
        """
        if self.machine_halted:
            return False
        state = self.machine_state
        trace_len = self._trace_len
        nmi = self.nmi_pending
        self.boot_image(self._ram, start_pc=state["pc"], state=state)
        self._trace_len = trace_len
        self.nmi_pending = nmi
        self.checkpoints += 1
        return True

    def machine_step(self) -> TraceRow:
        if (
            self._checkpoint_tokens
            and self.pos - self._seed_pos >= self._checkpoint_tokens
        ):
            self.checkpoint()
        return self._machine_step_cpp()

    def run_image(self, memory_image, *, start_pc: int, max_steps: int) -> Trace:
        self.boot_image(list(memory_image), start_pc=start_pc)
        trace = Trace()
        for _ in range(max_steps):
            trace.append(self.machine_step())
            if self.machine_halted:
                break
        return trace

    # ── C++ run loop (KernelBus shape) ──────────────────────────────
    # Native streaming API used by Klaus, nestest, and KernelBus.

    @property
    def supports_run_loop(self) -> bool:
        return self._engine.has_run_loop

    def _seed_run_state(self) -> None:
        st = self.mstate
        self._engine.set_run_state(
            st["pc"], st["a"], st["x"], st["y"], st["s"], st["p"],
            st["ir"], st["oplo"], st["ophi"], st["ea"], st["data"], st["halted"],
        )

    def _pull_run_state(self) -> None:
        """Copy the engine's architectural state back into the Python mirror
        (called after a native run so machine protocol / checkpointing see it)."""
        s = self._engine.get_run_state()
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
        self._jit_tick()
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
        st = dict(self.mstate)
        for k in st:
            if k in state:
                st[k] = int(state[k])
        self._engine.set_run_state(
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
