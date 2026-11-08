"""Construct and run the 6502 analytic transformer weights backend.

The graph is scheduled, converted to analytic weights and compiled through scalar SSA.
Python owns construction, caching and the public machine API; the C++ shell
owns scheduling, architectural readout and device arbitration.
"""

from __future__ import annotations

import argparse
import functools
import gc
import logging
import time

from transformer_nes.cpu6502.engine import (
    GeneratedWeightsEngine,
    engine_config_key,
)
from transformer_nes.cpu6502.execution import step_spec_base
from transformer_nes.cpu6502.experts import build_core_experts
from transformer_nes.cpu6502.graph import RUNTIME_READ_DIMS, Cpu6502Graph
from transformer_nes.cpu6502.shapes import match_shape_windows, with_loops
from transformer_nes.cpu6502.spec import FLAG_I, FLAG_U
from transformer_nes.cpu6502.trace import Cpu6502State, MemWrite, Trace, TraceRow
from transformer_nes.graph.fold import GraphFolder
from transformer_nes.model.schedule import asap_schedule
from transformer_nes.model.weights import build_model

logger = logging.getLogger(__name__)


class Cpu6502TransformerMachine:
    """The 6502 as actual transformer weights, stepped position by position."""

    def __init__(
        self,
        *,
        bcd: bool = False,
        union_only: bool = False,
        mirror_cpu_ram: bool = False,
    ):
        self.bcd = bool(bcd)
        self.mirror_cpu_ram = bool(mirror_cpu_ram)
        # The shell embeds inputs and commits neural readouts. The complete
        # immutable Torch bank is attached before the machine can execute.
        cache_key = engine_config_key(
            bcd=bcd,
            union_only=union_only,
            mirror_cpu_ram=self.mirror_cpu_ram,
        )
        engine = GeneratedWeightsEngine.from_cache(cache_key)
        if engine is not None:
            logger.info("engine cache hit (%d experts)", engine.n_experts)
        if engine is None:
            # Union diagnostics run whole instructions until a device read
            # requires the same retained continuation as production.
            core, meta = self.core_experts()
            experts, shapes_meta = (
                (
                    [
                        e | {"hook_id": i, "external_only": e.get("read_resume", False)}
                        for i, e in enumerate(core)
                        if e["kind"] == "read_request" or e.get("read_resume")
                    ],
                    [],
                )
                if union_only
                else (core, meta)
            )
            engine = GeneratedWeightsEngine(
                self.model,
                step_spec=self.step_spec(),
                experts=experts,
                cache_key=cache_key,
                shapes_meta=shapes_meta,
                mirror_cpu_ram=self.mirror_cpu_ram,
                union_id=len(core),
            )
            # Execution owns its compiled weights and embedding metadata.
            # Rebuild the dense reference model only if a caller asks for it.
            for cached in ("built", "execution_fold", "dim_by_name", "pg", "graph"):
                self.__dict__.pop(cached, None)
            del core, meta, experts, shapes_meta
            # Collect compiler graph cycles before the first gameplay frame.
            gc.collect()
        self.engine = engine
        from transformer_nes.cpu6502.bank import ForwardBank

        self.forwards = ForwardBank(self)

    # ── The graph and the union model, built when needed ────────────
    # (a cached engine needs neither: the folds, a rebuild and the X-ray do)

    @functools.cached_property
    def graph(self) -> Cpu6502Graph:
        return Cpu6502Graph(bcd=self.bcd, wide_bits=True)

    @functools.cached_property
    def pg(self):
        return self.graph.program_graph()

    @functools.cached_property
    def execution_fold(self):
        return GraphFolder(self.pg, root_names=RUNTIME_READ_DIMS, structural_reads=True).fold(
            {}, label="union"
        )

    @functools.cached_property
    def dim_by_name(self) -> dict:
        return self.execution_fold.dim_of_name

    @functools.cached_property
    def built(self) -> tuple:
        """(the union model, its token rows, its build info)."""
        graph = self.execution_fold.program_graph
        sched = asap_schedule(graph)
        return build_model(program_graph=graph, schedule=sched)

    model = property(lambda self: self.built[0])
    slot_of = property(lambda self: self.built[2].slot_of)
    lives = property(lambda self: self.built[2].lives)  # residual live ranges (nes/xray.py)

    def core_experts(self) -> tuple[list[dict], list[dict]]:
        """The core engine's experts in its build order (the forward hook's
        ids 0..), and the shapes' metadata (experts.py:build_core_experts;
        folds cached)."""
        # Imported here: the fold cache shapes no engine (transformer_nes/build.py).
        from transformer_nes.cpu6502.folds import cached_fold

        experts, meta = cached_fold(f"core_bcd{int(self.bcd)}", lambda: build_core_experts(self.pg))
        return with_loops(experts), meta

    def step_spec(self) -> dict:
        """Slot map for the generated C++ union step (engine.py)."""
        spec = step_spec_base(lambda name: self.slot_of[self.dim_by_name[name]])
        spec["outputs"] = {name: self.slot_of[self.dim_by_name[name]] for name in RUNTIME_READ_DIMS}
        return spec

    # ── Machine protocol used by the NES bus and test harness ────────

    def load_image(self, memory_image, *, start_pc: int) -> None:
        """Reset the machine and seed its memory and architectural state."""
        self.engine.reset()
        ram = bytearray(0x10000)
        ram[: len(memory_image)] = bytes(memory_image)
        if self.mirror_cpu_ram:
            for addr in range(0x0800, 0x2000):
                ram[addr] = ram[addr & 0x07FF]
        self.engine.seed_image(bytes(ram))
        self.engine.set_run_state(start_pc, 0, 0, 0, 0xFD, FLAG_U | FLAG_I, 0, 0, 0, 0, 0, 0)
        self.engine.clear_windows()
        for sid, head in match_shape_windows(ram, self.engine.shapes_meta):
            self.engine.add_window(sid, head)

    def trace_union(self):
        return self.forwards.trace_union(), 1

    def set_nmi(self):
        """The interrupt sequence itself runs inside the weights."""
        self.engine.set_nmi()

    def set_irq(self):
        self.engine.set_irq()

    def clear_irq(self):
        self.engine.clear_irq()

    def inject_write(self, addr: int, value: int):
        self.engine.inject_write(addr, value)

    def map_region(self, base: int, data: bytes):
        """Replace a mapped cartridge region without creating WRITE tokens
        (the engine's map_write stamps state in place; no sequence advance)."""
        self.engine.map_region(base, data)
        for sid, head in self.region_windows(base, data):
            self.engine.add_window(sid, head)

    def region_windows(self, base: int, data: bytes) -> tuple:
        """Match newly mapped code. The native console owns bank-content caching."""
        return tuple(match_shape_windows(data, self.engine.shapes_meta, base=base))

    def machine_abi(self) -> tuple[int, int, int]:
        """(version, engine, table) for native callers (native/machine_abi.h).
        This object must outlive its native callers."""
        return self.engine.machine_abi()

    def run_image(self, memory_image, *, start_pc: int, max_steps: int) -> Trace:
        """The native run's arrays as the canonical reference trace."""
        out = self.run_image_arrays(memory_image, start_pc=start_pc, max_steps=max_steps)
        writes = [[] for _ in out["pc"]]
        for step, addr, value, original in zip(
            out["w_row"], out["w_addr"], out["w_value"], out["w_original"], strict=True
        ):
            writes[int(step)].append(MemWrite(int(step), int(addr), int(value), int(original)))
        return Trace(
            [
                TraceRow(
                    i,
                    Cpu6502State(
                        pc=int(out["pc"][i]),
                        a=int(out["a"][i]),
                        x=int(out["x"][i]),
                        y=int(out["y"][i]),
                        s=int(out["s"][i]),
                        p=int(out["p"][i]),
                        ir=int(out["ir"][i]),
                        operand_lo=int(out["oplo"][i]),
                        operand_hi=int(out["ophi"][i]),
                        ea=int(out["ea"][i]),
                        data=int(out["data"][i]),
                        halted=bool(out["halted"][i]),
                    ),
                    tuple(writes[i]),
                )
                for i in range(len(out["pc"]))
            ]
        )

    def run_image_arrays(
        self, memory_image, *, start_pc: int, max_steps: int, stop_on_trap: bool = False
    ) -> dict:
        """Boot and run up to max_steps entirely in the C++ run loop.

        Returns the run-protocol result dict: per-step
        trace arrays (pc/a/x/y/s/p/ir/oplo/ophi/ea/data/halted), write arrays
        (w_row 0-based within this call, w_addr, w_value), and the run's
        "reason"/"cycles".  Materializes every step's arrays, so it is for
        bounded runs (nestest, a Klaus prefix); use iter_run_arrays to stream
        the full suite.
        """
        self.load_image(memory_image, start_pc=start_pc)
        return self.engine.run(max_steps, stop_on_trap, 0)

    def iter_run_arrays(
        self,
        memory_image,
        *,
        start_pc: int,
        max_steps: int,
        chunk: int = 200_000,
        stop_on_trap: bool = False,
    ):
        """Stream the C++ run loop in fixed-size chunks (w_row rebased to 0
        per chunk), so the full Klaus suite verifies without materializing a
        30M-row trace.  Yields the run-protocol result dict for each chunk.

        The latest-written byte image has fixed size, so execution and
        streaming stay bounded independently of the token log.
        """
        self.load_image(memory_image, start_pc=start_pc)
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

    def run(
        self, max_steps: int, *, stop_on_trap: bool = False, max_cycles: int = 0, trace: bool = True
    ) -> dict:
        """Resume the native run loop from the engine's persistent state
        (no reboot), pausing on halt/trap/watch/cycle budget."""
        return self.engine.run(max_steps, stop_on_trap, max_cycles, trace)

    def watch_read(self, lo: int, hi: int) -> None:
        self.engine.watch_read(lo, hi)

    def watch_write(self, lo: int, hi: int) -> None:
        self.engine.watch_write(lo, hi)

    def read_ram(self, addr: int) -> int:
        return self.engine.read_ram(addr)

    def dump_ram(self) -> bytes:
        return self.engine.dump_ram()

    def get_state(self) -> dict:
        return dict(self.engine.get_run_state())


# ── CLI verification program ────────────────────────────────────

VERIFY_START = 0x0400


def verify_program() -> bytes:
    """Loop + rewrites + stack + subroutine + arithmetic, all below $0500."""
    image = bytearray(0x10000)
    code = bytes([
        0xA9, 0x05,        # LDA #$05
        0x85, 0x10,        # STA $10
        0xA2, 0x03,        # LDX #$03
        0xE6, 0x10,        # loop: INC $10
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
        0x4C, 0x1B, 0x04,  # trap: JMP $041B
    ])
    image[VERIFY_START : VERIFY_START + len(code)] = code
    subroutine = bytes([0xA0, 0x07, 0x84, 0x13, 0x60])  # LDY #7; STY $13; RTS
    image[0x0420 : 0x0420 + len(subroutine)] = subroutine
    return bytes(image)


def main() -> None:
    from transformer_nes.cpu6502 import reference

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Verify the 6502 running as real transformer weights."
    )
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    machine = Cpu6502TransformerMachine(bcd=False)
    logger.info(
        "transformer: d_model=%d exact-int32",
        machine.engine.d_model,
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
        status,
        len(trace.rows),
        run_elapsed,
        len(trace.rows) / run_elapsed,
    )
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
