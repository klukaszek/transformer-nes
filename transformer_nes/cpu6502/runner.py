"""Run the 6502 as the NES model's CPU chip.

The weights (cpu6502/chip.py) compute every instruction; the native host
(native/console/cpu.cpp) keeps the registers between steps and arbitrates
device reads, watched writes, interrupts and budgets. This module is its
Python machine API: for the console's bus, conformance runs and the X-ray.
"""

from __future__ import annotations

import argparse
import functools
import logging
import time
from pathlib import Path

from transformer_nes import build
from transformer_nes.cpu6502 import chip
from transformer_nes.cpu6502.shapes import match_shape_windows
from transformer_nes.cpu6502.spec import FLAG_I, FLAG_U
from transformer_nes.cpu6502.trace import Cpu6502State, MemWrite, Trace, TraceRow

logger = logging.getLogger(__name__)


@functools.cache
def decimal_model():
    """A 6502 with decimal mode, for 6502 conformance (the NES's 2A03 has
    none): a model of its own beside the console's."""
    from transformer_nes.nes.model import CACHE_DIR, NES

    directory = Path(CACHE_DIR) / "decimal"
    return NES.load(directory, {"cpu": functools.partial(chip.chip, bcd=True)}).compile(directory)


class Cpu6502TransformerMachine:
    """The 6502 as transformer weights, run instruction by instruction."""

    def __init__(self, *, bcd: bool = False, union_only: bool = False, mirror_cpu_ram: bool = False,
                 shapes: bool = True):
        from transformer_nes.nes.model import compiled

        self.bcd, self.union_only, self.shapes = bool(bcd), bool(union_only), bool(shapes)
        self.mirror_cpu_ram = bool(mirror_cpu_ram)
        self.chip = (decimal_model() if self.bcd else compiled()).chip("cpu")
        hardware = build.load_nes_extension("hardware")
        self.engine = hardware.Cpu(self.chip, chip.tables(self.chip, union_only=union_only), self.mirror_cpu_ram)

    @property
    def windows(self) -> list[dict]:
        """The shape windows to find in code (shapes.py:match_shape_windows);
        none when instructions run one by one (`shapes` off, or the union)."""
        return [] if self.union_only or not self.shapes else self.chip.chip.constants["windows"]

    # ── The X-ray: the union circuit, model 0 ───────────────────────

    @functools.cached_property
    def layout(self):
        """The union circuit's construction info (which residual slot holds
        each named value, and when): reconstructed on demand."""
        from transformer_nes.cpu6502.graph import RUNTIME_READ_DIMS, Cpu6502Graph
        from transformer_nes.graph.fold import GraphFolder
        from transformer_nes.model.construct import construct

        pg = Cpu6502Graph(bcd=self.bcd, wide_bits=True).program_graph()
        fold = GraphFolder(pg, root_names=RUNTIME_READ_DIMS, structural_reads=True).fold({}, label="union")
        return construct(fold.program_graph)[2]

    slot_of = property(lambda self: self.layout.slot_of)
    lives = property(lambda self: self.layout.lives)  # residual live ranges (frontend/xray.py)

    def trace_union(self):
        """The next instruction through the union circuit: every layer's
        residual stream, and its fraction bits."""
        union = self.chip.chip.tokens["union"].general
        return self.chip.engine.trace(self.engine.union_inputs()), union.f_bits

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
        for sid, head in match_shape_windows(ram, self.windows):
            self.engine.add_window(sid, head)

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
        """Replace a mapped cartridge region in place: no CPU write happens."""
        self.engine.map_region(base, data)
        for sid, head in self.region_windows(base, data):
            self.engine.add_window(sid, head)

    def region_windows(self, base: int, data: bytes) -> tuple:
        """Match newly mapped code. The native console owns bank-content caching."""
        return tuple(match_shape_windows(data, self.windows, base=base))

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

        Memory is a fixed byte image, so execution and streaming stay bounded
        however long the run.
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
