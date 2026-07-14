"""Run the Klaus Dormann 6502 functional test suite.

The suite is a 64 KiB memory image that starts at $0400 and signals
completion by trapping in a jump-to-self.  A trap at SUCCESS_PC means every
test passed; a trap anywhere else identifies the failing test case.

Runners:
    reference -- the plain Python oracle core (full suite to the trap)
    graph     -- the computation-graph evaluator, checked in lockstep
                 against the reference core for a configurable number of
                 steps (the full suite is ~26M instructions, far beyond
                 what the graph evaluator can cover in reasonable time)
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections import defaultdict

from transformer_nes.cpu6502.backends.reference import Cpu6502

logger = logging.getLogger(__name__)

KLAUS_BIN = os.path.join(os.path.dirname(__file__), "6502_functional_test.bin")
START_PC = 0x0400
SUCCESS_PC = 0x3469
DEFAULT_MAX_STEPS = 100_000_000
# Instruction count of a full passing run, measured on the reference core.
KLAUS_EXPECTED_STEPS = 30_646_177


def load_image(path: str = KLAUS_BIN) -> bytes:
    with open(path, "rb") as f:
        image = f.read()
    if len(image) != 0x10000:
        raise ValueError(f"expected a 64 KiB image, got {len(image)} bytes")
    return image


def run_reference(image: bytes, *, max_steps: int = DEFAULT_MAX_STEPS) -> tuple[bool, int, float]:
    """Run the suite on the reference core until it traps.

    Returns (passed, steps_executed, elapsed_seconds).
    """
    cpu = Cpu6502(memory=bytearray(image))

    start = time.perf_counter()
    steps = cpu.run_traps(start_pc=START_PC, max_steps=max_steps)
    elapsed = time.perf_counter() - start

    passed = cpu.pc == SUCCESS_PC
    if not passed:
        if steps >= max_steps:
            logger.error("reference did not trap within %d steps (PC=%04X)", steps, cpu.pc)
        else:
            logger.error("reference trapped at PC=%04X after %d steps", cpu.pc, steps)
    return passed, steps, elapsed


def run_graph(image: bytes, *, max_steps: int) -> tuple[bool, int, float, float, float]:
    """Run the first max_steps of the suite on the graph evaluator.

    Every step is compared against the reference core.  Returns
    (matched, steps_executed, build_seconds, prefix_seconds, run_seconds).
    """
    from transformer_nes.cpu6502.backends.graph import Cpu6502Runtime

    build_start = time.perf_counter()
    rt = Cpu6502Runtime()
    build_elapsed = time.perf_counter() - build_start

    run_start = time.perf_counter()
    graph_trace = rt.run_image(image, start_pc=START_PC, max_steps=max_steps)
    run_elapsed = time.perf_counter() - run_start
    prefix_elapsed = rt.last_prefix_seconds

    ref_cpu = Cpu6502(memory=bytearray(image))
    ref_cpu.pc = START_PC
    for i, row in enumerate(graph_trace):
        ref_row = ref_cpu.step()
        s1, s2 = ref_row.state, row.state
        if (
            s1.pc != s2.pc
            or s1.a != s2.a
            or s1.x != s2.x
            or s1.y != s2.y
            or s1.s != s2.s
            or s1.p != s2.p
            or s1.halted != s2.halted
        ):
            logger.error(
                "graph diverged from reference at step %d:\nREF:   PC=%04X A=%02X X=%02X Y=%02X S=%02X P=%02X\nGRAPH: PC=%04X A=%02X X=%02X Y=%02X S=%02X P=%02X",
                i, s1.pc, s1.a, s1.x, s1.y, s1.s, s1.p,
                s2.pc, s2.a, s2.x, s2.y, s2.s, s2.p,
            )
            return False, i, build_elapsed, prefix_elapsed, run_elapsed
        ref_writes = {(w.addr, w.value) for w in ref_row.writes}
        graph_writes = {(w.addr, w.value) for w in row.writes}
        if ref_writes != graph_writes:
            logger.error("graph writes diverged at step %d: ref=%s graph=%s", i, ref_writes, graph_writes)
            return False, i, build_elapsed, prefix_elapsed, run_elapsed

    return True, len(graph_trace), build_elapsed, prefix_elapsed, run_elapsed


def lockstep_arrays_vs_reference(ref: Cpu6502, out: dict) -> tuple[bool, int]:
    """Compare native run-loop result arrays against a prepared reference core."""
    writes_by_row: dict[int, set] = defaultdict(set)
    for row, addr, value in zip(out["w_row"], out["w_addr"], out["w_value"], strict=True):
        writes_by_row[int(row)].add((int(addr), int(value)))
    steps = len(out["pc"])
    for i in range(steps):
        row = ref.step()
        s = row.state
        got = (int(out["pc"][i]), int(out["a"][i]), int(out["x"][i]), int(out["y"][i]),
               int(out["s"][i]), int(out["p"][i]))
        if (s.pc, s.a, s.x, s.y, s.s, s.p) != got:
            logger.error(
                "engine diverged from reference at step %d:\nREF:    PC=%04X A=%02X X=%02X Y=%02X S=%02X P=%02X\nENGINE: PC=%04X A=%02X X=%02X Y=%02X S=%02X P=%02X",
                i, s.pc, s.a, s.x, s.y, s.s, s.p, *got,
            )
            return False, i
        if {(w.addr, w.value) for w in row.writes} != writes_by_row.get(i, set()):
            logger.error("engine writes diverged at step %d", i)
            return False, i
    return True, steps


def run_weights(image: bytes, *, max_steps: int,
                ) -> tuple[bool, int, float, float, str]:
    """Lockstep the real weights-based transformer against the reference.

    The generated engine runs the whole stream in C++, pausing only at the
    jump-to-self success trap; each native chunk locksteps against the
    streaming reference (engine state is O(1), no checkpointing) -- the
    full 30.6M-instruction suite fits in ~75 s.
    """
    from transformer_nes.cpu6502.backends.weights.runner import Cpu6502TransformerMachine

    build_start = time.perf_counter()
    machine = Cpu6502TransformerMachine(bcd=True)
    mode = "codegen-exact-int32-runloop"
    build_elapsed = time.perf_counter() - build_start

    ref_cpu = Cpu6502(memory=bytearray(image))
    ref_cpu.pc = START_PC

    run_start = time.perf_counter()
    steps = 0
    for out in machine.iter_run_arrays(list(image), start_pc=START_PC,
                                       max_steps=max_steps, chunk=500_000,
                                       stop_on_trap=True):
        matched, n = lockstep_arrays_vs_reference(ref_cpu, out)
        if not matched:
            return False, steps + n, build_elapsed, time.perf_counter() - run_start, mode
        steps += len(out["pc"])
        if steps % 1_000_000 < 500_000:
            elapsed = time.perf_counter() - run_start
            logger.info("  %.1fM steps, %.0f steps/s", steps / 1e6, steps / elapsed)
    return True, steps, build_elapsed, time.perf_counter() - run_start, mode


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run the Klaus 6502 functional test suite.")
    parser.add_argument(
        "--runner",
        choices=["reference", "graph", "weights", "both"],
        default="reference",
    )
    parser.add_argument("--image", default=KLAUS_BIN)
    parser.add_argument(
        "--graph-steps", type=int, default=5_000,
        help="instruction count for the graph evaluator run (default 5000)",
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = parser.parse_args()

    image = load_image(args.image)
    failed = False

    if args.runner in ("reference", "both"):
        passed, steps, elapsed = run_reference(image, max_steps=args.max_steps)
        rate = steps / elapsed if elapsed > 0 else 0.0
        status = "PASSED" if passed else "FAILED"
        logger.info(
            "reference: %s  %d steps in %.2fs (%.0f steps/s)", status, steps, elapsed, rate
        )
        failed |= not passed

    if args.runner in ("graph", "both"):
        matched, steps, build_s, prefix_s, run_s = run_graph(image, max_steps=args.graph_steps)
        step_s = run_s - prefix_s
        rate = steps / step_s if step_s > 0 else 0.0
        status = "MATCHED" if matched else "DIVERGED"
        logger.info(
            "graph: %s  %d steps in %.2fs (%.0f steps/s), image prefix %.2fs, graph build %.2fs",
            status, steps, step_s, rate, prefix_s, build_s,
        )
        failed |= not matched

    if args.runner == "weights":
        matched, steps, build_s, run_s, mode = run_weights(
            image, max_steps=args.graph_steps,
        )
        rate = steps / run_s if run_s > 0 else 0.0
        status = "MATCHED" if matched else "DIVERGED"
        logger.info(
            "weights transformer (%s): %s  %d steps in %.1fs (%.1f steps/s), weights build %.1fs",
            mode, status, steps, run_s, rate, build_s,
        )
        failed |= not matched

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
