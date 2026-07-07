"""Run the nestest CPU suite against the Nintendulator golden log.

nestest.nes is an NROM-128 iNES ROM.  In automated mode execution starts at
$C000 and requires no PPU; the CPU state before every instruction must match
the golden log line-for-line.  Result codes are left at $0002 (official
opcodes) and $0003 (unofficial opcodes); 0x00 means every test passed.

The golden log was produced by a 2A03 (NES) CPU, which has no decimal mode,
so the reference core runs with bcd_enabled=False.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

from transformer_vm.cpu6502.reference import Cpu6502

logger = logging.getLogger(__name__)

# The ROM asset lives with the other NES ROMs (data-only dependency; the
# cpu6502 code itself never imports from transformer_vm.nes).
NESTEST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nes", "examples")
NESTEST_ROM = os.path.join(NESTEST_DIR, "nestest.nes")
NESTEST_LOG = os.path.join(NESTEST_DIR, "nestest.log")
START_PC = 0xC000


def load_image(rom_path: str = NESTEST_ROM) -> bytearray:
    """Build a 64 KiB memory image from the NROM-128 iNES file."""
    with open(rom_path, "rb") as f:
        rom = f.read()
    if rom[:4] != b"NES\x1a":
        raise ValueError("not an iNES file")
    prg_banks = rom[4]
    if prg_banks != 1:
        raise ValueError(f"expected NROM-128 (1 PRG bank), got {prg_banks}")
    prg = rom[16:16 + 0x4000]
    image = bytearray(0x10000)
    image[0x8000:0xC000] = prg
    image[0xC000:0x10000] = prg  # NROM-128 mirrors the single bank
    return image


def load_log(log_path: str = NESTEST_LOG) -> list[tuple[int, int, int, int, int, int]]:
    """Parse the golden log into (pc, a, x, y, p, s) tuples."""
    rows = []
    with open(log_path) as f:
        for line in f:
            pc = int(line[0:4], 16)
            a = int(line[line.index("A:") + 2:][:2], 16)
            x = int(line[line.index("X:") + 2:][:2], 16)
            y = int(line[line.index("Y:") + 2:][:2], 16)
            p = int(line[line.index("P:") + 2:][:2], 16)
            s = int(line[line.index("SP:") + 3:][:2], 16)
            rows.append((pc, a, x, y, p, s))
    return rows


def _new_cpu(image: bytearray) -> Cpu6502:
    cpu = Cpu6502(memory=bytearray(image), bcd_enabled=False)
    cpu.pc = START_PC
    return cpu


def run_reference(image: bytearray, log_rows) -> tuple[bool, int, float]:
    """Step the reference core against the golden log.

    Returns (passed, lines_matched, elapsed_seconds).
    """
    cpu = _new_cpu(image)
    start = time.perf_counter()
    for i, (pc, a, x, y, p, s) in enumerate(log_rows):
        got = (cpu.pc, cpu.a, cpu.x, cpu.y, cpu.p, cpu.s)
        if got != (pc, a, x, y, p, s):
            logger.error(
                "mismatch at log line %d:\nexpected PC=%04X A=%02X X=%02X Y=%02X P=%02X SP=%02X\ngot      PC=%04X A=%02X X=%02X Y=%02X P=%02X SP=%02X",
                i + 1, pc, a, x, y, p, s, *got,
            )
            return False, i, time.perf_counter() - start
        cpu.step()
    elapsed = time.perf_counter() - start

    official = cpu.memory[0x0002]
    unofficial = cpu.memory[0x0003]
    if official or unofficial:
        logger.error("nestest result codes: $02=%02X $03=%02X", official, unofficial)
        return False, len(log_rows), elapsed
    return True, len(log_rows), elapsed


def run_graph(image: bytearray, log_rows, *, max_steps: int) -> tuple[bool, int, float, float]:
    """Run the graph evaluator in lockstep with the reference core.

    The reference is the oracle here; it is itself validated against the
    golden log by run_reference.  Returns (matched, steps, build_s, run_s).
    """
    from transformer_vm.cpu6502.evaluator import Cpu6502Runtime

    build_start = time.perf_counter()
    rt = Cpu6502Runtime(bcd=False)
    build_elapsed = time.perf_counter() - build_start

    run_start = time.perf_counter()
    graph_trace = rt.run_image(bytes(image), start_pc=START_PC, max_steps=max_steps)
    run_elapsed = time.perf_counter() - run_start

    ref = _new_cpu(image)
    for i, row in enumerate(graph_trace):
        ref_row = ref.step()
        s1, s2 = ref_row.state, row.state
        if (
            s1.pc != s2.pc or s1.a != s2.a or s1.x != s2.x or s1.y != s2.y
            or s1.s != s2.s or s1.p != s2.p
        ):
            logger.error(
                "graph diverged from reference at step %d:\nREF:   PC=%04X A=%02X X=%02X Y=%02X S=%02X P=%02X\nGRAPH: PC=%04X A=%02X X=%02X Y=%02X S=%02X P=%02X",
                i, s1.pc, s1.a, s1.x, s1.y, s1.s, s1.p,
                s2.pc, s2.a, s2.x, s2.y, s2.s, s2.p,
            )
            return False, i, build_elapsed, run_elapsed
        if {(w.addr, w.value) for w in ref_row.writes} != {(w.addr, w.value) for w in row.writes}:
            logger.error(
                "graph writes diverged at step %d: ref=%s graph=%s",
                i, ref_row.writes, row.writes,
            )
            return False, i, build_elapsed, run_elapsed
    return True, len(graph_trace), build_elapsed, run_elapsed


def run_cpp(image: bytearray, *, max_steps: int) -> tuple[bool, int, float, float]:
    """Run the C++ kernel in lockstep with the reference core (2A03 mode)."""
    from transformer_vm.cpu6502.kernel import Cpu6502CppRuntime
    from transformer_vm.cpu6502.klaus import lockstep_arrays_vs_reference

    build_start = time.perf_counter()
    rt = Cpu6502CppRuntime(bcd=False)
    build_elapsed = time.perf_counter() - build_start
    out = rt.run_image_arrays(bytes(image), start_pc=START_PC, max_steps=max_steps)
    matched, steps = lockstep_arrays_vs_reference(_new_cpu(image), out)
    return matched, steps, build_elapsed, rt.last_step_seconds


def run_weights(image: bytearray, *, max_steps: int,
                matmul: str = "codegen") -> tuple[bool, int, float, float, str]:
    """Run the real weights-based transformer in lockstep with the reference."""
    from transformer_vm.cpu6502.weights_runner import Cpu6502TransformerMachine

    build_start = time.perf_counter()
    machine = Cpu6502TransformerMachine(bcd=False, matmul=matmul)
    arith = getattr(machine._engine, "arith", None)
    mode = f"{machine.matmul}-{arith}" if arith else f"{machine.matmul}/{machine.dtype}"
    build_elapsed = time.perf_counter() - build_start

    run_start = time.perf_counter()
    trace = machine.run_image(bytes(image), start_pc=START_PC, max_steps=max_steps)
    run_elapsed = time.perf_counter() - run_start

    ref = _new_cpu(image)
    for i, row in enumerate(trace):
        ref_row = ref.step()
        s1, s2 = ref_row.state, row.state
        if (
            s1.pc != s2.pc or s1.a != s2.a or s1.x != s2.x or s1.y != s2.y
            or s1.s != s2.s or s1.p != s2.p
        ):
            logger.error(
                "transformer diverged from reference at step %d:\nREF:         PC=%04X A=%02X X=%02X Y=%02X S=%02X P=%02X\nTRANSFORMER: PC=%04X A=%02X X=%02X Y=%02X S=%02X P=%02X",
                i, s1.pc, s1.a, s1.x, s1.y, s1.s, s1.p,
                s2.pc, s2.a, s2.x, s2.y, s2.s, s2.p,
            )
            return False, i, build_elapsed, run_elapsed, mode
        if {(w.addr, w.value) for w in ref_row.writes} != {(w.addr, w.value) for w in row.writes}:
            logger.error("transformer writes diverged at step %d", i)
            return False, i, build_elapsed, run_elapsed, mode
    return True, len(trace.rows), build_elapsed, run_elapsed, mode


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run the nestest CPU suite.")
    parser.add_argument(
        "--runner", choices=["reference", "graph", "cpp", "weights", "both"], default="reference"
    )
    parser.add_argument(
        "--graph-steps", type=int, default=8991,
        help="instruction count for the graph run (default: the full log)",
    )
    parser.add_argument(
        "--matmul", choices=["codegen", "engine", "sparse", "dense"], default="codegen",
        help="weights runner: engine (generic CSR C++) or codegen "
             "(generated straight-line C++); sparse/dense are diagnostics",
    )
    args = parser.parse_args()

    image = load_image()
    log_rows = load_log()
    failed = False

    if args.runner in ("reference", "both"):
        passed, lines, elapsed = run_reference(image, log_rows)
        status = "PASSED" if passed else "FAILED"
        logger.info("reference: %s  %d/%d log lines in %.2fs", status, lines, len(log_rows), elapsed)
        failed |= not passed

    if args.runner in ("graph", "both"):
        matched, steps, build_s, run_s = run_graph(image, log_rows, max_steps=args.graph_steps)
        status = "MATCHED" if matched else "DIVERGED"
        rate = steps / run_s if run_s > 0 else 0.0
        logger.info(
            "graph: %s  %d steps in %.2fs (%.0f steps/s), graph build %.2fs",
            status, steps, run_s, rate, build_s,
        )
        failed |= not matched

    if args.runner in ("cpp", "both"):
        matched, steps, build_s, step_s = run_cpp(image, max_steps=args.graph_steps)
        status = "MATCHED" if matched else "DIVERGED"
        rate = steps / step_s if step_s > 0 else 0.0
        logger.info(
            "cpp: %s  %d steps in %.3fs (%.0f steps/s), kernel build %.2fs",
            status, steps, step_s, rate, build_s,
        )
        failed |= not matched

    if args.runner == "weights":
        matched, steps, build_s, run_s, mode = run_weights(
            image, max_steps=args.graph_steps, matmul=args.matmul,
        )
        status = "MATCHED" if matched else "DIVERGED"
        rate = steps / run_s if run_s > 0 else 0.0
        logger.info(
            "weights transformer (%s): %s  %d steps in %.1fs (%.1f steps/s), weights build %.1fs",
            mode, status, steps, run_s, rate, build_s,
        )
        failed |= not matched

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
