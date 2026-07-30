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

from transformer_nes.cpu6502.backends.reference import Cpu6502

logger = logging.getLogger(__name__)

# Keep the CPU harness outside nes/; only its ROM and golden log are NES assets.
NESTEST_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "nes", "examples")
)
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
    prg = rom[16 : 16 + 0x4000]
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
            a = int(line[line.index("A:") + 2 :][:2], 16)
            x = int(line[line.index("X:") + 2 :][:2], 16)
            y = int(line[line.index("Y:") + 2 :][:2], 16)
            p = int(line[line.index("P:") + 2 :][:2], 16)
            s = int(line[line.index("SP:") + 3 :][:2], 16)
            rows.append((pc, a, x, y, p, s))
    return rows


def new_cpu(image: bytearray) -> Cpu6502:
    cpu = Cpu6502(memory=bytearray(image), bcd_enabled=False)
    cpu.pc = START_PC
    return cpu


def run_reference(image: bytearray, log_rows) -> tuple[bool, int, float]:
    """Step the reference core against the golden log.

    Returns (passed, lines_matched, elapsed_seconds).
    """
    cpu = new_cpu(image)
    start = time.perf_counter()
    for i, (pc, a, x, y, p, s) in enumerate(log_rows):
        got = (cpu.pc, cpu.a, cpu.x, cpu.y, cpu.p, cpu.s)
        if got != (pc, a, x, y, p, s):
            logger.error(
                "mismatch at log line %d:\nexpected PC=%04X A=%02X X=%02X Y=%02X P=%02X SP=%02X\ngot      PC=%04X A=%02X X=%02X Y=%02X P=%02X SP=%02X",
                i + 1,
                pc,
                a,
                x,
                y,
                p,
                s,
                *got,
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


def run_weights(
    image: bytearray, *, max_steps: int, union_only: bool = False,
) -> tuple[bool, int, float, float, str]:
    """Run the real weights-based transformer in lockstep with the reference."""
    from transformer_nes.cpu6502.backends.weights.runner import Cpu6502TransformerMachine
    from transformer_nes.cpu6502.examples.klaus.harness import lockstep_arrays_vs_reference

    build_start = time.perf_counter()
    machine = Cpu6502TransformerMachine(bcd=False, union_only=union_only)
    mode = "union-only" if union_only else "codegen-exact-int32-runloop"
    build_elapsed = time.perf_counter() - build_start

    # Whole suite through the C++ run loop, then lockstep the arrays.
    run_start = time.perf_counter()
    out = machine.run_image_arrays(bytes(image), start_pc=START_PC, max_steps=max_steps)
    run_elapsed = time.perf_counter() - run_start

    ref = new_cpu(image)
    matched, n = lockstep_arrays_vs_reference(ref, out)
    return matched, n, build_elapsed, run_elapsed, mode


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run the nestest CPU suite.")
    parser.add_argument(
        "--runner", choices=["reference", "weights", "both"], default="reference"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8991,
        help="maximum instructions for the weights run (default: the full log)",
    )
    parser.add_argument(
        "--union-only",
        action="store_true",
        help="build the weights engine with no experts, so every opcode "
             "executes on the union proof model",
    )
    args = parser.parse_args()

    image = load_image()
    log_rows = load_log()
    failed = False

    if args.runner in ("reference", "both"):
        passed, lines, elapsed = run_reference(image, log_rows)
        status = "PASSED" if passed else "FAILED"
        logger.info(
            "reference: %s  %d/%d log lines in %.2fs", status, lines, len(log_rows), elapsed
        )
        failed |= not passed

    if args.runner in ("weights", "both"):
        matched, steps, build_s, run_s, mode = run_weights(
            image,
            max_steps=args.max_steps,
            union_only=args.union_only,
        )
        status = "MATCHED" if matched else "DIVERGED"
        rate = steps / run_s if run_s > 0 else 0.0
        logger.info(
            "weights transformer (%s): %s  %d steps in %.1fs (%.1f steps/s), weights build %.1fs",
            mode,
            status,
            steps,
            run_s,
            rate,
            build_s,
        )
        failed |= not matched

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
