"""Graph-based 6502 evaluator CLI.

This mirrors ``6502-reference`` but runs the CPU through the Percepta
computation graph using exact arithmetic, so it can be compared trace-for-
trace with the reference core.
"""

from __future__ import annotations

import argparse
import logging

from transformer_vm.cpu6502.evaluator import run_program


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Evaluate a 6502 binary through the Percepta graph (exact arithmetic)."
    )
    parser.add_argument("program", help="Path to a .bin file")
    parser.add_argument("--start-pc", type=lambda x: int(x, 0), default=0x8000)
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()

    trace = run_program(
        args.program,
        start_pc=args.start_pc,
        max_steps=args.max_steps,
    )
    print(trace.format())


if __name__ == "__main__":
    main()
