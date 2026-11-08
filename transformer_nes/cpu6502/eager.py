"""The 6502 run by the NES model's eager PyTorch forward alone.

Each instruction is one forward of the CPU chip's union circuit
(cpu6502/chip.py), computed by model/transformer.py over the saved model
(``nes.json``, ``nes.safetensors``). This module imports only the standard
library, Torch, safetensors and model/transformer.py: no construction, no
compiler, no native host; a test enforces it. The loop below embeds the
registers, reads the next state, the writes and the cycle count out of the
residual stream and applies the writes to memory -- the whole claim,
checkable: a transformer of ordinary layers, given these weights, runs a 6502.

    python -m transformer_nes.cpu6502.eager [--steps N] [--model DIR]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from transformer_nes.model.transformer import Chip, Named, unpack_state

FLAGS = ("in_p_c", "in_p_z", "in_p_i", "in_p_d", "in_p_b", "in_p_u", "in_p_v", "in_p_n")


def load_cpu(directory: Path) -> Chip:
    """The CPU chip of a saved NES model (nes/model.py:NES.save)."""
    directory = Path(directory)
    structure = json.loads((directory / "nes.json").read_text())
    model = nn.Module()
    model.chips = Named({name: Chip.empty(chip) for name, chip in structure.items()})
    model.load_state_dict(unpack_state(model.state_dict(), load_file(directory / "nes.safetensors")))
    return model.chips["cpu"]


def run(cpu: Chip, image: bytes, *, pc: int, steps: int, s: int = 0xFD,
        p: int = 0x24) -> Iterator[dict]:
    """The state before each of `steps` instructions from `pc` (pc, a, x, y,
    p, s and the cycles so far), each instruction one union forward."""
    memory = torch.tensor(list(image), dtype=torch.uint8)
    state = {"pc": pc, "a": 0, "x": 0, "y": 0, "s": s, "p": p, "cycles": 0}
    inputs, outputs = cpu.io("union")
    index = {name: i for i, name in enumerate(inputs)}
    out = {name: i for i, name in enumerate(outputs)}
    for _ in range(steps):
        yield dict(state)
        x = torch.zeros(len(inputs), dtype=torch.int32)
        x[index["is_state"]] = 1
        x[index["in_pc_lo"]], x[index["in_pc_hi"]] = state["pc"] & 0xFF, state["pc"] >> 8
        for r in "axys":
            x[index[f"in_{r}"]] = state[r]
        for bit, name in enumerate(FLAGS):
            x[index[name]] = (state["p"] >> bit) & 1
        for bit in range(8):
            x[index[f"in_a_bit{bit}"]] = (state["a"] >> bit) & 1
        o = cpu("union", x, memory).tolist()
        for store, address, value in (("is_store", "write_addr_out", "write_value_out"),
                                      ("is_store2", "write_addr2_out", "write_value2_out"),
                                      ("is_store3", "write_addr3_out", "write_value3_out")):
            if o[out[store]] & 0xFF:
                memory[o[out[address]] & 0xFFFF] = o[out[value]] & 0xFF
        state = {
            "pc": o[out["next_pc"]] & 0xFFFF, "a": o[out["new_a"]] & 0xFF,
            "x": o[out["new_x"]] & 0xFF, "y": o[out["new_y"]] & 0xFF,
            "s": o[out["new_s"]] & 0xFF, "p": o[out["new_p"]] & 0xFF,
            "cycles": state["cycles"] + (o[out["cycles"]] & 0xFF),
        }


def main() -> None:
    from transformer_nes.cpu6502.conformance.nestest import START_PC, load_image, load_log

    parser = argparse.ArgumentParser(description="Run nestest on the eager CPU chip.")
    parser.add_argument("--steps", type=int, default=8991)
    parser.add_argument("--model", type=Path, help="a saved model (default: the process's own)")
    args = parser.parse_args()
    if args.model is None:
        from transformer_nes.nes.model import model_directory

        args.model = model_directory()
    log = load_log()
    rows = run(load_cpu(args.model), bytes(load_image()), pc=START_PC, steps=args.steps)
    for i, (state, (pc, a, x, y, p, s, cycles)) in enumerate(zip(rows, log, strict=False)):
        want = {"pc": pc, "a": a, "x": x, "y": y, "p": p, "s": s, "cycles": cycles - 7}
        if state != want:
            raise SystemExit(f"instruction {i}: eager {state}, nestest {want}")
    print(f"eager PyTorch CPU: MATCHED {min(args.steps, len(log))} nestest instructions")


if __name__ == "__main__":
    main()
