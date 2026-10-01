"""Run the exported 6502 checkpoint as a plain PyTorch transformer.

Everything here comes from the two files model/export.py writes --
``weights.safetensors`` and ``machine.json`` -- and nothing else from this
repository: no graph, no compiler, no generated engine.  This file imports
only torch and safetensors, so it is the whole claim made checkable: a
decoder-only transformer with ordinary layers, given these weights, runs a
6502.

The model (machine.json "format" has the contract):

- a residual stream of d_model exact integers (units of 2**-scale_bits);
  a token embeds as its named input values written into their slots;
- per layer, one attention block -- Q/K/V projections, 2-dimensional
  heads, hardmax with exact ties going to the latest token, the current
  token included, some heads masking tokens by a "clear gate" -- then a
  ReGLU feed-forward block, each added to the residual;
- no normalization and no position encoding: order enters only through
  the latest-wins tie-break.

The program is a token stream.  WRITE tokens lay down memory; a STATE
token carries the CPU registers, and its forward reads the next state,
the instruction's cycle count and up to three memory writes out of the
residual stream.  The harness appends those writes as WRITE tokens and
the next state as the next STATE token -- autoregressive decoding.

Two exact shortcuts keep the KV cache small, both read off the weights:
a head whose key ignores every input but the constant ties on every
token, so it always attends to the current one (no cache at all); and a
cached token whose key equals a newer one's can never win again, so each
head keeps one entry per distinct key.

    python -m transformer_nes.model.checkpoint_runner CHECKPOINT \\
        --nestest nestest.nes nestest.log
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from safetensors.torch import load_file

# The residual is int32, as in the native engine; an attention score (a
# key against a query, both up to an address times a head's temperature)
# can pass 2**31, so scores, and the keys they come from, are int64.
INT = torch.int32
SCORE = torch.int64


class SparseLinear(torch.nn.Module):
    """y = W x for a CSR matrix of exact integer coefficients, as a gather
    and a scatter-add: PyTorch's sparse kernels have no integer path."""

    def __init__(self, tensors: dict, name: str, rows: int):
        super().__init__()
        counts = tensors[f"{name}.indptr"].long().diff()
        self.register_buffer("rows", torch.repeat_interleave(torch.arange(rows), counts))
        self.register_buffer("cols", tensors[f"{name}.indices"].long())
        self.register_buffer("data", tensors[f"{name}.data"].to(INT))
        self.out_features = rows

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros(self.out_features, dtype=INT)
        return y.index_add_(0, self.rows, self.data * x[self.cols])


class HeadCache:
    """One attention head's keys and values: one entry per distinct key,
    holding the latest token's value and position (for the tie-break)."""

    def __init__(self):
        self.slot_of_key: dict[tuple[int, int], int] = {}
        self.keys = torch.zeros((64, 2), dtype=SCORE)
        self.values = torch.zeros((64, 2), dtype=INT)
        self.positions = torch.zeros(64, dtype=torch.long)

    def insert(self, key: torch.Tensor, value: torch.Tensor, position: int) -> None:
        k = (int(key[0]), int(key[1]))
        slot = self.slot_of_key.get(k)
        if slot is None:
            slot = len(self.slot_of_key)
            if slot == len(self.keys):
                grow = lambda t: torch.cat([t, torch.zeros_like(t)])  # noqa: E731
                self.keys, self.values, self.positions = map(
                    grow, (self.keys, self.values, self.positions)
                )
            self.slot_of_key[k] = slot
            self.keys[slot] = key
        self.values[slot] = value
        self.positions[slot] = position

    def attend(self, query: torch.Tensor) -> torch.Tensor:
        n = len(self.slot_of_key)
        if n == 0:
            return torch.zeros(2, dtype=INT)
        scores = self.keys[:n] @ query.to(SCORE)
        best = scores == scores.max()
        latest = torch.where(best, self.positions[:n], -1).argmax()
        return self.values[latest]


class Layer(torch.nn.Module):
    def __init__(self, tensors: dict, li: int, heads: int, ffn: int, d_model: int, one: int):
        super().__init__()
        self.heads, self.ffn = heads, ffn
        self.qkv = SparseLinear(tensors, f"L{li}.in_proj", 6 * heads)
        self.out_proj = SparseLinear(tensors, f"L{li}.out_proj", d_model)
        self.ff_in = SparseLinear(tensors, f"L{li}.ff_in", 2 * ffn)
        self.ff_out = SparseLinear(tensors, f"L{li}.ff_out", d_model)
        # Clear gates sit on key rows: a token is masked for the gate's
        # head when the gate's input column is non-zero.
        self.register_buffer("gate_heads", (tensors[f"L{li}.clear_rows"].long() - 2 * heads) // 2)
        self.register_buffer("gate_cols", tensors[f"L{li}.clear_cols"].long())
        # Heads that need a cache: a key that varies with the input, or a
        # clear gate.  Every other head attends to the current token.
        rows, cols = self.qkv.rows, self.qkv.cols
        varying = rows[(rows >= 2 * heads) & (rows < 4 * heads) & (cols != one)]
        self.cached = sorted({(r - 2 * heads) // 2 for r in varying.tolist()}
                             | set(self.gate_heads.tolist()))
        self.caches = {h: HeadCache() for h in self.cached}

    def forward(self, x: torch.Tensor, position: int, shift: int) -> torch.Tensor:
        h2 = 2 * self.heads
        qkv = self.qkv(x) >> shift
        q, k, v = qkv[:h2], qkv[h2 : 2 * h2], qkv[2 * h2 :]
        out = v.clone()  # attending to itself: the token's own value
        masked = set(self.gate_heads[x[self.gate_cols] != 0].tolist())
        for h in self.cached:
            pair = slice(2 * h, 2 * h + 2)
            if h not in masked:
                self.caches[h].insert(k[pair], v[pair], position)
            out[pair] = self.caches[h].attend(q[pair])
        x = x + (self.out_proj(out) >> shift)
        gates_values = self.ff_in(x) >> shift
        gate, value = gates_values[: self.ffn], gates_values[self.ffn :]
        act = torch.where(gate > 0, (gate * value) >> shift, 0)
        return x + (self.ff_out(act) >> shift)


class Transformer(torch.nn.Module):
    def __init__(self, checkpoint: str):
        super().__init__()
        with open(os.path.join(checkpoint, "machine.json")) as f:
            self.meta = json.load(f)
        tensors = load_file(os.path.join(checkpoint, "weights.safetensors"))
        m = self.meta
        self.shift, self.d_model = m["scale_bits"], m["d_model"]
        self.layers = torch.nn.ModuleList(
            Layer(tensors, li, m["layer_heads"][li], m["layer_ffn"][li], self.d_model, m["one_slot"])
            for li in range(m["n_layers"])
        )
        self.register_buffer("head", tensors["head"].to(INT))
        self.position = 0

    def forward(self, token: dict[str, int]) -> torch.Tensor:
        """Append one token; return the residual stream after the last
        layer (in units of 2**-scale_bits)."""
        x = torch.zeros(self.d_model, dtype=INT)
        x[self.meta["one_slot"]] = 1 << self.shift
        for name, value in token.items():
            x[self.meta["input_slots"][name]] = value << self.shift
        for layer in self.layers:
            x = layer(x, self.position, self.shift)
        self.position += 1
        return x


FLAGS = ("c", "z", "i", "d", "b", "u", "v", "n")  # bit 0 .. bit 7


class Cpu:
    """The 6502 as autoregressive decoding over the checkpoint."""

    def __init__(self, model: Transformer, image: bytes, pc: int, *, s: int = 0xFD, p: int = 0x24):
        self.model = model
        for addr, byte in enumerate(image):
            if byte:  # unwritten memory reads 0
                self.write(addr, byte)
        self.state = {"pc": pc, "a": 0, "x": 0, "y": 0, "s": s, "p": p,
                      "ir": 0, "oplo": 0, "ophi": 0, "ea": 0, "data": 0, "halted": 0}

    def write(self, addr: int, value: int) -> None:
        self.model({"is_memory_write": 1, "addr_lo": addr & 0xFF, "addr_hi": addr >> 8,
                    "byte_value": value})

    def step(self) -> int:
        """One instruction: one STATE token, then its writes as WRITE
        tokens.  Returns the instruction's cycle count."""
        st = self.state
        token = {
            "is_state": 1, "in_pc_lo": st["pc"] & 0xFF, "in_pc_hi": st["pc"] >> 8,
            "in_a": st["a"], "in_x": st["x"], "in_y": st["y"], "in_s": st["s"],
            "in_ir": st["ir"], "in_oplo": st["oplo"], "in_ophi": st["ophi"],
            "in_ea_lo": st["ea"] & 0xFF, "in_ea_hi": st["ea"] >> 8,
            "in_data": st["data"], "in_halted": st["halted"],
            **{f"in_p_{f}": (st["p"] >> b) & 1 for b, f in enumerate(FLAGS)},
        }
        x = self.model(token)
        slots, shift = self.model.meta["readout_slots"], self.model.shift

        def read(name: str) -> int:
            return int(x[slots[name]]) >> shift

        for suffix in ("", "2", "3"):
            if read(f"is_store{suffix}"):
                self.write(read(f"write_addr{suffix}_out"), read(f"write_value{suffix}_out"))
        logits = self.model.head @ x
        halt = logits[self.model.meta["halt_row"]] > logits[self.model.meta["state_row"]]
        self.state = {
            "pc": read("next_pc"), "a": read("new_a"), "x": read("new_x"), "y": read("new_y"),
            "s": read("new_s"), "p": read("new_p"), "ir": read("fetched_opcode"),
            "oplo": read("fetched_operand_lo"), "ophi": read("fetched_operand_hi"),
            "ea": read("ea"), "data": read("data_out"),
            "halted": int(read("new_halted") != 0 or bool(halt)),
        }
        return read("cycles")


def nestest(model: Transformer, rom: str, log: str, steps: int | None = None) -> int:
    """Run nestest in automation mode from $C000 and check every
    instruction against the golden log: PC, A, X, Y, P, SP, and CYC (the
    cycle count before it, from the model's own per-instruction cycles).
    Returns the number of instructions checked."""
    with open(rom, "rb") as f:
        prg = f.read()[16 : 16 + 0x4000]
    image = bytearray(0x10000)
    image[0x8000:0xC000] = image[0xC000:] = prg  # NROM-128 mirrors its bank
    with open(log) as f:
        lines = f.readlines()[:steps]
    cpu = Cpu(model, bytes(image), 0xC000)
    cycles = 7  # the reset sequence
    for n, line in enumerate(lines):
        def field(tag: str, line=line) -> int:
            return int(line[line.index(tag) + len(tag) :].split()[0], 16 if tag != "CYC:" else 10)

        want = (int(line[:4], 16), field("A:"), field("X:"), field("Y:"), field("P:"),
                field("SP:"), field("CYC:"))
        st = cpu.state
        got = (st["pc"], st["a"], st["x"], st["y"], st["p"], st["s"], cycles)
        if got != want:
            raise AssertionError(f"instruction {n}: got {got}, golden log {want}\n  {line}")
        cycles += cpu.step()
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", help="directory with weights.safetensors and machine.json")
    parser.add_argument("--nestest", nargs=2, metavar=("ROM", "LOG"), required=True)
    parser.add_argument("--steps", type=int, help="check only the first N log lines")
    args = parser.parse_args()
    model = Transformer(args.checkpoint)
    start = time.perf_counter()
    with torch.inference_mode():
        n = nestest(model, *args.nestest, args.steps)
    elapsed = time.perf_counter() - start
    print(f"nestest: {n} instructions match the golden log (registers and cycles), "
          f"{model.position} tokens in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
