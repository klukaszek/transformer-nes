"""Run exported opcode/interrupt experts with PyTorch and safetensors.

The PC byte routes execution. Declared memory queries read the byte image;
step_spec supplies the embedding and architectural readout contract.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from safetensors.torch import load_file

# The residual stream is the native engine's: int32, in units of
# 2**-scale_bits.  Every value the engine computes fits it, and so does
# every value here -- the same sums, products and shifts.
INT = torch.int32
PASSTHROUGH, LOOKUP = 2, 1
FLAG_MASKS = {"p_c": 0x01, "p_z": 0x02, "p_i": 0x04, "p_d": 0x08,
              "p_b": 0x10, "p_v": 0x40, "p_n": 0x80}
REGISTERS = {"new_pc": ("pc", 0xFFFF), "new_a": ("a", 0xFF), "new_x": ("x", 0xFF),
             "new_y": ("y", 0xFF), "new_s": ("s", 0xFF)}


def dense(tensors: dict, name: str, rows: int, cols: int) -> torch.Tensor:
    """A CSR matrix as a dense one.  Its entries are each coefficient in
    units of 2**-scale_bits: integers, and halves where they are odd."""
    counts = tensors[f"{name}.indptr"].long().diff()
    W = torch.zeros(rows, cols, dtype=INT)
    W.index_put_(
        (torch.repeat_interleave(torch.arange(rows), counts), tensors[f"{name}.indices"].long()),
        tensors[f"{name}.data"].to(INT), accumulate=True,
    )
    return W


def read_codes(outputs) -> list[int]:
    """A memory head's two output components as codes: the byte (-1), the
    address (-2), bit n of the byte (n), or nothing (-3)."""
    codes = {"value": -1, "address": -2, **{f"bit{n}": n for n in range(8)}}
    kinds = list(outputs or ("value", "address"))
    return [codes[k] for k in kinds] + [-3] * (2 - len(kinds))


def share_queries(w_in: torch.Tensor, w_out: torch.Tensor, read_kind: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Share complete query pairs and live (query, value kind) channels.

    Input/output coefficients use half units here. Summing duplicate value
    columns before splitting whole/half terms preserves the read projection's
    single rounding. Passthrough values retain their separate rounding.
    Query equality includes its scale, not just its address on sampled inputs.
    """
    queries, rows, routes, columns = {}, [], [], []
    channels = {}
    for h, kinds in enumerate(read_kind.tolist()):
        pair = w_in[2 * h:2 * h + 2]
        key = tuple(pair.flatten().tolist())
        if key not in queries:
            queries[key] = len(rows)
            rows.append(pair)
        query = queries[key]
        for c, kind in enumerate(kinds):
            column = w_out[:, 2 * h + c]
            if kind == -3 or not column.any():
                continue
            value = (query, kind)
            if value not in channels:
                channels[value] = len(columns)
                columns.append(column.to(torch.int64))
                routes.append(value)
            else:
                columns[channels[value]] += column
    projected = torch.cat([*rows, w_in[2 * len(read_kind):]])
    values = torch.stack(columns, 1) if columns else w_out[:, :0]
    out = torch.cat([values, w_out[:, 2 * len(read_kind):]], 1).to(torch.int64)
    if ((out < -(2**31)) | (out >= 2**31)).any():
        raise ValueError("shared value projection exceeds int32")
    return projected, out.to(w_out.dtype), torch.tensor(routes, dtype=INT).reshape(-1, 2)

def touched(W: torch.Tensor, dim: int) -> torch.Tensor:
    """The rows (dim 1) or columns (dim 0) of W holding a nonzero."""
    return (W != 0).any(dim=dim).nonzero().flatten().to(INT)


# A layer's weights, in layer()'s order; HALVES only for a layer with half
# coefficients (a handful: the engine's guarded halvings).
WEIGHTS = ("in_slots", "w_in", "read_kind", "out_slots", "w_out", "ff_in_slots", "w_ff",
           "ff_out_slots", "w_down")
HALVES = ("h_out", "h_ff", "h_down")


def linear(W: torch.Tensor, x: torch.Tensor, halves: torch.Tensor | None = None) -> torch.Tensor:
    """W @ x, plus the half coefficients as one floor shift: exact, and
    never larger than the engine's own sums.  A multiply and a row sum,
    which compiles into the surrounding kernel (an integer matmul would be
    a library call of its own)."""
    y = (W * x).sum(1, dtype=INT)
    return y if halves is None else y + ((halves * x).sum(1, dtype=INT) >> 1)


def indexed_read(memory: torch.Tensor, address: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each address's byte and whether the address exists: attention over
    memory (attention_read) evaluated by indexing, which it equals."""
    valid = (address >= 0) & (address < memory.numel())
    return memory[address.clamp(0, memory.numel() - 1)].to(INT) * valid, valid


def attention_read(memory: torch.Tensor, address: torch.Tensor,
                   temperature: float = 32.0) -> tuple[torch.Tensor, torch.Tensor]:
    """indexed_read as softmax attention: memory is the KV cache, one entry
    per address k with key [4k, k - 2k^2] and value (byte, k); a query for
    address q is [q, 1].  No two keys tie and the best leads by >= 1, so
    the rounded weighted values are the best key's exactly.  A query for no
    address gets its nearest key, whose address differs: masked to 0, as
    the graph masks a read whose returned address differs from the query."""
    keys = torch.arange(memory.numel(), dtype=torch.float64)
    q = address.to(torch.float64)[:, None]
    scores = 4 * q * keys + keys - 2 * keys * keys
    weights = torch.softmax(temperature * (scores - scores.max(dim=1, keepdim=True).values), dim=1)
    byte = torch.round(weights @ memory.to(torch.float64)).to(INT)
    found = torch.round(weights @ keys).to(address.dtype)
    valid = found == address
    return byte * valid, valid


def layer(x, memory, shift, in_slots, w_in, read_kind, out_slots, w_out, ff_in_slots, w_ff,
          ff_out_slots, w_down, h_out=None, h_ff=None, h_down=None, *, read=indexed_read):
    """Apply one layer and return its distinct memory-read addresses.

    Projections touch only live slots. read_kind selects each query's byte,
    address or bit; w_ff holds ReGLU gates followed by values.
    """
    n_values = read_kind.shape[0]
    query_rows = w_in.shape[0] - w_out.shape[1] + n_values
    projected = linear(w_in, x[in_slots])
    torch._check(projected.shape[0] >= query_rows)
    qx, qy = projected[0:query_rows:2], projected[1:query_rows:2]
    address = torch.div(qx, torch.where(qy == 0, 1, qy), rounding_mode="floor")
    byte, valid = read(memory, address)
    addresses_read = torch.where(valid, address, -1)
    address = address * valid
    # Each memory value channel serves the byte (-1), the
    # address (-2) or bit n of the byte (n).
    query, kinds = read_kind[:, 0], read_kind[:, 1]
    byte, address = byte[query], address[query]
    served = torch.where(
        kinds == -1, byte,
        torch.where(kinds == -2, address, (byte >> kinds.clamp(min=0)) & 1))
    reads, values = w_out[:, :n_values], w_out[:, n_values:]
    read_halves, value_halves = (None, None) if h_out is None else (h_out[:, :n_values], h_out[:, n_values:])
    attention = (linear(reads, served << shift, read_halves)
                 + linear(values, projected[query_rows:], value_halves))
    x = x.index_add(0, out_slots, attention)
    n_ffn = w_down.shape[1]
    gate_value = linear(w_ff, x[ff_in_slots], h_ff)
    torch._check(gate_value.shape[0] == 2 * n_ffn)
    gate, value = gate_value[:n_ffn], gate_value[n_ffn:]
    act = torch.where(gate > 0, (gate * value) >> shift, 0)
    return x.index_add(0, ff_out_slots, linear(w_down, act, h_down)), addresses_read


class ExpertLayer(torch.nn.Module):
    """One layer's weights (WEIGHTS, and HALVES if it has any), cut from
    the checkpoint's: Q/K/V and output projections, 2-dimensional heads
    that either attend to their own token (passthrough) or read the byte
    their query names (a memory head), then a ReGLU feed-forward block."""

    def __init__(self, tensors: dict, stem: str, heads: int, ffn: int, d_model: int,
                 kinds: list[int], outputs: list):
        super().__init__()
        w_in = dense(tensors, f"{stem}.in_proj", 6 * heads, d_model)
        w_out = dense(tensors, f"{stem}.out_proj", d_model, 2 * heads)
        w_ff = dense(tensors, f"{stem}.ff_in", 2 * ffn, d_model)
        w_down = dense(tensors, f"{stem}.ff_out", d_model, ffn)
        # Head h's query is in-projection rows 2h, 2h + 1, its value rows
        # 4H + 2h, + 1; its output, out-projection columns 2h, 2h + 1.
        reads = [2 * h + c for h in range(heads) if kinds[h] == LOOKUP for c in range(2)]
        passes = [2 * h + c for h in range(heads) if kinds[h] == PASSTHROUGH for c in range(2)]
        if (w_in % 2).any():
            raise ValueError(f"{stem}: half coefficients in the in-projection")
        w_in = w_in[reads + [4 * heads + c for c in passes]]
        w_out = w_out[:, reads + passes]
        w_in, w_out, read_kind = share_queries(w_in, w_out, torch.tensor(
            [read_codes(outputs[h]) for h in range(heads) if kinds[h] == LOOKUP],
            dtype=INT).reshape(-1, 2))
        in_slots, out_slots = touched(w_in, 0), touched(w_out, 1)
        ff_in_slots, ff_out_slots = touched(w_ff, 0), touched(w_down, 1)
        # The checkpoint's coefficients are in units of a half: split each
        # matrix into its whole coefficients and the halves left over.
        doubled = {"w_in": w_in[:, in_slots], "w_out": w_out[out_slots],
                   "w_ff": w_ff[:, ff_in_slots], "w_down": w_down[ff_out_slots]}
        halves = {f"h_{name[2:]}": W - 2 * (W >> 1) for name, W in doubled.items() if name != "w_in"}
        self.has_halves = any(h.any() for h in halves.values())
        weights = {
            "in_slots": in_slots, "out_slots": out_slots, "ff_in_slots": ff_in_slots,
            "ff_out_slots": ff_out_slots,
            "read_kind": read_kind,
            **{name: W >> 1 for name, W in doubled.items()},
        }
        for name in WEIGHTS + (HALVES if self.has_halves else ()):
            self.register_buffer(name, (weights | halves)[name].contiguous())

    def weights(self) -> list[torch.Tensor]:
        """layer()'s weight arguments."""
        return [getattr(self, name) for name in WEIGHTS + (HALVES if self.has_halves else ())]

    def forward(self, x: torch.Tensor, memory: torch.Tensor, shift: int,
                read=indexed_read) -> tuple[torch.Tensor, torch.Tensor]:
        return layer(x, memory, shift, *self.weights(), read=read)


class Expert(torch.nn.Module):
    """One exported expert (``E<index>.*`` tensors and its experts.json
    entry): its layers, and its step ABI (``spec``)."""

    def __init__(self, tensors: dict, meta: dict):
        super().__init__()
        self.meta, self.spec = meta, meta["step_spec"]
        self.layers = torch.nn.ModuleList(
            ExpertLayer(tensors, f"E{meta['index']}.L{li}", meta["layer_heads"][li],
                        meta["layer_ffn"][li], meta["d_model"],
                        meta["layer_head_kinds"][li], meta["head_memory_outputs"][li])
            for li in range(meta["n_layers"])
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor, shift: int,
                read=indexed_read) -> torch.Tensor:
        for expert_layer in self.layers:
            x, _addresses = expert_layer(x, memory, shift, read)
        return x


class Cpu:
    """The 6502 as routed expert forwards over a flat 64 KiB memory.  Every
    read of it -- the experts' memory heads and the router's opcode byte
    -- goes through `memory_read`
    (indexed_read, or attention_read, which it equals)."""

    def __init__(self, checkpoint: str, image: bytes, pc: int, *, s: int = 0xFD, p: int = 0x24,
                 memory_read=indexed_read):
        self.memory_read = memory_read
        with open(os.path.join(checkpoint, "experts.json")) as f:
            meta = json.load(f)
        if meta.get("execution_abi") != 3:
            raise ValueError("unsupported expert execution ABI; export a new checkpoint")
        tensors = load_file(os.path.join(checkpoint, "expert_weights.safetensors"))
        self.shift = meta["scale_bits"]
        experts = [Expert(tensors, m) for m in meta["experts"]]
        self.by_opcode = {e.meta["byte"]: e for e in experts if e.meta["kind"] == "exec_opcode"}
        self.interrupt = {e.meta["nmi"]: e for e in experts if e.meta["kind"] == "exec_int"}
        self.memory = torch.tensor(list(image), dtype=INT)
        self.state = {"pc": pc, "a": 0, "x": 0, "y": 0, "s": s, "p": p, "halted": 0}

    def step(self, nmi: bool = False, irq: bool = False) -> int:
        """One instruction (or interrupt sequence) through its expert;
        returns its cycle count."""
        st, memory = self.state, self.memory
        pc = st["pc"]
        expert = self.interrupt[nmi] if nmi or irq else self.by_opcode[self.peek(pc)]
        spec, shift = expert.spec, self.shift
        x = torch.zeros(expert.meta["d_model"], dtype=INT)
        x[expert.meta["one_slot"]] = 1 << shift
        inputs = {"is_state": 1, "pc_lo": pc & 0xFF, "pc_hi": pc >> 8, "a": st["a"],
                  "x": st["x"], "y": st["y"], "s": st["s"], "halted": st["halted"],
                  **{f"a_bit{b}": (st["a"] >> b) & 1 for b in range(8)}}
        for name, slot in spec["inputs"].items():
            x[slot] = inputs[name] << shift
        for slot, mask in spec["p_flags"]:
            x[slot] = (1 << shift) if st["p"] & mask else 0
        x = expert(x, memory, shift, self.memory_read)

        def read(slot: int) -> int:
            return int(x[slot]) >> shift

        final = spec["sequence_final"]
        for name, (register, mask) in REGISTERS.items():
            if name in final["model"]:
                st[register] = read(final["model"][name]) & mask
            elif name in final["constant"]:
                st[register] = final["constant"][name] & mask
        for name, mask in FLAG_MASKS.items():
            if name in final["model"]:
                bit = read(final["model"][name]) & 1
            elif name in final["constant"]:
                bit = final["constant"][name]
            else:
                continue
            st["p"] = (st["p"] & ~mask) | (mask if bit else 0)
        if spec["sequence_halt"] is not None:
            st["halted"] = read(spec["sequence_halt"]) & 1
        for step_writes in spec["sequence_writes"]:
            for write in step_writes:
                memory[read(write["address"]) & 0xFFFF] = read(write["value"]) & 0xFF
        cycles = spec["sequence_cycles"]
        return cycles["constant"] + sum(read(slot) for slot in cycles["model"])

    def peek(self, address: int) -> int:
        """One byte of memory, read as the experts read it."""
        byte, _valid = self.memory_read(self.memory, torch.tensor([address & 0xFFFF], dtype=INT))
        return int(byte[0])

def nestest(checkpoint: str, rom: str, log: str, steps: int | None = None, *,
            attention: bool = False) -> int:
    """nestest from $C000 against the golden log: PC, A, X, Y, P, SP and
    CYC before every instruction, memory read by `attention` softmax or by
    index.  Returns the instructions checked."""
    with open(rom, "rb") as f:
        prg = f.read()[16 : 16 + 0x4000]
    image = bytearray(0x10000)
    image[0x8000:0xC000] = image[0xC000:] = prg
    with open(log) as f:
        lines = f.readlines()[:steps]
    cpu = Cpu(checkpoint, bytes(image), 0xC000,
              memory_read=attention_read if attention else indexed_read)
    cycles = 7
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
    parser.add_argument("checkpoint", help="directory with expert_weights.safetensors and experts.json")
    parser.add_argument("--nestest", nargs=2, metavar=("ROM", "LOG"), required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--attention", action="store_true",
                        help="read memory by softmax attention over all of it, not by index")
    args = parser.parse_args()
    start = time.perf_counter()
    with torch.inference_mode():
        n = nestest(args.checkpoint, *args.nestest, args.steps, attention=args.attention)
    elapsed = time.perf_counter() - start
    print(f"nestest: {n} instructions match the golden log (registers and cycles) "
          f"in {elapsed:.1f}s, {n / elapsed:.0f} instructions/s")


if __name__ == "__main__":
    main()
