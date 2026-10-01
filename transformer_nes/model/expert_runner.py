"""Run the exported folded 6502 experts as a mixture of small PyTorch
transformers, routed by opcode.

The union checkpoint (checkpoint_runner.py) is the whole 6502 in one
63-layer model.  ``6502-export --experts`` writes what the generated
engine actually runs instead: one small model per opcode (and per
interrupt), each the union graph partially evaluated with that opcode
pinned -- a handful of layers and a few hundred weights.  Running them is
a mixture of experts whose router is the byte at PC.

Two things make an expert cheap besides its size, both in experts.json:

- its memory reads are *structural*: a lookup head's query names one
  address (``qx / qy``), so it reads that byte -- or the address, or one
  of the byte's bits -- from a flat memory, and a write is an assignment;
- its ``step_spec`` is the instruction's ABI: which registers it reads,
  which outputs are the next registers, flags, writes, control flow,
  halt and cycle count (folded to constants where the opcode decides
  them).

Like checkpoint_runner.py this imports only torch and safetensors.
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
REGISTERS = {"new_a": "a", "new_x": "x", "new_y": "y", "new_s": "s"}


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


def layer(x, memory, shift, in_slots, w_in, read_kind, out_slots, w_out, ff_in_slots, w_ff,
          ff_out_slots, w_down, h_out=None, h_ff=None, h_down=None):
    """One layer on the residual `x`: its output, and the address each
    memory head read (-1 where its query names none; heads sharing a query
    read the same one).

    Neither kind of head needs keys: a passthrough head's output is its
    own value, and a memory head's query s * [address, 1] names the byte
    the hardmax picks.  So the in-projection w_in is the memory heads'
    queries, then the passthrough heads' values; the out-projection w_out
    takes what the reads serve, then those values; and w_ff is the
    ReGLU's gates, then its values.  A layer also reads and writes few of
    the residual's slots: each projection has only the slots it reads
    (gathered) and writes (added back).  (Each part is used where it lies,
    never copied out, so the compiled layer makes few intermediates.)"""
    n_queries = 2 * read_kind.shape[0]
    projected = linear(w_in, x[in_slots])
    torch._check(projected.shape[0] >= n_queries)
    qx, qy = projected[0:n_queries:2], projected[1:n_queries:2]
    address = torch.div(qx, torch.where(qy == 0, 1, qy), rounding_mode="floor")
    valid = (address >= 0) & (address < memory.numel())
    byte = memory[address.clamp(0, memory.numel() - 1)].to(INT) * valid
    read = torch.where(valid, address, -1)
    address = address * valid
    # Each memory head's two output components serve the byte (-1), the
    # address (-2) or bit n of the byte (n).
    served = torch.where(
        read_kind == -1, byte[:, None],
        torch.where(read_kind == -2, address[:, None],
                    torch.where(read_kind >= 0, (byte[:, None] >> read_kind.clamp(min=0)) & 1, 0)))
    reads, values = w_out[:, :n_queries], w_out[:, n_queries:]
    read_halves, value_halves = (None, None) if h_out is None else (h_out[:, :n_queries], h_out[:, n_queries:])
    attention = (linear(reads, served.reshape(-1) << shift, read_halves)
                 + linear(values, projected[n_queries:], value_halves))
    x = x.index_add(0, out_slots, attention)
    n_ffn = w_down.shape[1]
    gate_value = linear(w_ff, x[ff_in_slots], h_ff)
    torch._check(gate_value.shape[0] == 2 * n_ffn)
    gate, value = gate_value[:n_ffn], gate_value[n_ffn:]
    act = torch.where(gate > 0, (gate * value) >> shift, 0)
    return x.index_add(0, ff_out_slots, linear(w_down, act, h_down)), read


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
            "read_kind": torch.tensor([read_codes(outputs[h]) for h in range(heads) if kinds[h] == LOOKUP],
                                      dtype=INT).reshape(-1, 2),
            **{name: W >> 1 for name, W in doubled.items()},
        }
        for name in WEIGHTS + (HALVES if self.has_halves else ()):
            self.register_buffer(name, (weights | halves)[name].contiguous())

    def weights(self) -> list[torch.Tensor]:
        """layer()'s weight arguments."""
        return [getattr(self, name) for name in WEIGHTS + (HALVES if self.has_halves else ())]

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                shift: int) -> tuple[torch.Tensor, torch.Tensor]:
        return layer(x, memory, shift, *self.weights())


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

    def forward(self, x: torch.Tensor, memory: torch.Tensor, shift: int) -> torch.Tensor:
        return self.forward_reads(x, memory, shift)[0]

    def forward_reads(self, x: torch.Tensor, memory: torch.Tensor,
                      shift: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The forward, and each memory head's address, layer by layer
        (layer())."""
        reads = []
        for expert_layer in self.layers:
            x, read = expert_layer(x, memory, shift)
            reads.append(read)
        return x, torch.cat(reads) if reads else torch.zeros(0, dtype=INT)


class Cpu:
    """The 6502 as routed expert forwards over a flat 64 KiB memory."""

    def __init__(self, checkpoint: str, image: bytes, pc: int, *, s: int = 0xFD, p: int = 0x24):
        with open(os.path.join(checkpoint, "experts.json")) as f:
            meta = json.load(f)
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
        expert = (self.interrupt[nmi] if nmi or irq
                  else self.by_opcode[int(memory[pc])])
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
        x = expert(x, memory, shift)

        def read(slot: int) -> int:
            return int(x[slot]) >> shift

        final = spec["sequence_final"]
        for name, register in REGISTERS.items():
            if name in final["model"]:
                st[register] = read(final["model"][name]) & 0xFF
            elif name in final["constant"]:
                st[register] = final["constant"][name] & 0xFF
        for name, mask in FLAG_MASKS.items():
            if name in final["model"]:
                bit = read(final["model"][name]) & 1
            elif name in final["constant"]:
                bit = final["constant"][name]
            else:
                continue
            st["p"] = (st["p"] & ~mask) | (mask if bit else 0)
        st["pc"] = self.successor(spec["sequence_control"], pc, read) & 0xFFFF
        if spec["sequence_halt"] is not None:
            st["halted"] = read(spec["sequence_halt"]) & 1
        for step_writes in spec["sequence_writes"]:
            for addr_slot, value_slot in step_writes:
                memory[read(addr_slot) & 0xFFFF] = read(value_slot) & 0xFF
        cycles = spec["sequence_cycles"]
        return cycles["constant"] + sum(read(slot) & 0xFF for slot in cycles["model"])

    def successor(self, control: dict, pc: int, read) -> int:
        """The next PC from the expert's control ABI (an operand byte, a
        model output, or a fixed offset; branches by the flag they test)."""
        memory = self.memory
        if control["target_dimension"] is not None:
            target = fallthrough = read(control["target_dimension"])
        elif control["absolute_operand_offset"] is not None:
            at = pc + control["absolute_operand_offset"]
            target = fallthrough = int(memory[at & 0xFFFF]) | int(memory[(at + 1) & 0xFFFF]) << 8
        elif control["relative_to_head"]:
            fallthrough, target = pc + control["fallthrough"], pc + control["target"]
        else:
            fallthrough, target = control["fallthrough"], control["target"]
        if control["relative_operand_offset"] is not None:
            offset = int(memory[(pc + control["relative_operand_offset"]) & 0xFFFF])
            target = pc + control["fallthrough"] + (offset - 256 if offset >= 128 else offset)
        if control["flag_mask"]:
            mask = control["flag_mask"]
            taken = (self.state["p"] & mask) == (mask if control["flag_value"] else 0)
        elif control["predicate"] is not None:
            taken = read(control["predicate"]) & 1
        else:
            return fallthrough
        return target if taken else fallthrough


def nestest(checkpoint: str, rom: str, log: str, steps: int | None = None) -> int:
    """nestest from $C000 against the golden log: PC, A, X, Y, P, SP and
    CYC before every instruction.  Returns the instructions checked."""
    with open(rom, "rb") as f:
        prg = f.read()[16 : 16 + 0x4000]
    image = bytearray(0x10000)
    image[0x8000:0xC000] = image[0xC000:] = prg
    with open(log) as f:
        lines = f.readlines()[:steps]
    cpu = Cpu(checkpoint, bytes(image), 0xC000)
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
    args = parser.parse_args()
    start = time.perf_counter()
    with torch.inference_mode():
        n = nestest(args.checkpoint, *args.nestest, args.steps)
    elapsed = time.perf_counter() - start
    print(f"nestest: {n} instructions match the golden log (registers and cycles) "
          f"in {elapsed:.1f}s, {n / elapsed:.0f} instructions/s")


if __name__ == "__main__":
    main()
