"""The NES's transformers as PyTorch modules.

Every chip is a set of token types over one memory. A token type's circuit is
a decoder stack over a residual stream of exact int32 values, in units of
2**-f_bits (float32 for the audio mixer), with no normalization and no
position encoding. Each layer is

    attention: an in-projection forms memory-read queries (an address each,
               exact; the attention over the WRITE log evaluates to the byte
               the address names) and passthrough values; an out-projection
               adds the read bytes, addresses or bits and the values back
    feed-forward: ReGLU, relu(gate) * value, then a down-projection

The weights are the nonzero coefficients alone. A neuron reads a handful of
slots, so each projection is stored sparse, as (row, column, coefficient)
triples over residual slots, and applied with index_add: exact, since integer
arithmetic wraps at 32 bits as the native kernels do. A few coefficients are
halves, kept apart (``h_*``) so each sum of them rounds once. Float32 sums run
in the native kernels' exact IEEE order (``ordered``).

A specialization is the same token type folded with some of its values
pinned (graph/fold.py): a smaller circuit, valid wherever the pins hold.
Choosing one is the caller's job. A token type whose callers always choose
one has no full circuit at all.

These modules hold weights only. Memory is state: the caller owns it and
passes it to every forward, as a language model's KV cache is the runtime's.
This file imports only torch.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

INT = torch.int32
MATRICES = ("w_in", "w_out", "w_ff", "w_down")
HALVES = ("h_out", "h_ff", "h_down")


def project(rows: Tensor, cols: Tensor, coef: Tensor, n: int, x: Tensor) -> Tensor:
    """y[r] = sum of coef * x[c] over a projection's (r, c, coef) triples."""
    return torch.zeros(n, dtype=INT).index_add(0, rows, coef * x[cols])


def ordered(rows: Tensor, cols: Tensor, coef: Tensor, x: Tensor) -> list[tuple[int, Tensor]]:
    """(row, sum) for each row with coefficients, in float32 and the native
    kernels' order: columns ascending, the first term unadded, a unit
    coefficient unmultiplied. Triples are stored row-major in that order."""
    sums: dict[int, Tensor] = {}
    for r, c, k in zip(rows.tolist(), cols.tolist(), coef, strict=True):
        term = x[c] if k == 1 else x[c] * k
        sums[r] = term if r not in sums else sums[r] + term
    return list(sums.items())


class Layer(nn.Module):
    """One decoder layer: memory-read attention and passthrough values, then ReGLU.

    Buffers: ``read_kind`` (each read channel's query and what it serves),
    ``dims`` (in-projection rows, out-projection channels, ReGLU neurons),
    and ``<matrix>_rows/_cols/_coef`` per projection: w_in maps residual slots
    to query pairs then passthrough values; w_out maps read channels then
    those values to slots; w_ff maps slots to ReGLU gates then values; w_down
    maps neurons to slots.
    """

    def __init__(self, weights: dict[str, Tensor]):
        super().__init__()
        self.has_halves = f"{HALVES[0]}_coef" in weights
        for name in ("read_kind", "dims"):
            self.register_buffer(name, weights[name].contiguous())
        for matrix in MATRICES + (HALVES if self.has_halves else ()):
            for part in ("rows", "cols", "coef"):
                self.register_buffer(f"{matrix}_{part}", weights[f"{matrix}_{part}"].contiguous())

    def linear(self, matrix: str, n: int, x: Tensor, keep=None) -> Tensor:
        rows, cols, coef = (getattr(self, f"{matrix}_{p}") for p in ("rows", "cols", "coef"))
        if keep is not None:
            rows, cols, coef = rows[keep], cols[keep], coef[keep]
        return project(rows, cols, coef, n, x)

    def halves(self, matrix: str, n: int, x: Tensor, keep=None) -> Tensor:
        return self.linear(matrix, n, x, keep) >> 1 if self.has_halves else 0

    def forward(self, x: Tensor, memory: Tensor, f_bits: int) -> Tensor:
        if x.dtype != INT:
            return self.forward_float(x)
        n_rows, n_channels, n_ffn = self.dims.tolist()
        n_values = self.read_kind.shape[0]
        query_rows = n_rows - n_channels + n_values
        projected = self.linear("w_in", n_rows, x)
        channels = projected[query_rows:]
        if n_values:
            # Each query pair (qx, qy) names the address qx / qy; a read
            # outside memory gives byte 0 and address 0.
            qx, qy = projected[0:query_rows:2], projected[1:query_rows:2]
            address = torch.div(qx, torch.where(qy == 0, 1, qy), rounding_mode="floor")
            valid = (address >= 0) & (address < memory.numel())
            byte = memory[address.clamp(0, memory.numel() - 1)].to(INT) * valid
            address = address * valid
            # Each read channel serves its query's byte (-1), address (-2) or bit n.
            query, kind = self.read_kind[:, 0], self.read_kind[:, 1]
            byte, address = byte[query], address[query]
            served = torch.where(
                kind == -1, byte, torch.where(kind == -2, address, (byte >> kind.clamp(min=0)) & 1)
            )
            channels = torch.cat([served << f_bits, channels])
        d_model = x.shape[0]
        # Read and value halves round separately, as their sums are separate.
        read_half = self.h_out_cols < n_values if self.has_halves else None
        x = x + self.linear("w_out", d_model, channels)
        if self.has_halves:
            x = x + self.halves("h_out", d_model, channels, read_half)
            x = x + self.halves("h_out", d_model, channels, ~read_half)
        gate_value = self.linear("w_ff", 2 * n_ffn, x) + self.halves("h_ff", 2 * n_ffn, x)
        gate, value = gate_value[:n_ffn], gate_value[n_ffn:]
        act = torch.where(gate > 0, (gate * value) >> f_bits, 0)
        return x + self.linear("w_down", d_model, act) + self.halves("h_down", d_model, act)

    def forward_float(self, x: Tensor) -> Tensor:
        """The float32 layer: passthrough values only, no memory reads."""
        n_rows, n_channels, n_ffn = self.dims.tolist()
        if self.read_kind.shape[0]:
            raise ValueError("float32 circuits read no memory")
        triples = {m: (getattr(self, f"{m}_rows"), getattr(self, f"{m}_cols"), getattr(self, f"{m}_coef"))
                   for m in MATRICES}
        projected = torch.zeros(n_rows, dtype=x.dtype)
        for r, v in ordered(*triples["w_in"], x):
            projected[r] = v
        values = projected[n_rows - n_channels:]
        x = x.clone()
        zero = torch.zeros((), dtype=x.dtype)
        for r, v in ordered(*triples["w_out"], values):
            x[r] = x[r] + (zero + v)  # the kernel adds the (empty) read sum first
        gate_value = torch.zeros(2 * n_ffn, dtype=x.dtype)
        for r, v in ordered(*triples["w_ff"], x):
            gate_value[r] = v
        act = torch.relu(gate_value[:n_ffn]) * gate_value[n_ffn:]
        for r, v in ordered(*triples["w_down"], act):
            x[r] = x[r] + v
        return x

    def matrices(self) -> tuple[str, ...]:
        return MATRICES + (HALVES if self.has_halves else ())

    def weight_count(self) -> int:
        """Its coefficients: every stored weight is nonzero."""
        return sum(getattr(self, f"{m}_coef").numel() for m in self.matrices())

    def structure(self) -> dict:
        """The shapes of its weights: read channels and nonzeros per projection."""
        return {
            "reads": self.read_kind.shape[0],
            "nonzero": {m: getattr(self, f"{m}_coef").numel() for m in self.matrices()},
        }

    @classmethod
    def empty(cls, structure: dict, dtype: torch.dtype = INT) -> Layer:
        """A layer of zeros shaped by ``structure()``, to load weights into."""
        weights = {"read_kind": torch.zeros(structure["reads"], 2, dtype=INT), "dims": torch.zeros(3, dtype=INT)}
        for matrix, n in structure["nonzero"].items():
            for part in ("rows", "cols", "coef"):
                weights[f"{matrix}_{part}"] = torch.zeros(n, dtype=dtype if part == "coef" else INT)
        return cls(weights)


class Transformer(nn.Module):
    """One circuit: embedding into residual slots, the layers, readout slots."""

    def __init__(self, layers: list[dict[str, Tensor] | Layer], *, d_model: int, one_slot: int,
                 input_slots: Tensor, output_slots: Tensor, dtype: torch.dtype = INT):
        super().__init__()
        self.d_model, self.one_slot, self.dtype = d_model, one_slot, dtype
        self.layers = nn.ModuleList(w if isinstance(w, Layer) else Layer(w) for w in layers)
        # input_slots[i] is input i's slot, or -1 when the circuit never reads it.
        self.register_buffer("input_slots", input_slots.to(INT))
        self.register_buffer("output_slots", output_slots.to(INT))

    def forward(self, inputs: Tensor, memory: Tensor, f_bits: int) -> Tensor:
        x = torch.zeros(self.d_model, dtype=self.dtype)
        live = self.input_slots >= 0
        values = inputs.to(self.dtype)[live]
        x[self.input_slots[live]] = values << f_bits if self.dtype == INT else values
        x[self.one_slot] = 1 << f_bits
        for layer in self.layers:
            x = layer(x, memory, f_bits)
        out = x[self.output_slots]
        if self.dtype == INT and f_bits:
            if (out & ((1 << f_bits) - 1)).any():
                raise ValueError("fractional readout value")
            out = out >> f_bits
        return out

    def weight_count(self) -> int:
        return sum(layer.weight_count() for layer in self.layers)


class Specializations(nn.ModuleDict):
    """Circuits by specialization name. A name may coincide with a dict
    method (a PPU event called ``clear``): entries are only ever indexed."""

    def __setitem__(self, name: str, module: nn.Module) -> None:
        if not name or "." in name:
            raise KeyError(f"bad specialization name {name!r}")
        self._modules[name] = module


class TokenType(nn.Module):
    """A kind of token: named inputs and outputs, the full circuit, and its
    specializations by name (each with the pins that make it valid).

    ``reassociate_residuals`` is a compiler hint, not semantics: False keeps
    a hot circuit's measured projection grouping when it is lowered."""

    def __init__(self, inputs: list[str], outputs: list[str], general: Transformer | None,
                 specialized: dict[str, Transformer], pins: dict[str, dict[str, float]],
                 *, reassociate_residuals: bool = True):
        super().__init__()
        if general is None and not specialized:
            raise ValueError("a token type needs a circuit")
        self.inputs, self.outputs, self.pins = list(inputs), list(outputs), dict(pins)
        self.reassociate_residuals = reassociate_residuals
        self.general = general  # None: callers always choose a specialization
        self.specialized = Specializations(specialized)

    def circuits(self) -> list[tuple[str | None, Transformer]]:
        """The full circuit if it has one, then each specialization."""
        return ([] if self.general is None else [(None, self.general)]) + list(self.specialized.items())

    def forward(self, inputs: Tensor, memory: Tensor, f_bits: int, specialization: str | None = None):
        if specialization is None and self.general is None:
            raise ValueError("this token type runs only its specializations; choose one")
        circuit = self.general if specialization is None else self.specialized[specialization]
        return circuit(inputs, memory, f_bits)


class Chip(nn.Module):
    """A chip: its token types over one memory of ``memory_size`` bytes.

    ``constants`` are facts about its circuits that their callers need, such
    as the longest span the APU's oscillator circuit covers."""

    def __init__(self, tokens: dict[str, TokenType], *, memory_size: int, f_bits: int,
                 dtype: torch.dtype = INT, constants: dict[str, int] | None = None):
        super().__init__()
        self.memory_size, self.f_bits, self.dtype = memory_size, f_bits, dtype
        self.constants = dict(constants or {})
        self.tokens = nn.ModuleDict(tokens)

    def circuits(self) -> list[tuple[str, str | None, Transformer]]:
        """Every circuit in model-id order: token by token, full circuit first."""
        return [(name, spec, c) for name, token in self.tokens.items() for spec, c in token.circuits()]

    def memory(self) -> Tensor:
        """A fresh, zeroed memory for this chip."""
        return torch.zeros(self.memory_size, dtype=torch.uint8)

    def forward(self, token: str, inputs: Tensor, memory: Tensor, specialization: str | None = None):
        return self.tokens[token](inputs, memory, self.f_bits, specialization)

    def weight_count(self) -> int:
        return sum(c.weight_count() for _t, _s, c in self.circuits())

    # ── Persistence: the structure as JSON-able data, the weights as tensors ──

    def structure(self) -> dict:
        return {
            "memory_size": self.memory_size,
            "f_bits": self.f_bits,
            "dtype": str(self.dtype).removeprefix("torch."),
            "constants": self.constants,
            "tokens": {
                name: {
                    "inputs": token.inputs,
                    "outputs": token.outputs,
                    "pins": token.pins,
                    "reassociate_residuals": token.reassociate_residuals,
                    "circuits": {
                        "general" if spec is None else f"specialized.{spec}": {
                            "d_model": c.d_model,
                            "one_slot": c.one_slot,
                            "layers": [layer.structure() for layer in c.layers],
                        }
                        for spec, c in token.circuits()
                    },
                }
                for name, token in self.tokens.items()
            },
        }

    @classmethod
    def empty(cls, structure: dict) -> Chip:
        """A chip of zeros shaped by ``structure()``, to load weights into."""

        dtype = getattr(torch, structure["dtype"])

        def circuit(token: dict, meta: dict) -> Transformer:
            return Transformer(
                [Layer.empty(layer, dtype) for layer in meta["layers"]], d_model=meta["d_model"],
                one_slot=meta["one_slot"], input_slots=torch.zeros(len(token["inputs"]), dtype=INT),
                output_slots=torch.zeros(len(token["outputs"]), dtype=INT), dtype=dtype,
            )

        tokens = {}
        for name, t in structure["tokens"].items():
            circuits = {key: circuit(t, meta) for key, meta in t["circuits"].items()}
            tokens[name] = TokenType(
                t["inputs"], t["outputs"], circuits.pop("general", None),
                {key.removeprefix("specialized."): c for key, c in circuits.items()}, t["pins"],
                reassociate_residuals=t["reassociate_residuals"],
            )
        return cls(tokens, memory_size=structure["memory_size"], f_bits=structure["f_bits"],
                   dtype=dtype, constants=structure["constants"])


# ── The saved form: one flat buffer per integer width ──────────────
# Each tensor of a state_dict, in its order, goes into the buffer of the
# narrowest integer type that holds it exactly; ``widths`` records which.
# Keys and shapes are the model's own: load into an ``empty()`` model.


def narrow(t: Tensor) -> Tensor:
    lo, hi = (int(t.min()), int(t.max())) if t.numel() else (0, 0)
    for dtype in (torch.int8, torch.int16, torch.int32):
        info = torch.iinfo(dtype)
        if info.min <= lo and hi <= info.max:
            return t.to(dtype)
    raise OverflowError("an integer weight exceeds int32")


# Buffer of each width code: integers by bits, 0 for float32 coefficients.
BUFFERS = {8: torch.int8, 16: torch.int16, 32: torch.int32, 0: torch.float32}
WIDTHS = {dtype: code for code, dtype in BUFFERS.items()}


def pack_state(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Buffers by width (``int8``, ``int16``, ``int32``, ``float32``) and
    ``widths``, each tensor's width code."""
    parts: dict[int, list[Tensor]] = {code: [] for code in BUFFERS}
    widths = []
    for t in state.values():
        packed = (t if t.dtype == torch.float32 else narrow(t)).flatten()
        widths.append(WIDTHS[packed.dtype])
        parts[widths[-1]].append(packed)
    buffers = {
        str(dtype).removeprefix("torch."): torch.cat(parts[code]) if parts[code] else torch.zeros(0, dtype=dtype)
        for code, dtype in BUFFERS.items()
    }
    return buffers | {"widths": torch.tensor(widths, dtype=torch.uint8)}


def unpack_state(shapes: dict[str, Tensor], buffers: dict[str, Tensor]) -> dict[str, Tensor]:
    """The state_dict of a model whose state_dict is ``shapes`` (an
    ``empty()`` model's), from its ``pack_state`` buffers."""
    widths = buffers["widths"].tolist()
    if len(widths) != len(shapes):
        raise ValueError("saved weights do not fit this model's structure")
    offsets, state = dict.fromkeys(BUFFERS, 0), {}
    for (key, like), width in zip(shapes.items(), widths, strict=True):
        n = like.numel()
        buffer = buffers[str(BUFFERS[width]).removeprefix("torch.")]
        flat = buffer[offsets[width] : offsets[width] + n]
        offsets[width] += n
        state[key] = flat.to(like.dtype).reshape(like.shape)
    return state
