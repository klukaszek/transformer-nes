"""Construct chips' transformers from their circuits.

A token type is a circuit (a ProgramGraph built with the graph DSL), the
caller's input and output order, and named specializations: pins under which
partial evaluation (graph/fold.py) folds the circuit to a smaller one. Each
circuit is scheduled into layers (schedule.py), turned into analytic weights
(weights.py) and cut into the per-layer arrays a model.transformer.Layer
holds. The result is a model.transformer.Chip: PyTorch modules with no
construction code behind them.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from transformer_nes.graph.core import (
    Expression,
    InputDimension,
    ProgramGraph,
    project_program_graph,
)
from transformer_nes.graph.fold import GraphFolder
from transformer_nes.model.compiler import layer_arrays
from transformer_nes.model.schedule import asap_schedule
from transformer_nes.model.transformer import Chip, TokenType, Transformer
from transformer_nes.model.weights import build_model


@dataclass(frozen=True)
class TokenGraph:
    """A token type's circuit, the caller's input and output order for it,
    and its specializations (name -> pins: persist dim name -> constant)."""

    program_graph: ProgramGraph
    inputs: Sequence[str]
    outputs: Sequence[str]
    experts: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    # Reassociation is exact for int32, but changes native register allocation.
    # Hot tokens can retain their measured projection grouping.
    reassociate_residuals: bool = True
    # False when callers always choose a specialization: the full circuit
    # would never run, so it is not built.
    full_circuit: bool = True

    def project(self, outputs: Sequence[str]) -> TokenGraph:
        """Only the dependency closure of selected readouts."""
        if not set(outputs) <= set(self.outputs):
            raise ValueError("projection must select existing token outputs")
        dims = {d.name: d for d in self.program_graph.all_dims}
        readout = Expression({dims[name]: 1e-6 for name in outputs})
        projected = project_program_graph(self.program_graph, {"READOUT": readout})
        # A pin on a value the projection no longer computes is moot.
        kept = {d.name for d in projected.all_dims if d.kind == "persist"}
        experts = {name: {k: v for k, v in pins.items() if k in kept} for name, pins in self.experts.items()}
        return replace(self, program_graph=projected, outputs=tuple(outputs), experts=experts)


def construct(program_graph: ProgramGraph):
    """(weights, token rows, build info) of a scheduled program graph."""
    schedule = asap_schedule(program_graph, log=lambda _: None)
    return build_model(program_graph=program_graph, schedule=schedule)


def describe(token: TokenGraph, program_graph: ProgramGraph, dims, *, dtype: str = "int32") -> dict:
    """One circuit's weights and its token ABI: which residual slot each
    named input and output occupies."""
    model, _rows, info = construct(program_graph)
    by_name = {}
    for d in dims:
        by_name.setdefault(d.name, d)
    inputs = {d.name: d for d in program_graph.all_dims if isinstance(d, InputDimension)}
    return {
        "model": model,
        "dtype": dtype,
        "reassociate_residuals": token.reassociate_residuals,
        # Integer coefficients preserve integer residuals. Fractional
        # projections need the same half-unit encoding as the CPU.
        "fraction_bits": int(dtype == "int32" and any(
            (weight.weight != np.rint(weight.weight)).any()
            for weight in [*model.ff_in, *model.ff_out, *(attn.out_proj for attn in model.attn)]
        )),
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "input_slots": [info.slot_of.get(inputs[n], -1) if n in inputs else -1 for n in token.inputs],
        "output_slots": [info.slot_of[by_name[n]] for n in token.outputs],
    }


def circuits(name: str, token: TokenGraph, *, dtype: str = "int32") -> list[tuple[str | None, dict]]:
    """The full circuit, then each specialization, described.

    Pins name persist dims; partial evaluation cannot fold an input or an
    unknown name, so such a specialization would be a copy of the full circuit.
    """
    persists = {d.name for d in token.program_graph.all_dims if d.kind == "persist"}
    named = Counter(d.name for d in token.program_graph.all_dims)
    if ambiguous := sorted(o for o in token.outputs if named[o] > 1):
        raise ValueError(f"{name}: outputs {ambiguous} each name more than one value")
    for expert, pins in token.experts.items():
        if unfoldable := sorted(set(pins) - persists):
            raise ValueError(f"{name}/{expert}: pins {unfoldable} name no persist dim to fold")
    if not token.full_circuit and not token.experts:
        raise ValueError(f"{name}: a token without its full circuit needs specializations")
    roots = list(token.outputs)
    out = []
    if token.full_circuit:
        full = GraphFolder(token.program_graph, root_names=roots, structural_reads=True).fold({}, label=name)
        out.append((None, describe(token, full.program_graph, list(full.dim_of_name.values()), dtype=dtype)))
    if token.experts:
        # Structural folds: each memory read becomes one engine-served read,
        # its annotated bits served with it.
        folder = GraphFolder(token.program_graph, root_names=roots, structural_reads=True)
        for expert, pins in token.experts.items():
            fold = folder.fold(dict(pins), label=expert)
            out.append((expert, describe(token, fold.program_graph, list(fold.dim_of_name.values()), dtype=dtype)))
    return out


# Each projection's block in layer_arrays: (matrix, the slots its rows
# index or None, the slots its columns index or None).
BLOCKS = {
    "w_in": (None, "in_slots"), "w_out": ("out_slots", None), "h_out": ("out_slots", None),
    "w_ff": (None, "ff_in_slots"), "h_ff": (None, "ff_in_slots"),
    "w_down": ("ff_out_slots", None), "h_down": ("ff_out_slots", None),
}


def sparse_layer(arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """A layer's compiler blocks (compiler.layer_arrays) as the model's sparse
    weights: each projection's nonzero coefficients over residual slots."""
    weights = {
        "read_kind": torch.from_numpy(arrays["read_kind"]),
        "dims": torch.tensor(
            [arrays["w_in"].shape[0], arrays["w_out"].shape[1], arrays["w_down"].shape[1]],
            dtype=torch.int32,
        ),
    }
    for matrix, (row_slots, col_slots) in BLOCKS.items():
        if matrix not in arrays:
            continue
        block = arrays[matrix]
        rows, cols = np.nonzero(block)
        weights[f"{matrix}_rows"] = torch.from_numpy((arrays[row_slots][rows] if row_slots else rows).astype(np.int32))
        weights[f"{matrix}_cols"] = torch.from_numpy((arrays[col_slots][cols] if col_slots else cols).astype(np.int32))
        weights[f"{matrix}_coef"] = torch.from_numpy(np.ascontiguousarray(block[rows, cols]))
    return weights


def transformer(description: dict) -> Transformer:
    """A described circuit as a model.transformer.Transformer."""
    model = description["model"]
    floating = description["dtype"] == "float32"
    return Transformer(
        [sparse_layer(layer_arrays(model, li, floating=floating)) for li in range(len(model.attn))],
        d_model=model.tok.weight.shape[1],
        one_slot=model.one_slot,
        input_slots=torch.tensor(description["input_slots"]),
        output_slots=torch.tensor(description["output_slots"]),
        dtype=torch.float32 if floating else torch.int32,
    )


def circuit_key(circuit: Transformer) -> bytes:
    """A circuit's identity: its shape and every weight."""
    h = hashlib.sha256(f"{circuit.d_model},{circuit.one_slot}".encode())
    for key, tensor in circuit.state_dict().items():
        h.update(f"{key}{tuple(tensor.shape)}{tensor.dtype}".encode())
        h.update(tensor.contiguous().numpy().tobytes())
    return h.digest()


def construct_chip(tokens: Mapping[str, TokenGraph], *, memory_size: int, dtype: str = "int32",
                   constants: Mapping[str, int] | None = None,
                   views: Mapping[str, Mapping] | None = None) -> Chip:
    """A chip's token types as one model.transformer.Chip. ``views`` name
    token types read off another's circuits at fewer outputs (``base``,
    ``outputs``, optionally the base ``specializations`` read): they are
    compiled from the base's weights and store none of their own."""
    described = {name: circuits(name, token, dtype=dtype) for name, token in tokens.items()}
    f_bits = max(d["fraction_bits"] for cs in described.values() for _, d in cs)
    token_types = {}
    for name, token in tokens.items():
        # Different pins can fold to the same circuit (a projection drops the
        # values that told them apart): store each circuit once.
        built, same, first = {}, {}, {}
        for spec, d in described[name]:
            circuit = transformer(d)
            key = circuit_key(circuit)
            if key in first:
                same[spec] = first[key]
            else:
                first[key], built[spec] = spec, circuit
        token_types[name] = TokenType(
            token.inputs, token.outputs, built.pop(None, None), built,
            {name: dict(pins) for name, pins in token.experts.items()},
            same=same, reassociate_residuals=token.reassociate_residuals,
        )
    return Chip(
        token_types, memory_size=memory_size, f_bits=f_bits,
        dtype=torch.float32 if dtype == "float32" else torch.int32, constants=constants,
        views={name: {"base": v["base"], "outputs": list(v["outputs"]),
                      "specializations": v.get("specializations")} for name, v in (views or {}).items()},
    )
