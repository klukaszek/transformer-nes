"""Construct the 6502 computation graph used by the weights pipeline.

This module is a construction IR boundary. It builds the graph DSL and
exposes the dimension handles needed by scheduling, folding, and codegen;
execution happens only in the generated transformer engine.
"""

from __future__ import annotations

from transformer_nes.cpu6502.interpreter import build as build_6502_graph
from transformer_nes.graph import core as graph_core
from transformer_nes.graph.core import InputDimension, reset_graph

RUNTIME_READ_DIMS = (
    "current_pc", "new_a", "new_x", "new_y", "new_s", "new_p", "new_halted",
    "fetched_opcode", "fetched_operand_lo", "fetched_operand_hi",
    "ea", "data_out", "next_pc",
    "is_store", "write_addr_out", "write_value_out",
    "is_store2", "write_addr2_out", "write_value2_out",
    "is_store3", "write_addr3_out", "write_value3_out",
    "cycles",
)


def _find_input_dim(name: str):
    for dim in graph_core.graph_dims:
        if isinstance(dim, InputDimension) and dim.name == name:
            return dim
    raise KeyError(name)


class Cpu6502Graph:
    """Build the 6502 graph and expose its stable dimension handles."""

    def __init__(self, *, bcd: bool = True, wide_bits: bool = False):
        reset_graph()
        self.input_tokens, self.output_tokens = build_6502_graph(
            bcd=bcd, wide_bits=wide_bits,
        )
        self.all_dims = list(graph_core.graph_dims)
        self.dim_id_by_name: dict[str, int] = {}
        for dim in self.all_dims:
            if dim.name not in self.dim_id_by_name:
                self.dim_id_by_name[dim.name] = dim.id

        self.one = graph_core.one
        self.is_start = _find_input_dim("is_start")
        self.start_pc_lo = _find_input_dim("start_pc_lo")
        self.start_pc_hi = _find_input_dim("start_pc_hi")
