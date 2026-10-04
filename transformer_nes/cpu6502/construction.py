"""Construct the 6502 computation graph used by the weights pipeline.

This module is a construction IR boundary. It builds the graph DSL and
exposes the dimension handles needed by scheduling, folding, and codegen;
execution happens only in the generated transformer engine.
"""

from __future__ import annotations

from transformer_nes.cpu6502.interpreter import build as build_6502_graph
from transformer_nes.graph.core import InputDimension, ProgramGraph, construction, current_graph

RUNTIME_READ_DIMS = (
    "current_pc",
    "new_a",
    "new_x",
    "new_y",
    "new_s",
    "new_p",
    "new_halted",
    "fetched_opcode",
    "fetched_operand_lo",
    "fetched_operand_hi",
    "ea",
    "data_out",
    "next_pc",
    "is_store",
    "write_addr_out",
    "write_value_out",
    "is_store2",
    "write_addr2_out",
    "write_value2_out",
    "is_store3",
    "write_addr3_out",
    "write_value3_out",
    "cycles",
)


class Cpu6502Graph:
    """Build the 6502 graph and expose its stable dimension handles."""

    @construction
    def __init__(self, *, bcd: bool = True, wide_bits: bool = False):
        self.input_tokens, self.output_tokens = build_6502_graph(
            bcd=bcd,
            wide_bits=wide_bits,
        )
        self.program = current_graph().capture(self.input_tokens, self.output_tokens)
        self.all_dims = self.program.all_dims
        self.dim_id_by_name: dict[str, int] = {}
        for dim in self.all_dims:
            if dim.name not in self.dim_id_by_name:
                self.dim_id_by_name[dim.name] = dim.id

        self.one = self.program.one
        inputs = {d.name: d for d in self.all_dims if isinstance(d, InputDimension)}
        self.is_start = inputs["is_start"]
        self.start_pc_lo = inputs["start_pc_lo"]
        self.start_pc_hi = inputs["start_pc_hi"]

    def program_graph(self) -> ProgramGraph:
        """The graph the engine runs.  The run loop reads the machine's
        outputs (RUNTIME_READ_DIMS) from the residual stream, so they must
        survive to the last layer: a synthetic READOUT token marks them as
        outputs for the scheduler (its head row is ignored: only STATE vs
        HALT compete)."""
        dims = {dim.id: dim for dim in self.all_dims}
        output_tokens = dict(self.output_tokens)
        readout = 0
        for name in RUNTIME_READ_DIMS:
            readout = readout + 1e-6 * dims[self.dim_id_by_name[name]]
        output_tokens["READOUT"] = readout
        return self.program.with_outputs(output_tokens)
