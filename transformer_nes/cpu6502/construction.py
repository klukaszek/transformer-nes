"""Construct the 6502 computation graph used by the weights pipeline.

This module is a construction IR boundary. It builds the graph DSL and
exposes the dimension handles needed by scheduling, folding, and codegen;
execution happens only in the generated transformer engine.
"""

from __future__ import annotations

from transformer_nes.cpu6502.interpreter import build as build_6502_graph
from transformer_nes.graph.core import ProgramGraph, construction, current_graph

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
    "write_original_out",
)


class Cpu6502Graph:
    """Build the 6502 graph and expose its stable dimension handles."""

    @construction
    def __init__(self, *, bcd: bool = True, wide_bits: bool = False):
        input_tokens, output_tokens = build_6502_graph(
            bcd=bcd,
            wide_bits=wide_bits,
        )
        self.program = current_graph().capture(input_tokens, output_tokens)
        self.dim_of_name = {}
        for dim in self.program.all_dims:
            self.dim_of_name.setdefault(dim.name, dim)

    def program_graph(self) -> ProgramGraph:
        """The graph the engine runs.  The run loop reads the machine's
        outputs (RUNTIME_READ_DIMS) from the residual stream, so they must
        survive to the last layer: a synthetic READOUT token marks them as
        outputs for the scheduler (its head row is ignored: only STATE vs
        HALT compete)."""
        output_tokens = dict(self.program.output_tokens)
        readout = 0
        for name in RUNTIME_READ_DIMS:
            readout = readout + 1e-6 * self.dim_of_name[name]
        output_tokens["READOUT"] = readout
        return self.program.with_outputs(output_tokens)
