"""Compile the PPU graphs into native engines (model/graph_engine.py):
the register file for ``Console.set_ppu_register_engine`` and background
rendering, with its folded experts, for ``Console.set_ppu_render_engine``.
"""

from __future__ import annotations

from transformer_nes.graph.core import ProgramGraph, reset_graph
from transformer_nes.model.graph_engine import CompiledGraph
from transformer_nes.ppu2c02 import registers, render, space

# The register engine's token contract: ACCESS inputs and outputs in order.
ENGINE_INPUTS = ("is_access", *registers.ACCESS_FIELDS, *registers.STATE_FIELDS)


def register_graph() -> ProgramGraph:
    reset_graph()
    return ProgramGraph(*registers.build())


def register_engine(program_graph: ProgramGraph | None = None) -> CompiledGraph:
    return CompiledGraph(
        program_graph or register_graph(),
        list(ENGINE_INPUTS),
        list(registers.OUTPUT_FIELDS),
        tag="ppu_registers",
        memory_size=space.memory_size(),
    )


def render_graph(chr_full: int = 0) -> ProgramGraph:
    reset_graph()
    return ProgramGraph(*render.build(chr_full))


class RenderEngine(CompiledGraph):
    """The compiled render graph and its folded experts, with the table the
    native PPU picks a model per slot from (``render.dispatch``)."""

    @property
    def dispatch(self) -> list[dict]:
        return render.dispatch(self.expert_names)


def render_engine(chr_full: int = 0) -> RenderEngine:
    """The compiled render graph, for ``Console.set_ppu_render_engine``;
    `chr_full` is the CHR size an MMC5 cart needs mapped whole (its
    extended attributes), else 0 (``Console.ppu_graph_chr_full``)."""
    return RenderEngine(
        render_graph(chr_full),
        list(render.INPUTS),
        list(render.OUTPUTS),
        tag=f"ppu_render_{chr_full}",
        experts=render.EXPERTS,
        memory_size=space.memory_size(chr_full),
    )
