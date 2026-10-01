"""Compile the PPU graphs into one native engine (model/graph_engine.py)
for ``Console.set_ppu_engine``: the PPU as one transformer over one
memory, with a token type per kind of work -- a register access, a tile
slot, the sprite unit's evaluation and fetch -- each with its folded
experts.
"""

from __future__ import annotations

from transformer_nes.graph.core import ProgramGraph, reset_graph
from transformer_nes.model.graph_engine import CompiledGraph, TokenGraph
from transformer_nes.ppu2c02 import registers, render, space, sprites


def register_graph(chr_full: int = 0) -> ProgramGraph:
    reset_graph()
    return ProgramGraph(*registers.build(chr_full))


def render_graph(chr_full: int = 0) -> ProgramGraph:
    reset_graph()
    return ProgramGraph(*render.build(chr_full))


def sprite_graphs(chr_full: int = 0) -> dict[str, TokenGraph]:
    """The sprite unit's evaluation and fetch as token graphs.  MMC5
    builds have no CHR latches, so their fetch experts pin no latch
    logic."""
    reset_graph()
    evaluate = ProgramGraph(*sprites.build_evaluation(chr_full))
    reset_graph()
    fetch = ProgramGraph(*sprites.build_fetch(chr_full))
    fetch_experts = (
        {
            name: {pin: pins[pin] for pin in pins if pin == "tall" or pin.startswith("present")}
            for name, pins in sprites.FETCH_EXPERTS.items()
        }
        if chr_full
        else sprites.FETCH_EXPERTS
    )
    return {
        "evaluate": TokenGraph(
            evaluate, sprites.EVALUATE_INPUTS, sprites.EVALUATE_OUTPUTS, sprites.EVALUATE_EXPERTS
        ),
        "fetch": TokenGraph(fetch, sprites.FETCH_INPUTS, sprites.FETCH_OUTPUTS, fetch_experts),
    }


class PpuEngine(CompiledGraph):
    """The compiled PPU graphs and their folded experts, with the tables
    the native PPU picks a model per token from."""

    @property
    def dispatch(self) -> list[dict]:
        return render.dispatch(self.expert_names)

    @property
    def sprite_dispatch(self) -> dict:
        """Model ids of the sprite unit: evaluation's full model, then its
        experts by sprite size; fetch's full model, then by sprite size
        and count its experts (only for carts without CHR latches)."""
        return {
            "evaluate": [self.model_id("evaluate")]
            + [self.model_id("evaluate", f"tall{t}") for t in range(2)],
            "fetch": [self.model_id("fetch")]
            + [
                self.model_id("fetch", f"tall{t}_count{n}")
                for t in range(2)
                for n in range(sprites.SPRITES + 1)
            ],
        }

    @property
    def register_dispatch(self) -> list[list[int]]:
        """Model ids of the register experts, [is_write][register]."""
        return [
            [self.model_id("access", f"{'write' if w else 'read'}{r}") for r in range(8)]
            for w in range(2)
        ]


def ppu_engine(chr_full: int = 0) -> PpuEngine:
    """The compiled PPU, for ``Console.set_ppu_engine``; `chr_full` is
    the CHR size an MMC5 cart needs mapped whole (its extended
    attributes, sprite banks and $2007 reads), else 0
    (``Console.ppu_graph_chr_full``).  The tile slot is the primary
    graph: model 0, the one the X-ray traces and ablates."""
    tile = TokenGraph(render_graph(chr_full), render.INPUTS, render.OUTPUTS, render.EXPERTS)
    access = TokenGraph(
        register_graph(chr_full), registers.INPUTS, registers.OUTPUTS, registers.EXPERTS
    )
    return PpuEngine(
        {"render": tile, **sprite_graphs(chr_full), "access": access},
        tag=f"ppu_{chr_full}",
        memory_size=space.memory_size(chr_full),
    )
