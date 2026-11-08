"""Compile the PPU graphs into one native engine (model/graph_engine.py)
for ``Console.set_ppu_engine``: the PPU as one transformer over one
memory, with a token type per kind of work -- a register access, a tile
slot, the sprite unit's evaluation and fetch -- each with its folded
experts. One engine serves every cartridge: MMC5's whole-CHR addressing
and extended attributes are inputs, with experts for each.
"""

from __future__ import annotations

from transformer_nes.graph.core import GraphContext, ProgramGraph, construction, current_graph
from transformer_nes.model.graph_engine import CompiledGraph, TokenGraph
from transformer_nes.ppu2c02 import registers, render, space, sprites


@construction
def register_graph() -> ProgramGraph:
    return current_graph().capture(*registers.build())


@construction
def render_graph() -> ProgramGraph:
    return current_graph().capture(*render.build())


def sprite_graphs() -> dict[str, TokenGraph]:
    """The sprite unit's evaluation and fetch as token graphs."""
    with GraphContext().activate() as context:
        evaluate = context.capture(*sprites.build_evaluation())
    with GraphContext().activate() as context:
        fetch = context.capture(*sprites.build_fetch())
    with GraphContext().activate() as context:
        overflow = context.capture(*sprites.build_overflow())
    fetch_token = TokenGraph(
        fetch, sprites.FETCH_INPUTS, sprites.FETCH_OUTPUTS,
        sprites.FETCH_EXPERTS | sprites.DEBUG_FETCH_EXPERTS,
    )
    return {
        "evaluate": TokenGraph(
            evaluate, sprites.EVALUATE_INPUTS, sprites.EVALUATE_OUTPUTS, sprites.EVALUATE_EXPERTS
        ),
        "fetch": fetch_token,
        "fetch_bus": fetch_token.project(sprites.FETCH_OUTPUTS[72:]),
        "fetch_status": fetch_token.project(
            sprites.FETCH_OUTPUTS[:8] + sprites.FETCH_OUTPUTS[64:65] + sprites.FETCH_OUTPUTS[72:]
        ),
        "overflow": TokenGraph(
            overflow, sprites.OVERFLOW_INPUTS, sprites.OVERFLOW_OUTPUTS, sprites.OVERFLOW_EXPERTS
        ),
    }


class PpuEngine(CompiledGraph):
    """The compiled PPU graphs and their folded experts, with the tables
    the native PPU picks a model per token from: for a cart addressing
    the whole CHR through MMC5's banks when `whole_chr`."""

    whole_chr = False

    @property
    def dispatch(self) -> list[dict]:
        return render.dispatch(self.expert_names)

    @property
    def sprite_dispatch(self) -> dict:
        """Model ids of the sprite unit: evaluation's full model, then its
        experts by sprite size; fetch's general model for this cart (the
        live-latch windows, or whole CHR), then by sprite size and count."""
        prefix = "mmc5_" if self.whole_chr else ""

        def fetches(token: str) -> list[int]:
            general = self.model_id(token) if self.whole_chr else self.model_id(token, "window")
            return [general] + [
                self.model_id(token, f"{prefix}tall{t}_count{n}")
                for t in range(2)
                for n in range(sprites.SPRITES + 1)
            ]

        return {
            "evaluate": [self.model_id("evaluate")]
            + [self.model_id("evaluate", f"tall{t}") for t in range(2)]
            + [self.model_id("evaluate", f"remaining_tall{t}") for t in range(2)],
            "overflow": [self.model_id("overflow")]
            + [self.model_id("overflow", f"tall{t}") for t in range(2)],
            "fetch": fetches("fetch"),
            "fetch_bus": fetches("fetch_bus"),
            "fetch_status": fetches("fetch_status"),
            "debug_fetch": [
                self.model_id("fetch", f"mmc5_tall{t}_count8" if self.whole_chr else f"debug_tall{t}")
                for t in range(2)
            ],
        }

    @property
    def register_dispatch(self) -> list[list[int]]:
        """Model ids of the register experts, [is_write][register]."""
        prefix = "mmc5_" if self.whole_chr else ""
        return [
            [self.model_id("access", f"{prefix}{'write' if w else 'read'}{r}") for r in range(8)]
            for w in range(2)
        ]

    @property
    def status_render_dispatch(self) -> list[int]:
        return [self.model_id("render_status")] + [
            self.model_id("render_status", name) for name in self.expert_names
        ]


def ppu_engine(whole_chr: bool = False) -> PpuEngine:
    """The compiled PPU, for ``Console.set_ppu_engine``, dispatching for a
    cart that addresses the whole CHR through MMC5's banks when
    `whole_chr` (``Console.ppu_whole_chr``). The tile slot is the primary
    graph: model 0, the one the X-ray traces and ablates."""

    def graphs() -> dict[str, TokenGraph]:
        # Preserve the hot tile path's projection grouping: collecting its
        # residual terms changes fusion and regresses measured frame latency.
        tile = TokenGraph(
            render_graph(), render.INPUTS, render.OUTPUTS, render.EXPERTS,
            reassociate_residuals=False,
        )
        access = TokenGraph(register_graph(), registers.INPUTS, registers.OUTPUTS, registers.EXPERTS)
        return {
            "render": tile,
            **sprite_graphs(),
            "access": access,
            "render_status": tile.project(render.OUTPUTS[8:]),
        }

    engine = PpuEngine(graphs, tag="ppu", sources=(__name__,), memory_size=space.MEMORY_SIZE)
    engine.whole_chr = bool(whole_chr)
    return engine
