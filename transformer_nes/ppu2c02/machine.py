"""The PPU as a chip of the NES model (nes/model.py): one transformer over
one memory, with a token type per kind of work -- a register access, a tile
slot, the sprite unit's evaluation and fetch -- each with the specializations
that pay for themselves. One chip serves every cartridge: MMC5's whole-CHR
addressing and extended attributes are inputs.

``PpuEngine`` is what ``Console.set_ppu_engine`` attaches: the compiled
``ppu`` and ``ppu_control`` chips and the tables the native PPU picks a model
from per token, for a cart that does or does not address the whole CHR.
"""

from __future__ import annotations

from functools import cached_property

from transformer_nes.graph.core import GraphContext, ProgramGraph, construction, current_graph
from transformer_nes.model.construct import TokenGraph, construct_chip
from transformer_nes.model.transformer import Chip
from transformer_nes.ppu2c02 import registers, render, space, sprites


@construction
def register_graph() -> ProgramGraph:
    return current_graph().capture(*registers.build())


@construction
def render_graph() -> ProgramGraph:
    return current_graph().capture(*render.build())


def render_token() -> TokenGraph:
    # The tile slot keeps its projection grouping: as one combination per
    # value (reassociated) it lowers to fewer operations but runs 11-22%
    # slower per frame.
    return TokenGraph(
        render_graph(), render.INPUTS, render.OUTPUTS, render.EXPERTS, reassociate_residuals=False
    )


def sprite_graphs() -> dict[str, TokenGraph]:
    """The sprite unit's evaluation, overflow scan and per-slot fetch as
    token types."""
    with GraphContext().activate() as context:
        evaluate = context.capture(*sprites.build_evaluation())
    with GraphContext().activate() as context:
        sprite = context.capture(*sprites.build_sprite())
    with GraphContext().activate() as context:
        overflow = context.capture(*sprites.build_overflow())
    # Every fetch meets one specialization's pins: no full circuit.
    fetch = TokenGraph(
        sprite, sprites.SPRITE_INPUTS, sprites.SPRITE_OUTPUTS, sprites.SPRITE_EXPERTS,
        full_circuit=False,
    )
    return {
        # Evaluation runs at most once a line per OAM change, the overflow
        # scan only past eight sprites: their full circuits.
        "evaluate": TokenGraph(evaluate, sprites.EVALUATE_INPUTS, sprites.EVALUATE_OUTPUTS),
        "overflow": TokenGraph(overflow, sprites.OVERFLOW_INPUTS, sprites.OVERFLOW_OUTPUTS),
        "sprite": fetch,
    }


def chip() -> Chip:
    """The PPU. The tile slot's full circuit is model 0, the one the X-ray
    traces and ablates."""
    tile = render_token()
    return construct_chip(
        {
            "render": tile,
            **sprite_graphs(),
            "access": TokenGraph(
                register_graph(), registers.INPUTS, registers.OUTPUTS, registers.EXPERTS,
                full_circuit=False,
            ),
        },
        memory_size=space.MEMORY_SIZE,
        # Previews read the tile slot's status effects and a fetch's bus
        # effects alone, from the same circuits.
        views={
            "render_status": {"base": "render", "outputs": render.OUTPUTS[8:]},
            "sprite_bus": {"base": "sprite", "outputs": sprites.SPRITE_BUS},
        },
    )


class PpuEngine:
    """The compiled PPU and PPU control chips, with the model tables the
    native PPU dispatches by for a cart that addresses the whole CHR through
    MMC5's banks when `whole_chr` (``Console.ppu_whole_chr``)."""

    def __init__(self, nes, whole_chr: bool = False):
        self.ppu = nes.chip("ppu")
        self.control = nes.chip("ppu_control")
        self.whole_chr = bool(whole_chr)

    # ── Dispatch ──

    @property
    def expert_names(self) -> list[str]:
        """The render circuits by model id: model k runs expert_names[k - 1]
        (a specialization folding to the same circuit as another runs it)."""
        return list(self.ppu.graphs["render"].specialized)

    @property
    def dispatch(self) -> list[dict]:
        return render.dispatch(lambda name: self.ppu.model_id("render", name))

    @property
    def status_render_dispatch(self) -> list[int]:
        """render_status's model for each render model."""
        return [self.ppu.model_id("render_status")] + [
            self.ppu.model_id("render_status", name) for name in self.expert_names
        ]

    @property
    def sprite_dispatch(self) -> dict:
        """Model ids of the sprite unit: evaluation, the overflow scan, a
        slot's fetch and its bus effects alone, [CHR latches active][8x16],
        for this cart, and an empty slot's dummy fetch."""
        model = self.ppu.model_id

        def fetches(token: str) -> list[list[int]]:
            if self.whole_chr:
                return [[model(token, f"mmc5_tall{t}") for t in range(2)]] * 2
            return [[model(token, f"{kind}tall{t}") for t in range(2)] for kind in ("", "window_")]

        return {
            "evaluate": model("evaluate"),
            "overflow": model("overflow"),
            "fetch": fetches("sprite"),
            "fetch_bus": fetches("sprite_bus"),
            "absent": model("sprite_bus", "absent"),
        }

    @property
    def register_dispatch(self) -> list[list[int]]:
        """Model ids of the register specializations, [is_write][register]."""
        return [
            [
                self.ppu.model_id(
                    "access",
                    f"{'mmc5_' if self.whole_chr and r == 7 else ''}{'write' if w else 'read'}{r}",
                )
                for r in range(8)
            ]
            for w in range(2)
        ]

    # ── The X-ray: model 0, the tile slot's full circuit ──

    @property
    def f_bits(self) -> int:
        return self.ppu.chip.f_bits

    @property
    def n_layers(self) -> int:
        return len(self.ppu.chip.tokens["render"].general.layers)

    @property
    def d_model(self) -> int:
        return self.ppu.chip.tokens["render"].general.d_model

    @cached_property
    def layout(self):
        """The tile slot's full circuit's construction info (which residual
        slot holds each named value, and when): reconstructed on demand."""
        from transformer_nes.graph.fold import GraphFolder
        from transformer_nes.model.construct import construct

        token = render_token()
        fold = GraphFolder(
            token.program_graph, root_names=list(token.outputs), structural_reads=True
        ).fold({}, label="render")
        return construct(fold.program_graph)[2]

    @property
    def slot_of(self) -> dict:
        return self.layout.slot_of

    @property
    def lives(self) -> list:
        return self.layout.lives

    def ablate(self, slot: int = 0, row: int = -1, value: int = 0) -> None:
        """Force residual `slot` to `value` after layer `row` - 1 whenever
        model 0 runs; row -1 restores it."""
        self.ppu.engine.ablate(slot, row, value)
