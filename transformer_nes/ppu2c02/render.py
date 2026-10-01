"""Background rendering and compositing as a computation graph.

One step per tile slot -- the native PPU's Slot event (nes/ppu.h): output
the slot's 8 pixels, fetch the next background tile at v, and step v.
Slots 0-31 are the visible slots (the last also steps v vertically);
slots 32-33 are the next line's two prefetches.  A SLOT token carries
the loopy v fields, fine x, ctrl, mask, the nametable arrangement, the
16 fetched background pixels under the slot (as palette indices), the 8
sprite-line pixels (as the sprite fetch leaves them, ppu2c02/sprites.py:
palette index, behind-background bit, sprite-0 bit), and whether sprite 0
already hit.

The step reads nametable, attribute, pattern and palette bytes from the
WRITE-token log (ppu2c02/space.py) and outputs the 8 pixel colors, the
fetched tile as 8 palette indices, v after the slot, the sprite-0 hits,
and the MMC2/MMC4 latches after the fetch.  The sprite rows come from
the sprite graphs; the priority rules they feed are here.

Nearly every product here has a 0/1 factor, so the graph uses the
one-unit gated forms (graph/ops.py: gate/both/either/select/equals).
"""

from __future__ import annotations

from transformer_nes.graph import core as graph
from transformer_nes.graph.core import (
    Expression,
    InputDimension,
    auto_name,
    persist,
    stepglu,
    to_expr,
)
from transformer_nes.graph.ops import both, either, equals, extract_bits, gate, select
from transformer_nes.ppu2c02.space import (
    BASE_GROUPS,
    CHR,
    CHR_ALT,
    CHR_ALT_BASE,
    CHR_FULL_BASE,
    PAGES,
    PALETTE,
    Memory,
    chr_full_groups,
    in_palette_region,
    nametable_page,
    palette_offset,
)

LOOPY_FIELDS = ("cx", "cy", "nt", "fy")
INPUTS = (
    "is_slot",
    "slot",
    "hit_set",
    "fine_x",
    "ctrl",
    "mask",
    *(f"nt_page{k}" for k in range(4)),
    # MMC2/MMC4 latches (0 FD, 1 FE) per pattern-table half, and whether
    # the low half triggers on all rows (MMC4) or row 0 only (MMC2).
    "latch_lo",
    "latch_hi",
    "wide_low",
    # MMC5 extended attributes: per-tile CHR bank and palette from ExRAM.
    "ext_mode",
    "chr_upper",
    *(f"v_{f}" for f in LOOPY_FIELDS),
    *(f"bg{i}" for i in range(16)),
    *(f"sprite{i}" for i in range(8)),
)
OUTPUTS = (
    *(f"pixel{i}" for i in range(8)),
    *(f"tile{i}" for i in range(8)),
    *(f"new_v_{f}" for f in LOOPY_FIELDS),
    "fetched",
    *(f"hit{i}" for i in range(8)),
    "new_latch_lo",
    "new_latch_hi",
)

# Folded experts, in model order after the full graph (model 0).  The
# native PPU picks one per slot (nes/ppu.cpp:render_slot) only when its
# pins hold -- background on (so rendering), no greyscale, no extended
# attributes -- and then by slot: an inner slot without sprite pixels
# gets one expert per background pattern table and fine x (the pixel
# window folds to plain wiring), with sprites the sprite expert; the
# edge slots 0 and 31 without sprites have their own.  Each comes twice:
# with the CHR latches pinned off (carts without them), then with them.
_PLAIN = {
    "maskbit3": 1, "rendering": 1, "maskbit0": 0, "first_slot": 0, "last_slot": 0,
    "extended": 0,
}
_SLOT_EXPERTS = {
    **{
        f"no_sprites_bg{table}_fine_x{f}": {
            **_PLAIN,
            "ctrlbit4": table,
            **{f"sprite_pixel{i}": 0 for i in range(8)},
            **{f"fine_x_bit{b}": (f >> b) & 1 for b in range(3)},
        }
        for table in range(2)
        for f in range(8)
    },
    "sprites": dict(_PLAIN),
    # The line's edge slots without sprites: slot 0 (left-edge clipping)
    # and slot 31 (the vertical step), per background table.
    **{
        f"no_sprites_bg{table}_{edge}": {
            **_PLAIN,
            "first_slot": int(edge == "first"),
            "last_slot": int(edge == "last"),
            "ctrlbit4": table,
            **{f"sprite_pixel{i}": 0 for i in range(8)},
        }
        for edge in ("first", "last")
        for table in range(2)
    },
}
_NO_LATCHES = {"latch": 0, "latch_set_fd": 0, "latch_set_fe": 0}
EXPERTS = {
    **{name: {**pins, **_NO_LATCHES} for name, pins in _SLOT_EXPERTS.items()},
    **{f"{name}_latched": pins for name, pins in _SLOT_EXPERTS.items()},
}


def dispatch(names: list[str]) -> list[dict]:
    """The native per-slot choice (nes/ppu.cpp:render_slot) as model ids
    (k runs names[k - 1]), without and then with the CHR latches: inner
    slots per background table and fine x, sprite slots, and the edge
    slots 0 and 31 per table."""
    model = {name: k for k, name in enumerate(names, start=1)}

    def variant(suffix: str) -> dict:
        return {
            "inner": [
                [model[f"no_sprites_bg{t}_fine_x{f}{suffix}"] for f in range(8)] for t in range(2)
            ],
            "sprites": model[f"sprites{suffix}"],
            "first": [model[f"no_sprites_bg{t}_first{suffix}"] for t in range(2)],
            "last": [model[f"no_sprites_bg{t}_last{suffix}"] for t in range(2)],
        }

    return [variant(""), variant("_latched")]


def low_bits(value, width: int) -> list[Expression]:
    """Bits width-1..0 of a small non-negative integer, MSB first."""
    one = graph.one
    bits, rest = [], to_expr(value)
    for b in range(width - 1, -1, -1):
        bit = stepglu(one, rest - (1 << b))
        bits.append(bit)
        rest = rest - (1 << b) * bit
    return bits


def build(chr_full: int = 0) -> tuple[dict[str, Expression], dict[str, Expression]]:
    """`chr_full`: bytes of whole-CHR memory for MMC5 extended attributes
    (a power of two), or 0 to leave that path out."""
    one = graph.one
    inp = {name: InputDimension(name) for name in INPUTS}
    memory_inputs = [InputDimension(n) for n in ("is_mem_write", "mem_addr", "mem_value")]
    input_tokens = {
        "SLOT": sum((to_expr(d) for d in inp.values()), Expression()),
        "WRITE": sum((to_expr(d) for d in memory_inputs), Expression()),
    }
    full_groups = chr_full_groups(chr_full)
    memory = Memory(*memory_inputs, groups=full_groups[-1] + 1 if chr_full else BASE_GROUPS)
    v = {f: inp[f"v_{f}"] for f in LOOPY_FIELDS}

    # Every condition a folded expert pins is a named persist.
    ctrl_bits = extract_bits(inp["ctrl"], "ctrl")  # MSB first
    mask_bits = extract_bits(inp["mask"], "mask")
    bg_table = ctrl_bits[3]
    greyscale, bg_left, sprite_left = mask_bits[7], mask_bits[6], mask_bits[5]
    bg_on, sprite_on = mask_bits[4], mask_bits[3]
    rendering = persist(either(bg_on, sprite_on), name="rendering")
    first_slot = persist(equals(inp["slot"], 0), name="first_slot")
    last_visible_slot = persist(equals(inp["slot"], 31), name="last_slot")
    fine_x_bits = [
        persist(b, name=f"fine_x_bit{2 - k}") for k, b in enumerate(low_bits(inp["fine_x"], 3))
    ]
    extended = persist(inp["ext_mode"] if chr_full else Expression(), name="extended")

    # ── Fetch the tile at v ─────────────────────────────────────────
    cy_bits = low_bits(v["cy"], 5)
    cx_bits = low_bits(v["cx"], 5)
    page = nametable_page(v["nt"], [inp[f"nt_page{k}"] for k in range(4)])
    table = 0x2000 + 0x400 * page
    offset = 32 * v["cy"] + v["cx"]
    tile_no = memory.read(table + offset, PAGES, "tile_no")
    attr_offset = (
        0x3C0
        + 8 * (4 * cy_bits[0] + 2 * cy_bits[1] + cy_bits[2])
        + (4 * cx_bits[0] + 2 * cx_bits[1] + cx_bits[2])
    )
    attr_bits = memory.bits(memory.read(table + attr_offset, PAGES, "attr"), "attr")
    quadrant_y, quadrant_x = cy_bits[3], cx_bits[3]
    quadrants = [
        both(1 - quadrant_y, 1 - quadrant_x),
        both(1 - quadrant_y, quadrant_x),
        both(quadrant_y, 1 - quadrant_x),
        both(quadrant_y, quadrant_x),
    ]
    palette_select = sum(
        (
            gate(q, 2 * attr_bits[7 - (2 * k + 1)] + attr_bits[7 - 2 * k])
            for k, q in enumerate(quadrants)
        ),
        Expression(),
    )
    # The pattern bytes come through the half's latch: FE-latched banks
    # live in the alternate CHR window.
    pattern = 0x1000 * bg_table + 16 * tile_no + v["fy"]
    latch = persist(select(bg_table, inp["latch_hi"], inp["latch_lo"]), name="latch")
    pattern_addr = pattern + gate(latch, CHR_ALT_BASE)
    groups = CHR + CHR_ALT
    if chr_full:
        # MMC5 extended attributes: ExRAM byte e (the ExRAM page, same
        # offset) gives the palette (bits 7-6) and the tile's 4 KiB CHR
        # bank (chr_upper * 64 + bits 5-0, wrapped to the CHR size).
        ex_bits = memory.bits(memory.read(0x2800 + offset, PAGES, "ex"), "ex")
        bank_bits = (chr_full >> 12).bit_length() - 1
        upper_bits = low_bits(inp["chr_upper"], 2)  # MSB first
        bank = sum(
            (
                (1 << j) * (ex_bits[7 - j] if j < 6 else upper_bits[7 - j])
                for j in range(min(bank_bits, 8))
            ),
            Expression(),
        )
        full_addr = CHR_FULL_BASE + 0x1000 * bank + 16 * tile_no + v["fy"]
        pattern_addr = select(extended, full_addr, pattern_addr)
        palette_select = select(extended, 2 * ex_bits[0] + ex_bits[1], palette_select)
        groups = groups + full_groups
    lo_bits = memory.bits(memory.read(pattern_addr, groups, "pattern_lo"), "pattern_lo")
    hi_bits = memory.bits(memory.read(pattern_addr + 8, groups, "pattern_hi"), "pattern_hi")
    tile = [
        lo_bits[c] + 2 * hi_bits[c] + 4 * gate(either(lo_bits[c], hi_bits[c]), palette_select)
        for c in range(8)
    ]
    # Latch triggers: the high byte of tile $FD/$FE sets FD/FE; MMC2's low
    # half only on row 0 ($0FD8/$0FE8).  Extended fetches bypass them.
    row_ok = either(bg_table, either(inp["wide_low"], equals(v["fy"], 0)))
    fetches = both(rendering, 1 - extended)
    set_fd = persist(both(both(equals(tile_no, 0xFD), row_ok), fetches), name="latch_set_fd")
    set_fe = persist(both(both(equals(tile_no, 0xFE), row_ok), fetches), name="latch_set_fe")
    latched = latch - both(set_fd, latch) + set_fe - both(set_fe, latch)
    new_latch_lo = select(bg_table, inp["latch_lo"], latched)
    new_latch_hi = select(bg_table, latched, inp["latch_hi"])

    # ── Step v: coarse x every slot, y after the last visible one ───
    x_wrap = stepglu(one, v["cx"] - 31)
    nt_b1 = stepglu(one, v["nt"] - 2)
    nt_b0 = v["nt"] - 2 * nt_b1
    fy_wrap = both(last_visible_slot, stepglu(one, v["fy"] - 7))
    cy_29 = both(fy_wrap, equals(v["cy"], 29))
    cy_31 = both(fy_wrap, stepglu(one, v["cy"] - 31))
    stepped = {
        "cx": v["cx"] + 1 - 32 * x_wrap,
        "nt": v["nt"] + gate(x_wrap, 1 - 2 * nt_b0) + gate(cy_29, 2 - 4 * nt_b1),
        "fy": v["fy"] + last_visible_slot - 8 * fy_wrap,
        "cy": v["cy"] + fy_wrap - 30 * cy_29 - 32 * cy_31,
    }
    new_v = {f: select(rendering, stepped[f], v[f]) for f in LOOPY_FIELDS}

    # ── Pixels ──────────────────────────────────────────────────────
    # The 16 buffered background pixels under the slot, shifted by fine
    # x: a three-stage barrel shifter (shift by 4, 2, 1).
    window = [to_expr(inp[f"bg{j}"]) for j in range(16)]
    for bit, shift in ((0, 4), (1, 2), (2, 1)):
        take = fine_x_bits[bit]  # MSB first
        window = [select(take, window[j + shift], window[j]) for j in range(len(window) - shift)]
    bg_gate = gate(bg_on, 1 - both(first_slot, 1 - bg_left))
    sprite_gate = gate(sprite_on, 1 - both(first_slot, 1 - sprite_left))
    not_hit_yet = 1 - inp["hit_set"]
    backdrop = gate(in_palette_region(v), palette_offset(v["cx"]))
    pixels, hits = [], []
    for i in range(8):
        bg = gate(bg_gate, window[i])
        bg_opaque = stepglu(one, bg - 1)
        sprite = persist(inp[f"sprite{i}"], name=f"sprite_pixel{i}")
        s_b7 = stepglu(one, sprite - 128)
        s_b6 = stepglu(one, sprite - 128 * s_b7 - 64)
        sprite_index = sprite - 128 * s_b7 - 64 * s_b6
        shown = both(sprite_gate, stepglu(one, sprite - 1))
        wins = shown - both(shown, both(s_b6, bg_opaque))
        index = select(rendering, bg + gate(wins, sprite_index - bg), backdrop)
        # Sprite 0 hits on opaque-over-opaque, never at x=255.
        hit = both(both(s_b7, shown), both(bg_opaque, not_hit_yet))
        if i == 7:
            hit = both(hit, 1 - last_visible_slot)
        hits.append(hit)
        # Palette RAM is 6 bits wide; greyscale keeps the two high ones.
        color = memory.read(0x3F00 + index, PALETTE, f"color{i}")
        c5 = stepglu(one, color - 32)
        c4 = stepglu(one, color - 32 * c5 - 16)
        pixels.append(select(greyscale, 32 * c5 + 16 * c4, color))

    outputs = {
        **{f"pixel{i}": p for i, p in enumerate(pixels)},
        **{f"tile{i}": t for i, t in enumerate(tile)},
        **{f"new_v_{f}": new_v[f] for f in LOOPY_FIELDS},
        "fetched": rendering,
        **{f"hit{i}": h for i, h in enumerate(hits)},
        "new_latch_lo": new_latch_lo,
        "new_latch_hi": new_latch_hi,
    }
    readout = Expression()
    for name, expr in outputs.items():
        readout = readout + 1e-6 * persist(expr, name=name)
    output_tokens = {"READOUT": readout}
    auto_name(locals())
    return input_tokens, output_tokens
