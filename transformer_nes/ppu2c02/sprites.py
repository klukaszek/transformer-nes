"""Sprite selection, overflow and pattern fetch as transformer tokens.

Evaluation (an EVALUATE token at dot 65 of a visible line): which of the
64 OAM sprites cover the line, and the first eight of them in OAM order.
Each sprite's Y arrives by attention over the OAM bytes in the WRITE-token
log (ppu2c02/space.py).  With P_i the in-range sprites among 0..i, the
k-th selected sprite is the count of i with P_i <= k: a prefix sum and
512 thresholds, no sort or loop.

Fetch (a FETCH token at dot 257): each selected sprite's OAM bytes and
pattern row -- vertical flip, 8x16 tiles and their pattern table, the
MMC2/MMC4 latches each fetch flips in order, MMC5's sprite banks -- and
the row as 8 screen-ordered pixels in the encoding the render graph
composites (render.py: palette index, behind-background bit, sprite-0
bit), with where it goes and the pattern table every fetch reads.

Overflow reads the diagonal OAM addresses after the eighth sprite and
reports the first matching index. The host stamps its dot and A12 edges
and places fetched rows in the line buffer, first opaque pixel wins.
"""

from __future__ import annotations

from transformer_nes.graph import core as graph
from transformer_nes.graph.core import (
    Expression,
    InputDimension,
    auto_name,
    persist,
    reglu,
    stepglu,
    to_expr,
)
from transformer_nes.graph.ops import both, either, equals, extract_bits, gate, gate_nonzero, select
from transformer_nes.ppu2c02.render import low_bits
from transformer_nes.ppu2c02.space import (
    CHR,
    CHR_ALT,
    CHR_ALT_BASE,
    CHR_FULL,
    CHR_FULL_BASE,
    OAM,
    Memory,
)

SPRITES = 8  # per line, the hardware's limit

EVALUATE_INPUTS = ("is_evaluate", "line", "ctrl", "start")
EVALUATE_OUTPUTS = ("count", *(f"index{k}" for k in range(SPRITES)))
OVERFLOW_INPUTS = ("is_overflow", "line", "ctrl", "start")
OVERFLOW_OUTPUTS = ("index",)

FETCH_INPUTS = (
    "is_fetch",
    "line",
    "ctrl",
    "count",
    *(f"index{k}" for k in range(SPRITES)),
    # MMC2/MMC4 latches (0 FD, 1 FE) per pattern-table half, and whether
    # the low half triggers on all rows (MMC4) or row 0 only (MMC2).
    "latch_lo",
    "latch_hi",
    "wide_low",
    # MMC5's sprite banks ($5120-$5127), in 1 KiB units of the whole CHR,
    # and whether CHR is addressed through them (else the CHR windows).
    *(f"bank{s}" for s in range(8)),
    "whole_chr",
)
FETCH_OUTPUTS = (
    *(f"sprite{k}_pixel{c}" for k in range(SPRITES) for c in range(8)),
    *(f"sprite{k}_x" for k in range(SPRITES)),
    *(f"sprite{k}_fetch" for k in range(SPRITES)),
    *(f"sprite{k}_a12" for k in range(SPRITES)),
    "new_latch_lo",
    "new_latch_hi",
)

# Folded experts: evaluation by sprite size; fetch by sprite size and
# count (the slots past it fold to dummy fetches) with the CHR latches
# pinned off (every cart without them), the same through MMC5's banks
# (mmc5_*), and the CHR windows with live latches (MMC2/MMC4).
EVALUATE_EXPERTS = {
    **{f"tall{t}": {"tall": t, "evaluation_start": 0} for t in range(2)},
    **{f"remaining_tall{t}": {"tall": t} for t in range(2)},
}
OVERFLOW_EXPERTS = {f"tall{t}": {"tall": t} for t in range(2)}
_COUNTS = {
    f"tall{t}_count{n}": {"tall": t, **{f"present{k}": int(k < n) for k in range(SPRITES)}}
    for t in range(2)
    for n in range(SPRITES + 1)
}
_NO_LATCHES = {
    "latch_in_lo": 0,
    "latch_in_hi": 0,
    **{f"set_fd{k}": 0 for k in range(SPRITES)},
    **{f"set_fe{k}": 0 for k in range(SPRITES)},
}
FETCH_EXPERTS = {
    "window": {"whole_chr": 0},
    **{name: {**pins, **_NO_LATCHES, "whole_chr": 0} for name, pins in _COUNTS.items()},
    **{f"mmc5_{name}": {**pins, "whole_chr": 1} for name, pins in _COUNTS.items()},
}
# Extra debug rows read the current latch window without changing it.
DEBUG_FETCH_EXPERTS = {
    f"debug_tall{t}": {
        "tall": t,
        "whole_chr": 0,
        **{f"set_fd{k}": 0 for k in range(SPRITES)},
        **{f"set_fe{k}": 0 for k in range(SPRITES)},
    }
    for t in range(2)
}


def sprite_memory():
    inputs = [InputDimension(n) for n in ("is_mem_write", "mem_addr", "mem_value")]
    return inputs, Memory(*inputs)


def sprite_tokens(name: str, inp: dict, memory_inputs: list, outputs: dict):
    """The token inputs, and the outputs as one READOUT token (persisted
    by name, so the engine reads them out of the residual stream)."""
    input_tokens = {
        name: sum((to_expr(d) for d in inp.values()), Expression()),
        "WRITE": sum((to_expr(d) for d in memory_inputs), Expression()),
    }
    readout = Expression()
    for out_name, expr in outputs.items():
        readout = readout + 1e-6 * persist(expr, name=out_name)
    return input_tokens, {"READOUT": readout}


def build_evaluation():
    one = graph.current_graph().one
    inp = {name: InputDimension(name) for name in EVALUATE_INPUTS}
    memory_inputs, memory = sprite_memory()
    tall = persist(extract_bits(inp["ctrl"], "ctrl")[2], name="tall")
    start = persist(inp["start"], name="evaluation_start")
    height = 8 + 8 * tall

    # P_i: sprites 0..i whose rows cover the line, in two levels so each
    # sum stays short: 8-sprite block totals, then each prefix as the
    # blocks before it plus its own block's partial sum.
    covers = []
    for i in range(64):
        row = inp["line"] - memory.read(0x4000 + 4 * i, OAM, f"y{i}")
        covers.append(
            persist(
                both(stepglu(one, i - start), stepglu(one, row) - stepglu(one, row - height)),
                name=f"covers{i}",
            )
        )
    blocks = [
        persist(sum(covers[b : b + 8], Expression()), name=f"block{b // 8}")
        for b in range(0, 64, 8)
    ]
    prefix = [
        persist(
            sum(blocks[: i // 8], Expression()) + sum(covers[i - i % 8 : i + 1], Expression()),
            name=f"covered{i}",
        )
        for i in range(64)
    ]
    covered = prefix[-1]
    # The k-th selected sprite: how many sprites precede it, i.e. have
    # P_i <= k (64 past the last).  Each threshold is a difference of two
    # ReGLU units (shared between neighbouring k) summed inside the FFN:
    # only the eight indices take residual slots.
    index = [
        sum((reglu(1, k + 1 - p) - reglu(1, k - p) for p in prefix), Expression())
        for k in range(SPRITES)
    ]
    found = SPRITES - stepglu(SPRITES - covered, SPRITES - covered)  # min(P_63, 8)

    outputs = {"count": found, **{f"index{k}": index[k] for k in range(SPRITES)}}
    auto_name(locals())
    return sprite_tokens("EVALUATE", inp, memory_inputs, outputs)


def build_overflow():
    """First diagonal-scan match in eight candidates, or sentinel 64.

    The host advances start by eight after a miss, preserving the byte
    phase (j-start) mod 4. Each token reads only its eight candidates;
    early exit and a short final batch need no extra expert variants.
    """
    one = graph.current_graph().one
    inp = {name: InputDimension(name) for name in OVERFLOW_INPUTS}
    memory_inputs, memory = sprite_memory()
    tall = persist(extract_bits(inp["ctrl"], "ctrl")[2], name="tall")
    height = 8 + 8 * tall
    hits = []
    for j in range(8):
        present = stepglu(one, 63 - inp["start"] - j)
        address = 0x4000 + gate(present, 4 * (inp["start"] + j) + j % 4)
        y = memory.read(address, OAM, f"diagonal{j}")
        row = inp["line"] - y
        hits.append(
            persist(both(present, stepglu(one, row) - stepglu(one, row - height)), name=f"hit{j}")
        )
    first_match = sum(
        (1 - stepglu(one, sum(hits[: j + 1], Expression()) - 1) for j in range(8)), Expression()
    )
    result = select(stepglu(one, 7 - first_match), inp["start"] + first_match, 64)
    auto_name(locals())
    return sprite_tokens("OVERFLOW", inp, memory_inputs, {"index": result})


def encode_row(lo, hi, flip_h, shown, meta):
    """Encode eight binary pattern lanes, then optionally reverse the row.

    Pattern bits, flip_h and shown are 0/1. A transparent pixel remains zero;
    opaque pixels carry the same palette/priority/sprite-zero metadata.
    """
    pixels = []
    for low, high in zip(lo, hi, strict=True):
        value = low + 2 * high  # 0..3
        pixels.append(gate_nonzero(shown, value, meta + value, zero_payload=meta))
    result = [None] * 8
    # One gated difference supplies both sides of a controlled swap.
    for c in range(4):
        opposite = 7 - c
        delta = gate(flip_h, pixels[opposite] - pixels[c])
        result[c], result[opposite] = pixels[c] + delta, pixels[opposite] - delta
    return result


def build_fetch():
    """Sprite rows through MMC5's sprite banks into the whole CHR, or through
    the CHR windows and their MMC2/MMC4 latches (the whole_chr input)."""
    one = graph.current_graph().one
    inp = {name: InputDimension(name) for name in FETCH_INPUTS}
    memory_inputs, memory = sprite_memory()
    ctrl_bits = extract_bits(inp["ctrl"], "ctrl")  # MSB first
    tall = persist(ctrl_bits[2], name="tall")
    table8 = ctrl_bits[4]  # 8x8 sprites' pattern table
    height = 8 + 8 * tall
    draw = stepglu(one, 238 - inp["line"])  # the next line is visible
    latch = {
        "lo": persist(inp["latch_lo"], name="latch_in_lo"),
        "hi": persist(inp["latch_hi"], name="latch_in_hi"),
    }
    whole_chr = persist(inp["whole_chr"], name="whole_chr")

    outputs: dict[str, Expression] = {}
    for k in range(SPRITES):
        present = persist(stepglu(one, inp["count"] - k - 1), name=f"present{k}")
        index = gate(present, inp[f"index{k}"])
        base = 0x4000 + 4 * index
        y = memory.read(base, OAM, f"y{k}")
        tile = memory.read(base + 1, OAM, f"tile{k}")
        tile_bits = Memory.bits(tile, f"tile{k}")
        attr_bits = Memory.bits(memory.read(base + 2, OAM, f"attr{k}"), f"attr{k}")
        x = memory.read(base + 3, OAM, f"x{k}")
        flip_v, flip_h, behind = attr_bits[0], attr_bits[1], attr_bits[2]

        # A sprite whose row left the line since evaluation (OAM or the
        # sprite size changed) is skipped, fetch and all.
        row = inp["line"] - y
        valid = both(present, stepglu(one, row) - stepglu(one, row - height))
        row = gate(valid, select(flip_v, height - 1 - row, row))
        row_bits = low_bits(row, 4)  # MSB first
        fine = row - 8 * row_bits[0]
        tile_b0 = tile_bits[7]
        half = select(tall, tile_b0, table8)  # address bit 12
        # The tile the fetch reads, as an 8x8 tile number in its half.
        fetched = select(tall, tile - tile_b0 + row_bits[0], tile)
        address = 0x1000 * half + 16 * fetched + fine

        # MMC5: the 1 KiB slot's sprite bank in the whole CHR.
        slot = 4 * half + 2 * tile_bits[0] + tile_bits[1]
        bank = sum((gate(equals(slot, s), inp[f"bank{s}"]) for s in range(8)), Expression())
        full_address = CHR_FULL_BASE + 0x400 * bank + (address - 0x400 * slot)
        # Otherwise the half's latch picks the FD or FE window, then the
        # fetch may flip it (high byte of tile $FD/$FE; MMC2's low half on
        # row 0 only) for the fetches after it.
        current = select(half, latch["hi"], latch["lo"])
        window_address = address + gate(current, CHR_ALT_BASE)
        row_ok = either(half, either(inp["wide_low"], equals(fine, 0)))
        set_fd = persist(both(both(equals(fetched, 0xFD), row_ok), valid), name=f"set_fd{k}")
        set_fe = persist(both(both(equals(fetched, 0xFE), row_ok), valid), name=f"set_fe{k}")
        flipped = current - both(set_fd, current) + set_fe - both(set_fe, current)
        latch = {
            "lo": select(half, latch["lo"], flipped),
            "hi": select(half, flipped, latch["hi"]),
        }
        chr_address = select(whole_chr, full_address, window_address)
        groups = CHR + CHR_ALT + CHR_FULL
        lo = Memory.bits(memory.read(chr_address, groups, f"pattern_lo{k}"), f"pattern_lo{k}")
        hi = Memory.bits(memory.read(chr_address + 8, groups, f"pattern_hi{k}"), f"pattern_hi{k}")

        sprite0 = both(present, equals(inp["index0"], 0)) if k == 0 else 0
        meta = 16 + 8 * attr_bits[6] + 4 * attr_bits[7] + 64 * behind + 128 * sprite0
        shown = both(draw, valid)
        for c, pixel in enumerate(encode_row(lo, hi, flip_h, shown, meta)):
            outputs[f"sprite{k}_pixel{c}"] = pixel
        outputs[f"sprite{k}_x"] = x
        # A selected sprite fetches when valid; the empty slots past the
        # count fetch a dummy tile from $1000 under 8x16 or table 1.
        outputs[f"sprite{k}_fetch"] = valid + 1 - present
        outputs[f"sprite{k}_a12"] = select(present, half, either(tall, table8))

    # Whole-CHR fetches never flip a latch.
    outputs["new_latch_lo"] = select(whole_chr, inp["latch_lo"], latch["lo"])
    outputs["new_latch_hi"] = select(whole_chr, inp["latch_hi"], latch["hi"])
    outputs = {name: outputs[name] for name in FETCH_OUTPUTS}
    auto_name(locals())
    return sprite_tokens("FETCH", inp, memory_inputs, outputs)
