"""Neural transitions for CPU accesses to $2000-$2007.

Tokens carry the register file, status and rendering context. Readouts return
new registers, read values, writes and timing effects. Scroll addresses use
loopy fields; CHR reads honor MMC2/MMC4 latches and MMC5 CPU banks.
Thresholds require integer inputs; byte extraction requires values in 0..255.
"""

from __future__ import annotations

from transformer_nes.graph import core as graph
from transformer_nes.graph.core import (
    Expression,
    ExprLike,
    InputDimension,
    auto_name,
    persist,
    stepglu,
    to_expr,
)
from transformer_nes.graph.ops import both, equals, extract_bits, gate, select, wrap8
from transformer_nes.ppu2c02.space import (
    CHR,
    CHR_ALT,
    CHR_ALT_BASE,
    CHR_FULL,
    CHR_FULL_BASE,
    OAM,
    PAGES,
    PALETTE,
    Memory,
    in_palette_region,
    nametable_page,
    palette_offset,
)

LOOPY_FIELDS = ("cx", "cy", "nt", "fy")
# Register-file fields an ACCESS token carries and the step outputs.
STATE_FIELDS = (
    *(f"t_{f}" for f in LOOPY_FIELDS),
    *(f"v_{f}" for f in LOOPY_FIELDS),
    "fine_x",
    "w",
    "ctrl",
    "mask",
    "oam_addr",
    "read_buffer",
)
# Access context: the access itself and the timing side's view.
# nt_page0-3: the source page each logical nametable reads (space.py).
NT_PAGES = tuple(f"nt_page{k}" for k in range(4))
ACCESS_FIELDS = (
    "reg",
    "value",
    "is_write",
    "status",
    "rendering",
    *NT_PAGES,
    "chr_writable",
    # MMC2/MMC4 latches (0 FD, 1 FE) per pattern-table half; MMC5's banks.
    "latch_lo",
    "latch_hi",
    *(f"bank{s}" for s in range(8)),
)
# whole_chr: CHR is addressed through MMC5's banks into the whole CHR,
# rather than through the CHR windows.
INPUTS = ("is_access", *ACCESS_FIELDS, *STATE_FIELDS, "whole_chr")
EFFECT_FIELDS = (
    "read_value",
    "mem_write",
    "mem_write_addr",
    "mem_write_value",
    "clear_vblank",
    "nmi_rise",
    "nmi_fall",
)
OUTPUTS = tuple(f"new_{f}" for f in STATE_FIELDS) + EFFECT_FIELDS

# Folded experts, one per register and direction: an access always meets
# exactly one's pins, so the full circuit is never built. Only $2007 reaches
# CHR, so only it comes twice: through the CHR windows and (mmc5_*) through
# MMC5's banks into the whole CHR.
EXPERTS = {
    **{
        f"{'write' if writing else 'read'}{r}": {
            "writing": writing,
            **{f"reg{j}": int(j == r) for j in range(8)},
            **({"whole_chr": 0} if r == 7 else {}),
        }
        for writing in range(2)
        for r in range(8)
    },
    **{
        f"mmc5_{'write' if writing else 'read'}7": {
            "writing": writing, **{f"reg{j}": int(j == 7) for j in range(8)}, "whole_chr": 1,
        }
        for writing in range(2)
    },
}


def gated(default: ExprLike, cases: list[tuple[ExprLike, ExprLike]]) -> Expression:
    """default, or the value of the one case whose 0/1 gate is set (the
    gates must be mutually exclusive): a parallel gated sum."""
    out = to_expr(default)
    for g, value in cases:
        out = out + gate(g, to_expr(value) - default)
    return out


def build() -> tuple[dict[str, Expression], dict[str, Expression]]:
    one = graph.current_graph().one
    inp = {name: InputDimension(name) for name in (*ACCESS_FIELDS, *STATE_FIELDS)}
    is_access = InputDimension("is_access")
    is_mem_write = InputDimension("is_mem_write")
    mem_addr = InputDimension("mem_addr")
    mem_value = InputDimension("mem_value")
    whole_input = InputDimension("whole_chr")

    input_tokens = {
        "ACCESS": is_access + sum((to_expr(d) for d in inp.values()), Expression()) + whole_input,
        "WRITE": is_mem_write + mem_addr + mem_value,
    }

    reg, value = inp["reg"], inp["value"]
    is_write = persist(inp["is_write"], name="writing")
    t = {f: inp[f"t_{f}"] for f in LOOPY_FIELDS}
    v = {f: inp[f"v_{f}"] for f in LOOPY_FIELDS}
    w = inp["w"]
    oam_addr = inp["oam_addr"]
    read_buffer = inp["read_buffer"]
    rendering = inp["rendering"]

    # ── Decode ──────────────────────────────────────────────────────
    # Every product below has a 0/1 factor: one-unit gated forms.
    value_bits = extract_bits(value, "value")  # MSB first
    bit = {7 - i: b for i, b in enumerate(value_bits)}

    def value_field(lo: int, width: int) -> Expression:
        return sum((bit[lo + k] * (1 << k) for k in range(width)), Expression())

    ctrl_bits = extract_bits(inp["ctrl"], "ctrl")
    ctrl_nmi = ctrl_bits[0]  # bit 7
    ctrl_inc32 = ctrl_bits[5]  # bit 2

    selected = [persist(equals(reg, r), name=f"reg{r}") for r in range(8)]
    wr = [both(g, is_write) for g in selected]
    rd = [g - wr_r for g, wr_r in zip(selected, wr, strict=True)]
    first = 1 - w  # the shared $2005/$2006 write toggle
    scroll_x = both(wr[5], first)
    scroll_y = wr[5] - scroll_x
    addr_hi = both(wr[6], first)
    addr_lo = wr[6] - addr_hi

    # ── t, fine x, w ────────────────────────────────────────────────
    t_cy_b4 = stepglu(one, t["cy"] - 16)
    t_cy_b3 = stepglu(one, t["cy"] - 16 * t_cy_b4 - 8)
    t_cy_high = 16 * t_cy_b4 + 8 * t_cy_b3
    new_t = {
        "nt": gated(t["nt"], [(wr[0], value_field(0, 2)), (addr_hi, value_field(2, 2))]),
        "cx": gated(t["cx"], [(scroll_x, value_field(3, 5)), (addr_lo, value_field(0, 5))]),
        "cy": gated(
            t["cy"],
            [
                (scroll_y, value_field(3, 5)),
                (addr_hi, t["cy"] - t_cy_high + 8 * value_field(0, 2)),
                (addr_lo, t_cy_high + value_field(5, 3)),
            ],
        ),
        # $2006's high write clears bit 14 (fine y's top bit).
        "fy": gated(t["fy"], [(scroll_y, value_field(0, 3)), (addr_hi, value_field(4, 2))]),
    }
    new_fine_x = gated(inp["fine_x"], [(scroll_x, value_field(0, 3))])
    new_w = gated(w, [(rd[2], 0), (wr[5] + wr[6], first)])

    # ── The $2007 address and its physical location ─────────────────
    fy_low = v["fy"] - 4 * stepglu(one, v["fy"] - 4)  # address bits 12-13
    nt_b1 = stepglu(one, v["nt"] - 2)
    nt_b0 = v["nt"] - 2 * nt_b1
    address = 4096 * fy_low + 1024 * v["nt"] + 32 * v["cy"] + v["cx"]
    chr_region = 1 - stepglu(one, fy_low - 2)
    palette_region = in_palette_region(v)
    nt_region = 1 - chr_region - palette_region
    page = nametable_page(v["nt"], [inp[n] for n in NT_PAGES])
    nt_phys = 0x2000 + 0x400 * page + 32 * v["cy"] + v["cx"]
    palette_phys = 0x3F00 + palette_offset(v["cx"])
    phys = gate(chr_region, address) + gate(nt_region, nt_phys) + gate(palette_region, palette_phys)
    # A palette read still refills the buffer from the nametable byte
    # "underneath" ($2F00-$2FFF shares the nametable fields).
    # CHR as $2007 sees it (a CHR address has fy_low 0 or 1, its bit 12):
    # through MMC5's 1 KiB banks into the whole CHR, or through the windows.
    slot = 4 * fy_low + v["nt"]  # the address's 1 KiB slot
    bank = sum((gate(equals(slot, s), inp[f"bank{s}"]) for s in range(8)), Expression())
    full_phys = CHR_FULL_BASE + 0x400 * bank + 32 * v["cy"] + v["cx"]
    latch = select(fy_low, inp["latch_hi"], inp["latch_lo"])
    window_phys = address + gate(latch, CHR_ALT_BASE)
    chr_phys = select(persist(whole_input, name="whole_chr"), full_phys, window_phys)
    chr_groups = CHR + CHR_ALT + CHR_FULL
    buffer_phys = gate(chr_region, chr_phys) + gate(1 - chr_region, nt_phys)

    # ── Memory reads ────────────────────────────────────────────────
    # The harness seeds every byte of PPU memory, so reads are dense, and
    # each read only spans the groups its address can fall in.
    memory = Memory(is_mem_write, mem_addr, mem_value)
    buffer_fill = memory.read(buffer_phys, tuple(sorted(chr_groups + PAGES)), "buffer_fill")
    direct_addr = select(selected[4], 0x4000 + oam_addr, palette_phys)
    direct_byte = memory.read(direct_addr, PALETTE + OAM, "direct_byte")

    # ── The $2007 increment ─────────────────────────────────────────
    # Outside rendering: +1 or +32 across the 15-bit register, as field
    # carries.  During rendering the access collides with the fetch
    # pipeline and bumps coarse x and y instead (loopy's increments).
    step1 = 1 - ctrl_inc32
    x_wrap = stepglu(one, v["cx"] - 31)
    y_wrap = stepglu(one, v["cy"] - 31)
    fy_wrap = stepglu(one, v["fy"] - 7)
    carry_x = both(step1, x_wrap)
    carry_y_in = carry_x + ctrl_inc32
    carry_y = both(carry_y_in, y_wrap)
    carry_nt = both(carry_y, stepglu(one, v["nt"] - 3))
    carry_fy = both(carry_nt, fy_wrap)
    stepped = {
        "cx": v["cx"] + step1 - 32 * carry_x,
        "cy": v["cy"] + carry_y_in - 32 * carry_y,
        "nt": v["nt"] + carry_y - 4 * carry_nt,
        "fy": v["fy"] + carry_nt - 8 * carry_fy,
    }
    cy_29 = both(fy_wrap, equals(v["cy"], 29))
    cy_31 = both(fy_wrap, y_wrap)
    bumped = {
        "cx": v["cx"] + 1 - 32 * x_wrap,
        "cy": v["cy"] + fy_wrap - 30 * cy_29 - 32 * cy_31,
        "nt": v["nt"] + gate(x_wrap, 1 - 2 * nt_b0) + gate(cy_29, 2 - 4 * nt_b1),
        "fy": v["fy"] + 1 - 8 * fy_wrap,
    }
    new_v = {
        f: gated(
            v[f],
            [
                (addr_lo, new_t[f]),
                (selected[7], select(rendering, bumped[f], stepped[f])),
            ],
        )
        for f in LOOPY_FIELDS
    }

    # ── Remaining registers and effects ─────────────────────────────
    new_ctrl = gated(inp["ctrl"], [(wr[0], value)])
    new_mask = gated(inp["mask"], [(wr[1], value)])
    new_oam_addr = gated(oam_addr, [(wr[3], value), (wr[4], wrap8(oam_addr + 1))])
    new_read_buffer = gated(read_buffer, [(rd[7], buffer_fill)])
    read_value = (
        gate(rd[2], inp["status"])
        + gate(rd[4], direct_byte)
        + gate(rd[7], select(palette_region, direct_byte, read_buffer))
    )
    # Writes to CHR ROM, and to MMC5's fill-mode page, reach no memory.
    fill_page = both(nt_region, stepglu(one, page - 3))
    vram_write = wr[7] - both(wr[7], both(chr_region, 1 - inp["chr_writable"]) + fill_page)
    outputs = {
        **{f"new_t_{f}": new_t[f] for f in LOOPY_FIELDS},
        **{f"new_v_{f}": new_v[f] for f in LOOPY_FIELDS},
        "new_fine_x": new_fine_x,
        "new_w": new_w,
        "new_ctrl": new_ctrl,
        "new_mask": new_mask,
        "new_oam_addr": new_oam_addr,
        "new_read_buffer": new_read_buffer,
        "read_value": read_value,
        "mem_write": wr[4] + vram_write,
        "mem_write_addr": gate(wr[4], 0x4000 + oam_addr) + gate(wr[7], phys),
        # Palette RAM is 6 bits wide.
        "mem_write_value": value - gate(both(wr[7], palette_region), 128 * bit[7] + 64 * bit[6]),
        "clear_vblank": rd[2],
        "nmi_rise": both(both(wr[0], bit[7]), 1 - ctrl_nmi),
        "nmi_fall": both(both(wr[0], ctrl_nmi), 1 - bit[7]),
    }
    readout = Expression()
    for name, expr in outputs.items():
        persisted = persist(expr, name=name)
        readout = readout + 1e-6 * persisted

    # The step has no successor token to predict: READOUT keeps the
    # outputs alive to the last layer, where the harness reads them.
    output_tokens = {"READOUT": readout}
    auto_name(locals())
    return input_tokens, output_tokens
