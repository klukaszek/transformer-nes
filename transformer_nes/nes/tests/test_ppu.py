"""PPU register semantics and renderer checks (nes/ppu.cpp)."""

from __future__ import annotations

import numpy as np

from transformer_nes.nes.cartridge import Cartridge
from transformer_nes.nes.ppu import make_ppu as load_ppu


def make_ppu(mirroring: str = "horizontal", chr_rom: bytes = b""):
    cart = Cartridge(prg=bytes(0x4000), chr=chr_rom, mirroring=mirroring, mapper=0)
    return load_ppu(cart)


def test_ppu_views_keep_native_owner_alive():
    ppu = make_ppu()
    palette = ppu.palette
    assert palette.base is ppu
    del ppu
    palette[3] = 0x2A
    assert palette[3] == 0x2A


def test_mmc2_pattern_fetches_switch_chr_latches():
    chr_rom = b"".join(bytes([bank + 1]) * 0x1000 for bank in range(4))
    ppu = make_ppu(chr_rom=chr_rom)
    ppu.set_mmc2_banks([0, 1, 2, 3])

    assert ppu.vram_read(0x0000) == 2
    assert ppu.vram_read(0x1000) == 4
    assert ppu.vram_read(0x0FD8) == 2         # trigger tile uses old bank
    assert ppu.vram_read(0x0000) == 1
    assert ppu.vram_read(0x1FD8) == 4         # then subsequent reads switch
    assert ppu.vram_read(0x1000) == 3
    assert ppu.vram_read(0x0FE8) == 1
    assert ppu.vram_read(0x0000) == 2


def test_mmc2_hidden_background_prefetch_switches_next_scanline():
    banks = [bytearray(0x1000) for _ in range(4)]
    # Upper FE bank: tile 1 is opaque; upper FD bank leaves it blank.
    banks[3][16:24] = b"\xff" * 8
    chr_rom = b"".join(bytes(bank) for bank in banks)
    ppu = make_ppu(mirroring="vertical", chr_rom=chr_rom)
    ppu.set_mmc2_banks([0, 1, 2, 3])
    for col in range(32):
        ppu.vram_write(0x2000 + col, 1)
    # First hidden prefetch crosses into vertical-mirrored nametable 1.
    ppu.vram_write(0x2400, 0xFD)
    ppu.vram_write(0x3F00, 0)
    ppu.vram_write(0x3F01, 1)
    ppu.write(0x2000, 0x10)  # background patterns at $1000
    ppu.write(0x2001, 0x0A)  # background + left edge
    frame = np.asarray(ppu.render_frame())

    assert frame[0, 0] == 1               # row 0 fetched from FE bank
    assert frame[1, 0] == 0               # hidden $FD selected FD for row 1


def test_mmc4_low_pattern_latch_responds_on_every_tile_row():
    chr_rom = b"".join(bytes([bank + 1]) * 0x1000 for bank in range(4))
    ppu = make_ppu(chr_rom=chr_rom)
    ppu.set_mmc4_banks([0, 1, 2, 3])

    assert ppu.vram_read(0x0000) == 2
    assert ppu.vram_read(0x0FDF) == 2     # unlike MMC2, row 7 triggers FD
    assert ppu.vram_read(0x0000) == 1
    assert ppu.vram_read(0x0FEF) == 1
    assert ppu.vram_read(0x0000) == 2


def test_mmc5_exram_and_fill_nametable_substitution():
    ppu = make_ppu()
    # Table 0 = ExRAM, table 1 = fill, tables 2/3 = CIRAM 0.
    ppu.set_mmc5_state(0, 0x0E, 0x37, 2, 0)
    ppu.set_mmc5_exram(5, 0xA4)
    assert ppu.vram_read(0x2005) == 0xA4
    assert ppu.vram_read(0x2400) == 0x37
    assert ppu.vram_read(0x27C0) == 0xAA


def test_mmc5_chr_banks_latch_at_irq_split():
    """MMC5 swaps CHR at a mid-frame IRQ split like scroll: rows above the
    split render from the frame-start windows, rows below from live."""
    ppu = make_ppu()
    ppu.set_mmc5_state(0, 0, 0, 0, 0)     # MMC5 active, all CIRAM0, exram off
    ppu.set_chr_banks(list(range(8)))     # identity, so vram writes hit banks
    for r in range(8):
        ppu.vram_write(0x0400 + r, 0xFF)      # phys bank 1, tile 0 plane 0: pix 1
        ppu.vram_write(0x0808 + r, 0xFF)      # phys bank 2, tile 0 plane 1: pix 2
    ppu.vram_write(0x3F01, 0x11)
    ppu.vram_write(0x3F02, 0x22)
    ppu.mask = 0x0A                        # background + leftmost column
    ppu.ctrl = 0x20                        # 8x16: background uses $5128-$512B
    ppu.set_mmc5_chr([0] * 8, [1] * 8)     # above the split -> bank 1
    ppu.end_vblank()                       # latch the frame-start windows
    ppu.set_mmc5_chr([0] * 8, [3] * 8)     # chained IRQ's temporary blank bank
    ppu.mark_split(100)
    ppu.set_mmc5_chr([0] * 8, [2] * 8)     # below the split -> bank 2

    frame = np.asarray(ppu.render_frame())
    assert frame[10][0] == 0x11            # frozen bank 1 above the split
    assert frame[200][0] == 0x22           # live bank 2 below the split
    assert ppu.mmc5_split_bg()[0] == 3     # IRQ snapshot stays distinct


def test_mmc5_chained_irqs_preserve_each_raster_band():
    ppu = make_ppu()
    ppu.set_mmc5_state(0, 0, 0, 0, 0)
    ppu.set_chr_banks(list(range(8)))
    for row in range(8):
        ppu.vram_write(0x0400 + row, 0xFF)       # bank 1 -> pixel 1
        ppu.vram_write(0x0808 + row, 0xFF)       # bank 2 -> pixel 2
        ppu.vram_write(0x0C00 + row, 0xFF)       # bank 3 -> pixel 3
        ppu.vram_write(0x0C08 + row, 0xFF)
    ppu.vram_write(0x3F01, 0x11)
    ppu.vram_write(0x3F02, 0x12)
    ppu.vram_write(0x3F03, 0x13)
    ppu.mask = 0x0A
    ppu.ctrl = 0x20
    ppu.set_mmc5_chr([0] * 8, [1] * 8)
    ppu.end_vblank()
    ppu.mark_split(41)                       # frame bank 1 through line 41
    ppu.set_mmc5_chr([0] * 8, [2] * 8)
    ppu.mark_split(42)                       # bank 2 on line 42
    ppu.set_mmc5_chr([0] * 8, [3] * 8)
    ppu.mark_split(52)                       # bank 3 on lines 43..52
    ppu.set_mmc5_chr([0] * 8, [4] * 8)       # live below line 52

    frame = np.asarray(ppu.render_frame())
    assert frame[20, 0] == 0x11
    assert frame[42, 0] == 0x12
    assert frame[50, 0] == 0x13
    assert frame[100, 0] == 0


def test_mmc5_raster_journal_handles_more_than_eight_splits():
    """Render scratch capacity must match the PPU's raster journal."""
    ppu = make_ppu()
    ppu.set_mmc5_state(0, 0, 0, 0, 0)
    ppu.mask = 0x0A
    ppu.end_vblank()
    for scanline in range(10, 130, 10):
        ppu.mark_split(scanline)

    frame = np.asarray(ppu.render_frame())
    assert frame.shape == (240, 256)


def test_mmc5_v_reload_scrolls_relative_to_moving_irq_boundary():
    ppu = make_ppu()
    ppu.set_mmc5_state(0, 0x44, 0, 0, 0)
    ppu.set_chr_banks(list(range(8)))
    for row in range(8):
        ppu.vram_write(0x0410 + row, 0xFF)  # tile 1 -> pixel 1
        ppu.vram_write(0x0428 + row, 0xFF)  # tile 2 -> pixel 2
    for col in range(32):
        ppu.vram_write(0x2800 + col, 1)
        ppu.vram_write(0x2820 + col, 2)
    ppu.vram_write(0x3F01, 0x11)
    ppu.vram_write(0x3F02, 0x22)
    ppu.mask = 0x0A
    ppu.ctrl = 0x20
    ppu.set_mmc5_chr([0] * 8, [1] * 8)
    ppu.end_vblank()
    ppu.vram_addr = 0x2000
    ppu.mark_split(41)
    ppu.mark_split(42)
    ppu.mark_split(46)                    # moving camera boundary
    ppu.t_addr = 0x2800
    ppu.vram_addr = 0x2800                # handler reloads v
    ppu.mark_split(53)                    # next IRQ observes the new v

    frame = np.asarray(ppu.render_frame())
    assert frame[47, 0] == 0x11            # v=$2800 starts at the boundary
    assert frame[55, 0] == 0x22            # relative advance reaches row 1


def test_mmc5_nametable_substitution_survives_mid_frame_map_change():
    """A band's latched $5105 map must keep decoding ExRAM/fill sources
    even after the live map changes (CV3 rewrites $5105 mid-frame)."""
    ppu = make_ppu()
    for r in range(8):
        ppu.vram_write(0x0010 + r, 0xFF)   # tile 1 plane 0: pix 1
    ppu.vram_write(0x3F01, 0x15)
    ppu.mask = 0x0A
    ppu.set_mmc5_state(0, 0xFF, 1, 0, 0)   # all four tables = fill tile 1
    ppu.end_vblank()                       # latch the span map
    ppu.set_mmc5_state(0, 0x00, 1, 0, 0)   # live map now all CIRAM 0
    frame = np.asarray(ppu.render_frame())
    assert frame[100][100] == 0x15         # fill tile still renders above


def test_mmc5_8x8_sprites_render_background_from_sprite_set():
    """In 8x8 sprite mode $5128-$512B are ignored; background fetches use
    the $5120-$5127 set."""
    ppu = make_ppu()
    ppu.set_mmc5_state(0, 0, 0, 0, 0)
    ppu.set_chr_banks(list(range(8)))
    for r in range(8):
        ppu.vram_write(0x0400 + r, 0xFF)   # phys bank 1, tile 0: pix 1
        ppu.vram_write(0x0808 + r, 0xFF)   # phys bank 2, tile 0: pix 2
    ppu.vram_write(0x3F01, 0x11)
    ppu.vram_write(0x3F02, 0x22)
    ppu.mask = 0x0A
    ppu.ctrl = 0x00                        # 8x8 sprites
    ppu.set_mmc5_chr([1] * 8, [2] * 8)
    ppu.end_vblank()
    frame = np.asarray(ppu.render_frame())
    assert frame[10][0] == 0x11            # sprite set, not the ignored bg set


def test_status_read_clears_vblank_and_latch():
    ppu = make_ppu()
    ppu.begin_vblank()
    ppu.write(0x2006, 0x21)  # first write flips the latch
    assert ppu.write_latch is True
    value = ppu.read(0x2002)
    assert value & 0x80
    assert ppu.vblank is False
    assert ppu.write_latch is False
    assert ppu.read(0x2002) & 0x80 == 0  # second read: flag stays clear


def test_addr_latch_and_data_write_increment():
    ppu = make_ppu()
    ppu.write(0x2006, 0x21)  # high
    ppu.write(0x2006, 0x08)  # low
    ppu.write(0x2007, 0xAB)
    assert ppu.vram_read(0x2108) == 0xAB
    assert ppu.vram_addr == 0x2109  # +1 mode
    ppu.write(0x2000, 0x04)  # +32 mode
    ppu.write(0x2007, 0xCD)
    assert ppu.vram_read(0x2109) == 0xCD
    assert ppu.vram_addr == 0x2109 + 32


def test_data_read_is_buffered_but_palette_is_direct():
    ppu = make_ppu()
    ppu.vram_write(0x2100, 0x55)
    ppu.write(0x2006, 0x21)
    ppu.write(0x2006, 0x00)
    first = ppu.read(0x2007)   # stale buffer
    second = ppu.read(0x2007)  # now the real byte
    assert first == 0x00
    assert second == 0x55

    ppu.palette[ppu.palette_index(0x3F07)] = 0x2A
    ppu.write(0x2006, 0x3F)
    ppu.write(0x2006, 0x07)
    assert ppu.read(0x2007) == 0x2A  # palette reads bypass the buffer


def test_palette_sprite_mirrors():
    ppu = make_ppu()
    ppu.write(0x2006, 0x3F)
    ppu.write(0x2006, 0x10)
    ppu.write(0x2007, 0x21)
    assert ppu.palette[0x00] == 0x21  # $3F10 mirrors $3F00


def test_nametable_mirroring():
    horizontal = make_ppu("horizontal")
    horizontal.vram_write(0x2000, 7)
    assert horizontal.vram_read(0x2400) == 7      # $2400 mirrors $2000
    assert horizontal.vram_read(0x2800) == 0

    vertical = make_ppu("vertical")
    vertical.vram_write(0x2000, 9)
    assert vertical.vram_read(0x2800) == 9        # $2800 mirrors $2000
    assert vertical.vram_read(0x2400) == 0


def test_register_mirrors_every_8_bytes():
    ppu = make_ppu()
    ppu.begin_vblank()
    assert ppu.read(0x3FFA) & 0x80  # $3FFA mirrors $2002
    assert ppu.vblank is False


def test_oam_dma_wraps_at_oamaddr():
    ppu = make_ppu()
    ppu.write(0x2003, 0xFE)  # OAMADDR
    page = bytes(range(256))
    ppu.oam_dma(page)
    assert ppu.oam[0xFE] == 0
    assert ppu.oam[0xFF] == 1
    assert ppu.oam[0x00] == 2


def test_nmi_edge_on_enable_during_vblank():
    ppu = make_ppu()
    ppu.begin_vblank()
    ppu.write(0x2000, 0x80)
    assert ppu.take_nmi_edge() is True
    assert ppu.take_nmi_edge() is False  # consumed
    ppu.write(0x2000, 0x80)              # no rising edge
    assert ppu.take_nmi_edge() is False


def test_render_sprites_priority_flips_and_sprite0():
    """A register-built scene: patterned CHR RAM, both nametables +
    attributes, coarse scroll, a flipped sprite, a behind-background
    sprite, and a sprite-0 overlap with a known scanline."""
    ppu = make_ppu()
    for tile in range(3):
        for row in range(8):
            ppu.vram_write(tile * 16 + row, 0xFF if tile else 0x00)  # plane 0
            ppu.vram_write(tile * 16 + 8 + row, 0x0F if tile == 2 else 0x00)

    ppu.write(0x2006, 0x20)  # tile row 2: an opaque strip of tile 1
    ppu.write(0x2006, 0x40)
    for _ in range(32):
        ppu.write(0x2007, 1)
    ppu.write(0x2006, 0x3F)  # palette: distinct entries
    ppu.write(0x2006, 0x00)
    for i in range(0x20):
        ppu.write(0x2007, (i * 3 + 1) & 0x3F)

    ppu.write(0x2003, 0)     # sprite 0 at (8, y=15): overlaps the strip
    for value in (15, 2, 0x00, 8):
        ppu.write(0x2004, value)
    for value in (99, 2, 0xC0, 60):   # sprite 1: h+v flipped, in the open
        ppu.write(0x2004, value)
    for value in (17, 2, 0x20, 40):   # sprite 2: behind the background
        ppu.write(0x2004, value)

    # $2006 writes share the scroll state (loopy t), so reset the address
    # before rendering, exactly as real games do after VRAM streaming.
    ppu.write(0x2006, 0x00)
    ppu.write(0x2006, 0x00)

    ppu.write(0x2001, 0x18)  # rendering on
    frame = np.asarray(ppu.render_frame())
    assert frame.shape == (240, 256)
    assert ppu.sprite0_scanline == 16              # strip starts at y=16
    # The $3F10 write mirrors onto $3F00, so the backdrop is entry $10's.
    assert frame[0, 0] == (0x10 * 3 + 1) & 0x3F
    assert frame[16, 8] == (0x11 * 3 + 1) & 0x3F   # sprite 0 in front of strip
    assert frame[100, 60] == (0x13 * 3 + 1) & 0x3F  # flipped corner pixel (pix=3)
    # Behind-background sprite: hidden over the strip, visible below it.
    assert frame[18, 40] == (1 * 3 + 1) & 0x3F     # strip color wins
    assert frame[24, 40] == (0x11 * 3 + 1) & 0x3F  # sprite shows over backdrop
    again = np.asarray(ppu.render_frame())
    assert np.array_equal(frame, again)            # renderer is deterministic
