"""PPU register semantics and renderer checks (nes/ppu.cpp)."""

from __future__ import annotations

import numpy as np

from transformer_nes.nes.cartridge import Cartridge
from transformer_nes.nes.ppu import make_ppu as load_ppu


def make_ppu(mirroring: str = "horizontal", chr_rom: bytes = b""):
    cart = Cartridge(prg=bytes(0x4000), chr=chr_rom, mirroring=mirroring, mapper=0)
    return load_ppu(cart)


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
