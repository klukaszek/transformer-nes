"""Unit tests for the frame-level PPU register semantics."""

from __future__ import annotations

from transformer_vm.nes.cartridge import Cartridge
from transformer_vm.nes.ppu import PPU


def _ppu(mirroring: str = "horizontal") -> PPU:
    cart = Cartridge(prg=bytes(0x4000), chr=b"", mirroring=mirroring, mapper=0)
    return PPU(cart)


def test_status_read_clears_vblank_and_latch():
    ppu = _ppu()
    ppu.begin_vblank()
    ppu.write(0x2006, 0x21)  # first write flips the latch
    assert ppu.write_latch is True
    value = ppu.read(0x2002)
    assert value & 0x80
    assert ppu.vblank is False
    assert ppu.write_latch is False
    assert ppu.read(0x2002) & 0x80 == 0  # second read: flag stays clear


def test_addr_latch_and_data_write_increment():
    ppu = _ppu()
    ppu.write(0x2006, 0x21)  # high
    ppu.write(0x2006, 0x08)  # low
    ppu.write(0x2007, 0xAB)
    assert ppu._vram_read(0x2108) == 0xAB
    assert ppu.vram_addr == 0x2109  # +1 mode
    ppu.write(0x2000, 0x04)  # +32 mode
    ppu.write(0x2007, 0xCD)
    assert ppu._vram_read(0x2109) == 0xCD
    assert ppu.vram_addr == 0x2109 + 32


def test_data_read_is_buffered_but_palette_is_direct():
    ppu = _ppu()
    ppu._vram_write(0x2100, 0x55)
    ppu.write(0x2006, 0x21)
    ppu.write(0x2006, 0x00)
    first = ppu.read(0x2007)   # stale buffer
    second = ppu.read(0x2007)  # now the real byte
    assert first == 0x00
    assert second == 0x55

    ppu.palette[ppu._palette_index(0x3F07)] = 0x2A
    ppu.write(0x2006, 0x3F)
    ppu.write(0x2006, 0x07)
    assert ppu.read(0x2007) == 0x2A  # palette reads bypass the buffer


def test_palette_sprite_mirrors():
    ppu = _ppu()
    ppu.write(0x2006, 0x3F)
    ppu.write(0x2006, 0x10)
    ppu.write(0x2007, 0x21)
    assert ppu.palette[0x00] == 0x21  # $3F10 mirrors $3F00


def test_nametable_mirroring():
    horizontal = _ppu("horizontal")
    horizontal._vram_write(0x2000, 7)
    assert horizontal._vram_read(0x2400) == 7      # $2400 mirrors $2000
    assert horizontal._vram_read(0x2800) == 0

    vertical = _ppu("vertical")
    vertical._vram_write(0x2000, 9)
    assert vertical._vram_read(0x2800) == 9        # $2800 mirrors $2000
    assert vertical._vram_read(0x2400) == 0


def test_register_mirrors_every_8_bytes():
    ppu = _ppu()
    ppu.begin_vblank()
    assert ppu.read(0x3FFA) & 0x80  # $3FFA mirrors $2002
    assert ppu.vblank is False


def test_oam_dma_wraps_at_oamaddr():
    ppu = _ppu()
    ppu.write(0x2003, 0xFE)  # OAMADDR
    page = bytes(range(256))
    ppu.oam_dma(page)
    assert ppu.oam[0xFE] == 0
    assert ppu.oam[0xFF] == 1
    assert ppu.oam[0x00] == 2


def test_nmi_edge_on_enable_during_vblank():
    ppu = _ppu()
    ppu.begin_vblank()
    ppu.write(0x2000, 0x80)
    assert ppu.take_nmi_edge() is True
    assert ppu.take_nmi_edge() is False  # consumed
    ppu.write(0x2000, 0x80)              # no rising edge
    assert ppu.take_nmi_edge() is False
