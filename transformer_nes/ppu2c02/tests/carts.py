"""Synthetic carts for the PPU graph tests (no game ROMs needed).

Each cart's main loop counts X up forever and drives the PPU registers
from it -- $2005/$2006 pairs, $2007 reads and writes across CHR,
nametables and palette, OAM access, status reads, ctrl and mask changes
that switch rendering, sprite size and pattern tables -- plus its
mapper's PPU-facing registers.
"""

import numpy as np


def load_reference_hardware():
    """Conventional PPU oracle; excluded from the production extension."""
    from transformer_nes.nes.build import _import_extension, _make

    _make("reference-hardware")
    return _import_extension("nes_hardware_reference")


def absolute(op: int, addr: int) -> list[int]:
    return [op, addr & 0xFF, addr >> 8]


def ppu_workout(mask_bits: int) -> list[int]:
    return (
        [0x8A]
        + absolute(0x8D, 0x2006)  # TXA; STA $2006
        + [0x49, 0x5A]
        + absolute(0x8D, 0x2006)  # EOR #$5A; STA $2006
        + absolute(0x8E, 0x2007)  # STX $2007
        + absolute(0xAD, 0x2007)
        + absolute(0xAD, 0x2007)  # LDA $2007 x2 (buffer)
        + absolute(0x8E, 0x2005)
        + absolute(0x8D, 0x2005)  # STX/STA $2005
        + [0x8A, 0x29, 0x3F]
        + absolute(0x8D, 0x2000)  # TXA; AND #$3F; STA $2000
        + [0x8A, 0x29, mask_bits]
        + absolute(0x8D, 0x2001)  # TXA; AND #mask; STA $2001
        + absolute(0x8E, 0x2003)
        + absolute(0x8D, 0x2004)  # STX $2003; STA $2004
        + absolute(0xAD, 0x2004)
        + absolute(0xAD, 0x2002)  # LDA $2004; LDA $2002
    )


def cart(mapper: int, prg: bytes, chr_: bytes, *, vertical: bool = False):
    flags6 = ((mapper & 0x0F) << 4) | int(vertical)
    header = bytes(
        (0x4E, 0x45, 0x53, 0x1A, len(prg) // 0x4000, len(chr_) // 0x2000, flags6, mapper & 0xF0)
    ) + bytes(8)
    return header + prg + chr_


def pattern_chr(size: int) -> bytes:
    return bytes((i * 29 + 7 + (i >> 12) * 13) & 0xFF for i in range(size))


def ppu_setup() -> list[int]:
    """Varied palette, all four nametables (and any page behind them),
    and OAM; then background and sprites on."""
    pal_loop = [0x98, 0x0A, 0x0A, 0x69, 0x05, 0x29, 0x3F] + absolute(0x8D, 0x2007) + [0xC8, 0xC0, 0x20]
    nt_loop = [0x8A, 0x49, 0xA5] + absolute(0x8D, 0x2007) + [0xE8]
    oam_loop = [0x8A] + absolute(0x8D, 0x2004) + [0xE8]
    return (
        [0xA9, 0x3F] + absolute(0x8D, 0x2006) + [0xA9, 0x00] + absolute(0x8D, 0x2006)
        + [0xA0, 0x00] + pal_loop + [0xD0, (-(len(pal_loop) + 2)) & 0xFF]  # BNE pal
        + [0xA9, 0x20] + absolute(0x8D, 0x2006) + [0xA9, 0x00] + absolute(0x8D, 0x2006)
        + [0xA2, 0x00, 0xA0, 0x10]  # LDX #0; LDY #16 (16 x 256 bytes)
        + nt_loop + [0xD0, (-(len(nt_loop) + 2)) & 0xFF]  # BNE nt
        + [0x88, 0xD0, (-(len(nt_loop) + 2 + 3)) & 0xFF]  # DEY; BNE nt
        + [0xA9, 0x00] + absolute(0x8D, 0x2003)
        + oam_loop + [0xD0, (-(len(oam_loop) + 2)) & 0xFF]  # BNE oam
        + [0xA9, 0x1E] + absolute(0x8D, 0x2001)  # rendering on
    )


def main_loop(start: int, per_pass: list[int], mask_bits: int) -> list[int]:
    """Scroll and the mapper move every pass; every 64th pass runs the
    register workout, then turns rendering back on."""
    workout = ppu_workout(mask_bits) + [0xA9, 0x1E] + absolute(0x8D, 0x2001)
    return (
        absolute(0x8E, 0x2005) + [0x8A, 0x0A] + absolute(0x8D, 0x2005)  # scroll x, 2x
        + per_pass
        + [0x8A, 0x29, 0x3F, 0xD0, len(workout)]  # every 64th pass:
        + workout
        + [0xE8] + absolute(0x4C, start)  # INX; JMP main
    )


def register_cart(hardware, *, vertical: bool, mask_bits: int = 0x18):
    """NROM: the PPU workout alone, rendering left to the workout."""
    code = [0xA2, 0x00] + ppu_workout(mask_bits) + [0xE8] + absolute(0x4C, 0x8002)
    prg = bytearray(0x4000)
    prg[: len(code)] = bytes(code)
    prg[-6:] = bytes((0x00, 0x80, 0x00, 0x80, 0x00, 0x80))
    return hardware.Cartridge.parse(cart(0, bytes(prg), pattern_chr(0x2000), vertical=vertical))


def last_bank_cart(mapper: int, setup: list[int], per_pass: list[int], chr_size: int, *, after_setup=()):
    """Code at $E000 in the last 8 KiB of a 32 KiB PRG (fixed under MMC2,
    MMC4 and MMC5's power-on mode)."""
    head = setup + ppu_setup() + list(after_setup) + [0xA2, 0x00]
    code = head + main_loop(0xE000 + len(head), per_pass, 0x1F)
    prg = bytearray(0x8000)
    prg[0x6000 : 0x6000 + len(code)] = bytes(code)
    prg[-6:] = bytes((0x00, 0xE0, 0x00, 0xE0, 0x00, 0xE0))
    return cart(mapper, bytes(prg), pattern_chr(chr_size))
def latch_cart(hardware, *, mapper: int):
    """MMC2 (9) / MMC4 (10): the four latch banks move every pass, and
    the workout's nametable and OAM bytes sweep through tiles $FD/$FE."""
    per_pass = []
    for k, reg in enumerate((0xB000, 0xC000, 0xD000, 0xE000)):
        per_pass += [0x8A, 0x69, 3 * k + 1, 0x29, 0x0F] + absolute(0x8D, reg)  # (X+c)&15
    per_pass += [0x8A, 0x29, 0x01] + absolute(0x8D, 0xF000)  # mirroring
    return hardware.Cartridge.parse(last_bank_cart(mapper, [], per_pass, 0x10000))


def mmc5_cart(hardware, *, exram_mode: int):
    """MMC5: every nametable map (CIRAM, ExRAM, fill), fill tile/color,
    ExRAM bytes, CHR upper bits and both CHR sets move every pass."""
    setup = [0xA9, exram_mode] + absolute(0x8D, 0x5104)  # ExRAM mode
    for k in range(12):  # CHR sets: $5120-$5127 sprites, $5128-$512B background
        setup += [0xA9, 5 * k + 3] + absolute(0x8D, 0x5120 + k)
    per_pass = (
        absolute(0x8E, 0x5105)  # STX $5105: the nametable map
        + absolute(0x8E, 0x5106)  # fill tile
        + absolute(0x8E, 0x5107)  # fill color
        + [0x8A, 0x29, 0x03]
        + absolute(0x8D, 0x5130)  # CHR upper bits
        + absolute(0x8E, 0x5C00)
        + absolute(0x8D, 0x5DA7)  # ExRAM bytes
        + [0x8A, 0x69, 0x11]
        + absolute(0x8D, 0x5128)  # a background bank
    )
    return hardware.Cartridge.parse(last_bank_cart(5, setup, per_pass, 0x20000))


def console_frames(hardware, cart, count, attach=None):
    """`count` frames of the cart under a scripted pad (Start taps, then
    A+Right), after `attach(console)` sets up any graph engines."""
    from transformer_nes.nes.backends import backend_factory

    console = hardware.Console(cart, backend_factory("weights"), 44100.0)
    if attach is not None:
        attach(console)
    out = []
    for i in range(count):
        console.set_buttons(0x08 if i % 120 in (40, 41) else 0x81 if i % 120 > 60 else 0)
        out.append(console.run_frame().copy())
    return np.array(out)
