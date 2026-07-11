"""NES mappers: PRG banking and mirroring control behind the MMIO bus.

Mapper registers are CPU writes into ROM space ($8000-$FFFF).  The bus
watches that range write-only (reads of ROM tables must stay native);
the machine's own store lands in its memory before the pause, so the
handler repairs the clobbered PRG byte and injects any bank change
through ``poke`` -- inject_write on the native machines, which keeps the
token log and memory structure consistent with every other bus write.

The console wires two hooks after boot/handoff: ``poke(addr, value)``
(backend-specific memory injection) and ``on_mirroring(mode)`` (the
PPU's runtime nametable arrangement, MMC1 encoding: 0/1 single-screen,
2 vertical, 3 horizontal).
"""

from __future__ import annotations

from transformer_nes.nes.cartridge import MIRROR_VERTICAL, Cartridge

BANK16 = 0x4000


class Nrom:
    """Mapper 0: fixed PRG, nothing to service."""

    banked = False

    def __init__(self, cart: Cartridge):
        self.cart = cart

    def cpu_image(self) -> bytes:
        image = bytearray(0x10000)
        image[0x8000:0x8000 + len(self.cart.prg)] = self.cart.prg
        if len(self.cart.prg) == BANK16:  # NROM-128: mirror at $C000
            image[0xC000:0x10000] = self.cart.prg
        return bytes(image)


class Mmc1:
    """Mapper 1 (SxROM): the serial 5-bit shift register at $8000-$FFFF.

    CHR-RAM carts only for now (Final Fantasy, Zelda, Metroid): PRG
    banking + mirroring; the CHR bank registers are accepted and ignored.
    """

    banked = True

    def __init__(self, cart: Cartridge):
        if cart.chr:
            raise NotImplementedError(
                "MMC1 CHR-ROM banking not supported yet (CHR-RAM carts only)")
        if len(cart.prg) % BANK16 or not cart.prg:
            raise ValueError("MMC1 PRG size must be a multiple of 16 KiB")
        self.prg = cart.prg
        self.n_banks = len(cart.prg) // BANK16
        self.control = 0x0C  # power-on: PRG mode 3 (last bank fixed at $C000)
        self.control |= {MIRROR_VERTICAL: 2}.get(cart.mirroring, 3)
        self.prg_reg = 0
        self.shift = 0x10
        self.poke = None           # set by the console per backend
        self.on_mirroring = None   # set by the console (PPU hook)

    # ── Mapping ─────────────────────────────────────────────────────

    def prg_banks(self) -> tuple[int, int]:
        """(16 KiB bank at $8000, bank at $C000) under the current mode."""
        mode = (self.control >> 2) & 3
        bank = (self.prg_reg & 0x0F) % self.n_banks
        if mode <= 1:  # 32 KiB switch
            return bank & 0x0E, (bank & 0x0E) | 1
        if mode == 2:  # first bank fixed at $8000
            return 0, bank
        return bank, self.n_banks - 1  # mode 3: last bank fixed at $C000

    def rom_byte(self, addr: int) -> int:
        lo, hi = self.prg_banks()
        bank = lo if addr < 0xC000 else hi
        return self.prg[bank * BANK16 + (addr & 0x3FFF)]

    def cpu_image(self) -> bytes:
        image = bytearray(0x10000)
        lo, hi = self.prg_banks()
        image[0x8000:0xC000] = self.prg[lo * BANK16:(lo + 1) * BANK16]
        image[0xC000:0x10000] = self.prg[hi * BANK16:(hi + 1) * BANK16]
        return bytes(image)

    # ── Device protocol ─────────────────────────────────────────────

    def read(self, addr: int) -> int:
        raise RuntimeError("mapper range is attached write-only")

    def write(self, addr: int, value: int) -> None:
        old_banks = self.prg_banks()
        old_mirroring = self.control & 3
        if value & 0x80:
            self.shift = 0x10
            self.control |= 0x0C
        else:
            complete = self.shift & 1
            self.shift = ((self.shift >> 1) | ((value & 1) << 4)) & 0x1F
            if complete:
                reg = (addr >> 13) & 3
                if reg == 0:
                    self.control = self.shift
                elif reg == 3:
                    self.prg_reg = self.shift
                # regs 1/2 are the CHR banks: ignored on CHR-RAM carts
                self.shift = 0x10
        new_banks = self.prg_banks()
        if new_banks[0] != old_banks[0]:
            self.load_bank(0x8000, new_banks[0])
        if new_banks[1] != old_banks[1]:
            self.load_bank(0xC000, new_banks[1])
        # The store landed in machine memory before the pause: repair it.
        self.poke(addr, self.rom_byte(addr))
        if (self.control & 3) != old_mirroring and self.on_mirroring is not None:
            self.on_mirroring(self.control & 3)

    def load_bank(self, base: int, bank: int) -> None:
        data = self.prg[bank * BANK16:(bank + 1) * BANK16]
        poke = self.poke
        for i, byte in enumerate(data):
            poke(base + i, byte)


def make_mapper(cart: Cartridge):
    if cart.mapper == 0:
        return Nrom(cart)
    if cart.mapper == 1:
        return Mmc1(cart)
    raise NotImplementedError(f"mapper {cart.mapper} not supported (NROM, MMC1)")
