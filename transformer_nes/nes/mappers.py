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

import logging

from transformer_nes.nes.cartridge import MIRROR_VERTICAL, Cartridge

logger = logging.getLogger(__name__)

BANK8 = 0x2000
BANK16 = 0x4000


def inject_region(mapper, base: int, data: bytes) -> None:
    """Bank-switch injection: bulk native call when the backend has one."""
    if mapper.poke_region is not None:
        mapper.poke_region(base, data)
    else:
        poke = mapper.poke
        for i, byte in enumerate(data):
            poke(base + i, byte)


class Mapper:
    """Common contract the console drives every mapper through: whether
    PRG banks switch at runtime, and this frame's scanline IRQ (if any)."""

    banked = False

    def irq_scanline(self) -> int | None:
        return None


class Nrom(Mapper):
    """Mapper 0: fixed PRG, nothing to service."""

    def __init__(self, cart: Cartridge):
        self.cart = cart

    def cpu_image(self) -> bytes:
        image = bytearray(0x10000)
        image[0x8000:0x8000 + len(self.cart.prg)] = self.cart.prg
        if len(self.cart.prg) == BANK16:  # NROM-128: mirror at $C000
            image[0xC000:0x10000] = self.cart.prg
        return bytes(image)


class Mmc1(Mapper):
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
        self.poke_region = None    # bulk variant (native on the engines)
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
        inject_region(self, base, self.prg[bank * BANK16:(bank + 1) * BANK16])


class Mmc3(Mapper):
    """Mapper 4 (TxROM): 8 KiB PRG banks + 2/1 KiB CHR banks, plus a
    scanline IRQ used for status-bar raster splits (SMB3)."""

    banked = True

    def __init__(self, cart: Cartridge):
        if len(cart.prg) % BANK8 or not cart.prg:
            raise ValueError("MMC3 PRG size must be a multiple of 8 KiB")
        self.prg = cart.prg
        self.n_banks = len(cart.prg) // BANK8
        self.chr_rom = bool(cart.chr)
        self.n_chr = max(len(cart.chr) // 0x400, 1)
        self.bank_select = 0
        self.regs = [0, 2, 4, 5, 6, 7, 0, 1]  # R0-R7
        self.irq_latch = 0
        self.irq_enabled = False
        self.poke = None            # set by the console per backend
        self.poke_region = None     # bulk variant (native on the engines)
        self.on_mirroring = None    # set by the console (PPU hook)
        self.set_chr_banks = None   # set by the console (PPU hook)
        self.on_irq_ack = None      # set by the console (clears the IRQ line)

    def irq_scanline(self) -> int | None:
        """The visible scanline this frame's IRQ fires at, or None.

        Frame model: games reload the counter each vblank ($C001) and the
        counter decrements once per rendered scanline, so an enabled IRQ
        with latch N fires at the end of scanline N.
        """
        if not self.irq_enabled or not 0 < self.irq_latch < 240:
            return None
        return self.irq_latch

    # ── Mapping ─────────────────────────────────────────────────────

    def prg_banks(self) -> tuple[int, int, int, int]:
        """8 KiB banks at $8000/$A000/$C000/$E000 under the current mode."""
        r6, r7 = self.regs[6] % self.n_banks, self.regs[7] % self.n_banks
        second_last, last = self.n_banks - 2, self.n_banks - 1
        if self.bank_select & 0x40:
            return second_last, r7, r6, last
        return r6, r7, second_last, last

    def chr_banks(self) -> list[int]:
        """Eight 1 KiB slot banks under the current A12 inversion."""
        r, n = self.regs, self.n_chr
        two = [(r[0] & 0xFE) % n, ((r[0] & 0xFE) | 1) % n,
               (r[1] & 0xFE) % n, ((r[1] & 0xFE) | 1) % n]
        one = [r[2] % n, r[3] % n, r[4] % n, r[5] % n]
        return one + two if self.bank_select & 0x80 else two + one

    def rom_byte(self, addr: int) -> int:
        bank = self.prg_banks()[(addr - 0x8000) >> 13]
        return self.prg[bank * BANK8 + (addr & 0x1FFF)]

    def cpu_image(self) -> bytes:
        image = bytearray(0x10000)
        for i, bank in enumerate(self.prg_banks()):
            base = 0x8000 + i * BANK8
            image[base:base + BANK8] = self.prg[bank * BANK8:(bank + 1) * BANK8]
        return bytes(image)

    # ── Device protocol ─────────────────────────────────────────────

    def read(self, addr: int) -> int:
        raise RuntimeError("mapper range is attached write-only")

    def write(self, addr: int, value: int) -> None:
        old_prg = self.prg_banks()
        reg = addr & 0xE001
        if reg == 0x8000:
            self.bank_select = value
        elif reg == 0x8001:
            self.regs[self.bank_select & 7] = value
        elif reg == 0xA000:
            self.on_mirroring(3 if value & 1 else 2)
        elif reg == 0xC000:
            self.irq_latch = value
        elif reg == 0xE000:
            self.irq_enabled = False
            if self.on_irq_ack is not None:
                self.on_irq_ack()  # $E000 also acks a delivered IRQ
        elif reg == 0xE001:
            self.irq_enabled = True
        # $A001 (PRG RAM protect) and $C001 (IRQ reload) need no action in
        # the frame model: irq_scanline() re-derives from the latch per frame.
        new_prg = self.prg_banks()
        for i, (old, new) in enumerate(zip(old_prg, new_prg, strict=True)):
            if old != new:
                self.load_bank(0x8000 + i * BANK8, new)
        if self.chr_rom and reg in (0x8000, 0x8001):
            self.set_chr_banks(self.chr_banks())
        # The store landed in machine memory before the pause: repair it.
        self.poke(addr, self.rom_byte(addr))

    def load_bank(self, base: int, bank: int) -> None:
        inject_region(self, base, self.prg[bank * BANK8:(bank + 1) * BANK8])


def make_mapper(cart: Cartridge):
    if cart.mapper == 0:
        return Nrom(cart)
    if cart.mapper == 1:
        return Mmc1(cart)
    if cart.mapper == 4:
        return Mmc3(cart)
    raise NotImplementedError(
        f"mapper {cart.mapper} not supported (NROM, MMC1, MMC3)")
