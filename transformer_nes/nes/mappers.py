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
from collections.abc import Callable

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
    PRG banks switch at runtime, and this frame's scanline IRQ (if any).
    The console wires the callbacks after boot/handoff; unwired ones stay
    None (tests wire only what they observe)."""

    banked = False
    poke: Callable[[int, int], None] | None = None
    poke_region: Callable[[int, bytes], None] | None = None
    on_mirroring: Callable[[int], None] | None = None
    set_chr_banks: Callable | None = None
    on_irq_ack: Callable[[], None] | None = None

    def irq_scanline(self) -> int | None:
        return None

    def prg_scan_regions(self) -> tuple[tuple[int, bytes], ...]:
        """Every (base, bank bytes) PRG mapping reachable at runtime, for
        the sequence-window scan; the boot image is scanned separately."""
        return ()


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

    Supports CHR-RAM boards and CHR-ROM in MMC1's switchable 8 KiB or
    independent 4 KiB modes.
    """

    banked = True

    def __init__(self, cart: Cartridge):
        if len(cart.prg) % BANK16 or not cart.prg:
            raise ValueError("MMC1 PRG size must be a multiple of 16 KiB")
        if len(cart.chr) % 0x1000:
            raise ValueError("MMC1 CHR ROM must use 4 KiB bank alignment")
        self.prg = cart.prg
        self.n_banks = len(cart.prg) // BANK16
        self.chr_rom = bool(cart.chr)
        self.n_chr = max(len(cart.chr) // 0x400, 1)
        self.control = 0x0C  # power-on: PRG mode 3 (last bank fixed at $C000)
        self.control |= {MIRROR_VERTICAL: 2}.get(cart.mirroring, 3)
        self.prg_reg = 0
        self.chr_regs = [0, 0]
        self.shift = 0x10
        self.poke = None           # set by the console per backend
        self.poke_region = None    # bulk variant (native on the engines)
        self.on_mirroring = None   # set by the console (PPU hook)
        self.set_chr_banks = None  # set by the console for CHR-ROM boards

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

    def chr_banks(self) -> list[int]:
        """Eight 1 KiB PPU slots under MMC1's 8/4 KiB CHR mode."""
        if self.control & 0x10:
            lower = self.chr_regs[0] * 4
            upper = self.chr_regs[1] * 4
            banks = range(lower, lower + 4), range(upper, upper + 4)
            return [bank % self.n_chr for half in banks for bank in half]
        # In 8 KiB mode CHR bank 0's low bit is ignored. Registers count
        # 4 KiB units, so an even register selects eight consecutive 1K slots.
        base = (self.chr_regs[0] & 0x1E) * 4
        return [(base + i) % self.n_chr for i in range(8)]

    def cpu_image(self) -> bytes:
        image = bytearray(0x10000)
        lo, hi = self.prg_banks()
        image[0x8000:0xC000] = self.prg[lo * BANK16:(lo + 1) * BANK16]
        image[0xC000:0x10000] = self.prg[hi * BANK16:(hi + 1) * BANK16]
        return bytes(image)

    def prg_scan_regions(self) -> tuple[tuple[int, bytes], ...]:
        return tuple(
            (base, self.prg[b * BANK16:(b + 1) * BANK16])
            for b in range(self.n_banks) for base in (0x8000, 0xC000)
        )

    # ── Device protocol ─────────────────────────────────────────────

    def read(self, addr: int) -> int:
        raise RuntimeError("mapper range is attached write-only")

    def write(self, addr: int, value: int) -> None:
        old_banks = self.prg_banks()
        old_mirroring = self.control & 3
        old_chr = self.chr_banks() if self.chr_rom else None
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
                elif reg in (1, 2):
                    self.chr_regs[reg - 1] = self.shift
                elif reg == 3:
                    self.prg_reg = self.shift
                self.shift = 0x10
        new_banks = self.prg_banks()
        if new_banks[0] != old_banks[0]:
            self.load_bank(0x8000, new_banks[0])
        if new_banks[1] != old_banks[1]:
            self.load_bank(0xC000, new_banks[1])
        # The store landed in machine memory before the pause: repair it.
        assert self.poke is not None  # console-wired
        self.poke(addr, self.rom_byte(addr))
        if (self.control & 3) != old_mirroring and self.on_mirroring is not None:
            self.on_mirroring(self.control & 3)
        if self.chr_rom and self.set_chr_banks is not None:
            new_chr = self.chr_banks()
            if new_chr != old_chr:
                self.set_chr_banks(new_chr)

    def load_bank(self, base: int, bank: int) -> None:
        inject_region(self, base, self.prg[bank * BANK16:(bank + 1) * BANK16])

    def native_attach(self, rt) -> None:
        """Hand the serial register and PRG ROM to the engine's native
        banking (console silicon, like the PPU); register writes then
        reach Python only as journal drains via apply_native."""
        lo, hi = self.prg_banks()
        rt.set_mapper_mmc1(self.prg, self.shift, self.control,
                           self.prg_reg, lo, hi)

    def apply_native(self, addr: int, value: int) -> None:
        """Journal drain: bank injection and ROM-byte repair already
        happened engine-side; only CHR/mirroring side effects run here."""
        old_mirroring = self.control & 3
        old_chr = self.chr_banks() if self.chr_rom else None
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
                elif reg in (1, 2):
                    self.chr_regs[reg - 1] = self.shift
                elif reg == 3:
                    self.prg_reg = self.shift
                self.shift = 0x10
        if (self.control & 3) != old_mirroring and self.on_mirroring is not None:
            self.on_mirroring(self.control & 3)
        if self.chr_rom and self.set_chr_banks is not None:
            new_chr = self.chr_banks()
            if new_chr != old_chr:
                self.set_chr_banks(new_chr)


class Mmc2(Mapper):
    """Mapper 9 (PxROM): Punch-Out's 8 KiB PRG switch and CHR latches.

    Four registers select 4 KiB CHR banks. The PPU chooses each pattern
    half's FD or FE register when it fetches the corresponding latch tile;
    ``set_mmc2_banks`` sends all four choices to that native latch model.
    """

    banked = True

    def __init__(self, cart: Cartridge):
        if len(cart.prg) % BANK8 or len(cart.prg) < 4 * BANK8:
            raise ValueError("MMC2 PRG size must be at least four 8 KiB banks")
        if len(cart.chr) % 0x1000 or not cart.chr:
            raise ValueError("MMC2 requires CHR ROM in 4 KiB banks")
        self.prg = cart.prg
        self.n_banks = len(cart.prg) // BANK8
        self.n_chr = len(cart.chr) // 0x1000
        self.prg_reg = 0
        self.chr_regs = [0, 0, 0, 0]  # lower FD/FE, upper FD/FE
        self.chr_rom = True
        self.poke = None
        self.poke_region = None
        self.on_mirroring = None
        self.set_mmc2_banks = None

    def prg_banks(self) -> tuple[int, int, int, int]:
        return (self.prg_reg % self.n_banks,
                self.n_banks - 3, self.n_banks - 2, self.n_banks - 1)

    def mmc2_banks(self) -> list[int]:
        return [bank % self.n_chr for bank in self.chr_regs]

    def rom_byte(self, addr: int) -> int:
        bank = self.prg_banks()[(addr - 0x8000) >> 13]
        return self.prg[bank * BANK8 + (addr & 0x1FFF)]

    def cpu_image(self) -> bytes:
        image = bytearray(0x10000)
        for i, bank in enumerate(self.prg_banks()):
            base = 0x8000 + i * BANK8
            image[base:base + BANK8] = self.prg[bank * BANK8:(bank + 1) * BANK8]
        return bytes(image)

    def prg_scan_regions(self) -> tuple[tuple[int, bytes], ...]:
        # Only the $8000 slot switches; the fixed upper banks are in the
        # boot image already.
        return tuple(
            (0x8000, self.prg[b * BANK8:(b + 1) * BANK8])
            for b in range(self.n_banks)
        )

    def read(self, addr: int) -> int:
        raise RuntimeError("mapper range is attached write-only")

    def write(self, addr: int, value: int) -> None:
        region = addr & 0xF000
        if region == 0xA000:
            old = self.prg_banks()[0]
            self.prg_reg = value & 0x0F
            new = self.prg_banks()[0]
            if new != old:
                self.load_bank(0x8000, new)
        elif 0xB000 <= region <= 0xE000:
            self.chr_regs[(region - 0xB000) >> 12] = value & 0x1F
            if self.set_mmc2_banks is not None:
                self.set_mmc2_banks(self.mmc2_banks())
        elif region == 0xF000 and self.on_mirroring is not None:
            self.on_mirroring(3 if value & 1 else 2)
        assert self.poke is not None  # console-wired
        self.poke(addr, self.rom_byte(addr))

    def load_bank(self, base: int, bank: int) -> None:
        inject_region(self, base, self.prg[bank * BANK8:(bank + 1) * BANK8])


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

    def prg_scan_regions(self) -> tuple[tuple[int, bytes], ...]:
        # Switchable 8 KiB slots at $8000/$A000/$C000 across both modes;
        # $E000 is always the fixed last bank (boot image).
        return tuple(
            (base, self.prg[b * BANK8:(b + 1) * BANK8])
            for b in range(self.n_banks)
            for base in (0x8000, 0xA000, 0xC000)
        )

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
            assert self.on_mirroring is not None  # console-wired
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
            assert self.set_chr_banks is not None  # console-wired
            self.set_chr_banks(self.chr_banks())
        # The store landed in machine memory before the pause: repair it.
        assert self.poke is not None  # console-wired
        self.poke(addr, self.rom_byte(addr))

    def load_bank(self, base: int, bank: int) -> None:
        inject_region(self, base, self.prg[bank * BANK8:(bank + 1) * BANK8])

    def native_attach(self, rt) -> None:
        """Hand the register file and PRG ROM to the engine's native
        banking; register writes then reach Python only as journal
        drains via apply_native."""
        rt.set_mapper_mmc3(self.prg, self.bank_select, bytes(self.regs),
                           bytes(self.prg_banks()))

    def apply_native(self, addr: int, value: int) -> None:
        """Journal drain: bank injection, ROM-byte repair, and the $E000
        line ack already happened engine-side; CHR/IRQ/mirroring side
        effects run here."""
        reg = addr & 0xE001
        if reg == 0x8000:
            self.bank_select = value
        elif reg == 0x8001:
            self.regs[self.bank_select & 7] = value
        elif reg == 0xA000:
            assert self.on_mirroring is not None  # console-wired
            self.on_mirroring(3 if value & 1 else 2)
        elif reg == 0xC000:
            self.irq_latch = value
        elif reg == 0xE000:
            self.irq_enabled = False
            if self.on_irq_ack is not None:
                self.on_irq_ack()
        elif reg == 0xE001:
            self.irq_enabled = True
        if self.chr_rom and reg in (0x8000, 0x8001):
            assert self.set_chr_banks is not None  # console-wired
            self.set_chr_banks(self.chr_banks())


class Mmc4(Mapper):
    """Mapper 10 (FxROM): 16 KiB PRG banking and symmetric CHR latches."""

    banked = True

    def __init__(self, cart: Cartridge):
        if len(cart.prg) % BANK16 or len(cart.prg) < 2 * BANK16:
            raise ValueError("MMC4 PRG size must be at least two 16 KiB banks")
        if len(cart.chr) % 0x1000 or not cart.chr:
            raise ValueError("MMC4 requires CHR ROM in 4 KiB banks")
        self.prg = cart.prg
        self.n_banks = len(cart.prg) // BANK16
        self.n_chr = len(cart.chr) // 0x1000
        self.prg_reg = 0
        self.chr_regs = [0, 0, 0, 0]
        self.chr_rom = True
        self.poke = None
        self.poke_region = None
        self.on_mirroring = None
        self.set_mmc4_banks = None

    def prg_banks(self) -> tuple[int, int]:
        return self.prg_reg % self.n_banks, self.n_banks - 1

    def mmc4_banks(self) -> list[int]:
        return [bank % self.n_chr for bank in self.chr_regs]

    def rom_byte(self, addr: int) -> int:
        bank = self.prg_banks()[0 if addr < 0xC000 else 1]
        return self.prg[bank * BANK16 + (addr & 0x3FFF)]

    def cpu_image(self) -> bytes:
        image = bytearray(0x10000)
        lo, hi = self.prg_banks()
        image[0x8000:0xC000] = self.prg[lo * BANK16:(lo + 1) * BANK16]
        image[0xC000:0x10000] = self.prg[hi * BANK16:(hi + 1) * BANK16]
        return bytes(image)

    def prg_scan_regions(self) -> tuple[tuple[int, bytes], ...]:
        # Only the $8000 slot switches; $C000 is the fixed last bank.
        return tuple(
            (0x8000, self.prg[b * BANK16:(b + 1) * BANK16])
            for b in range(self.n_banks)
        )

    def read(self, addr: int) -> int:
        raise RuntimeError("mapper range is attached write-only")

    def write(self, addr: int, value: int) -> None:
        region = addr & 0xF000
        if region == 0xA000:
            old = self.prg_banks()[0]
            self.prg_reg = value & 0x0F
            new = self.prg_banks()[0]
            if new != old:
                inject_region(self, 0x8000,
                              self.prg[new * BANK16:(new + 1) * BANK16])
        elif 0xB000 <= region <= 0xE000:
            self.chr_regs[(region - 0xB000) >> 12] = value & 0x1F
            if self.set_mmc4_banks is not None:
                self.set_mmc4_banks(self.mmc4_banks())
        elif region == 0xF000 and self.on_mirroring is not None:
            self.on_mirroring(3 if value & 1 else 2)
        assert self.poke is not None  # console-wired
        self.poke(addr, self.rom_byte(addr))


class Mmc5(Mapper):
    """Mapper 5 (ExROM), initially covering Castlevania III's core path."""

    banked = True
    register_ranges = ((0x5000, 0x7FFF),)
    # Source register feeding each 8 KiB slot per prg_mode.  $5117 (index 3)
    # is always ROM; the others map PRG-RAM when their bit 7 is clear.
    PRG_SLOT_SOURCE = {0: (3, 3, 3, 3), 1: (1, 1, 3, 3),
                       2: (1, 1, 2, 3), 3: (0, 1, 2, 3)}

    def __init__(self, cart: Cartridge):
        if len(cart.prg) % BANK8 or not cart.prg or len(cart.chr) % 0x400:
            raise ValueError("MMC5 requires 8 KiB PRG and 1 KiB CHR bank alignment")
        self.prg, self.n_banks = cart.prg, len(cart.prg) // BANK8
        self.n_chr = max(len(cart.chr) // 0x400, 1)
        self.chr_rom = bool(cart.chr)
        self.prg_mode, self.chr_mode = 3, 3
        self.prg_regs = [0x80, 0x80, 0x80, 0xFF]
        self.chr_regs = list(range(8)) + [0, 1, 2, 3]
        self.chr_upper = 0
        self.irq_target = 0
        self.irq_enabled = False
        self.irq_pending = False
        self.mul_a = self.mul_b = 0xFF
        self.exram = bytearray(0x400)
        self.prg_ram = bytearray(0x10000)
        self.prg_ram_bank = 0
        self.ram_protect1 = self.ram_protect2 = 0
        self.exram_mode = 0
        self.nametable_map = 0
        self.fill_tile = 0
        self.fill_color = 0
        self.poke = self.poke_region = self.on_mirroring = None
        self.set_chr_banks = self.on_irq_ack = None
        self.set_mmc5_state = self.set_mmc5_exram = None
        self.set_mmc5_chr = None

    def prg_banks(self) -> tuple[int, int, int, int]:
        r = [x & 0x7F for x in self.prg_regs]
        if self.prg_mode == 0:
            b = r[3] & ~3
            return b, b + 1, b + 2, b + 3
        if self.prg_mode == 1:
            a, b = r[1] & ~1, r[3] & ~1
            return a, a + 1, b, b + 1
        if self.prg_mode == 2:
            a = r[1] & ~1
            return a, a + 1, r[2], r[3]
        return tuple(r)  # type: ignore[return-value]

    def chr_banks(self) -> list[int]:
        """Sprite-set 1 KiB slots.  Register values are bank numbers in the
        size selected by $5101 (unlike MMC3's fixed 1 KiB indexing)."""
        n = self.n_chr
        r = [(self.chr_upper << 8) | x for x in self.chr_regs[:8]]
        if self.chr_mode == 0:
            return [(r[7] * 8 + i) % n for i in range(8)]
        if self.chr_mode == 1:
            out = []
            for b in (r[3] * 4, r[7] * 4):
                out.extend((b + i) % n for i in range(4))
            return out
        if self.chr_mode == 2:
            out = []
            for j in (1, 3, 5, 7):
                b = r[j] * 2
                out.extend((b % n, (b + 1) % n))
            return out
        return [x % n for x in r]

    def background_chr_banks(self) -> list[int]:
        n = self.n_chr
        r = [(self.chr_upper << 8) | x for x in self.chr_regs[8:12]]
        if self.chr_mode == 0:
            return [(r[3] * 8 + i) % n for i in range(8)]
        if self.chr_mode == 1:
            out = []
            for b in (r[3] * 4, r[3] * 4):
                out.extend((b + i) % n for i in range(4))
            return out
        if self.chr_mode == 2:
            out = []
            for j in (1, 3, 1, 3):
                b = r[j] * 2
                out.extend((b % n, (b + 1) % n))
            return out
        return [r[i & 3] % n for i in range(8)]

    def sync_mmc5_chr(self) -> None:
        if self.set_mmc5_chr:
            self.set_mmc5_chr(self.chr_banks(), self.background_chr_banks())

    def ram_writable(self) -> bool:
        return self.ram_protect1 == 2 and self.ram_protect2 == 1

    def prg_ram_slots(self) -> tuple[bool, bool, bool, bool]:
        src = self.PRG_SLOT_SOURCE[self.prg_mode]
        return tuple(j != 3 and not self.prg_regs[j] & 0x80 for j in src)  # type: ignore[return-value]

    def prg_layout(self) -> tuple[tuple[int, bool], ...]:
        """Per-slot (bank, is_ram) -- the injection identity of each window."""
        return tuple(zip(self.prg_banks(), self.prg_ram_slots(), strict=True))

    def prg_slot_region(self, i: int) -> bytes:
        bank, is_ram = self.prg_layout()[i]
        if is_ram:
            base = (bank & 7) * BANK8
            return bytes(self.prg_ram[base:base + BANK8])
        base = (bank % self.n_banks) * BANK8
        return self.prg[base:base + BANK8]

    def rom_byte(self, addr: int) -> int:
        bank, is_ram = self.prg_layout()[(addr - 0x8000) >> 13]
        if is_ram:
            return self.prg_ram[(bank & 7) * BANK8 + (addr & 0x1FFF)]
        return self.prg[(bank % self.n_banks) * BANK8 + (addr & 0x1FFF)]

    def cpu_image(self) -> bytes:
        image = bytearray(0x10000)
        for i in range(4):
            image[0x8000 + i * BANK8:0xA000 + i * BANK8] = self.prg_slot_region(i)
        return bytes(image)

    def prg_scan_regions(self) -> tuple[tuple[int, bytes], ...]:
        # ROM banks in any of the four 8 KiB slots; PRG-RAM windows are
        # dynamic content, left to the JIT.
        return tuple(
            (base, self.prg[b * BANK8:(b + 1) * BANK8])
            for b in range(self.n_banks)
            for base in (0x8000, 0xA000, 0xC000, 0xE000)
        )

    def irq_scanline(self) -> int | None:
        return self.irq_target if self.irq_enabled and self.irq_target else None

    def read(self, addr: int) -> int:
        if 0x6000 <= addr <= 0x7FFF:
            return self.prg_ram[(self.prg_ram_bank & 7) * BANK8 + (addr & 0x1FFF)]
        if addr >= 0x8000:
            return self.rom_byte(addr)
        if addr == 0x5204:
            value = (0x80 if self.irq_pending else 0) | 0x40
            self.irq_pending = False
            if self.on_irq_ack:
                self.on_irq_ack()
            return value
        if addr == 0x5205:
            return (self.mul_a * self.mul_b) & 0xFF
        if addr == 0x5206:
            return (self.mul_a * self.mul_b) >> 8
        if 0x5C00 <= addr <= 0x5FFF:
            return self.exram[addr - 0x5C00]
        return 0

    def write(self, addr: int, value: int) -> None:
        value &= 0xFF
        old = self.prg_layout()
        if addr == 0x5100:
            self.prg_mode = value & 3
        elif addr == 0x5101:
            self.chr_mode = value & 3
        elif addr == 0x5102:
            self.ram_protect1 = value & 3
        elif addr == 0x5103:
            self.ram_protect2 = value & 3
        elif addr == 0x5104:
            self.exram_mode = value & 3
        elif addr == 0x5105:
            self.nametable_map = value
            assert self.on_mirroring is not None  # console-wired
            self.on_mirroring(value)
        elif addr == 0x5106:
            self.fill_tile = value
        elif addr == 0x5107:
            self.fill_color = value & 3
        elif addr == 0x5113:
            self.prg_ram_bank = value & 7
        elif 0x5114 <= addr <= 0x5117:
            self.prg_regs[addr - 0x5114] = value
        elif 0x5120 <= addr <= 0x512B:
            self.chr_regs[addr - 0x5120] = value
            if self.set_chr_banks:
                self.set_chr_banks(self.chr_banks())
            self.sync_mmc5_chr()
        elif addr == 0x5130:
            self.chr_upper = value & 3
            if self.set_chr_banks:
                self.set_chr_banks(self.chr_banks())
            self.sync_mmc5_chr()
        elif addr == 0x5203:
            self.irq_target = value
        elif addr == 0x5204:
            self.irq_enabled = bool(value & 0x80)
        elif addr == 0x5205:
            self.mul_a = value
        elif addr == 0x5206:
            self.mul_b = value
        elif 0x5C00 <= addr <= 0x5FFF:
            self.exram[addr - 0x5C00] = value
            if self.set_mmc5_exram:
                self.set_mmc5_exram(addr - 0x5C00, value)
        elif 0x6000 <= addr <= 0x7FFF and self.ram_writable():
            offset = (self.prg_ram_bank & 7) * BANK8 + (addr & 0x1FFF)
            self.prg_ram[offset] = value
        if ((0x5104 <= addr <= 0x5107 or addr == 0x5130)
                and self.set_mmc5_state):
            self.set_mmc5_state(self.exram_mode, self.nametable_map,
                                self.fill_tile, self.fill_color, self.chr_upper)
        new = self.prg_layout()
        for i, (a, b) in enumerate(zip(old, new, strict=True)):
            if a != b:
                inject_region(self, 0x8000 + i * BANK8, self.prg_slot_region(i))
        if addr >= 0x8000:
            i = (addr - 0x8000) >> 13
            bank, is_ram = new[i]
            if is_ram and self.ram_writable():
                self.prg_ram[(bank & 7) * BANK8 + (addr & 0x1FFF)] = value
                assert self.poke is not None  # console-wired
                self.poke(addr, value)
            else:
                assert self.poke is not None  # console-wired
                self.poke(addr, self.rom_byte(addr))


def make_mapper(cart: Cartridge):
    if cart.mapper == 0:
        return Nrom(cart)
    if cart.mapper == 1:
        return Mmc1(cart)
    if cart.mapper == 9:
        return Mmc2(cart)
    if cart.mapper == 4:
        return Mmc3(cart)
    if cart.mapper == 10:
        return Mmc4(cart)
    if cart.mapper == 5:
        return Mmc5(cart)
    raise NotImplementedError(
        f"mapper {cart.mapper} not supported (NROM, MMC1, MMC2, MMC3, MMC4, MMC5)")
