"""Frame-level NES PPU model (2C02), lives behind the MMIO bus.

The CPU-in-the-graph talks to this PPU exclusively through memory-mapped
registers $2000-$2007 (mirrored through $3FFF) -- every access arrives via
the bus's watch/inject machinery, exactly like the real chip sits on the
address bus.  Rendering is frame-at-once: the console runs the CPU for a
visible-frame cycle budget, asks for one rendered frame, then opens
vblank.

Faithful register semantics: $2002 read clears vblank and the shared
write latch; $2005/$2006 use the shared two-write latch; $2007 reads are
buffered (palette reads bypass the buffer) and increment by 1 or 32;
nametables honor cartridge H/V mirroring; palette RAM mirrors $3F10/14/
18/1C onto $3F00/04/08/0C.

Frame-level approximations (documented, revisit when a game needs them):
base-cycle frame pacing, sprite-0 hit raised at its scanline's cycle
offset from bg/sprite overlap of the *rendered* frame, sprite-overflow
flag always 0, no mid-frame scroll/bank changes (no raster effects), no
left-edge masking, no color emphasis/greyscale.
"""

from __future__ import annotations

import numpy as np

from transformer_nes.nes.cartridge import MIRROR_VERTICAL, Cartridge

# Standard 2C02 palette (RGB triples for the 64 NES colors).
NES_PALETTE = np.array([
    (84, 84, 84), (0, 30, 116), (8, 16, 144), (48, 0, 136),
    (68, 0, 100), (92, 0, 48), (84, 4, 0), (60, 24, 0),
    (32, 42, 0), (8, 58, 0), (0, 64, 0), (0, 60, 0),
    (0, 50, 60), (0, 0, 0), (0, 0, 0), (0, 0, 0),
    (152, 150, 152), (8, 76, 196), (48, 50, 236), (92, 30, 228),
    (136, 20, 176), (160, 20, 100), (152, 34, 32), (120, 60, 0),
    (84, 90, 0), (40, 114, 0), (8, 124, 0), (0, 118, 40),
    (0, 102, 120), (0, 0, 0), (0, 0, 0), (0, 0, 0),
    (236, 238, 236), (76, 154, 236), (120, 124, 236), (176, 98, 236),
    (228, 84, 236), (236, 88, 180), (236, 106, 100), (212, 136, 32),
    (160, 170, 0), (116, 196, 0), (76, 208, 32), (56, 204, 108),
    (56, 180, 204), (60, 60, 60), (0, 0, 0), (0, 0, 0),
    (236, 238, 236), (168, 204, 236), (188, 188, 236), (212, 178, 236),
    (236, 174, 236), (236, 174, 212), (236, 180, 176), (228, 196, 144),
    (204, 210, 120), (180, 222, 120), (168, 226, 144), (152, 226, 180),
    (160, 214, 228), (160, 162, 160), (0, 0, 0), (0, 0, 0),
], dtype=np.uint8)


class PPU:
    """Register file, VRAM/OAM, and a frame-at-once renderer."""

    def __init__(self, cart: Cartridge):
        self.chr = bytearray(cart.chr) if cart.chr else bytearray(0x2000)
        self.chr_writable = not cart.chr  # CHR RAM when the cart has no CHR ROM
        self.mirroring = cart.mirroring
        self.nametables = bytearray(0x800)
        self.palette = bytearray(0x20)
        self.oam = bytearray(0x100)

        self.ctrl = 0
        self.mask = 0
        self.oam_addr = 0
        self.scroll_x = 0
        self.scroll_y = 0
        self.vram_addr = 0
        self.write_latch = False  # False: next $2005/$2006 write is the first
        self.read_buffer = 0

        self.vblank = False
        self.sprite0_hit = False
        self.nmi_edge = False  # NMI enable rose while vblank was set
        # Scanline of the sprite-0/background overlap in the last rendered
        # frame, or None; the console turns it into a mid-frame hit event.
        self.sprite0_scanline: int | None = None

    # ── Device protocol (registers $2000-$3FFF, mirrored every 8) ──

    def read(self, addr: int) -> int:
        reg = (addr - 0x2000) & 7
        if reg == 2:  # PPUSTATUS
            value = (0x80 if self.vblank else 0) | (0x40 if self.sprite0_hit else 0)
            self.vblank = False
            self.write_latch = False
            return value
        if reg == 4:  # OAMDATA
            return self.oam[self.oam_addr]
        if reg == 7:  # PPUDATA (buffered below $3F00, direct for palette)
            addr = self.vram_addr & 0x3FFF
            if addr >= 0x3F00:
                value = self.palette[self.palette_index(addr)] & 0x3F
                # The buffer still loads from the nametable "underneath".
                self.read_buffer = self.vram_read(addr - 0x1000)
            else:
                value = self.read_buffer
                self.read_buffer = self.vram_read(addr)
            self.vram_addr = (self.vram_addr + self.increment) & 0x3FFF
            return value
        return 0  # write-only registers read as open bus; model as 0

    def read_lease(self, addr: int) -> int | None:
        """Value repeated $2002 reads return for the rest of the visible
        span, or None.  Sound because status changes only at the sprite-0
        and vblank cycles (the console clears the lease there) and the
        read side effects (vblank flag, write latch) are already applied
        by the granting read; PPU writes clear the lease at the bus."""
        if (addr - 0x2000) & 7 != 2 or self.vblank:
            return None
        return 0x40 if self.sprite0_hit else 0

    def write(self, addr: int, value: int) -> None:
        reg = (addr - 0x2000) & 7
        value &= 0xFF
        if reg == 0:  # PPUCTRL
            rising = (value & 0x80) and not (self.ctrl & 0x80)
            self.ctrl = value
            if rising and self.vblank:
                self.nmi_edge = True
        elif reg == 1:  # PPUMASK
            self.mask = value
        elif reg == 3:  # OAMADDR
            self.oam_addr = value
        elif reg == 4:  # OAMDATA
            self.oam[self.oam_addr] = value
            self.oam_addr = (self.oam_addr + 1) & 0xFF
        elif reg == 5:  # PPUSCROLL (x then y)
            if self.write_latch:
                self.scroll_y = value
            else:
                self.scroll_x = value
            self.write_latch = not self.write_latch
        elif reg == 6:  # PPUADDR (high then low, shared latch with $2005)
            if self.write_latch:
                self.vram_addr = ((self.vram_addr & 0x3F00) | value) & 0x3FFF
            else:
                self.vram_addr = ((value & 0x3F) << 8) | (self.vram_addr & 0x00FF)
            self.write_latch = not self.write_latch
        elif reg == 7:  # PPUDATA
            self.vram_write(self.vram_addr & 0x3FFF, value)
            self.vram_addr = (self.vram_addr + self.increment) & 0x3FFF

    # ── Console-side control ────────────────────────────────────────

    @property
    def nmi_enabled(self) -> bool:
        return bool(self.ctrl & 0x80)

    @property
    def rendering_enabled(self) -> bool:
        return bool(self.mask & 0x18)

    def begin_vblank(self) -> None:
        self.vblank = True

    def end_vblank(self) -> None:
        self.vblank = False
        self.sprite0_hit = False

    def take_nmi_edge(self) -> bool:
        edge = self.nmi_edge
        self.nmi_edge = False
        return edge

    def oam_dma(self, page: bytes) -> None:
        """$4014: copy 256 bytes into OAM starting at OAMADDR."""
        for i, b in enumerate(page):
            self.oam[(self.oam_addr + i) & 0xFF] = b

    # ── VRAM mapping ────────────────────────────────────────────────

    @property
    def increment(self) -> int:
        return 32 if self.ctrl & 0x04 else 1

    def nt_offset(self, addr: int) -> int:
        a = (addr - 0x2000) & 0xFFF
        table, offset = a >> 10, a & 0x3FF
        phys = table & 1 if self.mirroring == MIRROR_VERTICAL else table >> 1
        return phys * 0x400 + offset

    @staticmethod
    def palette_index(addr: int) -> int:
        index = (addr - 0x3F00) & 0x1F
        if index in (0x10, 0x14, 0x18, 0x1C):  # sprite mirrors of bg entries
            index -= 0x10
        return index

    def vram_read(self, addr: int) -> int:
        addr &= 0x3FFF
        if addr < 0x2000:
            return self.chr[addr]
        if addr < 0x3F00:
            return self.nametables[self.nt_offset(addr)]
        return self.palette[self.palette_index(addr)]

    def vram_write(self, addr: int, value: int) -> None:
        addr &= 0x3FFF
        if addr < 0x2000:
            if self.chr_writable:
                self.chr[addr] = value
        elif addr < 0x3F00:
            self.nametables[self.nt_offset(addr)] = value
        else:
            self.palette[self.palette_index(addr)] = value

    # ── Rendering ───────────────────────────────────────────────────

    def tile_cache(self) -> np.ndarray:
        """Decode both pattern tables: (512, 8, 8) array of 2-bit pixels."""
        raw = np.frombuffer(bytes(self.chr), dtype=np.uint8).reshape(512, 2, 8, 1)
        bits = np.unpackbits(raw, axis=3)  # MSB first: bit 7 is the left pixel
        return (bits[:, 0] + 2 * bits[:, 1]).astype(np.uint8)

    def render_background(self, tiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Render the visible 256x240 background with coarse scroll.

        Returns (color indices 240x256 in NES palette space, opacity mask).
        """
        backdrop = self.palette[0] & 0x3F
        if not self.mask & 0x08:
            frame = np.full((240, 256), backdrop, dtype=np.uint8)
            return frame, np.zeros((240, 256), dtype=bool)

        # Render the full 2x2 nametable canvas (512x480), then slice it at
        # the scroll position with wraparound.
        bg_table = 256 if self.ctrl & 0x10 else 0
        nt = np.frombuffer(bytes(self.nametables), dtype=np.uint8)
        canvas_pix = np.zeros((480, 512), dtype=np.uint8)   # 2-bit pattern
        canvas_pal = np.zeros((480, 512), dtype=np.uint8)   # palette select
        for logical in range(4):
            base = self.nt_offset(0x2000 + logical * 0x400)
            tile_ids = nt[base:base + 0x3C0].reshape(30, 32).astype(np.intp)
            attrs = nt[base + 0x3C0:base + 0x400].reshape(8, 8)
            pattern = tiles[bg_table + tile_ids]              # (30, 32, 8, 8)
            pix = pattern.transpose(0, 2, 1, 3).reshape(240, 256)
            # Attribute bytes: one per 4x4 tile block, 2 bits per 2x2 quadrant.
            blocks = np.repeat(np.repeat(attrs, 4, axis=0), 4, axis=1)[:30, :32]
            rows = np.arange(30)[:, None] % 4 // 2
            cols = np.arange(32)[None, :] % 4 // 2
            shift = (rows * 4 + cols * 2).astype(np.uint8)
            palsel = (blocks >> shift) & 3
            palsel = np.repeat(np.repeat(palsel, 8, axis=0), 8, axis=1)
            y0, x0 = (logical >> 1) * 240, (logical & 1) * 256
            canvas_pix[y0:y0 + 240, x0:x0 + 256] = pix
            canvas_pal[y0:y0 + 240, x0:x0 + 256] = palsel

        sx = (self.scroll_x + (self.ctrl & 1) * 256) % 512
        sy = (self.scroll_y + ((self.ctrl >> 1) & 1) * 240) % 480
        xs = (np.arange(256) + sx) % 512
        ys = (np.arange(240) + sy) % 480
        pix = canvas_pix[np.ix_(ys, xs)]
        palsel = canvas_pal[np.ix_(ys, xs)]

        pal = np.frombuffer(bytes(self.palette), dtype=np.uint8) & 0x3F
        entry = np.where(pix == 0, 0, palsel * 4 + pix)
        frame = pal[entry]
        frame[pix == 0] = backdrop
        return frame, pix != 0

    def render_frame(self) -> np.ndarray:
        """Render one frame; returns 240x256 NES color indices.

        Also computes the sprite-0/background overlap scanline for the
        console's hit-event scheduling.
        """
        tiles = self.tile_cache()
        frame, bg_opaque = self.render_background(tiles)
        self.sprite0_scanline = None
        if not self.mask & 0x10:
            return frame

        pal = np.frombuffer(bytes(self.palette), dtype=np.uint8) & 0x3F
        tall = bool(self.ctrl & 0x20)  # 8x16 sprites
        # Draw from sprite 63 down so lower-index sprites win overlaps.
        for i in range(63, -1, -1):
            y, tile, attr, x = self.oam[i * 4:i * 4 + 4]
            top = y + 1
            if top >= 240:
                continue
            if tall:
                table = (tile & 1) * 256
                first = tile & 0xFE
                pattern = np.concatenate([tiles[table + first], tiles[table + first + 1]])
            else:
                table = 256 if self.ctrl & 0x08 else 0
                pattern = tiles[table + tile]
            if attr & 0x40:
                pattern = pattern[:, ::-1]
            if attr & 0x80:
                pattern = pattern[::-1, :]
            h = pattern.shape[0]
            rows = min(240 - top, h)
            cols = min(256 - x, 8)
            if cols <= 0 or rows <= 0:
                continue
            block = pattern[:rows, :cols]
            opaque = block != 0
            if attr & 0x20:  # behind background: only over transparent bg
                opaque = opaque & ~bg_opaque[top:top + rows, x:x + cols]
            colors = pal[0x10 + (attr & 3) * 4 + block]
            region = frame[top:top + rows, x:x + cols]
            region[opaque] = colors[opaque]
            if i == 0:
                hit = (block != 0) & bg_opaque[top:top + rows, x:x + cols]
                if hit.any():
                    self.sprite0_scanline = top + int(np.argmax(hit.any(axis=1)))
        return frame

    @staticmethod
    def to_rgb(frame: np.ndarray) -> np.ndarray:
        """Convert a frame of NES color indices to (240, 256, 3) RGB."""
        return NES_PALETTE[frame]
