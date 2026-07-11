// Native frame-level NES PPU (2C02).  Registers $2000-$2007 behind the
// MMIO bus, VRAM/OAM, and a frame-at-once renderer; ported from a Python
// reference and proven pixel-identical before that reference was deleted
// (receipts in ROADMAP.md, 2026-07-11), then upgraded same day with the
// shared loopy-t scroll state and the sprite-0 split render (see ppu.py's
// docstring for the model).  Plain fixed silicon -- nothing here is
// model-generated -- compiled once and cached by content digest.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace py = pybind11;

struct PPU {
    // Full CHR data (ROM of any 1 KiB-multiple size, or 8 KiB CHR RAM);
    // the PPU's $0000-$1FFF window maps through eight 1 KiB bank slots.
    std::vector<uint8_t> chr;
    int chr_bank[8];
    bool chr_writable;
    // Nametable arrangement, runtime-switchable by mappers (MMC1
    // encoding): 0/1 single-screen low/high, 2 vertical, 3 horizontal.
    int mirroring;
    uint8_t nametables[0x800];
    uint8_t palette[0x20];
    uint8_t oam[0x100];

    int ctrl = 0, mask = 0, oam_addr = 0;
    int vram_addr = 0;
    // Shared scroll/address state (loopy t + fine x): $2000 sets its
    // nametable bits, $2005 its scroll bits, $2006 overwrites it as an
    // address -- so an address-then-scroll reset sequence really does
    // reset the nametable, which frame-start rendering depends on.
    int t_addr = 0, fine_x = 0;
    bool write_latch = false;
    int read_buffer = 0;

    bool vblank = false, sprite0_hit = false, nmi_edge = false;
    int sprite0_scanline = -1;  // -1: none
    bool bg_opaque[240][256];
    // Per-pixel sprite ownership: the lowest-index opaque sprite claims
    // the pixel, and only its priority bit decides front-vs-behind.
    bool sprite_claimed[240][256];

    // Background state as of the visible span's start (latched at
    // end_vblank): rows above the sprite-0 hit render from these, rows
    // below from the live registers -- the sprite-0 split.
    int frame_ctrl = 0, frame_mask = 0, frame_mirroring = 3;
    int frame_t_addr = 0, frame_fine_x = 0;
    int frame_chr_bank[8];
    bool frame_latch_valid = false;

    // A second mid-frame split (mapper IRQ): the console latches the state
    // at delivery; rows above the split render from it, rows below from
    // the live (post-IRQ-handler) registers.  Consumed per render.
    int mid_scanline = -1;
    int mid_ctrl = 0, mid_mask = 0, mid_mirroring = 3;
    int mid_t_addr = 0, mid_fine_x = 0, mid_v_addr = 0;
    int mid_chr_bank[8];

    void mark_split(int scanline) {
        mid_scanline = scanline;
        mid_ctrl = ctrl;
        mid_mask = mask;
        mid_mirroring = mirroring;
        mid_t_addr = t_addr;
        mid_fine_x = fine_x;
        mid_v_addr = vram_addr;
        std::memcpy(mid_chr_bank, chr_bank, sizeof(chr_bank));
    }

    // Everything a background row depends on: pattern/enable/edge bits,
    // nametable arrangement, scroll, and CHR banks.  y_base >= 0 pins the
    // band's FIRST row to an absolute nametable y (a mid-frame $2006 v
    // load -- the hardware y-jump; y otherwise derives from t).
    struct BandState {
        int ctrl, mask, mirroring, t_addr, fine_x, y_base;
        const int* banks;
    };

    static bool same_band(const BandState& a, const BandState& b) {
        return (a.ctrl & 0x10) == (b.ctrl & 0x10)
            && (a.mask & 0x0A) == (b.mask & 0x0A)
            && a.mirroring == b.mirroring
            && (a.t_addr & 0x7FFF) == (b.t_addr & 0x7FFF)
            && a.fine_x == b.fine_x
            && a.y_base == b.y_base
            && std::memcmp(a.banks, b.banks, 8 * sizeof(int)) == 0;
    }

    int scroll_px_x() const { return ((t_addr & 0x1F) << 3) | fine_x; }
    int scroll_px_y() const { return scroll_px_y_for(t_addr); }

    // Coarse + fine y as an on-screen pixel row (0-239) from a loopy addr.
    static int scroll_px_y_for(int addr) {
        return (((addr >> 5) & 0x1F) << 3) | ((addr >> 12) & 7);
    }
    // ... plus the nametable-y bit, giving the 0-479 two-screen row.
    static int scroll_row_480(int addr) {
        return scroll_px_y_for(addr) + ((addr >> 11) & 1) * 240;
    }

    PPU(py::bytes chr_data, bool writable, bool vertical) {
        std::string c = chr_data;
        chr.assign(c.begin(), c.end());
        if (chr.size() < 0x2000) chr.resize(0x2000, 0);
        for (int i = 0; i < 8; ++i) chr_bank[i] = i * 0x400;
        chr_writable = writable;
        mirroring = vertical ? 2 : 3;
        std::memset(nametables, 0, sizeof(nametables));
        std::memset(palette, 0, sizeof(palette));
        std::memset(oam, 0, sizeof(oam));
    }

    // ── Device protocol (registers $2000-$3FFF, mirrored every 8) ──

    int increment() const { return (ctrl & 0x04) ? 32 : 1; }

    static int palette_index(int addr) {
        int index = (addr - 0x3F00) & 0x1F;
        if (index == 0x10 || index == 0x14 || index == 0x18 || index == 0x1C)
            index -= 0x10;
        return index;
    }

    static int nt_offset_for(int addr, int mirr) {
        const int a = (addr - 0x2000) & 0xFFF;
        const int table = a >> 10, offset = a & 0x3FF;
        int phys;
        switch (mirr) {
            case 0:  phys = 0; break;           // single-screen low
            case 1:  phys = 1; break;           // single-screen high
            case 2:  phys = table & 1; break;   // vertical
            default: phys = table >> 1; break;  // horizontal
        }
        return phys * 0x400 + offset;
    }

    int nt_offset(int addr) const { return nt_offset_for(addr, mirroring); }

    int chr_at(int addr) const {
        return chr[chr_bank[(addr >> 10) & 7] + (addr & 0x3FF)];
    }

    void set_chr_banks(const std::vector<int>& banks_1k) {
        if (banks_1k.size() != 8) throw std::runtime_error("expected 8 CHR banks");
        const int n = (int)(chr.size() >> 10);
        for (int i = 0; i < 8; ++i) chr_bank[i] = (banks_1k[i] % n) * 0x400;
    }

    int vram_read(int addr) const {
        addr &= 0x3FFF;
        if (addr < 0x2000) return chr_at(addr);
        if (addr < 0x3F00) return nametables[nt_offset(addr)];
        return palette[palette_index(addr)];
    }

    void vram_write(int addr, int value) {
        addr &= 0x3FFF;
        if (addr < 0x2000) {
            if (chr_writable)
                chr[chr_bank[(addr >> 10) & 7] + (addr & 0x3FF)] = (uint8_t)value;
        } else if (addr < 0x3F00) {
            nametables[nt_offset(addr)] = (uint8_t)value;
        } else {
            palette[palette_index(addr)] = (uint8_t)value;
        }
    }

    int read(int addr) {
        const int reg = (addr - 0x2000) & 7;
        if (reg == 2) {  // PPUSTATUS
            const int value = (vblank ? 0x80 : 0) | (sprite0_hit ? 0x40 : 0);
            vblank = false;
            write_latch = false;
            return value;
        }
        if (reg == 4) return oam[oam_addr];  // OAMDATA
        if (reg == 7) {  // PPUDATA (buffered below $3F00, direct for palette)
            const int a = vram_addr & 0x3FFF;
            int value;
            if (a >= 0x3F00) {
                value = palette[palette_index(a)] & 0x3F;
                // The buffer still loads from the nametable "underneath".
                read_buffer = vram_read(a - 0x1000);
            } else {
                value = read_buffer;
                read_buffer = vram_read(a);
            }
            vram_addr = (vram_addr + increment()) & 0x3FFF;
            return value;
        }
        return 0;  // write-only registers read as open bus; model as 0
    }

    std::optional<int> read_lease(int addr) const {
        if (((addr - 0x2000) & 7) != 2 || vblank) return std::nullopt;
        return sprite0_hit ? 0x40 : 0;
    }

    void write(int addr, int value) {
        const int reg = (addr - 0x2000) & 7;
        value &= 0xFF;
        if (reg == 0) {  // PPUCTRL
            const bool rising = (value & 0x80) && !(ctrl & 0x80);
            ctrl = value;
            t_addr = (t_addr & ~0x0C00) | ((value & 3) << 10);
            if (rising && vblank) nmi_edge = true;
        } else if (reg == 1) {  // PPUMASK
            mask = value;
        } else if (reg == 3) {  // OAMADDR
            oam_addr = value;
        } else if (reg == 4) {  // OAMDATA
            oam[oam_addr] = (uint8_t)value;
            oam_addr = (oam_addr + 1) & 0xFF;
        } else if (reg == 5) {  // PPUSCROLL (x then y)
            if (write_latch) {
                t_addr = (t_addr & ~0x73E0) | ((value & 7) << 12) | ((value >> 3) << 5);
            } else {
                fine_x = value & 7;
                t_addr = (t_addr & ~0x001F) | (value >> 3);
            }
            write_latch = !write_latch;
        } else if (reg == 6) {  // PPUADDR (high then low, shared latch + t)
            if (write_latch) {
                t_addr = (t_addr & 0x7F00) | value;
                vram_addr = t_addr & 0x3FFF;
            } else {
                t_addr = (t_addr & 0x00FF) | ((value & 0x3F) << 8);
            }
            write_latch = !write_latch;
        } else if (reg == 7) {  // PPUDATA
            vram_write(vram_addr & 0x3FFF, value);
            vram_addr = (vram_addr + increment()) & 0x3FFF;
        }
    }

    // ── Console-side control ────────────────────────────────────────

    bool nmi_enabled() const { return (ctrl & 0x80) != 0; }
    bool rendering_enabled() const { return (mask & 0x18) != 0; }
    void begin_vblank() { vblank = true; }
    void end_vblank() {
        vblank = false;
        sprite0_hit = false;
        frame_ctrl = ctrl;
        frame_mask = mask;
        frame_mirroring = mirroring;
        frame_t_addr = t_addr;
        frame_fine_x = fine_x;
        std::memcpy(frame_chr_bank, chr_bank, sizeof(chr_bank));
        frame_latch_valid = true;
    }

    bool take_nmi_edge() {
        const bool edge = nmi_edge;
        nmi_edge = false;
        return edge;
    }

    void oam_dma(py::bytes page) {
        std::string p = page;
        for (size_t i = 0; i < p.size(); ++i)
            oam[(oam_addr + i) & 0xFF] = (uint8_t)p[i];
    }

    // ── Rendering ───────────────────────────────────────────────────

    int tile_pixel(const int* banks, int tile, int row, int col) const {
        const int base = tile * 16 + row;
        const int b0 = (chr[banks[(base >> 10) & 7] + (base & 0x3FF)] >> (7 - col)) & 1;
        const int b1 = (chr[banks[((base + 8) >> 10) & 7] + ((base + 8) & 0x3FF)] >> (7 - col)) & 1;
        return b0 | (b1 << 1);
    }

    void render_bg_rows(uint8_t* frame, const uint8_t* pal, uint8_t backdrop,
                        const BandState& band, int y0, int y1) {
        if (!(band.mask & 0x08)) {  // background disabled in this band
            for (int y = y0; y < y1; ++y) {
                std::memset(frame + y * 256, backdrop, 256);
                std::memset(bg_opaque[y], 0, 256);
            }
            return;
        }
        const int bctrl = band.ctrl, bt_addr = band.t_addr;
        const int bg_table = (bctrl & 0x10) ? 256 : 0;
        const int px_x = ((bt_addr & 0x1F) << 3) | band.fine_x;
        const int sx = (px_x + ((bt_addr >> 10) & 1) * 256) % 512;
        // y: relative to the screen from t, or pinned absolute at the
        // band's first row (a mid-frame $2006 v load; y self-increments
        // from there on hardware, so the band scans forward from it).
        const int sy = band.y_base >= 0
            ? ((band.y_base - y0) % 480 + 480) % 480
            : scroll_row_480(bt_addr) % 480;
        // Left-edge masking: the leftmost 8 pixels show the backdrop (and
        // count as transparent for sprite priority) with the bit clear.
        const int x0 = (band.mask & 0x02) ? 0 : 8;
        for (int y = y0; y < y1; ++y) {
            const int cy = (y + sy) % 480;
            const int ly = cy % 240, quad_y = cy / 240;
            for (int x = 0; x < x0; ++x) {
                frame[y * 256 + x] = backdrop;
                bg_opaque[y][x] = false;
            }
            for (int x = x0; x < 256; ++x) {
                const int cx = (x + sx) % 512;
                const int lx = cx % 256, quad_x = cx / 256;
                const int base = nt_offset_for(
                    0x2000 + (quad_y * 2 + quad_x) * 0x400, band.mirroring);
                const int trow = ly >> 3, tcol = lx >> 3;
                const int tile = nametables[base + trow * 32 + tcol];
                const int pix = tile_pixel(band.banks, bg_table + tile, ly & 7, lx & 7);
                const uint8_t attr =
                    nametables[base + 0x3C0 + (trow >> 2) * 8 + (tcol >> 2)];
                const int shift = ((trow % 4) / 2) * 4 + ((tcol % 4) / 2) * 2;
                const int palsel = (attr >> shift) & 3;
                frame[y * 256 + x] = pix ? pal[palsel * 4 + pix] : backdrop;
                bg_opaque[y][x] = pix != 0;
            }
        }
    }

    // First scanline where an opaque sprite-0 pixel overlaps opaque bg.
    int detect_sprite0(const int* banks) const {
        const bool tall = (ctrl & 0x20) != 0;
        const int y = oam[0], tile = oam[1], attr = oam[2], x = oam[3];
        const int top = y + 1;
        if (top >= 240) return -1;
        const int h = tall ? 16 : 8;
        const int rows = (240 - top) < h ? (240 - top) : h;
        const int cols = (256 - x) < 8 ? (256 - x) : 8;
        for (int r = 0; r < rows; ++r) {
            const int src_r = (attr & 0x80) ? (h - 1 - r) : r;
            int t, prow;
            if (tall) {
                t = (tile & 1) * 256 + (tile & 0xFE) + (src_r >> 3);
                prow = src_r & 7;
            } else {
                t = ((ctrl & 0x08) ? 256 : 0) + tile;
                prow = src_r;
            }
            for (int c = 0; c < cols; ++c) {
                // No hit in the left 8 columns when either side is masked.
                if (x + c < 8 && (mask & 0x06) != 0x06) continue;
                const int src_c = (attr & 0x40) ? (7 - c) : c;
                if (tile_pixel(banks, t, prow, src_c) && bg_opaque[top + r][x + c])
                    return top + r;
            }
        }
        return -1;
    }

    py::array_t<uint8_t> render_frame() {
        auto out = py::array_t<uint8_t>({240, 256});
        uint8_t* frame = out.mutable_data();

        uint8_t pal[0x20];
        for (int i = 0; i < 0x20; ++i) pal[i] = palette[i] & 0x3F;
        const uint8_t backdrop = pal[0];

        // Piecewise state: rows through the sprite-0 hit render from the
        // visible-span-start state, rows through the IRQ split (when the
        // console marked one) from the state latched at delivery, rows
        // below from the live registers.  Each band carries its own
        // enable/edge bits, mirroring, scroll, and CHR banks.
        // A live v differing from the one latched at the split means the
        // IRQ handler loaded v via a $2006 pair -- the hardware mid-frame
        // y-jump; the post-split band pins its y to it.
        int live_y_base = -1;
        if (mid_scanline >= 0 && (vram_addr & 0x7FFF) != (mid_v_addr & 0x7FFF)) {
            live_y_base = scroll_row_480(vram_addr);
        }
        const BandState live{ctrl, mask, mirroring, t_addr, fine_x,
                             live_y_base, chr_bank};
        const BandState span = frame_latch_valid
            ? BandState{frame_ctrl, frame_mask, frame_mirroring,
                        frame_t_addr, frame_fine_x, -1, frame_chr_bank}
            : BandState{ctrl, mask, mirroring, t_addr, fine_x, -1, chr_bank};

        render_bg_rows(frame, pal, backdrop, span, 0, 240);

        sprite0_scanline = -1;
        if (mask & 0x10) sprite0_scanline = detect_sprite0(span.banks);

        const int s0 = sprite0_scanline;
        const bool have_mid = mid_scanline >= 0 && mid_scanline + 1 < 240
            && (s0 < 0 || mid_scanline > s0);
        const BandState mid_band = have_mid
            ? BandState{mid_ctrl, mid_mask, mid_mirroring,
                        mid_t_addr, mid_fine_x, -1, mid_chr_bank}
            : live;

        // The frame's ordered background bands, each starting at its first
        // row: span from the top, then the post-sprite-0 and post-IRQ
        // splits (mid_band collapses to live when there is no IRQ split).
        struct Segment { int y0; const BandState* band; };
        Segment segs[3];
        int nseg = 0;
        segs[nseg++] = {0, &span};
        if (s0 >= 0 && s0 + 1 < 240) segs[nseg++] = {s0 + 1, &mid_band};
        if (have_mid) segs[nseg++] = {mid_scanline + 1, &live};
        mid_scanline = -1;

        // Repaint each lower band over the full-frame span pass, skipping
        // any whose state matches the band above it.
        for (int i = 1; i < nseg; ++i) {
            const int end = (i + 1 < nseg) ? segs[i + 1].y0 : 240;
            if (!same_band(*segs[i].band, *segs[i - 1].band))
                render_bg_rows(frame, pal, backdrop, *segs[i].band, segs[i].y0, end);
        }
        if (!(mask & 0x10)) return out;

        // Sprite pixels fetch through the banks of the band their row
        // falls in (per-scanline fetches on hardware -- how games blank
        // sprites below the split by switching to an empty CHR page).
        auto banks_for_row = [&](int row) -> const int* {
            const BandState* b = segs[0].band;
            for (int i = 1; i < nseg; ++i)
                if (row >= segs[i].y0) b = segs[i].band;
            return b->banks;
        };

        const bool tall = (ctrl & 0x20) != 0;
        const int sprite_x_min = (mask & 0x04) ? 0 : 8;
        std::memset(sprite_claimed, 0, sizeof(sprite_claimed));
        // Low index first: the first opaque sprite claims the pixel, and
        // only its priority bit decides front-vs-behind (an obscured
        // behind-background sprite hides higher-index sprites too).
        for (int i = 0; i < 64; ++i) {
            const int y = oam[i * 4], tile = oam[i * 4 + 1];
            const int attr = oam[i * 4 + 2], x = oam[i * 4 + 3];
            const int top = y + 1;
            if (top >= 240) continue;
            const int h = tall ? 16 : 8;
            const int rows = (240 - top) < h ? (240 - top) : h;
            const int cols = (256 - x) < 8 ? (256 - x) : 8;
            if (rows <= 0 || cols <= 0) continue;
            for (int r = 0; r < rows; ++r) {
                const int src_r = (attr & 0x80) ? (h - 1 - r) : r;
                int t, prow;
                if (tall) {
                    t = (tile & 1) * 256 + (tile & 0xFE) + (src_r >> 3);
                    prow = src_r & 7;
                } else {
                    t = ((ctrl & 0x08) ? 256 : 0) + tile;
                    prow = src_r;
                }
                const int* sbanks = banks_for_row(top + r);
                for (int c = 0; c < cols; ++c) {
                    if (x + c < sprite_x_min) continue;
                    const int src_c = (attr & 0x40) ? (7 - c) : c;
                    const int pix = tile_pixel(sbanks, t, prow, src_c);
                    if (!pix) continue;
                    if (sprite_claimed[top + r][x + c]) continue;
                    sprite_claimed[top + r][x + c] = true;
                    if ((attr & 0x20) && bg_opaque[top + r][x + c])
                        continue;  // behind background
                    frame[(top + r) * 256 + (x + c)] = pal[0x10 + (attr & 3) * 4 + pix];
                }
            }
        }
        return out;
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<PPU>(m, "PPU", py::module_local())
        .def(py::init<py::bytes, bool, bool>())
        .def("read", &PPU::read)
        .def("write", &PPU::write)
        .def("read_lease", &PPU::read_lease)
        .def("vram_read", &PPU::vram_read)
        .def("vram_write", &PPU::vram_write)
        .def("set_chr_banks", &PPU::set_chr_banks)
        .def_static("palette_index", &PPU::palette_index)
        .def("begin_vblank", &PPU::begin_vblank)
        .def("end_vblank", &PPU::end_vblank)
        .def("mark_split", &PPU::mark_split)
        .def("take_nmi_edge", &PPU::take_nmi_edge)
        .def("oam_dma", &PPU::oam_dma)
        .def("render_frame", &PPU::render_frame)
        .def_property_readonly("nmi_enabled", &PPU::nmi_enabled)
        .def_property_readonly("rendering_enabled", &PPU::rendering_enabled)
        .def_readwrite("ctrl", &PPU::ctrl)
        .def_readwrite("mirroring", &PPU::mirroring)
        .def_readwrite("mask", &PPU::mask)
        .def_readwrite("oam_addr", &PPU::oam_addr)
        .def_property_readonly("scroll_x", &PPU::scroll_px_x)
        .def_property_readonly("scroll_y", &PPU::scroll_px_y)
        .def_readwrite("t_addr", &PPU::t_addr)
        .def_readwrite("fine_x", &PPU::fine_x)
        .def_readwrite("vram_addr", &PPU::vram_addr)
        .def_readwrite("write_latch", &PPU::write_latch)
        .def_readwrite("read_buffer", &PPU::read_buffer)
        .def_readwrite("vblank", &PPU::vblank)
        .def_readwrite("sprite0_hit", &PPU::sprite0_hit)
        .def_readwrite("frame_ctrl", &PPU::frame_ctrl)
        .def_readwrite("frame_mask", &PPU::frame_mask)
        .def_readwrite("frame_mirroring", &PPU::frame_mirroring)
        .def_readwrite("frame_t_addr", &PPU::frame_t_addr)
        .def_readwrite("frame_fine_x", &PPU::frame_fine_x)
        .def_readwrite("frame_latch_valid", &PPU::frame_latch_valid)
        .def_readwrite("mid_ctrl", &PPU::mid_ctrl)
        .def_readwrite("mid_mask", &PPU::mid_mask)
        .def_readwrite("mid_mirroring", &PPU::mid_mirroring)
        .def_readwrite("mid_t_addr", &PPU::mid_t_addr)
        .def_readwrite("mid_fine_x", &PPU::mid_fine_x)
        .def_property_readonly("chr_banks", [](PPU& p) {
            return std::vector<int>(p.chr_bank, p.chr_bank + 8);
        })
        .def_property_readonly("sprite0_scanline", [](PPU& p) -> std::optional<int> {
            if (p.sprite0_scanline < 0) return std::nullopt;
            return p.sprite0_scanline;
        })
        .def_property_readonly("nametables", [](PPU& p) {
            return py::memoryview::from_memory(p.nametables, sizeof(p.nametables));
        })
        .def_property_readonly("palette", [](PPU& p) {
            return py::memoryview::from_memory(p.palette, sizeof(p.palette));
        })
        .def_property_readonly("oam", [](PPU& p) {
            return py::memoryview::from_memory(p.oam, sizeof(p.oam));
        });
}
