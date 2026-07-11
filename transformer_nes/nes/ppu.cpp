// Native frame-level NES PPU (2C02).  Registers $2000-$2007 behind the
// MMIO bus, VRAM/OAM, and a frame-at-once renderer; ported from a Python
// reference and proven pixel-identical before that reference was deleted
// (receipts in ROADMAP.md, 2026-07-11).  Plain fixed silicon -- nothing
// here is model-generated -- compiled once and cached by content digest.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>

namespace py = pybind11;

struct PPU {
    uint8_t chr[0x2000];
    bool chr_writable;
    bool vertical_mirroring;
    uint8_t nametables[0x800];
    uint8_t palette[0x20];
    uint8_t oam[0x100];

    int ctrl = 0, mask = 0, oam_addr = 0;
    int scroll_x = 0, scroll_y = 0, vram_addr = 0;
    bool write_latch = false;
    int read_buffer = 0;

    bool vblank = false, sprite0_hit = false, nmi_edge = false;
    int sprite0_scanline = -1;  // -1: none
    bool bg_opaque[240][256];

    PPU(py::bytes chr_data, bool writable, bool vertical) {
        std::string c = chr_data;
        std::memset(chr, 0, sizeof(chr));
        std::memcpy(chr, c.data(), c.size() < sizeof(chr) ? c.size() : sizeof(chr));
        chr_writable = writable;
        vertical_mirroring = vertical;
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

    int nt_offset(int addr) const {
        const int a = (addr - 0x2000) & 0xFFF;
        const int table = a >> 10, offset = a & 0x3FF;
        const int phys = vertical_mirroring ? (table & 1) : (table >> 1);
        return phys * 0x400 + offset;
    }

    int vram_read(int addr) const {
        addr &= 0x3FFF;
        if (addr < 0x2000) return chr[addr];
        if (addr < 0x3F00) return nametables[nt_offset(addr)];
        return palette[palette_index(addr)];
    }

    void vram_write(int addr, int value) {
        addr &= 0x3FFF;
        if (addr < 0x2000) {
            if (chr_writable) chr[addr] = (uint8_t)value;
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
            if (rising && vblank) nmi_edge = true;
        } else if (reg == 1) {  // PPUMASK
            mask = value;
        } else if (reg == 3) {  // OAMADDR
            oam_addr = value;
        } else if (reg == 4) {  // OAMDATA
            oam[oam_addr] = (uint8_t)value;
            oam_addr = (oam_addr + 1) & 0xFF;
        } else if (reg == 5) {  // PPUSCROLL (x then y)
            if (write_latch) scroll_y = value;
            else scroll_x = value;
            write_latch = !write_latch;
        } else if (reg == 6) {  // PPUADDR (high then low, shared latch)
            if (write_latch) vram_addr = ((vram_addr & 0x3F00) | value) & 0x3FFF;
            else vram_addr = ((value & 0x3F) << 8) | (vram_addr & 0x00FF);
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
    void end_vblank() { vblank = false; sprite0_hit = false; }

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

    int tile_pixel(int tile, int row, int col) const {
        const int base = tile * 16 + row;
        const int b0 = (chr[base] >> (7 - col)) & 1;
        const int b1 = (chr[base + 8] >> (7 - col)) & 1;
        return b0 | (b1 << 1);
    }

    py::array_t<uint8_t> render_frame() {
        auto out = py::array_t<uint8_t>({240, 256});
        uint8_t* frame = out.mutable_data();

        uint8_t pal[0x20];
        for (int i = 0; i < 0x20; ++i) pal[i] = palette[i] & 0x3F;
        const uint8_t backdrop = pal[0];

        if (!(mask & 0x08)) {
            std::memset(frame, backdrop, 240 * 256);
            std::memset(bg_opaque, 0, sizeof(bg_opaque));
        } else {
            const int bg_table = (ctrl & 0x10) ? 256 : 0;
            const int sx = (scroll_x + (ctrl & 1) * 256) % 512;
            const int sy = (scroll_y + ((ctrl >> 1) & 1) * 240) % 480;
            for (int y = 0; y < 240; ++y) {
                const int cy = (y + sy) % 480;
                const int ly = cy % 240, quad_y = cy / 240;
                for (int x = 0; x < 256; ++x) {
                    const int cx = (x + sx) % 512;
                    const int lx = cx % 256, quad_x = cx / 256;
                    const int base = nt_offset(0x2000 + (quad_y * 2 + quad_x) * 0x400);
                    const int trow = ly >> 3, tcol = lx >> 3;
                    const int tile = nametables[base + trow * 32 + tcol];
                    const int pix = tile_pixel(bg_table + tile, ly & 7, lx & 7);
                    const uint8_t attr =
                        nametables[base + 0x3C0 + (trow >> 2) * 8 + (tcol >> 2)];
                    const int shift = ((trow % 4) / 2) * 4 + ((tcol % 4) / 2) * 2;
                    const int palsel = (attr >> shift) & 3;
                    frame[y * 256 + x] = pix ? pal[palsel * 4 + pix] : backdrop;
                    bg_opaque[y][x] = pix != 0;
                }
            }
        }

        sprite0_scanline = -1;
        if (!(mask & 0x10)) return out;

        const bool tall = (ctrl & 0x20) != 0;
        // Draw from sprite 63 down so lower-index sprites win overlaps.
        for (int i = 63; i >= 0; --i) {
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
                for (int c = 0; c < cols; ++c) {
                    const int src_c = (attr & 0x40) ? (7 - c) : c;
                    const int pix = tile_pixel(t, prow, src_c);
                    if (i == 0 && pix && bg_opaque[top + r][x + c]
                        && sprite0_scanline < 0)
                        sprite0_scanline = top + r;
                    if (!pix) continue;
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
        .def_static("palette_index", &PPU::palette_index)
        .def("begin_vblank", &PPU::begin_vblank)
        .def("end_vblank", &PPU::end_vblank)
        .def("take_nmi_edge", &PPU::take_nmi_edge)
        .def("oam_dma", &PPU::oam_dma)
        .def("render_frame", &PPU::render_frame)
        .def_property_readonly("nmi_enabled", &PPU::nmi_enabled)
        .def_property_readonly("rendering_enabled", &PPU::rendering_enabled)
        .def_readwrite("ctrl", &PPU::ctrl)
        .def_readwrite("mask", &PPU::mask)
        .def_readwrite("oam_addr", &PPU::oam_addr)
        .def_readwrite("scroll_x", &PPU::scroll_x)
        .def_readwrite("scroll_y", &PPU::scroll_y)
        .def_readwrite("vram_addr", &PPU::vram_addr)
        .def_readwrite("write_latch", &PPU::write_latch)
        .def_readwrite("read_buffer", &PPU::read_buffer)
        .def_readwrite("vblank", &PPU::vblank)
        .def_readwrite("sprite0_hit", &PPU::sprite0_hit)
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
