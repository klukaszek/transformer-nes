#pragma once

// Native frame-level NES PPU (2C02): MMIO registers, VRAM/OAM, and rendering.

#include "nes/chr.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace transformer_nes::nes {

using std::array;
using std::optional;
using std::size_t;
using std::span;
using std::uint16_t;
using std::uint8_t;
using std::uint64_t;
using std::vector;

class Ppu {
public:
  static constexpr size_t frame_width = 256;
  static constexpr size_t frame_height = 240;
  static constexpr size_t MAX_RASTER_SPLITS = 240;
  using Frame = array<uint8_t, frame_width * frame_height>;

  Ppu(vector<uint8_t> chr_data, bool writable, bool vertical_mirroring);

  // ── Device protocol (registers $2000-$3FFF, mirrored every 8) ──

  [[nodiscard]] int read(int address);
  // Register write stamped with the visible-span scanline it executed
  // on (-1 outside the span): mid-frame split and forced-blank rows.
  void write(int address, int value, int scanline = -1);
  [[nodiscard]] optional<int> read_lease(int address) const;
  void oam_dma(span<const uint8_t, 256> page) {
    for (size_t i = 0; i < page.size(); ++i)
      oam[(oam_addr + i) & 0xFF] = page[i];
  }

  // Console-side control

  [[nodiscard]] Frame render_frame();
  void begin_vblank() {
    vblank = true;
    in_span = false;
  }
  void end_vblank();
  void mark_split(int scanline);
  // Arm at the sprite-0 event so only post-hit scroll writes count as
  // the game's split batch.
  void begin_span_watch() {
    span_watch = true;
    span_scroll_row = -1;
    span_v_load = -1;
  }
  [[nodiscard]] bool take_nmi_edge() {
    const bool edge = nmi_edge;
    nmi_edge = false;
    return edge;
  }
  [[nodiscard]] bool nmi_enabled() const { return (ctrl & 0x80) != 0; }
  [[nodiscard]] bool rendering_enabled() const { return (mask & 0x18) != 0; }

  void set_chr_banks(const ChrBanks& banks_1k) {
    const size_t count = chr.size() >> 10;
    for (size_t i = 0; i < chr_bank.size(); ++i)
      chr_bank[i] = (banks_1k[i] % count) * 0x400;
  }
  void set_chr_latch_banks(const ChrLatchBanks& banks_4k, bool wide_low) {
    chr_latch_active = true;
    chr_latch.wide_low = wide_low;
    chr_latch.configure(banks_4k, chr.size());
  }
  void set_mmc5_state(int exram_mode, int nametable_map, int fill_tile, int fill_color,
                      int chr_upper) {
    mmc5.configure(exram_mode, nametable_map, fill_tile, fill_color, chr_upper);
    mirroring = nametable_map;
  }
  void set_mmc5_exram(int offset, int value) { mmc5.write_exram(offset, value); }
  void set_mmc5_chr(const ChrBanks& sprite, const ChrBanks& background) {
    mmc5.configure_chr(sprite, background);
  }

  // State

  // Full CHR data (ROM of any 1 KiB-multiple size, or 8 KiB CHR RAM);
  // the PPU's $0000-$1FFF window maps through eight 1 KiB bank slots.
  vector<uint8_t> chr;
  ChrBanks chr_bank{};
  ChrLatch chr_latch;
  bool chr_latch_active = false;
  Mmc5Ppu mmc5;
  bool chr_writable;
  // Nametable arrangement, runtime-switchable by mappers (MMC1
  // encoding): 0/1 single-screen low/high, 2 vertical, 3 horizontal.
  int mirroring;
  array<uint8_t, 0x800> nametables{};
  array<uint8_t, 0x20> palette{};
  array<uint8_t, 0x100> oam{};

  int ctrl = 0, mask = 0;
  uint8_t oam_addr = 0;
  uint16_t vram_addr = 0;
  uint64_t v_reload_generation = 0;
  // Shared scroll/address state (loopy t + fine x): $2000 sets its
  // nametable bits, $2005 its scroll bits, $2006 overwrites it as an
  // address -- so an address-then-scroll reset sequence really does
  // reset the nametable, which frame-start rendering depends on.
  uint16_t t_addr = 0;
  int fine_x = 0;
  bool write_latch = false;
  int read_buffer = 0;

  bool vblank = false, sprite0_hit = false, nmi_edge = false;
  bool sprite_limit = true;  // hardware 8-per-scanline cap (debug toggle)
  int sprite0_scanline = -1; // -1: none
  bool bg_opaque[240][256]{};
  // Per-pixel sprite ownership: the lowest-index opaque sprite claims
  // the pixel, and only its priority bit decides front-vs-behind.
  bool sprite_claimed[240][256]{};

  // Background state as of the visible span's start (latched at
  // end_vblank): rows above the sprite-0 hit render from these, rows
  // below from the live registers -- the sprite-0 split.
  int frame_ctrl = 0, frame_mask = 0, frame_mirroring = 3;
  int frame_t_addr = 0, frame_fine_x = 0;
  uint64_t frame_v_reload_generation = 0;
  ChrBanks frame_chr_bank{};
  bool frame_latch_valid = false;

  // Mid-visible-span scroll writes (between end_vblank and the next
  // render): a $2006 second write reloads all 15 bits of v immediately
  // on hardware, so the post-sprite-0 band must pin its y to that load
  // (Zelda scrolls the playfield under its fixed HUD this way).  Rows
  // arrive exactly via the stamped write path.
  bool in_span = false;
  bool span_watch = false;
  bool prev_full_blank = false; // last render was a whole-frame blank
  // Row of the span's last rendering on->off write (-1: none); an
  // in-span re-enable cancels it.  Zelda blanks a few rows into the
  // span at NMI end whenever the coming vblank carries VRAM updates:
  // hardware shows content above that row and backdrop below.
  int span_blank_row = -1;
  int span_scroll_row = -1;
  int span_v_load = -1;
  // Scroll state as of the split batch's last write: the post-split
  // band renders from this, not from the live registers -- a spilled
  // NMI (the next frame's setup running past vblank in our compressed
  // clock) must not clobber the split's scroll.
  int span_ctrl = 0, span_t = 0, span_fx = 0;

  // A second mid-frame split (mapper IRQ): the console latches the state
  // at delivery; rows above the split render from it, rows below from
  // the live (post-IRQ-handler) registers.  Consumed per render.
  int mid_scanline = -1;
  int mid_ctrl = 0, mid_mask = 0, mid_mirroring = 3;
  int mid_t_addr = 0, mid_fine_x = 0, mid_v_addr = 0;
  ChrBanks mid_chr_bank{};

  struct RasterSplit {
    int line, ctrl, mask, mirroring, t_addr, fine_x, v_addr;
    uint64_t v_reload_generation;
    ChrBanks chr_banks{};
    Mmc5BankSnapshot mapper_banks;
  };
  array<RasterSplit, MAX_RASTER_SPLITS> raster_splits{};
  size_t raster_split_count = 0;

  // Everything a background row depends on: pattern/enable/edge bits,
  // nametable arrangement, scroll, and CHR banks.  y_base >= 0 pins the
  // band's FIRST row to an absolute nametable y (a mid-frame $2006 v
  // load -- the hardware y-jump; y otherwise derives from t).
  struct BandState {
    int ctrl, mask, mirroring, t_addr, fine_x, y_base;
    const size_t* banks;
  };

private:
  int increment() const { return (ctrl & 0x04) ? 32 : 1; }
  static size_t palette_index(int addr);
  static size_t nt_offset_for(int addr, int mirr);
  int nt_read_for(int addr, int arrangement) const;
  size_t chr_index(const size_t* banks, size_t address) const {
    if (chr_latch_active)
      return chr_latch.map(address);
    return banks[(address >> 10) & 7] + (address & 0x3FF);
  }
  int chr_at(int addr);
  int vram_read(int addr);
  void vram_write(int addr, int value);
  static bool same_band(const BandState& a, const BandState& b);
  // Coarse + fine y as an on-screen pixel row (0-239) from a loopy addr.
  static int scroll_px_y_for(int addr) { return (((addr >> 5) & 0x1F) << 3) | ((addr >> 12) & 7); }
  // ... plus the nametable-y bit, giving the 0-479 two-screen row.
  static int scroll_row_480(int addr) { return scroll_px_y_for(addr) + ((addr >> 11) & 1) * 240; }
  std::pair<uint8_t, uint8_t> fetch_pattern_pair(int pattern_addr);
  void render_ordered_bg_row(uint8_t* frame, const uint8_t* pal, uint8_t backdrop,
                             const BandState& band, int y);
  void render_bg_rows(uint8_t* frame, const uint8_t* pal, uint8_t backdrop, const BandState& band,
                      int y0, int y1);
  int detect_sprite0(const size_t* banks);
};

} // namespace transformer_nes::nes
