#pragma once

// CHR bank types and the PPU-side halves of the banking mappers: the
// FD/FE latch unit shared by MMC2/MMC4 and MMC5's substitution state.
// CPU-side register decoding stays under nes/mappers/.

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace transformer_nes::nes {

using std::array;
using std::size_t;
using std::uint8_t;

using ChrBanks = array<size_t, 8>;
using ChrLatchBanks = array<size_t, 4>;

// The two post-fetch FD/FE latches and their 4 KiB bank slots.  MMC2
// triggers the low pattern table on the exact $0FD8/$0FE8 fetches only;
// MMC4 (wide_low) responds to all eight rows on both tables.
struct ChrLatch {
  ChrLatchBanks banks{}; // byte offsets: lower FD/FE, upper FD/FE
  bool latch_fe[2] = {true, true};
  bool wide_low = false;

  void configure(const ChrLatchBanks& banks_4k, size_t chr_size) {
    const size_t count = chr_size >> 12;
    if (count == 0)
      throw std::runtime_error("CHR-latch mapper requires CHR ROM");
    for (size_t i = 0; i < banks_4k.size(); ++i)
      banks[i] = (banks_4k[i] % count) * 0x1000;
  }

  size_t map(size_t address) const {
    const size_t half = (address >> 12) & 1;
    return banks[half * 2 + (latch_fe[half] ? 1 : 0)] + (address & 0xFFF);
  }

  void after_fetch(size_t address) {
    address &= 0x1FFF;
    if (address >= 0x1FD8 && address <= 0x1FDF)
      latch_fe[1] = false;
    else if (address >= 0x1FE8 && address <= 0x1FEF)
      latch_fe[1] = true;
    else if (wide_low ? address >= 0x0FD8 && address <= 0x0FDF : address == 0x0FD8)
      latch_fe[0] = false;
    else if (wide_low ? address >= 0x0FE8 && address <= 0x0FEF : address == 0x0FE8)
      latch_fe[0] = true;
  }
};

struct Mmc5BankSnapshot {
  ChrBanks sprite{};
  ChrBanks background{};
};

// Native PPU-side MMC5 state. Mmc5 owns CPU register decoding and PRG
// banking; this class owns data substituted onto the PPU bus.
class Mmc5Ppu {
public:
  void configure(int exram_mode, int nametable_map, int fill_tile, int fill_color, int chr_upper) {
    active_ = true;
    exram_mode_ = exram_mode & 3;
    nametable_map_ = nametable_map & 0xFF;
    fill_tile_ = fill_tile & 0xFF;
    fill_color_ = fill_color & 3;
    chr_upper_ = chr_upper & 3;
  }

  bool active() const { return active_; }
  int exram_mode() const { return exram_mode_; }
  bool extended_attributes() const { return exram_mode_ == 1; }
  int extended_value(int tile_offset) const { return exram_[tile_offset & 0x3FF]; }
  size_t extended_chr_base(int tile_offset) const {
    return ((chr_upper_ << 6) | (extended_value(tile_offset) & 0x3F)) * 0x1000;
  }

  void write_exram(int offset, int value) { exram_[offset & 0x3FF] = static_cast<uint8_t>(value); }

  // The map is a caller-supplied $5105 value because render bands latch
  // it independently of the live register. Route the complete read here
  // so generic PPU code does not contain MMC5 source decoding.
  int read_nametable(int map, int table, int offset, const uint8_t* ciram) const {
    const int source = (map >> (table * 2)) & 3;
    if (source == 2) {
      return exram_mode_ <= 1 ? exram_[offset & 0x3FF] : 0;
    }
    if (source == 3) {
      if (offset < 0x3C0)
        return fill_tile_;
      return fill_color_ * 0x55; // replicate AA into all quadrants
    }
    return ciram[source * 0x400 + offset];
  }

  void write_nametable(int table, int offset, int value, uint8_t* ciram) {
    const int source = (nametable_map_ >> (table * 2)) & 3;
    if (source == 2) {
      write_exram(offset, value);
    } else if (source != 3) { // fill mode has no memory behind it
      ciram[source * 0x400 + offset] = static_cast<uint8_t>(value);
    }
  }

  void configure_chr(const ChrBanks& sprite, const ChrBanks& background) {
    for (size_t i = 0; i < sprite.size(); ++i) {
      sprite_banks_[i] = sprite[i] * 0x400;
      background_banks_[i] = background[i] * 0x400;
    }
  }

  const size_t* sprite_banks() const { return sprite_banks_.data(); }
  const size_t* background_banks() const { return background_banks_.data(); }

  Mmc5BankSnapshot bank_snapshot() const { return {sprite_banks_, background_banks_}; }

  const size_t* live_background_banks(int ctrl) const {
    return (ctrl & 0x20) ? background_banks_.data() : sprite_banks_.data();
  }

  // State after vblank programming, used from scanline 0 until the first
  // visible IRQ. This is distinct from an IRQ snapshot: chained handlers
  // may temporarily select blank banks between raster events.
  void snapshot_frame() {
    frame_sprite_banks_ = sprite_banks_;
    frame_background_banks_ = background_banks_;
  }
  const size_t* frame_sprite_banks() const { return frame_sprite_banks_.data(); }
  const size_t* frame_background_banks() const { return frame_background_banks_.data(); }
  const size_t* frame_background_banks(int ctrl) const {
    return (ctrl & 0x20) ? frame_background_banks_.data() : frame_sprite_banks_.data();
  }

  // Freeze the CHR windows in effect at a mid-frame IRQ split, so rows
  // above it render from these while later writes only affect below.
  void snapshot() {
    split_sprite_banks_ = sprite_banks_;
    split_background_banks_ = background_banks_;
  }
  const size_t* split_background_banks(int ctrl) const {
    return (ctrl & 0x20) ? split_background_banks_.data() : split_sprite_banks_.data();
  }

private:
  bool active_ = false;
  int exram_mode_ = 0;
  int nametable_map_ = 0;
  int fill_tile_ = 0;
  int fill_color_ = 0;
  int chr_upper_ = 0;
  array<uint8_t, 0x400> exram_{};
  ChrBanks sprite_banks_{};
  ChrBanks background_banks_{};
  ChrBanks split_sprite_banks_{};
  ChrBanks split_background_banks_{};
  ChrBanks frame_sprite_banks_{};
  ChrBanks frame_background_banks_{};
};

} // namespace transformer_nes::nes
