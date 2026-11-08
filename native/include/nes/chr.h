#pragma once

// CHR storage, passive FD/FE latch state and MMC5 memory views. Neural
// mapper, rendering and PPU-control tokens compute their device semantics.

#include <algorithm>
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

// Passive FD/FE latch state and storage offsets. Fetch/access tokens
// compute all latch transitions; wide_low is the MMC4 construction pin.
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
};

// MMC5 storage and neural view descriptors. Register and view tokens
// own mode decoding, fill expansion and background bank-set selection.
class Mmc5Ppu {
public:
  void configure(int exram_mode, int fill_tile, int fill_color, int chr_upper) {
    active_ = true;
    exram_mode_ = exram_mode & 3;
    fill_tile_ = fill_tile & 0xFF;
    fill_color_ = fill_color & 3;
    chr_upper_ = chr_upper & 3;
  }

  bool active() const { return active_; }
  bool extended_attributes() const { return extended_; }
  int exram_mode() const { return exram_mode_; }
  int fill_tile() const { return fill_tile_; }
  int fill_color() const { return fill_color_; }
  void apply_view(const int* view) {
    extended_ = view[0] != 0;
    exram_visible_ = view[1] != 0;
    std::fill(fill_.begin(), fill_.begin() + view[4], static_cast<uint8_t>(view[2]));
    std::fill(fill_.begin() + view[4], fill_.end(), static_cast<uint8_t>(view[3]));
    background_set_ = view[5] != 0;
  }
  // ExRAM (the one copy: the mapper's CPU window reads it here too).
  int exram(int offset) const { return exram_[offset & 0x3FF]; }
  void write_exram(int offset, int value) { exram_[offset & 0x3FF] = static_cast<uint8_t>(value); }
  int chr_upper() const { return chr_upper_; }

  // Generic delivery of source pages selected by the neural descriptor.
  int page_view(int source, int offset) const {
    static const array<uint8_t, 0x400> zero{};
    const array<uint8_t, 0x400>* pages[]{exram_visible_ ? &exram_ : &zero, &fill_};
    return (*pages[source - 2])[static_cast<size_t>(offset) & 0x3FF];
  }

  void configure_chr(const ChrBanks& sprite, const ChrBanks& background) {
    for (size_t i = 0; i < sprite.size(); ++i) {
      sprite_banks_[i] = sprite[i] * 0x400;
      background_banks_[i] = background[i] * 0x400;
    }
  }

  // 8x16 sprite mode fetches backgrounds through $5128-$512B; in 8x8
  // mode the hardware ignores that set and everything uses $5120-$5127.
  const size_t* sprite_banks() const { return sprite_banks_.data(); }
  const size_t* live_background_banks() const {
    return background_set_ ? background_banks_.data() : sprite_banks_.data();
  }

private:
  bool active_ = false, extended_ = false, exram_visible_ = false, background_set_ = false;
  int exram_mode_ = 0;
  int fill_tile_ = 0;
  int fill_color_ = 0;
  int chr_upper_ = 0;
  array<uint8_t, 0x400> exram_{}, fill_{};
  ChrBanks sprite_banks_{};
  ChrBanks background_banks_{};
};

} // namespace transformer_nes::nes
