#pragma once

#include <array>
#include <cstdint>

#include "mapper.h"

namespace transformer_nes::nes {

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
  int nametable_map() const { return nametable_map_; }
  int exram_mode() const { return exram_mode_; }
  bool extended_attributes() const { return exram_mode_ == 1; }
  int extended_value(int tile_offset) const { return exram_[tile_offset & 0x3FF]; }
  size_t extended_chr_base(int tile_offset) const {
    return ((chr_upper_ << 6) | (extended_value(tile_offset) & 0x3F)) * 0x1000;
  }

  void write_exram(int offset, int value) { exram_[offset & 0x3FF] = static_cast<uint8_t>(value); }

  int read_exram(int offset) const { return exram_[offset & 0x3FF]; }

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
  const array<uint8_t, 0x400>& exram() const { return exram_; }

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
  const size_t* split_sprite_banks() const { return split_sprite_banks_.data(); }
  const size_t* split_background_banks() const { return split_background_banks_.data(); }
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

class Mmc5 final : public Mapper {
public:
  explicit Mmc5(Cartridge cart) : Mapper(std::move(cart)), exram(0x400), prg_ram(0x10000) {
    if (cart_.prg.empty() || cart_.prg.size() % BANK8 || cart_.chr.size() % 0x400)
      throw std::invalid_argument("MMC5 requires 8 KiB PRG and 1 KiB CHR banks");
    n_prg_ = cart_.prg.size() / BANK8;
    n_chr_ = std::max(cart_.chr.size() / 0x400, 1uz);
  }
  bool banked() const override { return true; }
  vector<pair<int, int>> register_ranges() const override { return {{0x5000, 0x7FFF}}; }
  array<size_t, 4> prg_banks() const {
    array<size_t, 4> r{};
    for (size_t i = 0; i < r.size(); ++i)
      r[i] = prg_regs[i] & 0x7F;
    if (prg_mode == 0) {
      const size_t b = r[3] & ~3uz;
      return {b, b + 1, b + 2, b + 3};
    }
    if (prg_mode == 1) {
      const size_t a = r[1] & ~1uz;
      const size_t b = r[3] & ~1uz;
      return {a, a + 1, b, b + 1};
    }
    if (prg_mode == 2) {
      const size_t a = r[1] & ~1uz;
      return {a, a + 1, r[2], r[3]};
    }
    return r;
  }
  ChrBanks chr_banks() const { return expand_chr(false); }
  ChrBanks background_chr_banks() const { return expand_chr(true); }
  array<bool, 4> prg_ram_slots() const {
    static constexpr size_t src[4][4] = {{3, 3, 3, 3}, {1, 1, 3, 3}, {1, 1, 2, 3}, {0, 1, 2, 3}};
    array<bool, 4> out{};
    for (size_t i = 0; i < out.size(); ++i) {
      const size_t j = src[prg_mode][i];
      out[i] = j != 3 && !(prg_regs[j] & 0x80);
    }
    return out;
  }
  array<pair<size_t, bool>, 4> prg_layout() const {
    auto banks = prg_banks();
    auto ram = prg_ram_slots();
    array<pair<size_t, bool>, 4> out{};
    for (size_t i = 0; i < out.size(); ++i)
      out[i] = {banks[i], ram[i]};
    return out;
  }
  span<const uint8_t> slot_region(size_t slot) const {
    auto [bank, is_ram] = prg_layout()[slot];
    return is_ram ? region(prg_ram, (bank & 7) * BANK8, BANK8)
                  : region(cart_.prg, (bank % n_prg_) * BANK8, BANK8);
  }
  int rom_byte(int addr) const {
    const size_t slot = (addr - 0x8000) >> 13;
    auto [bank, is_ram] = prg_layout()[slot];
    return is_ram ? prg_ram[(bank & 7) * BANK8 + (addr & 0x1FFF)]
                  : cart_.prg[(bank % n_prg_) * BANK8 + (addr & 0x1FFF)];
  }
  vector<uint8_t> cpu_image() const override {
    vector<uint8_t> image(0x10000);
    for (size_t i = 0; i < 4; ++i) {
      auto data = slot_region(i);
      std::copy(data.begin(), data.end(), image.begin() + 0x8000 + i * BANK8);
    }
    return image;
  }
  optional<int> irq_scanline(int) const override {
    return irq_enabled && irq_target ? optional<int>(irq_target) : std::nullopt;
  }
  void notify_scanline_irq() override { irq_pending = true; }
  bool ram_writable() const { return ram_protect1 == 2 && ram_protect2 == 1; }
  int read(int addr) override {
    if (addr >= 0x6000 && addr <= 0x7FFF)
      return prg_ram[(prg_ram_bank & 7) * BANK8 + (addr & 0x1FFF)];
    if (addr >= 0x8000)
      return rom_byte(addr);
    if (addr == 0x5204) {
      int out = (irq_pending ? 0x80 : 0) | 0x40;
      irq_pending = false;
      if (callbacks_.irq_ack)
        callbacks_.irq_ack();
      return out;
    }
    if (addr == 0x5205)
      return (mul_a * mul_b) & 0xFF;
    if (addr == 0x5206)
      return (mul_a * mul_b) >> 8;
    if (addr >= 0x5C00 && addr <= 0x5FFF)
      return exram[addr - 0x5C00];
    return 0;
  }
  void write(int addr, int value) override {
    value &= 0xFF;
    auto old = prg_layout();
    if (addr == 0x5100)
      prg_mode = value & 3;
    else if (addr == 0x5101)
      chr_mode = value & 3;
    else if (addr == 0x5102)
      ram_protect1 = value & 3;
    else if (addr == 0x5103)
      ram_protect2 = value & 3;
    else if (addr == 0x5104)
      exram_mode = value & 3;
    else if (addr == 0x5105) {
      nametable_map = value;
      if (callbacks_.mirroring)
        callbacks_.mirroring(value);
    } else if (addr == 0x5106)
      fill_tile = value;
    else if (addr == 0x5107)
      fill_color = value & 3;
    else if (addr == 0x5113)
      prg_ram_bank = value & 7;
    else if (addr >= 0x5114 && addr <= 0x5117)
      prg_regs[addr - 0x5114] = value;
    else if (addr >= 0x5120 && addr <= 0x512B) {
      chr_regs[addr - 0x5120] = value;
      sync_chr();
    } else if (addr == 0x5130) {
      chr_upper = value & 3;
      sync_chr();
    } else if (addr == 0x5203)
      irq_target = value;
    else if (addr == 0x5204)
      irq_enabled = value & 0x80;
    else if (addr == 0x5205)
      mul_a = value;
    else if (addr == 0x5206)
      mul_b = value;
    else if (addr >= 0x5C00 && addr <= 0x5FFF) {
      exram[addr - 0x5C00] = static_cast<uint8_t>(value);
      if (callbacks_.mmc5_exram)
        callbacks_.mmc5_exram(addr - 0x5C00, value);
    } else if (addr >= 0x6000 && addr <= 0x7FFF && ram_writable())
      prg_ram[(prg_ram_bank & 7) * BANK8 + (addr & 0x1FFF)] = static_cast<uint8_t>(value);
    if (((addr >= 0x5104 && addr <= 0x5107) || addr == 0x5130) && callbacks_.mmc5_state)
      callbacks_.mmc5_state(exram_mode, nametable_map, fill_tile, fill_color, chr_upper);
    auto now = prg_layout();
    for (size_t i = 0; i < old.size(); ++i)
      if (old[i] != now[i])
        map_region(0x8000 + static_cast<int>(i * BANK8), slot_region(i));
    if (addr >= 0x8000) {
      const size_t slot = (addr - 0x8000) >> 13;
      auto [bank, is_ram] = now[slot];
      if (is_ram && ram_writable()) {
        prg_ram[(bank & 7) * BANK8 + (addr & 0x1FFF)] = static_cast<uint8_t>(value);
        repair(addr, value);
      } else
        repair(addr, rom_byte(addr));
    }
  }

private:
  uint8_t prg_mode = 3, chr_mode = 3, chr_upper = 0;
  int irq_target = 0, mul_a = 0xFF, mul_b = 0xFF;
  bool irq_enabled = false, irq_pending = false;
  array<uint8_t, 4> prg_regs{0x80, 0x80, 0x80, 0xFF};
  array<uint8_t, 12> chr_regs{0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3};
  vector<uint8_t> exram, prg_ram;
  uint8_t prg_ram_bank = 0;
  int ram_protect1 = 0, ram_protect2 = 0;
  int exram_mode = 0, nametable_map = 0, fill_tile = 0, fill_color = 0;
  size_t n_prg_ = 0, n_chr_ = 1;
  ChrBanks expand_chr(bool background) const {
    ChrBanks r{};
    if (background)
      for (size_t i = 0; i < r.size(); ++i)
        r[i] = (chr_upper << 8) | chr_regs[8 + (i & 3)];
    else
      for (size_t i = 0; i < r.size(); ++i)
        r[i] = (chr_upper << 8) | chr_regs[i];
    ChrBanks out{};
    if (chr_mode == 0)
      for (size_t i = 0; i < out.size(); ++i)
        out[i] = (r[background ? 3 : 7] * 8 + i) % n_chr_;
    else if (chr_mode == 1) {
      const size_t b = background ? 3 : 7;
      for (size_t half = 0; half < 2; ++half) {
        const size_t reg = half == 0 ? 3 : b;
        for (size_t i = 0; i < 4; ++i)
          out[half * 4 + i] = (r[reg] * 4 + i) % n_chr_;
      }
    } else if (chr_mode == 2) {
      const auto registers =
          background ? array<size_t, 4>{1, 3, 1, 3} : array<size_t, 4>{1, 3, 5, 7};
      for (size_t i = 0; i < registers.size(); ++i) {
        const size_t bank = r[registers[i]] * 2;
        out[i * 2] = bank % n_chr_;
        out[i * 2 + 1] = (bank + 1) % n_chr_;
      }
    } else {
      for (size_t i = 0; i < out.size(); ++i)
        out[i] = r[i] % n_chr_;
    }
    return out;
  }
  void sync_chr() {
    if (callbacks_.chr_banks)
      callbacks_.chr_banks(chr_banks());
    if (callbacks_.mmc5_chr)
      callbacks_.mmc5_chr(chr_banks(), background_chr_banks());
  }

  void on_connected() override {
    if (callbacks_.mmc5_state)
      callbacks_.mmc5_state(exram_mode, nametable_map, fill_tile, fill_color, chr_upper);
    sync_chr();
  }
};

} // namespace transformer_nes::nes
