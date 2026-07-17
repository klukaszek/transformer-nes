#pragma once

#include "nes/mappers/mapper.h"

namespace transformer_nes::nes {

// Mappers 9 (PxROM, Punch-Out!!) and 10 (FxROM) share one register
// file -- $A000 PRG select, $B000-$E000 CHR-latch registers, $F000
// mirroring -- and differ only in PRG bank unit (8 KiB switchable +
// three fixed vs 16 KiB + one fixed) and the latch trigger width the
// PPU-side ChrLatch applies (chr.h).
class LatchMapper : public Mapper {
public:
  LatchMapper(Cartridge cart, size_t unit, bool wide_low, const char* name)
      : Mapper(std::move(cart)), unit_(unit), wide_low_(wide_low) {
    if (cart_.prg.size() < 0x8000 || cart_.prg.size() % unit_)
      throw std::invalid_argument(string(name) + " PRG bank layout is invalid");
    if (cart_.chr.empty() || cart_.chr.size() % 0x1000)
      throw std::invalid_argument(string(name) + " requires 4 KiB CHR ROM banks");
    n_prg_ = cart_.prg.size() / unit_;
    n_chr_ = cart_.chr.size() / 0x1000;
  }
  bool banked() const override { return true; }
  // Slot 0 switches; the remaining slots are the last banks in order.
  size_t prg_bank(size_t slot) const {
    return slot == 0 ? prg_reg_ % n_prg_ : n_prg_ - (slots() - slot);
  }
  ChrLatchBanks latch_banks() const {
    ChrLatchBanks out{};
    for (size_t i = 0; i < out.size(); ++i)
      out[i] = chr_regs_[i] % n_chr_;
    return out;
  }
  int rom_byte(int addr) const {
    const size_t slot = static_cast<size_t>(addr - 0x8000) / unit_;
    return cart_.prg[prg_bank(slot) * unit_ + (static_cast<size_t>(addr) & (unit_ - 1))];
  }
  vector<uint8_t> cpu_image() const override {
    vector<uint8_t> image(0x10000);
    for (size_t i = 0; i < slots(); ++i)
      std::copy_n(cart_.prg.begin() + prg_bank(i) * unit_, unit_,
                  image.begin() + 0x8000 + i * unit_);
    return image;
  }
  void write(int addr, int value, int) override {
    int reg = addr & 0xF000;
    if (reg == 0xA000) {
      const size_t old = prg_bank(0);
      prg_reg_ = value & 15;
      if (old != prg_bank(0))
        map_region(0x8000, region(cart_.prg, prg_bank(0) * unit_, unit_));
    } else if (reg >= 0xB000 && reg <= 0xE000) {
      chr_regs_[(reg - 0xB000) >> 12] = value & 0x1F;
      host_.ppu->set_chr_latch_banks(latch_banks(), wide_low_);
    } else if (reg == 0xF000)
      host_.ppu->mirroring = value & 1 ? 3 : 2;
    repair(addr, rom_byte(addr));
  }

private:
  size_t unit_;
  bool wide_low_;
  uint8_t prg_reg_ = 0;
  array<uint8_t, 4> chr_regs_{};
  size_t n_prg_ = 0, n_chr_ = 0;

  size_t slots() const { return 0x8000 / unit_; }

  void on_connected() override { host_.ppu->set_chr_latch_banks(latch_banks(), wide_low_); }
};

class Mmc2 final : public LatchMapper {
public:
  explicit Mmc2(Cartridge cart) : LatchMapper(std::move(cart), BANK8, false, "MMC2") {}
};

class Mmc4 final : public LatchMapper {
public:
  explicit Mmc4(Cartridge cart) : LatchMapper(std::move(cart), BANK16, true, "MMC4") {}
};

} // namespace transformer_nes::nes
