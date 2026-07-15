#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>

#include "mapper.h"

namespace transformer_nes::nes {

// Native PPU-side half of mapper 10. Unlike MMC2, both pattern-table
// halves respond to all eight rows of the FD and FE switching tiles.
class Mmc4Chr final : public ChrLatchMapper {
public:
  void configure(const ChrLatchBanks& banks_4k, size_t chr_size) override {
    const size_t count = chr_size >> 12;
    if (count == 0)
      throw std::runtime_error("MMC4 requires CHR ROM");
    for (size_t i = 0; i < banks_4k.size(); ++i)
      banks_[i] = (banks_4k[i] % count) * 0x1000;
  }

  size_t map(size_t address) const override {
    const size_t half = (address >> 12) & 1;
    const size_t slot = half * 2 + (latch_fe_[half] ? 1 : 0);
    return banks_[slot] + (address & 0xFFF);
  }

  void after_fetch(size_t address) override {
    address &= 0x1FFF;
    if (address >= 0x0FD8 && address <= 0x0FDF)
      latch_fe_[0] = false;
    else if (address >= 0x0FE8 && address <= 0x0FEF)
      latch_fe_[0] = true;
    else if (address >= 0x1FD8 && address <= 0x1FDF)
      latch_fe_[1] = false;
    else if (address >= 0x1FE8 && address <= 0x1FEF)
      latch_fe_[1] = true;
  }

  array<bool, 2> latches() const override { return {latch_fe_[0], latch_fe_[1]}; }

private:
  ChrLatchBanks banks_{};
  bool latch_fe_[2] = {true, true};
};

class Mmc4 final : public Mapper {
public:
  explicit Mmc4(Cartridge cart) : Mapper(std::move(cart)) {
    if (cart_.prg.size() < 2 * BANK16 || cart_.prg.size() % BANK16)
      throw std::invalid_argument("MMC4 PRG must contain two 16 KiB banks");
    if (cart_.chr.empty() || cart_.chr.size() % 0x1000)
      throw std::invalid_argument("MMC4 requires 4 KiB CHR ROM banks");
    n_prg_ = cart_.prg.size() / BANK16;
    n_chr_ = cart_.chr.size() / 0x1000;
  }
  bool banked() const override { return true; }
  array<size_t, 2> prg_banks() const { return {prg_reg % n_prg_, n_prg_ - 1}; }
  ChrLatchBanks mmc4_banks() const {
    ChrLatchBanks out{};
    for (size_t i = 0; i < out.size(); ++i)
      out[i] = chr_regs[i] % n_chr_;
    return out;
  }
  int rom_byte(int addr) const {
    auto b = prg_banks();
    const size_t bank = b[addr < 0xC000 ? 0 : 1];
    return cart_.prg[bank * BANK16 + (addr & 0x3FFF)];
  }
  vector<uint8_t> cpu_image() const override {
    vector<uint8_t> image(0x10000);
    auto b = prg_banks();
    for (size_t i = 0; i < b.size(); ++i)
      std::copy_n(cart_.prg.begin() + b[i] * BANK16, BANK16, image.begin() + 0x8000 + i * BANK16);
    return image;
  }
  void write(int addr, int value) override {
    int reg = addr & 0xF000;
    if (reg == 0xA000) {
      const size_t old = prg_banks()[0];
      prg_reg = value & 15;
      const size_t now = prg_banks()[0];
      if (old != now)
        map_region(0x8000, region(cart_.prg, now * BANK16, BANK16));
    } else if (reg >= 0xB000 && reg <= 0xE000) {
      const size_t chr_register = (reg - 0xB000) >> 12;
      chr_regs[chr_register] = value & 0x1F;
      if (callbacks_.mmc4_banks)
        callbacks_.mmc4_banks(mmc4_banks());
    } else if (reg == 0xF000 && callbacks_.mirroring)
      callbacks_.mirroring(value & 1 ? 3 : 2);
    repair(addr, rom_byte(addr));
  }

private:
  uint8_t prg_reg = 0;
  array<uint8_t, 4> chr_regs{};
  size_t n_prg_ = 0, n_chr_ = 0;

  void on_connected() override {
    if (callbacks_.mmc4_banks)
      callbacks_.mmc4_banks(mmc4_banks());
  }
};

} // namespace transformer_nes::nes
