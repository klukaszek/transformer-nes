#pragma once
#include "nes/mappers/mapper.h"

namespace transformer_nes::nes {

class Mmc1 final : public Mapper {
public:
  explicit Mmc1(Cartridge cart) : Mapper(std::move(cart)) {
    if (cart_.prg.empty() || cart_.prg.size() % BANK16)
      throw std::invalid_argument("MMC1 PRG must use 16 KiB banks");
    if (cart_.chr.size() % 0x1000)
      throw std::invalid_argument("MMC1 CHR must use 4 KiB banks");
    n_prg_ = cart_.prg.size() / BANK16;
    n_chr_ = std::max(cart_.chr.size() / 0x400, 1uz);
    chr_rom = !cart_.chr.empty();
    control = 0x0C | (cart_.mirroring == Mirroring::Vertical ? 2 : 3);
  }
  bool banked() const override { return true; }
  array<size_t, 2> prg_banks() const {
    const auto mode = (control >> 2) & 3;
    const size_t bank = (prg_reg & 15) % n_prg_;
    if (mode < 2)
      return {bank & ~1uz, (bank & ~1uz) | 1uz};
    if (mode == 2)
      return {0, bank};
    return {bank, n_prg_ - 1};
  }
  ChrBanks chr_banks() const {
    ChrBanks out{};
    if (control & 0x10) {
      for (size_t half = 0; half < 2; ++half)
        for (size_t i = 0; i < 4; ++i)
          out[half * 4 + i] = (chr_regs[half] * 4 + i) % n_chr_;
    } else {
      const size_t base = (chr_regs[0] & 0x1E) * 4;
      for (size_t i = 0; i < out.size(); ++i)
        out[i] = (base + i) % n_chr_;
    }
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
    std::copy_n(cart_.prg.begin() + b[0] * BANK16, BANK16, image.begin() + 0x8000);
    std::copy_n(cart_.prg.begin() + b[1] * BANK16, BANK16, image.begin() + 0xC000);
    return image;
  }
  void write(int addr, int value, int) override {
    auto old_prg = prg_banks();
    int old_mirror = control & 3;
    auto old_chr = chr_banks();
    update(addr, value);
    auto now = prg_banks();
    for (size_t i = 0; i < old_prg.size(); ++i)
      if (old_prg[i] != now[i])
        map_region(0x8000 + static_cast<int>(i * BANK16),
                   region(cart_.prg, now[i] * BANK16, BANK16));
    repair(addr, rom_byte(addr));
    notify(old_mirror, old_chr);
  }
  void write_rmw(int addr, int original, int value) override {
    write(addr, original, -1);
    if (value & 0x80)
      write(addr, value, -1);
  }

private:
  bool chr_rom = false;
  uint8_t control = 0, prg_reg = 0, shift = 0x10;
  array<uint8_t, 2> chr_regs{};
  size_t n_prg_ = 0, n_chr_ = 1;
  void update(int addr, int value) {
    if (value & 0x80) {
      shift = 0x10;
      control |= 0x0C;
      return;
    }
    bool complete = shift & 1;
    shift = ((shift >> 1) | ((value & 1) << 4)) & 0x1F;
    if (!complete)
      return;
    const size_t reg = (addr >> 13) & 3;
    if (!reg)
      control = shift;
    else if (reg < 3)
      chr_regs[reg - 1] = shift;
    else
      prg_reg = shift;
    shift = 0x10;
  }
  void notify(int old_mirror, const ChrBanks& old_chr) {
    if ((control & 3) != old_mirror)
      host_.ppu->mirroring = control & 3;
    auto now = chr_banks();
    if (chr_rom && now != old_chr)
      host_.ppu->set_chr_banks(now);
  }

  void on_connected() override {
    if (chr_rom)
      host_.ppu->set_chr_banks(chr_banks());
  }
};

} // namespace transformer_nes::nes
