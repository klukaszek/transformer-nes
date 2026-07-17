#pragma once
#include "nes/mappers/mapper.h"

namespace transformer_nes::nes {

class Mmc3 final : public Mapper {
public:
  explicit Mmc3(Cartridge cart) : Mapper(std::move(cart)) {
    if (cart_.prg.empty() || cart_.prg.size() % BANK8)
      throw std::invalid_argument("MMC3 PRG must use 8 KiB banks");
    n_prg_ = cart_.prg.size() / BANK8;
    n_chr_ = std::max(cart_.chr.size() / 0x400, 1uz);
    chr_rom = !cart_.chr.empty();
  }
  bool banked() const override { return true; }
  array<size_t, 4> prg_banks() const {
    const size_t r6 = regs[6] % n_prg_;
    const size_t r7 = regs[7] % n_prg_;
    return bank_select & 0x40 ? array<size_t, 4>{n_prg_ - 2, r7, r6, n_prg_ - 1}
                              : array<size_t, 4>{r6, r7, n_prg_ - 2, n_prg_ - 1};
  }
  ChrBanks chr_banks() const {
    const array<size_t, 4> two{(regs[0] & 0xFE) % n_chr_, ((regs[0] & 0xFE) | 1) % n_chr_,
                               (regs[1] & 0xFE) % n_chr_, ((regs[1] & 0xFE) | 1) % n_chr_};
    const array<size_t, 4> one{regs[2] % n_chr_, regs[3] % n_chr_, regs[4] % n_chr_,
                               regs[5] % n_chr_};
    ChrBanks out{};
    for (size_t i = 0; i < 4; ++i) {
      out[i] = (bank_select & 0x80) ? one[i] : two[i];
      out[i + 4] = (bank_select & 0x80) ? two[i] : one[i];
    }
    return out;
  }
  int rom_byte(int addr) const {
    const size_t slot = (addr - 0x8000) >> 13;
    const size_t bank = prg_banks()[slot];
    return cart_.prg[bank * BANK8 + (addr & 0x1FFF)];
  }
  vector<uint8_t> cpu_image() const override {
    vector<uint8_t> image(0x10000);
    auto b = prg_banks();
    for (size_t i = 0; i < b.size(); ++i)
      std::copy_n(cart_.prg.begin() + b[i] * BANK8, BANK8, image.begin() + 0x8000 + i * BANK8);
    return image;
  }
  // The counter reloads from the latch on the line after each expiry
  // and after a $C001 clear, so the latch is a DELTA from that anchor,
  // not an absolute line (NESDEV MMC3): SMB3's spade minigame chains
  // latch=52 handlers to split its strips and HUD at 32/85/138/191.
  // A vblank reload (anchor -1) fires at the latch's line; latch 0 is
  // the fire-next-scanline trick and needs a mid-frame anchor.
  optional<int> irq_scanline(int last_fired) const override {
    if (!irq_enabled)
      return std::nullopt;
    const int anchor = std::max(last_fired, reload_line);
    if (anchor < 0 && irq_latch <= 0)
      return std::nullopt;
    const int line = anchor < 0 ? irq_latch : anchor + 1 + irq_latch;
    return line < 240 ? optional<int>(line) : std::nullopt;
  }
  void notify_scanline_irq() override { irq_pending = true; }
  void begin_frame() override { reload_line = -1; }
  void write(int addr, int value, int scanline) override {
    auto old = prg_banks();
    int reg = addr & 0xE001;
    update(reg, value, scanline);
    auto now = prg_banks();
    for (size_t i = 0; i < old.size(); ++i)
      if (old[i] != now[i])
        map_region(0x8000 + static_cast<int>(i * BANK8), region(cart_.prg, now[i] * BANK8, BANK8));
    notify_chr(reg);
    repair(addr, rom_byte(addr));
  }

private:
  bool chr_rom = false, irq_enabled = false, irq_pending = false;
  uint8_t bank_select = 0;
  int irq_latch = 0;
  int reload_line = -1; // scanline of a mid-frame $C001 clear (-1: vblank)
  array<uint8_t, 8> regs{0, 2, 4, 5, 6, 7, 0, 1};
  size_t n_prg_ = 0, n_chr_ = 1;
  void update(int reg, int value, int scanline) {
    if (reg == 0x8000)
      bank_select = value;
    else if (reg == 0x8001)
      regs[bank_select & 7] = value;
    else if (reg == 0xA000)
      host_.ppu->mirroring = value & 1 ? 3 : 2;
    else if (reg == 0xC000)
      irq_latch = value;
    else if (reg == 0xC001)
      reload_line = scanline;
    else if (reg == 0xE000) {
      irq_enabled = false;
      host_.irq->set(IRQ_MAPPER, false);
    } else if (reg == 0xE001)
      irq_enabled = true;
    // The IRQ schedule may have moved (latch/reload/enable writes): the
    // console re-evaluates its frame events instead of running past a
    // mid-frame arm.
    if (reg >= 0xC000 && reg != 0xE000)
      host_.irq->sched_dirty = true;
  }
  void notify_chr(int reg) {
    if (chr_rom && (reg == 0x8000 || reg == 0x8001))
      host_.ppu->set_chr_banks(chr_banks());
  }

  void on_connected() override {
    if (chr_rom)
      host_.ppu->set_chr_banks(chr_banks());
  }
};

} // namespace transformer_nes::nes
