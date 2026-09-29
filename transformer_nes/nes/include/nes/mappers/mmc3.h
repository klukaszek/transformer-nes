#pragma once
#include "nes/mappers/mapper.h"

namespace transformer_nes::nes {

class Mmc3 final : public Mapper {
public:
  explicit Mmc3(Cartridge cart) : Mapper(std::move(cart)) {
    if (cart_.prg.empty() || cart_.prg.size() % BANK8)
      throw std::invalid_argument("MMC3 PRG must use 8 KiB banks");
    n_prg_ = cart_.prg.size() / BANK8;
    n_chr_ = cart_.chr.empty() ? CHR_RAM_BANKS : cart_.chr.size() / CHR_BANK1;
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
  bool wants_a12() const override { return true; }
  // Each filtered A12 rise clocks the counter (bg=$0000/sprites=$1000:
  // dot 260 of every rendered line); expiry with IRQs enabled asserts.
  void ppu_event(const PpuEvent& event) override {
    if (event.kind == PpuEvent::A12Rise && counter_.clock() && irq_enabled)
      host_.irq->set(IRQ_MAPPER, true);
  }
  std::unique_ptr<IrqProbe> irq_probe() const override {
    return irq_enabled ? std::make_unique<Probe>(counter_) : nullptr;
  }
  void write(int addr, int value) override {
    auto old = prg_banks();
    int reg = addr & 0xE001;
    update(reg, value);
    auto now = prg_banks();
    for (size_t i = 0; i < old.size(); ++i)
      if (old[i] != now[i])
        map_region(0x8000 + static_cast<int>(i * BANK8), region(cart_.prg, now[i] * BANK8, BANK8));
    notify_chr(reg);
    repair(addr, rom_byte(addr));
  }

private:
  // The scanline counter (NESDEV MMC3, Sharp behavior): a clock
  // reloads it from the latch when it is 0 or a reload is pending,
  // otherwise decrements; it fires on reaching 0.
  struct Counter {
    int latch = 0, value = 0;
    bool reload = false;
    bool clock() {
      if (value == 0 || reload) {
        value = latch;
        reload = false;
      } else {
        --value;
      }
      return value == 0;
    }
  };
  struct Probe final : IrqProbe {
    explicit Probe(Counter c) : counter(c) {}
    void ppu_event(const PpuEvent& event) override {
      if (event.kind == PpuEvent::A12Rise && counter.clock() && fire_dot < 0)
        fire_dot = event.dot;
    }
    Counter counter;
  };

  bool irq_enabled = false;
  Counter counter_;
  uint8_t bank_select = 0;
  array<uint8_t, 8> regs{0, 2, 4, 5, 6, 7, 0, 1};
  size_t n_prg_ = 0, n_chr_ = 1;
  void update(int reg, int value) {
    if (reg == 0x8000)
      bank_select = value;
    else if (reg == 0x8001)
      regs[bank_select & 7] = value;
    else if (reg == 0xA000)
      host_.ppu->mirroring = value & 1 ? 3 : 2;
    else if (reg == 0xC000)
      counter_.latch = value;
    else if (reg == 0xC001) {
      counter_.value = 0;
      counter_.reload = true;
    } else if (reg == 0xE000) {
      irq_enabled = false;
      host_.irq->set(IRQ_MAPPER, false);
    } else if (reg == 0xE001)
      irq_enabled = true;
    // The IRQ schedule may have moved (latch/reload/enable writes): the
    // console re-plans instead of running past the new expiry.
    if (reg >= 0xC000 && reg != 0xE000)
      host_.irq->sched_dirty = true;
  }
  void notify_chr(int reg) {
    if (reg == 0x8000 || reg == 0x8001)
      host_.ppu->set_chr_banks(chr_banks());
  }

  void on_connected() override {
    host_.ppu->set_chr_banks(chr_banks());
  }
};

} // namespace transformer_nes::nes
