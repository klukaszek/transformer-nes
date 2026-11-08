#pragma once

#include "nes/mappers/banked.h"

namespace transformer_nes::nes {

// MMC5 storage and effect delivery. Register transitions, permissions,
// bank arithmetic and the multiplier are supplied by the transformer.
class Mmc5 final : public Mapper {
public:
  explicit Mmc5(Cartridge cart) : Mapper(std::move(cart)), prg_ram_(0x10000) {
    if (cart_.prg.empty() || cart_.prg.size() % BANK16 || cart_.chr.size() % 0x2000)
      throw std::invalid_argument("MMC5 requires 16 KiB PRG and 8 KiB CHR units");
    bank_.chip = BankControl::Mmc5;
    bank_.unit = BANK8;
    bank_.n_prg = cart_.prg.size() / BANK8;
    bank_.n_chr = cart_.chr.empty() ? CHR_RAM_BANKS : cart_.chr.size() / CHR_BANK1;
    bank_.board = {static_cast<int>(cart_.prg.size() / BANK16),
                   static_cast<int>(cart_.chr.empty() ? 2 : cart_.chr.size() / 0x1000)};
    bank_.prg_count = 4;
    bank_.chr_count = 16;
    bank_.initial_mirror = 0;
    bank_.latches = false;
  }
  bool banked() const override { return true; }
  BankControl* bank_state() override { return &bank_; }
  MapperIrq* irq_state() override { return &irq_; }
  vector<pair<int, int>> register_ranges() const override { return {{0x5000, 0x7FFF}}; }
  vector<uint8_t> cpu_image() const override {
    if (!bank_.engine.step)
      throw std::runtime_error("mapper bank transformer is required");
    vector<uint8_t> image(0x10000);
    for (size_t i = 0; i < 4; ++i)
      std::ranges::copy(slot_region(i), image.begin() + 0x8000 + i * BANK8);
    return image;
  }
  void ppu_event(const PpuEvent& event) override {
    if (event.kind == PpuEvent::ScanlineStart)
      irq_event(irq_, MapperIrq::Clock);
    else if (event.kind == PpuEvent::Idle)
      irq_event(irq_, MapperIrq::Idle);
  }
  std::unique_ptr<IrqProbe> irq_probe() const override {
    return irq_.predictable ? std::make_unique<MapperIrqProbe>(irq_) : nullptr;
  }
  int read(int addr) override {
    const auto packet = bank_.access(addr, false);
    if (packet.irq_event >= 0)
      irq_event(irq_, static_cast<MapperIrq::Event>(packet.irq_event));
    return load(packet);
  }
  void write(int addr, int value) override {
    value &= 0xFF;
    const auto packet = bank_.access(addr, true);
    if (packet.port) {
      array<pair<int, int>, 4> before;
      for (size_t i = 0; i < before.size(); ++i)
        before[i] = {bank_.map[i], bank_.map[Ram + i]};
      const int dirty = bank_.transition(packet.event, packet.port, value);
      if (dirty & BankControl::Prg)
        for (size_t i = 0; i < 4; ++i)
          if (before[i] != pair{bank_.map[i], bank_.map[Ram + i]})
            map_region(0x8000 + static_cast<int>(i * BANK8), slot_region(i));
      if (dirty & BankControl::Chr)
        deliver_chr();
      if (dirty & BankControl::Mirror)
        deliver_ppu();
    }
    if (packet.irq_event >= 0)
      irq_event(irq_, static_cast<MapperIrq::Event>(packet.irq_event), value);
    host_.irq->sched_dirty |= packet.schedule != 0;
    if (packet.write == BankControl::Ram)
      prg_ram_[static_cast<size_t>(packet.index)] = static_cast<uint8_t>(value);
    else if (packet.write == BankControl::Exram)
      host_.ppu->set_mmc5_exram(packet.index, value);
    if (packet.repair)
      repair(addr, load(packet));
  }

private:
  enum Map { Mirror = 20, Ram = 21, Exram = 27, FillTile = 28, FillColor = 29, Upper = 30 };
  BankControl bank_;
  MapperIrq irq_{MapperIrq::Mmc5};
  vector<uint8_t> prg_ram_;

  int load(const BankControl::Access& packet) const {
    switch (packet.source) {
    case BankControl::Rom: return cart_.prg[static_cast<size_t>(packet.index)];
    case BankControl::Ram: return prg_ram_[static_cast<size_t>(packet.index)];
    case BankControl::Exram: return host_.ppu->mmc5.exram(packet.index);
    case BankControl::Irq: return irq_.read_value;
    default: return packet.value;
    }
  }
  span<const uint8_t> slot_region(size_t slot) const {
    const auto& storage = bank_.map[Ram + slot] ? prg_ram_ : cart_.prg;
    return region(storage, bank_.map[slot] * BANK8, BANK8);
  }
  void deliver_chr() {
    ChrBanks sprite{}, background{};
    std::copy_n(bank_.map.begin() + 4, 8, sprite.begin());
    std::copy_n(bank_.map.begin() + 12, 8, background.begin());
    host_.ppu->set_chr_banks(sprite);
    host_.ppu->set_mmc5_chr(sprite, background);
  }
  void deliver_ppu() {
    host_.ppu->set_mmc5_state(bank_.map[Exram], bank_.map[Mirror], bank_.map[FillTile],
                              bank_.map[FillColor], bank_.map[Upper]);
  }
  void on_connected() override {
    deliver_ppu();
    deliver_chr();
  }
};

} // namespace transformer_nes::nes
