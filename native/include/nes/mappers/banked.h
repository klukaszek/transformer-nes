#pragma once

#include "nes/mappers/mapper.h"
#include <bit>

namespace transformer_nes::nes {

// Opaque neural register state. A transition emits a dirty mask; only then
// is the bank-layout projection evaluated, with the board's ROM sizes as its
// last inputs. Neither this adapter nor BankMapper computes banking formulas;
// the chip-unit counts here only bound-check the circuit's bank numbers.
struct BankControl {
  enum Chip { Mmc1 = 1, Mmc3 = 4, Mmc5 = 5, Mmc2 = 9, Mmc4 = 10 };
  enum Effect { Prg = 1, Chr = 2, Mirror = 4 };
  // The chip's model ids (nes/mappers.py:chip).
  struct Engine {
    void* ctx = nullptr;
    void (*step)(void*, int, const int*, int*) = nullptr;
    int transition = 0, layout = 0;
    void* access_ctx = nullptr;
    void (*access_step)(void*, int, const int*, int*) = nullptr;
    array<int, 2> access_models{};
    int address_model = 0;
  } engine;
  int chip, state_count = 0, initial_mirror;
  size_t unit, n_prg, n_chr, prg_count, chr_count;
  // mappers.py BOARD: PRG in 16 KiB units, CHR in 4 KiB units (RAM counts as 8 KiB).
  array<int, 2> board{};
  bool latches;
  array<array<int, 36>, 2> states{};
  int active = 0;
  array<int, 40> map{}; // PRG banks, CHR banks, mirroring, chip-specific effects

  enum Storage { Literal, Rom, Ram, Exram, Irq };
  // port: the register written (its address), or 0.
  struct Access {
    int port, event, irq_event, schedule, source, index, value, write, repair;
  };
  array<int, 16> access_inputs(int addr, bool writing, int original = -1) const {
    array<int, 16> in{addr & 255, (addr >> 8) & 255, writing, original};
    std::copy_n(map.begin(), prg_count, in.begin() + 4);
    if (chip == Mmc5) {
      std::copy_n(map.begin() + 21, 6, in.begin() + 8); // RAM flags, bank, permission
      std::copy_n(map.begin() + 31, 2, in.begin() + 14); // multiplier bytes
    }
    return in;
  }
  Access access(int addr, bool writing, int original = -1) const {
    if (!engine.access_step)
      throw std::runtime_error("mapper access transformer is required");
    const auto in = access_inputs(addr, writing, original);
    array<int, 9> out;
    engine.access_step(engine.access_ctx, engine.access_models[writing], in.data(), out.data());
    static_assert(sizeof(Access) == sizeof(out));
    return std::bit_cast<Access>(out);
  }
  void refresh_address(Access& packet, int addr) const {
    const auto in = access_inputs(addr, true);
    int out[3];
    engine.access_step(engine.access_ctx, engine.address_model, in.data(), out);
    packet.source = out[0]; packet.index = out[1]; packet.value = out[2];
  }

  int transition(int event, int port, int value, int original = 0) {
    if (!engine.step)
      throw std::runtime_error("mapper bank transformer is required");
    auto& in = states[active];
    auto& out = states[active ^ 1];
    in[state_count] = event;
    in[state_count + 1] = port;
    in[state_count + 2] = value;
    in[state_count + 3] = original;
    engine.step(engine.ctx, engine.transition, in.data(), out.data());
    active ^= 1;
    const int dirty = out[state_count];
    if (dirty) {
      std::copy(board.begin(), board.end(), out.begin() + state_count);
      engine.step(engine.ctx, engine.layout, out.data(), map.data());
    }
    return dirty;
  }
  void reset() { transition(0, 0, initial_mirror); }
};

class BankMapper final : public Mapper {
public:
  BankMapper(Cartridge cart, int chip) : Mapper(std::move(cart)) {
    bank_.chip = chip;
    bank_.unit = chip == BankControl::Mmc1 || chip == BankControl::Mmc4 ? BANK16 : BANK8;
    bank_.latches = chip == BankControl::Mmc2 || chip == BankControl::Mmc4;
    if (cart_.prg.empty() || cart_.prg.size() % bank_.unit ||
        (bank_.latches && cart_.prg.size() < 0x8000))
      throw std::invalid_argument("invalid mapper PRG bank geometry");
    if (cart_.prg.size() % BANK16 || cart_.chr.size() % 0x1000)
      throw std::invalid_argument("cartridge ROM sizes must be whole 16 KiB PRG and 4 KiB CHR units");
    if ((chip == BankControl::Mmc1 && cart_.chr.size() % 0x1000) ||
        (bank_.latches && (cart_.chr.empty() || cart_.chr.size() % 0x1000)))
      throw std::invalid_argument("invalid mapper CHR bank geometry");
    bank_.n_prg = cart_.prg.size() / bank_.unit;
    bank_.n_chr = cart_.chr.empty() ? CHR_RAM_BANKS
                    : cart_.chr.size() / (bank_.latches ? 0x1000 : CHR_BANK1);
    if (!bank_.n_chr || (chip == BankControl::Mmc3 && bank_.n_prg < 2))
      throw std::invalid_argument("invalid mapper bank count");
    bank_.board = {static_cast<int>(cart_.prg.size() / BANK16),
                   static_cast<int>(cart_.chr.empty() ? 2 : cart_.chr.size() / 0x1000)};
    bank_.prg_count = 0x8000 / bank_.unit;
    bank_.chr_count = bank_.latches ? 4 : 8;
    bank_.initial_mirror = cart_.mirroring == Mirroring::Vertical ? 2 : 3;
    if (chip == BankControl::Mmc3)
      irq_ = std::make_unique<MapperIrq>(MapperIrq::Mmc3);
  }
  bool banked() const override { return true; }
  BankControl* bank_state() override { return &bank_; }
  MapperIrq* irq_state() override { return irq_.get(); }
  bool wants_a12() const override { return irq_ != nullptr; }
  void ppu_event(const PpuEvent& event) override {
    if (irq_ && event.kind == PpuEvent::A12Rise)
      irq_event(*irq_, MapperIrq::Clock);
  }
  std::unique_ptr<IrqProbe> irq_probe() const override {
    return irq_ && irq_->predictable ? std::make_unique<MapperIrqProbe>(*irq_) : nullptr;
  }
  span<const int> prg_banks() const { return {bank_.map.data(), bank_.prg_count}; }
  vector<uint8_t> cpu_image() const override {
    if (!bank_.engine.step)
      throw std::runtime_error("mapper bank transformer is required");
    vector<uint8_t> image(0x10000);
    for (size_t i = 0; i < bank_.prg_count; ++i)
      std::ranges::copy(prg_region(bank_.map[i]), image.begin() + 0x8000 + i * bank_.unit);
    return image;
  }
  void write(int addr, int value) override { apply_write(addr, value, -1); }
  void write_rmw(int addr, int value, int original) override {
    apply_write(addr, value, original);
  }

private:
  BankControl bank_;
  std::unique_ptr<MapperIrq> irq_;

  span<const uint8_t> prg_region(int bank) const {
    if (bank < 0 || static_cast<size_t>(bank) >= bank_.n_prg)
      throw std::out_of_range("PRG bank outside cartridge");
    return region(cart_.prg, static_cast<size_t>(bank) * bank_.unit, bank_.unit);
  }

  void deliver_chr() {
    if (bank_.latches) {
      ChrLatchBanks banks{};
      std::copy_n(bank_.map.begin() + bank_.prg_count, banks.size(), banks.begin());
      host_.ppu->set_chr_latch_banks(banks, bank_.chip == BankControl::Mmc4);
    } else {
      ChrBanks banks{};
      std::copy_n(bank_.map.begin() + bank_.prg_count, banks.size(), banks.begin());
      host_.ppu->set_chr_banks(banks);
    }
  }
  void on_connected() override {
    deliver_chr();
    host_.ppu->mirroring = bank_.map[bank_.prg_count + bank_.chr_count];
  }
  void apply_write(int addr, int value, int original) {
    auto packet = bank_.access(addr, true, original);
    int dirty = 0;
    if (packet.port) {
      array<int, 4> before;
      std::copy_n(bank_.map.begin(), bank_.prg_count, before.begin());
      dirty = bank_.transition(packet.event, packet.port, value & 0xFF, original);
      if (dirty & BankControl::Prg)
        for (size_t i = 0; i < bank_.prg_count; ++i)
          if (before[i] != bank_.map[i])
            map_region(0x8000 + static_cast<int>(i * bank_.unit),
                       prg_region(bank_.map[i]));
      if (dirty & BankControl::Chr)
        deliver_chr();
      if (dirty & BankControl::Mirror)
        host_.ppu->mirroring = bank_.map[bank_.prg_count + bank_.chr_count];
    }
    if (packet.irq_event >= 0)
      irq_event(*irq_, static_cast<MapperIrq::Event>(packet.irq_event), value);
    host_.irq->sched_dirty |= packet.schedule != 0;
    if (packet.repair) {
      if (dirty & BankControl::Prg)
        bank_.refresh_address(packet, addr);
      repair(addr, cart_.prg[static_cast<size_t>(packet.index)]);
    }
  }
};

} // namespace transformer_nes::nes
