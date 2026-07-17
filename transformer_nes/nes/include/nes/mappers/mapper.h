#pragma once

#include "nes/cartridge.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace transformer_nes::nes {

using std::array;
using std::function;
using std::optional;
using std::pair;
using std::size_t;
using std::span;
using std::uint8_t;
using std::vector;

constexpr size_t BANK8 = 0x2000;
constexpr size_t BANK16 = 0x4000;
using ChrBanks = array<size_t, 8>;
using ChrLatchBanks = array<size_t, 4>;

class ChrLatchMapper {
public:
  virtual ~ChrLatchMapper() = default;
  virtual void configure(const ChrLatchBanks& banks_4k, size_t chr_size) = 0;
  virtual size_t map(size_t address) const = 0;
  virtual void after_fetch(size_t address) = 0;
  virtual array<bool, 2> latches() const = 0;
};

struct MapperCallbacks {
  function<void(int, int)> poke;
  function<void(int, span<const uint8_t>)> map_region;
  function<void(int)> mirroring;
  function<void(const ChrBanks&)> chr_banks;
  function<void()> irq_ack;
  // The IRQ schedule may have moved (latch/reload/enable writes): the
  // console re-evaluates its frame events instead of running past a
  // mid-frame arm.
  function<void()> irq_changed;
  function<void(const ChrLatchBanks&)> mmc2_banks;
  function<void(const ChrLatchBanks&)> mmc4_banks;
  function<void(int, int, int, int, int)> mmc5_state;
  function<void(int, int)> mmc5_exram;
  function<void(const ChrBanks&, const ChrBanks&)> mmc5_chr;
};

class Mapper {
public:
  explicit Mapper(Cartridge cart) : cart_(std::move(cart)) {}
  virtual ~Mapper() = default;
  virtual bool banked() const { return false; }
  virtual vector<uint8_t> cpu_image() const = 0;
  virtual int read(int) { throw std::runtime_error("mapper range is write-only"); }
  virtual void write(int, int) { throw std::runtime_error("mapper is not writable"); }
  // Write with the visible-span scanline it executed on (-1 outside);
  // MMC3 anchors mid-frame $C001 reloads to it.
  virtual void write_at(int addr, int value, int) { write(addr, value); }
  virtual void write_rmw(int addr, int, int value) { write(addr, value); }
  // Next IRQ scanline given the line of the frame's previous IRQ (-1 at
  // frame start): MMC3-class counters chain in latch-sized deltas from
  // the previous expiry, MMC5 compares against an absolute line.
  virtual optional<int> irq_scanline(int) const { return std::nullopt; }
  virtual vector<pair<int, int>> register_ranges() const { return {}; }
  virtual void notify_scanline_irq() {}
  virtual void begin_frame() {}

  void connect(MapperCallbacks callbacks) {
    callbacks_ = std::move(callbacks);
    on_connected();
  }

  const Cartridge& cartridge() const { return cart_; }

protected:
  Cartridge cart_;
  MapperCallbacks callbacks_;

  virtual void on_connected() {}

  void repair(int addr, int value) {
    if (!callbacks_.poke)
      throw std::logic_error("mapper poke callback not attached");
    callbacks_.poke(addr, value & 0xFF);
  }
  void map_region(int base, span<const uint8_t> data) {
    if (callbacks_.map_region)
      callbacks_.map_region(base, data);
    else {
      if (!callbacks_.poke)
        throw std::logic_error("mapper callback not attached");
      for (size_t i = 0; i < data.size(); ++i)
        callbacks_.poke(base + static_cast<int>(i), data[i]);
    }
  }
  static span<const uint8_t> region(span<const uint8_t> data, size_t offset, size_t size) {
    return data.subspan(offset, size);
  }
};

class Nrom final : public Mapper {
public:
  using Mapper::Mapper;
  vector<uint8_t> cpu_image() const override {
    vector<uint8_t> image(0x10000);
    std::copy(cart_.prg.begin(), cart_.prg.end(), image.begin() + 0x8000);
    if (cart_.prg.size() == BANK16)
      std::copy(cart_.prg.begin(), cart_.prg.end(), image.begin() + 0xC000);
    return image;
  }
};

} // namespace transformer_nes::nes
