#pragma once

#include "nes/bus.h"
#include "nes/cartridge.h"
#include "nes/chr.h"
#include "nes/ppu.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace transformer_nes::nes {

using std::array;
using std::optional;
using std::pair;
using std::size_t;
using std::span;
using std::uint8_t;
using std::vector;

constexpr size_t BANK8 = 0x2000;
constexpr size_t BANK16 = 0x4000;

// Everything a mapper drives outside its own cartridge state: the PPU
// (mirroring, CHR banks, MMC5 substitution state), the machine image
// (repair pokes, bank region maps), and the shared IRQ line (acks +
// the sched_dirty re-plan signal).
struct MapperHost {
  Ppu* ppu = nullptr;
  ConsoleMachine* machine = nullptr;
  IrqLine* irq = nullptr;
};

class Mapper {
public:
  explicit Mapper(Cartridge cart) : cart_(std::move(cart)) {}
  virtual ~Mapper() = default;
  virtual bool banked() const { return false; }
  virtual vector<uint8_t> cpu_image() const = 0;
  virtual int read(int) { throw std::runtime_error("mapper range is write-only"); }
  // scanline: the visible-span row the write executed on (-1 outside
  // or unknown); MMC3 anchors mid-frame $C001 reloads to it.
  virtual void write(int, int, int) { throw std::runtime_error("mapper is not writable"); }
  virtual void write_rmw(int addr, int, int value) { write(addr, value, -1); }
  // Next IRQ scanline given the line of the frame's previous IRQ (-1 at
  // frame start): MMC3-class counters chain in latch-sized deltas from
  // the previous expiry, MMC5 compares against an absolute line.
  virtual optional<int> irq_scanline(int) const { return std::nullopt; }
  virtual vector<pair<int, int>> register_ranges() const { return {}; }
  virtual void notify_scanline_irq() {}
  virtual void begin_frame() {}

  void connect(MapperHost host) {
    host_ = host;
    on_connected();
  }

  const Cartridge& cartridge() const { return cart_; }

protected:
  Cartridge cart_;
  MapperHost host_;

  virtual void on_connected() {}

  void repair(int addr, int value) { host_.machine->poke(addr, value & 0xFF); }
  void map_region(int base, span<const uint8_t> data) { host_.machine->map_region(base, data); }
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
