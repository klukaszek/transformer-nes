#pragma once

#include <cstdint>
#include <fstream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace transformer_nes::nes {

using std::span;
using std::string;
using std::uint8_t;
using std::vector;

enum class Mirroring : uint8_t { Horizontal = 0, Vertical = 1 };

struct Cartridge {
  vector<uint8_t> prg;
  vector<uint8_t> chr;
  Mirroring mirroring = Mirroring::Horizontal;
  int mapper = 0;
  bool battery = false;

  static Cartridge parse(span<const uint8_t> data) {
    if (data.size() < 16 || data[0] != 'N' || data[1] != 'E' || data[2] != 'S' || data[3] != 0x1A)
      throw std::invalid_argument("not an iNES file");
    const uint8_t flags6 = data[6], flags7 = data[7];
    const size_t offset = 16 + ((flags6 & 0x04) ? 512 : 0);
    const size_t prg_size = size_t(data[4]) * 0x4000;
    const size_t chr_size = size_t(data[5]) * 0x2000;
    if (offset > data.size() || prg_size > data.size() - offset ||
        chr_size > data.size() - offset - prg_size)
      throw std::invalid_argument("truncated iNES file");
    Cartridge cart;
    const auto prg = data.subspan(offset, prg_size);
    const auto chr = data.subspan(offset + prg_size, chr_size);
    cart.prg.assign(prg.begin(), prg.end());
    cart.chr.assign(chr.begin(), chr.end());
    cart.mirroring = flags6 & 1 ? Mirroring::Vertical : Mirroring::Horizontal;
    cart.mapper = (flags7 & 0xF0) | (flags6 >> 4);
    cart.battery = flags6 & 0x02;
    return cart;
  }

  static Cartridge load(const string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
      throw std::runtime_error("cannot open ROM: " + path);
    const vector<uint8_t> data{std::istreambuf_iterator<char>(file), {}};
    return parse(data);
  }
};

} // namespace transformer_nes::nes
