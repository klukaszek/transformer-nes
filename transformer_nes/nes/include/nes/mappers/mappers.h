#pragma once

#include "nes/mappers/mapper.h"
#include "nes/mappers/banked.h"
#include "nes/mappers/mmc5.h"

#include <memory>

namespace transformer_nes::nes {

using std::unique_ptr;

inline unique_ptr<Mapper> make_mapper(Cartridge cart) {
  switch (cart.mapper) {
  case 0:
    return std::make_unique<Nrom>(std::move(cart));
  case 1:
  case 4:
  case 9:
  case 10: {
    const int chip = cart.mapper;
    return std::make_unique<BankMapper>(std::move(cart), chip);
  }
  case 5:
    return std::make_unique<Mmc5>(std::move(cart));
  default:
    throw std::invalid_argument("unsupported NES mapper " + std::to_string(cart.mapper));
  }
}

} // namespace transformer_nes::nes
