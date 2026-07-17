#pragma once

#include "nes/mappers/mapper.h"
#include "nes/mappers/mmc1.h"
#include "nes/mappers/mmc2_mmc4.h"
#include "nes/mappers/mmc3.h"
#include "nes/mappers/mmc5.h"

#include <memory>

namespace transformer_nes::nes {

using std::unique_ptr;

inline unique_ptr<Mapper> make_mapper(Cartridge cart) {
  switch (cart.mapper) {
  case 0:
    return std::make_unique<Nrom>(std::move(cart));
  case 1:
    return std::make_unique<Mmc1>(std::move(cart));
  case 4:
    return std::make_unique<Mmc3>(std::move(cart));
  case 5:
    return std::make_unique<Mmc5>(std::move(cart));
  case 9:
    return std::make_unique<Mmc2>(std::move(cart));
  case 10:
    return std::make_unique<Mmc4>(std::move(cart));
  default:
    throw std::invalid_argument("unsupported NES mapper " + std::to_string(cart.mapper));
  }
}

} // namespace transformer_nes::nes
