#pragma once

#include <cstddef>
#include <array>

class ChrLatchMapper {
public:
    virtual ~ChrLatchMapper() = default;
    virtual void configure(const std::array<int, 4>& banks_4k,
                           std::size_t chr_size) = 0;
    virtual int map(int addr) const = 0;
    virtual void after_fetch(int addr) = 0;
    virtual std::array<bool, 2> latches() const = 0;
};
