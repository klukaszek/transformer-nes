#pragma once

#include <cstddef>
#include <vector>

class ChrLatchMapper {
public:
    virtual ~ChrLatchMapper() = default;
    virtual void configure(const std::vector<int>& banks_4k,
                           std::size_t chr_size) = 0;
    virtual int map(int addr) const = 0;
    virtual void after_fetch(int addr) = 0;
    virtual std::vector<bool> latches() const = 0;
};
