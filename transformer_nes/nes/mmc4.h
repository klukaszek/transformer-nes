#pragma once

#include <cstddef>
#include <stdexcept>
#include <array>

#include "chr_latch.h"

// Native PPU-side half of mapper 10. Unlike MMC2, both pattern-table
// halves respond to all eight rows of the FD and FE switching tiles.
class Mmc4Chr final : public ChrLatchMapper {
public:
    void configure(const std::array<int, 4>& banks_4k,
                   std::size_t chr_size) override {
        const int count = static_cast<int>(chr_size >> 12);
        if (count == 0) throw std::runtime_error("MMC4 requires CHR ROM");
        for (int i = 0; i < 4; ++i)
            banks_[i] = (banks_4k[i] % count) * 0x1000;
    }

    int map(int addr) const override {
        const int half = (addr >> 12) & 1;
        const int slot = half * 2 + (latch_fe_[half] ? 1 : 0);
        return banks_[slot] + (addr & 0xFFF);
    }

    void after_fetch(int addr) override {
        addr &= 0x1FFF;
        if (addr >= 0x0FD8 && addr <= 0x0FDF) latch_fe_[0] = false;
        else if (addr >= 0x0FE8 && addr <= 0x0FEF) latch_fe_[0] = true;
        else if (addr >= 0x1FD8 && addr <= 0x1FDF) latch_fe_[1] = false;
        else if (addr >= 0x1FE8 && addr <= 0x1FEF) latch_fe_[1] = true;
    }

    std::array<bool, 2> latches() const override {
        return {latch_fe_[0], latch_fe_[1]};
    }

private:
    int banks_[4]{};
    bool latch_fe_[2] = {true, true};
};
