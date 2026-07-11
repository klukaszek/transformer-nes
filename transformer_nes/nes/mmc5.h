#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

// Native PPU-side MMC5 state. CPU register decoding and PRG banking live
// in mappers.py; this class owns data substituted onto the PPU bus.
class Mmc5Ppu {
public:
    void configure(int exram_mode, int nametable_map, int fill_tile,
                   int fill_color, int chr_upper) {
        active_ = true;
        exram_mode_ = exram_mode & 3;
        nametable_map_ = nametable_map & 0xFF;
        fill_tile_ = fill_tile & 0xFF;
        fill_color_ = fill_color & 3;
        chr_upper_ = chr_upper & 3;
    }

    bool active() const { return active_; }
    int nametable_map() const { return nametable_map_; }
    int exram_mode() const { return exram_mode_; }
    bool extended_attributes() const { return exram_mode_ == 1; }
    int extended_value(int tile_offset) const { return exram_[tile_offset & 0x3FF]; }
    int extended_chr_base(int tile_offset) const {
        return ((chr_upper_ << 6) | (extended_value(tile_offset) & 0x3F)) * 0x1000;
    }

    void write_exram(int offset, int value) {
        exram_[offset & 0x3FF] = static_cast<uint8_t>(value);
    }

    int read_exram(int offset) const { return exram_[offset & 0x3FF]; }

    // A value means MMC5 substitutes the bus; nullopt selects CIRAM 0/1.
    // The map is a caller-supplied $5105 value: render bands latch it, so
    // a mid-frame $5105 change must not re-decode through the live one.
    std::optional<int> nametable_byte(int map, int table, int offset) const {
        const int source = (map >> (table * 2)) & 3;
        if (source == 2) {
            return exram_mode_ <= 1 ? exram_[offset & 0x3FF] : 0;
        }
        if (source == 3) {
            if (offset < 0x3C0) return fill_tile_;
            return fill_color_ * 0x55;  // replicate AA into all quadrants
        }
        return std::nullopt;
    }

    void configure_chr(const std::vector<int>& sprite,
                       const std::vector<int>& background) {
        for (int i = 0; i < 8; ++i) {
            sprite_banks_[i] = sprite.at(i) * 0x400;
            background_banks_[i] = background.at(i) * 0x400;
        }
    }

    const int* sprite_banks() const { return sprite_banks_.data(); }
    const int* background_banks() const { return background_banks_.data(); }

    // State after vblank programming, used from scanline 0 until the first
    // visible IRQ. This is distinct from an IRQ snapshot: chained handlers
    // may temporarily select blank banks between raster events.
    void snapshot_frame() {
        frame_sprite_banks_ = sprite_banks_;
        frame_background_banks_ = background_banks_;
    }
    const int* frame_sprite_banks() const { return frame_sprite_banks_.data(); }
    const int* frame_background_banks() const {
        return frame_background_banks_.data();
    }

    // Freeze the CHR windows in effect at a mid-frame IRQ split, so rows
    // above it render from these while later writes only affect below.
    void snapshot() {
        split_sprite_banks_ = sprite_banks_;
        split_background_banks_ = background_banks_;
    }
    const int* split_sprite_banks() const { return split_sprite_banks_.data(); }
    const int* split_background_banks() const { return split_background_banks_.data(); }

private:
    bool active_ = false;
    int exram_mode_ = 0;
    int nametable_map_ = 0;
    int fill_tile_ = 0;
    int fill_color_ = 0;
    int chr_upper_ = 0;
    std::array<uint8_t, 0x400> exram_{};
    std::array<int, 8> sprite_banks_{};
    std::array<int, 8> background_banks_{};
    std::array<int, 8> split_sprite_banks_{};
    std::array<int, 8> split_background_banks_{};
    std::array<int, 8> frame_sprite_banks_{};
    std::array<int, 8> frame_background_banks_{};
};
