#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace transformer_nes::nes {

using std::array;
using std::optional;
using std::size_t;
using std::span;
using std::uint8_t;
using std::unique_ptr;
using std::vector;

class Ppu final {
public:
  static constexpr size_t frame_width = 256;
  static constexpr size_t frame_height = 240;
  using Frame = array<uint8_t, frame_width * frame_height>;

  Ppu(vector<uint8_t> chr, bool writable, bool vertical_mirroring);
  ~Ppu();

  Ppu(const Ppu&) = delete;
  Ppu& operator=(const Ppu&) = delete;
  Ppu(Ppu&&) noexcept;
  Ppu& operator=(Ppu&&) noexcept;

  [[nodiscard]] int read(int address);
  void write(int address, int value);
  // Register write stamped with the visible-span scanline it executed
  // on (-1 outside the span): mid-frame split and forced-blank rows.
  void write_stamped(int address, int value, int scanline);
  [[nodiscard]] optional<int> read_lease(int address) const;
  void oam_dma(span<const uint8_t, 256> page);

  [[nodiscard]] Frame render_frame();
  void begin_vblank();
  void end_vblank();
  void mark_split(int scanline);
  void begin_span_watch();
  [[nodiscard]] bool take_nmi_edge();
  [[nodiscard]] bool nmi_enabled() const;
  [[nodiscard]] bool rendering_enabled() const;
  [[nodiscard]] int sprite0_scanline() const;
  void set_sprite0_hit(bool value);
  // Debug/QoL: lift the hardware 8-per-scanline sprite cap.  Default is
  // hardware-exact; TLoZ's door masking depends on the cap staying on.
  void set_sprite_limit(bool enabled);

  void set_mirroring(int value);
  void set_chr_banks(const array<size_t, 8>& banks);
  void set_chr_latch_banks(const array<size_t, 4>& banks, bool wide_low);
  void set_mmc5_state(int exram_mode, int nametable_map, int fill_tile, int fill_color,
                      int chr_upper);
  void set_mmc5_exram(int address, int value);
  void set_mmc5_chr(const array<size_t, 8>& sprite, const array<size_t, 8>& background);

  [[nodiscard]] span<const uint8_t, 0x800> nametables() const;
  [[nodiscard]] span<const uint8_t, 0x20> palette() const;
  [[nodiscard]] span<const uint8_t, 0x100> oam() const;

private:
  struct Impl;
  unique_ptr<Impl> impl_;
};

} // namespace transformer_nes::nes
