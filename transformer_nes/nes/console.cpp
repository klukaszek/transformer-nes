#include "console.h"

#include <algorithm>

namespace transformer_nes::nes {

namespace {

constexpr int64_t vblank_start_cycles = 27'394;
constexpr int64_t frame_cycles = 29'781;
constexpr int max_frame_events = 8;
constexpr int max_split_scan_lines = 48;

constexpr int64_t scanline_cycle(int line) { return static_cast<int64_t>(line + 1) * 341 / 3; }

} // namespace

class Console::PpuDevice final : public Device {
public:
  explicit PpuDevice(Ppu& ppu) : ppu_(ppu) {}
  int read(int address) override { return ppu_.read(address); }
  void write(int address, int value) override { ppu_.write(address, value); }
  ReadLease read_lease(int a) override {
    auto v = ppu_.read_lease(a);
    return v ? ReadLease(*v) : ReadLease{};
  }

private:
  Ppu& ppu_;
};
class Console::ApuDevice final : public Device {
public:
  explicit ApuDevice(Apu& apu) : apu_(apu) {}
  int read(int address) override { return apu_.read(address); }
  void write(int address, int value) override { apu_.write(address, value); }

private:
  Apu& apu_;
};
class Console::MapperDevice final : public Device {
public:
  explicit MapperDevice(Mapper& m) : m_(m) {}
  int read(int a) override { return m_.read(a); }
  void write(int a, int v) override { m_.write(a, v); }
  void write_rmw(int a, int original, int value) override { m_.write_rmw(a, original, value); }

private:
  Mapper& m_;
};
class Console::FrameCounter final : public Device {
public:
  explicit FrameCounter(Apu& apu) : apu_(apu) {}
  int read(int address) override { return apu_.read(address); }
  void write(int address, int value) override { apu_.write(address, value); }
  ReadLease read_lease(int) override { return 0x40; }

private:
  Apu& apu_;
};

Console::~Console() = default;

Console::Console(unique_ptr<Mapper> mapper, unique_ptr<Ppu> ppu, unique_ptr<Apu> apu,
                 ConsoleMachine& machine)
    : mapper_(std::move(mapper)), ppu_(std::move(ppu)), apu_(std::move(apu)), machine_(machine) {
  ppu_device_ = std::make_unique<PpuDevice>(*ppu_);
  apu_device_ = std::make_unique<ApuDevice>(*apu_);
  mapper_device_ = std::make_unique<MapperDevice>(*mapper_);
  frame_counter_ = std::make_unique<FrameCounter>(*apu_);
  dma_ = std::make_unique<OamDma>([this](int a) { return machine_.read_ram(a); },
                                  [this](const auto& page) { ppu_->oam_dma(page); });
  attach_devices();
}
void Console::attach_devices() {
  machine_.attach(*ppu_device_, 0x2000, 0x3FFF);
  machine_.attach(*dma_, 0x4014, 0x4014);
  machine_.attach(*apu_device_, 0x4000, 0x4013);
  machine_.attach(*apu_device_, 0x4015, 0x4015);
  machine_.attach(controller_, 0x4016, 0x4016);
  machine_.attach(*frame_counter_, 0x4017, 0x4017);

  if (!mapper_->banked())
    return;
  for (auto [lo, hi] : mapper_->register_ranges())
    machine_.attach(*mapper_device_, lo, hi);
  machine_.attach(*mapper_device_, 0x8000, 0xFFFF, true);
  MapperCallbacks callbacks;
  callbacks.poke = [this](int a, int v) { machine_.poke(a, v); };
  callbacks.map_region = [this](int a, const auto& d) { machine_.map_region(a, d); };
  callbacks.mirroring = [this](int m) { ppu_->set_mirroring(m); };
  callbacks.chr_banks = [this](const auto& banks) { ppu_->set_chr_banks(banks); };
  callbacks.irq_ack = [this] { set_irq(IrqSource::mapper, false); };
  callbacks.mmc2_banks = [this](const auto& banks) { ppu_->set_mmc2_banks(banks); };
  callbacks.mmc4_banks = [this](const auto& banks) { ppu_->set_mmc4_banks(banks); };
  callbacks.mmc5_state = [this](int a, int b, int c, int d, int e) {
    ppu_->set_mmc5_state(a, b, c, d, e);
  };
  callbacks.mmc5_exram = [this](int a, int v) { ppu_->set_mmc5_exram(a, v); };
  callbacks.mmc5_chr = [this](const auto& sprite, const auto& background) {
    ppu_->set_mmc5_chr(sprite, background);
  };
  mapper_->connect(std::move(callbacks));
}
void Console::sync_irq() {
  if (irq_sources_ == 0)
    machine_.clear_irq();
  else
    machine_.set_irq();
}

void Console::set_irq(IrqSource source, bool active) {
  const auto bit = static_cast<uint8_t>(source);
  if (active)
    irq_sources_ |= bit;
  else
    irq_sources_ &= static_cast<uint8_t>(~bit);
  sync_irq();
}

void Console::check_interrupts() {
  if (ppu_->take_nmi_edge())
    machine_.set_nmi();
  if ((irq_sources_ & static_cast<uint8_t>(IrqSource::apu)) != 0 && !apu_->irq_pending())
    set_irq(IrqSource::apu, false);
}
void Console::run_until(int64_t target) {
  if (halted_ || target <= cycles_)
    return;
  auto out = machine_.run_cycles(target - cycles_, [this] { check_interrupts(); });
  cycles_ += out.cycles;
  halted_ = out.halted;
}
// The game's scroll writes land some scanlines after the sprite-0 hit
// (a poll loop plus a timed wait to its real split edge).  Journaled PPU
// writes drain at run() returns, so running scanline-sized slices bounds
// the write batch to one line; the scan stops at the first batch and its
// line becomes the render split row.  Mapper-IRQ games keep their splits
// from IRQ delivery instead, and a game that never writes costs at most
// max_split_scan_lines slices.
void Console::locate_span_split(int64_t base, int hit_line) {
  if (mapper_->irq_scanline(-1))
    return;
  ppu_->begin_span_watch(); // only post-hit writes can be the split batch
  const int last = std::min(hit_line + max_split_scan_lines, 238);
  int found = -1;
  for (int line = hit_line + 1; line <= last && found < 0; ++line) {
    run_until(std::min(base + scanline_cycle(line), base + vblank_start_cycles));
    if (ppu_->span_scroll_seen())
      found = line;
  }
  if (found < 0)
    return;
  // One guard slice: a $2006 pair can straddle the slice boundary.
  run_until(std::min(base + scanline_cycle(found + 1), base + vblank_start_cycles));
  // Slice attribution is +/- a line and games time split writes to take
  // effect at a tile-row boundary (Zelda writes during line ~61 for its
  // row-64 HUD edge): snap up to the next multiple of 8.
  ppu_->note_span_split((found + 7) & ~7);
}
Ppu::Frame Console::run_frame() {
  const int64_t base = frame_base_;
  bool fired = false;
  int last_irq = -1;
  for (int guard = 0; guard < max_frame_events; ++guard) {
    // Next event: the sprite-0 hit or the mapper IRQ, earliest first
    // (sprite-0 wins a tie -- the SMB-style poll precedes the handler).
    int line = -1, kind = -1;
    if (!fired && sprite0_scanline_ >= 0 && ppu_->rendering_enabled()) {
      line = sprite0_scanline_;
      kind = 0;
    }
    auto irq = mapper_->irq_scanline(last_irq);
    if (irq && *irq > last_irq && ppu_->rendering_enabled() && (kind < 0 || *irq < line)) {
      line = *irq;
      kind = 1;
    }
    if (kind < 0)
      break;
    const int64_t at = base + scanline_cycle(line);
    run_until(std::min(at, base + vblank_start_cycles));
    machine_.clear_read_lease();
    if (!kind) {
      fired = true;
      ppu_->set_sprite0_hit(true);
      locate_span_split(base, line);
      continue;
    }
    // Re-validate with the pre-event context: the delta chain's next
    // line is still computed from the PREVIOUS expiry.
    auto current = mapper_->irq_scanline(last_irq);
    last_irq = line;
    if (ppu_->rendering_enabled() && current && *current == line) {
      ppu_->mark_split(line);
      mapper_->notify_scanline_irq();
      set_irq(IrqSource::mapper, true);
      run_until(std::min(at + scanline_cycle(1), base + vblank_start_cycles));
    }
  }
  run_until(base + vblank_start_cycles);
  machine_.clear_read_lease();
  auto frame = ppu_->render_frame();
  sprite0_scanline_ = ppu_->sprite0_scanline();
  ppu_->begin_vblank();
  if (ppu_->nmi_enabled())
    machine_.set_nmi();
  run_until(base + frame_cycles);
  machine_.clear_read_lease();
  ppu_->end_vblank();
  frame_base_ += frame_cycles;
  ++frame_index_;
  last_audio_ =
      apu_->end_frame(frame_cycles, [this](int address) { return machine_.read_ram(address); });
  cycles_ += apu_->take_dma_stall();
  if (apu_->irq_pending())
    set_irq(IrqSource::apu, true);
  else
    set_irq(IrqSource::apu, false);
  return frame;
}
Ppu::Frame Console::run(int frames) {
  Ppu::Frame frame{};
  for (int i = 0; i < frames && !halted_; ++i)
    frame = run_frame();
  return frame;
}

} // namespace transformer_nes::nes
