#include "nes/console.h"

#include <algorithm>
#include <utility>

namespace transformer_nes::nes {

namespace {

constexpr int64_t vblank_start_cycles = 27'394;
constexpr int64_t frame_cycles = 29'781;
constexpr int max_frame_events = 32;

constexpr int64_t scanline_cycle(int line) { return static_cast<int64_t>(line + 1) * 341 / 3; }

// Visible-span scanline containing a console-clock cycle, -1 outside.
int span_scanline(int64_t cycle, int64_t frame_base) {
  const int64_t rel = cycle - frame_base;
  if (rel < 0 || rel >= vblank_start_cycles)
    return -1;
  const int row = static_cast<int>(rel * 3 / 341);
  return row < 240 ? row : -1;
}

} // namespace

class Console::PpuDevice final : public Device {
public:
  PpuDevice(Ppu& ppu, const int64_t& frame_base) : ppu_(ppu), frame_base_(frame_base) {}
  int read(int address) override { return ppu_.read(address); }
  void write(int address, int value) override { ppu_.write(address, value); }
  // Stamped writes carry the scanline they executed on (-1 outside the
  // visible span); the PPU uses it for the mid-frame split row.
  void write_at(int address, int value, int64_t cycle) override {
    ppu_.write_stamped(address, value, span_scanline(cycle, frame_base_));
  }
  ReadLease read_lease(int a) override {
    auto v = ppu_.read_lease(a);
    return v ? ReadLease(*v) : ReadLease{};
  }

private:
  Ppu& ppu_;
  const int64_t& frame_base_;
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
  MapperDevice(Mapper& m, const int64_t& frame_base) : m_(m), frame_base_(frame_base) {}
  int read(int a) override { return m_.read(a); }
  void write(int a, int v) override { m_.write(a, v); }
  void write_at(int a, int v, int64_t cycle) override {
    m_.write_at(a, v, span_scanline(cycle, frame_base_));
  }
  void write_rmw(int a, int original, int value) override { m_.write_rmw(a, original, value); }

private:
  Mapper& m_;
  const int64_t& frame_base_;
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
  ppu_device_ = std::make_unique<PpuDevice>(*ppu_, frame_base_);
  apu_device_ = std::make_unique<ApuDevice>(*apu_);
  mapper_device_ = std::make_unique<MapperDevice>(*mapper_, frame_base_);
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
  callbacks.irq_changed = [this] { irq_sched_dirty_ = true; };
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
// A replan run stops early (cycles_ short of target) when a mapper
// write moved the IRQ schedule: the event-planning loop re-evaluates.
// Plain runs always complete -- cutting the vblank run short would
// slip the NMI's tail into the next frame's visible span.
void Console::run_until(int64_t target, bool replan) {
  if (halted_ || target <= cycles_)
    return;
  auto out = machine_.run_cycles(
      target - cycles_,
      [this, replan] {
        check_interrupts();
        const bool dirty = std::exchange(irq_sched_dirty_, false);
        return replan && dirty;
      },
      cycles_);
  cycles_ += out.cycles;
  halted_ = out.halted;
}
Ppu::Frame Console::run_frame() {
  const int64_t base = frame_base_;
  const int64_t span_end = base + vblank_start_cycles;
  mapper_->begin_frame();
  bool fired = false;
  int last_irq = -1;
  for (int guard = 0; guard < max_frame_events && cycles_ < span_end && !halted_; ++guard) {
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
    if (kind < 0) {
      // No event pending; a mid-frame arm stops this run early and
      // re-enters the planning loop.
      run_until(span_end, true);
      continue;
    }
    const int64_t at = std::min(base + scanline_cycle(line), span_end);
    run_until(at, true);
    if (cycles_ < at)
      continue; // IRQ schedule moved mid-run; re-plan
    machine_.clear_read_lease();
    if (!kind) {
      fired = true;
      ppu_->set_sprite0_hit(true);
      // Only post-hit scroll writes can be the game's split batch; their
      // rows arrive exactly via the stamped write path.
      ppu_->begin_span_watch();
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
      run_until(std::min(at + scanline_cycle(1), span_end));
    }
  }
  run_until(span_end);
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
