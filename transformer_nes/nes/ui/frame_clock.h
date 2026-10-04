#pragma once

#include <algorithm>
#include <cmath>

namespace transformer_nes::ui {

// Wall time drives emulation; the display's smoothed cadence is only a
// rendering budget. Discarding overdue frames must retain sub-frame phase.
class FrameClock {
public:
  explicit FrameClock(double period) : period_(period) {}
  void advance(double elapsed) {
    accumulated_ += std::max(0.0, elapsed);
    if (accumulated_ >= 5 * period_)
      accumulated_ = 4 * period_ + std::fmod(accumulated_, period_);
  }
  bool ready() const { return accumulated_ >= period_; }
  void consume() { accumulated_ -= period_; }
  void discard_backlog() { accumulated_ = std::fmod(accumulated_, period_); }
  double remainder() const { return accumulated_; }

private:
  double period_, accumulated_ = 0.0;
};

} // namespace transformer_nes::ui
