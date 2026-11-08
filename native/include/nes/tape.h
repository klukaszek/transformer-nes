#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace transformer_nes::nes {

using std::string;
using std::string_view;
using std::uint8_t;
using std::vector;

class Console;

struct TapeEvent {
  int frame = 0;
  uint8_t buttons = 0;
};

struct InputTape {
  vector<TapeEvent> events;

  static InputTape load(const string& path);
  static InputTape parse(string_view text);
  uint8_t buttons_at(int frame) const;
  int last_frame() const { return events.empty() ? -1 : events.back().frame; }
};

struct TapeBenchmark {
  int frames = 0;
  double seconds = 0.0;
  bool halted = false;
  vector<double> frame_seconds;
  vector<double> frame_cpu_seconds;
};

TapeBenchmark replay_tape(Console& console, const InputTape& tape, int tail_frames = 120,
                          int max_frames = 0);

} // namespace transformer_nes::nes
