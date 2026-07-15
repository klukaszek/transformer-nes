#include "tape.h"

#include "console.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace transformer_nes::nes {

namespace {

[[noreturn]] void throw_tape_error(string_view message, int line) {
  throw std::runtime_error(string(message) + " on line " + std::to_string(line));
}

} // namespace

InputTape InputTape::load(const string& path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("could not open input tape: " + path);
  const string text{std::istreambuf_iterator<char>(input), {}};
  return parse(text);
}

InputTape InputTape::parse(string_view text) {
  std::istringstream input{string(text)};
  vector<TapeEvent> events;
  string line;
  int line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    std::istringstream fields(line);
    int frame;
    string mask;
    if (!(fields >> frame))
      continue;
    if (frame < 0 || !(fields >> mask))
      throw_tape_error("invalid input tape", line_number);
    unsigned int value = 0;
    const auto [end, error] = std::from_chars(mask.data(), mask.data() + mask.size(), value, 16);
    if (error != std::errc{} || end != mask.data() + mask.size() || value > 0xFF)
      throw_tape_error("invalid input tape mask", line_number);
    events.push_back({frame, static_cast<uint8_t>(value)});
  }
  if (!std::is_sorted(events.begin(), events.end(),
                      [](const auto& a, const auto& b) { return a.frame < b.frame; }))
    throw std::runtime_error("input tape frames must be ordered");
  return InputTape(std::move(events));
}

uint8_t InputTape::buttons_at(int frame) const {
  auto next =
      std::upper_bound(events_.begin(), events_.end(), frame,
                       [](int value, const TapeEvent& event) { return value < event.frame; });
  return next == events_.begin() ? 0 : std::prev(next)->buttons;
}

TapeBenchmark replay_tape(Console& console, const InputTape& tape, int tail_frames,
                          int max_frames) {
  if (tail_frames < 0 || max_frames < 0)
    throw std::invalid_argument("frame counts cannot be negative");
  const int start_frame = console.frame_index();
  int stop_frame = tape.last_frame() + 1 + tail_frames;
  if (max_frames)
    stop_frame = std::min(stop_frame, start_frame + max_frames);
  const auto start = std::chrono::steady_clock::now();
  while (!console.halted() && console.frame_index() < stop_frame) {
    const uint8_t buttons = tape.buttons_at(console.frame_index());
    for (int button = 0; button < 8; ++button)
      console.set_button(button, (buttons >> button) & 1);
    console.run_frame();
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return {console.frame_index() - start_frame, seconds, console.halted()};
}

} // namespace transformer_nes::nes
