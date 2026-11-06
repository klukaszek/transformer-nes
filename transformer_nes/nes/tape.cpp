#include "nes/tape.h"

#include "nes/console.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <time.h>

namespace transformer_nes::nes {

namespace {

[[noreturn]] void throw_tape_error(string_view message, int line) {
  throw std::runtime_error(string(message) + " on line " + std::to_string(line));
}

double thread_seconds() {
#ifdef CLOCK_THREAD_CPUTIME_ID
  timespec stamp;
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &stamp) == 0)
    return stamp.tv_sec + stamp.tv_nsec * 1e-9;
#endif
  return -1.0;
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
  return {std::move(events)};
}

uint8_t InputTape::buttons_at(int frame) const {
  auto next =
      std::upper_bound(events.begin(), events.end(), frame,
                       [](int value, const TapeEvent& event) { return value < event.frame; });
  return next == events.begin() ? 0 : std::prev(next)->buttons;
}

TapeBenchmark replay_tape(Console& console, const InputTape& tape, int tail_frames,
                          int max_frames) {
  if (tail_frames < 0 || max_frames < 0)
    throw std::invalid_argument("frame counts cannot be negative");
  const int start_frame = console.frame_index;
  int stop_frame = tape.last_frame() + 1 + tail_frames;
  if (max_frames)
    stop_frame = std::min(stop_frame, start_frame + max_frames);
  TapeBenchmark result;
  result.frame_seconds.reserve(std::max(0, stop_frame - start_frame));
  result.frame_cpu_seconds.reserve(std::max(0, stop_frame - start_frame));
  const auto start = std::chrono::steady_clock::now();
  while (!console.halted && console.frame_index < stop_frame) {
    const uint8_t buttons = tape.buttons_at(console.frame_index);
    for (int button = 0; button < 8; ++button)
      console.set_button(button, (buttons >> button) & 1);
    const double cpu_start = thread_seconds();
    const auto frame_start = std::chrono::steady_clock::now();
    console.run_frame();
    const auto frame_end = std::chrono::steady_clock::now();
    const double cpu_end = thread_seconds();
    result.frame_seconds.push_back(
        std::chrono::duration<double>(frame_end - frame_start).count());
    if (cpu_start >= 0.0 && cpu_end >= cpu_start)
      result.frame_cpu_seconds.push_back(cpu_end - cpu_start);
  }
  result.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  result.frames = console.frame_index - start_frame;
  result.halted = console.halted;
  return result;
}

} // namespace transformer_nes::nes
