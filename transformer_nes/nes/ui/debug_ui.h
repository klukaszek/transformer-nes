// Tilde-toggled Dear ImGui debug overlay: QoL toggles, frame stats, and
// a hex inspector over the CPU address space.  The app owns the sokol
// window/pass; this module owns everything ImGui.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

struct sapp_event;

namespace transformer_nes::ui {

// Emulator-side hooks; every callback may call into Python (the frame
// loop holds the GIL on this thread).
struct DebugHooks {
  std::function<std::string(std::uint32_t start, std::uint32_t count)> read_mem;
  std::function<std::vector<std::pair<std::string, std::string>>()> stats;
  std::function<std::vector<std::pair<std::string, bool>>()> options;
  std::function<void(const std::string& name, bool value)> set_option;
  std::function<void()> step_frame;
};

// Frontend-owned toggles the app reads back each frame.
struct UiState {
  bool open = false;
  // Keep the complete NES framebuffer visible by default. Some HUDs use the
  // first nametable column, which an 8-pixel overscan crop would remove.
  bool crop = false;
};

void setup();    // after sg_setup
void shutdown(); // before sg_shutdown
// True when the UI consumed the event (game input must skip it).
bool handle_event(const sapp_event* event, UiState& state);
// Build the overlay's frame; call once per frame BEFORE the render pass.
void draw(UiState& state, const DebugHooks& hooks);
// Submit the draw data; call inside the active render pass, after draw().
void render(const UiState& state);

} // namespace transformer_nes::ui
