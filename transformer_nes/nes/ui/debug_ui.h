// Tilde-toggled Dear ImGui debug overlay: frame stats, options, the
// transformer X-ray (nes/xray.py), and a hex inspector over the CPU
// address space.  The app owns the sokol window/pass; this module owns
// everything ImGui.
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

struct sapp_event;

namespace transformer_nes::ui {

using Counts = std::vector<std::pair<std::string, long long>>;

// The X-ray's view of the last frames.
struct XrayFrame {
  Counts cpu_paths;   // instructions retired per execution path
  Counts cpu_singles; // opcodes run one at a time, busiest first
  Counts ppu_models;  // 8-pixel slots per render model, busiest first
  std::string expert_map; // 240 x 32 render-model ids
  std::vector<std::string> expert_names;   // per model id - 1
  std::vector<std::uint32_t> expert_colors; // RGBA per model id (256)
  std::string read_heat; // 256 x 256 log-scaled attention reads per CPU address
  Counts hot_pages;      // busiest 256-byte pages, reads per frame
};

// One forward of a full model, layer by layer.
struct Dissection {
  std::string title;
  int layers = 0, d_model = 0;
  // (row, kind, name, detail, value): row 0 is the embedding, L + 1 the
  // output of layer L; kind is input/attention/ffn/linear; a memory
  // read's detail names the address it read.
  std::vector<std::tuple<int, std::string, std::string, std::string, double>> steps;
  std::vector<int> activity; // residual slots written by each layer
};

// Emulator-side hooks; every callback may call into Python (the frame
// loop holds the GIL on this thread).  The X-ray's CPU hooks exist only
// with the weights CPU, its pixel hooks only with the render graph.
struct DebugHooks {
  std::function<std::string(std::uint32_t start, std::uint32_t count)> read_mem;
  std::function<std::vector<std::pair<std::string, std::string>>()> stats;
  std::function<std::vector<std::pair<std::string, bool>>()> options;
  std::function<void(const std::string& name, bool value)> set_option;
  std::function<void()> step_frame;
  std::function<XrayFrame()> xray_frame;
  // The next instruction through the full CPU model.
  std::function<Dissection()> dissect;
  // Ask for the slot under (line, x) when it is next drawn, then poll.
  std::function<void(int line, int x)> trace_pixel;
  std::function<std::optional<Dissection>()> pixel_dissection;
  // Force a named render value to 0 in every slot (returns a status
  // line); restore undoes it.
  std::function<std::string(const std::string& name)> knock_out;
  std::function<void()> restore;
};

// Frontend-owned toggles the app reads back each frame.
struct UiState {
  bool open = false;
  // Keep the complete NES framebuffer visible by default. Some HUDs use the
  // first nametable column, which an 8-pixel overscan crop would remove.
  bool crop = false;
  bool xray = false;      // start with the X-ray section expanded
  bool tint_game = false; // the expert map over the game picture
  // Where the game picture lands, in ImGui points (set by the app).
  float game_x = 0, game_y = 0, game_w = 0, game_h = 0;
  // A click on the game picture (points), and the pixel it picked.
  bool click_pending = false;
  float click_x = 0, click_y = 0;
  int pick_line = -1, pick_x = -1;
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
