// Debug overlay implementation; also hosts the sokol_imgui glue
// (SOKOL_IMGUI_IMPL) so the ObjC/C sokol implementation unit stays
// ImGui-free.

#include "frontend/sokol_platform.h"

#include "sokol_app.h"
#include "sokol_gfx.h"
#include "sokol_log.h"

#include "imgui.h"

#define SOKOL_IMGUI_IMPL
#include "sokol_imgui.h"

#include "frontend/debug_ui.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace transformer_nes::ui {

namespace {

using std::uint32_t;

constexpr int frame_width = 256, frame_height = 240;
constexpr int slots_per_line = frame_width / 8;
constexpr int bytes_per_row = 16;
constexpr int address_rows = 0x10000 / bytes_per_row;
// A begun ImGui frame must be rendered even if the window's close
// button flipped state.open mid-draw.
bool frame_built = false;

void draw_hex_inspector(const DebugHooks& hooks) {
  if (!ImGui::CollapsingHeader("Memory (CPU $0000-$FFFF)") || !hooks.read_mem)
    return;
  static int jump_target = -1;
  static char jump_text[8] = "";
  ImGui::SetNextItemWidth(80.0f);
  if (ImGui::InputText("addr", jump_text, sizeof(jump_text),
                       ImGuiInputTextFlags_CharsHexadecimal |
                           ImGuiInputTextFlags_EnterReturnsTrue)) {
    jump_target = static_cast<int>(std::strtol(jump_text, nullptr, 16)) & 0xFFFF;
  }
  ImGui::SameLine();
  ImGui::TextDisabled("hex, enter to jump");
  ImGui::BeginChild("hexdump", ImVec2(0.0f, 320.0f), ImGuiChildFlags_Borders);
  if (jump_target >= 0) {
    ImGui::SetScrollY((jump_target / bytes_per_row) * ImGui::GetTextLineHeightWithSpacing());
    jump_target = -1;
  }
  ImGuiListClipper clipper;
  clipper.Begin(address_rows, ImGui::GetTextLineHeightWithSpacing());
  while (clipper.Step()) {
    const uint32_t start = static_cast<uint32_t>(clipper.DisplayStart) * bytes_per_row;
    const uint32_t count =
        static_cast<uint32_t>(clipper.DisplayEnd - clipper.DisplayStart) * bytes_per_row;
    const std::string block = hooks.read_mem(start, count);
    for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
      const auto* bytes = reinterpret_cast<const unsigned char*>(block.data()) +
                          (row - clipper.DisplayStart) * bytes_per_row;
      char line[96];
      int n = std::snprintf(line, sizeof(line), "%04X ", row * bytes_per_row);
      for (int i = 0; i < bytes_per_row; ++i)
        n += std::snprintf(line + n, sizeof(line) - static_cast<size_t>(n), " %02X", bytes[i]);
      n += std::snprintf(line + n, sizeof(line) - static_cast<size_t>(n), "  ");
      for (int i = 0; i < bytes_per_row; ++i)
        n += std::snprintf(line + n, sizeof(line) - static_cast<size_t>(n), "%c",
                           bytes[i] >= 0x20 && bytes[i] < 0x7F ? bytes[i] : '.');
      ImGui::TextUnformatted(line);
    }
  }
  ImGui::EndChild();
}

// ── Transformer X-ray ──

// The game picture on screen: frame pixels <-> ImGui points, with the
// overscan crop trimming 8 pixels off each edge.
struct GameView {
  float x0, y0, sx, sy, left, top;
  explicit GameView(const UiState& s)
      : x0(s.crop ? 8.0f : 0.0f), y0(s.crop ? 8.0f : 0.0f),
        sx(s.game_w / (frame_width - 2 * x0)), sy(s.game_h / (frame_height - 2 * y0)),
        left(s.game_x), top(s.game_y) {}
  ImVec2 to_screen(float px, float py) const {
    return ImVec2(left + (px - x0) * sx, top + (py - y0) * sy);
  }
  ImVec2 to_frame(float x, float y) const { return ImVec2((x - left) / sx + x0, (y - top) / sy + y0); }
};

// Paint the expert map (one cell per 8-pixel slot) with `cell(x, y)`
// giving a cell's top-left corner and `size` its extent.
template <class Cell>
void paint_expert_map(ImDrawList* draw, const XrayFrame& frame, Cell cell, ImVec2 size,
                      uint32_t alpha) {
  const auto* ids = reinterpret_cast<const unsigned char*>(frame.expert_map.data());
  for (int y = 0; y < frame_height; ++y)
    for (int x = 0; x < slots_per_line;) {
      const int id = ids[y * slots_per_line + x];
      int run = 1;
      while (x + run < slots_per_line && ids[y * slots_per_line + x + run] == id)
        ++run;
      const ImVec2 a = cell(x, y);
      const uint32_t color = (frame.expert_colors[static_cast<size_t>(id)] & ~IM_COL32_A_MASK) |
                             (alpha << IM_COL32_A_SHIFT);
      draw->AddRectFilled(a, ImVec2(a.x + run * size.x, a.y + size.y), color);
      x += run;
    }
}

bool has_expert_map(const XrayFrame& frame) {
  return frame.expert_map.size() == static_cast<size_t>(slots_per_line * frame_height) &&
         frame.expert_colors.size() == 256;
}

void draw_counts(const char* id, const Counts& counts, size_t limit) {
  long long total = 0;
  for (const auto& [name, n] : counts)
    total += n;
  if (total == 0) {
    ImGui::TextDisabled("(none yet)");
    return;
  }
  ImGui::PushID(id);
  for (size_t i = 0; i < counts.size() && i < limit; ++i) {
    const auto& [name, n] = counts[i];
    char label[96];
    std::snprintf(label, sizeof(label), "%s  %lld (%.1f%%)", name.c_str(), n,
                  100.0 * static_cast<double>(n) / static_cast<double>(total));
    ImGui::ProgressBar(static_cast<float>(n) / static_cast<float>(total), ImVec2(-1.0f, 0.0f),
                       label);
  }
  ImGui::PopID();
}

void draw_expert_map(const XrayFrame& frame) {
  if (!has_expert_map(frame))
    return;
  const ImVec2 cell(4.0f, 1.0f), origin = ImGui::GetCursorScreenPos();
  paint_expert_map(
      ImGui::GetWindowDrawList(), frame,
      [&](int x, int y) { return ImVec2(origin.x + x * cell.x, origin.y + y * cell.y); }, cell,
      255);
  ImGui::Dummy(ImVec2(slots_per_line * cell.x, frame_height * cell.y));
  if (!ImGui::IsItemHovered())
    return;
  const ImVec2 m = ImGui::GetMousePos();
  const int x = static_cast<int>((m.x - origin.x) / cell.x), y = static_cast<int>(m.y - origin.y);
  if (x < 0 || x >= slots_per_line || y < 0 || y >= frame_height)
    return;
  const int id = static_cast<unsigned char>(frame.expert_map[static_cast<size_t>(y * slots_per_line + x)]);
  const char* name = id == 255 ? "no output yet"
                     : id == 0 ? "full render graph"
                               : frame.expert_names[static_cast<size_t>(id - 1)].c_str();
  ImGui::SetTooltip("line %d, pixels %d-%d: %s", y, x * 8, x * 8 + 7, name);
}

// The expert map again, translucent over the game picture itself.
void tint_game(const XrayFrame& frame, const UiState& state) {
  if (!has_expert_map(frame) || state.game_w <= 0)
    return;
  const GameView view(state);
  ImDrawList* draw = ImGui::GetBackgroundDrawList();
  draw->PushClipRect(ImVec2(state.game_x, state.game_y),
                     ImVec2(state.game_x + state.game_w, state.game_y + state.game_h));
  paint_expert_map(
      draw, frame, [&](int x, int y) { return view.to_screen(x * 8.0f, static_cast<float>(y)); },
      ImVec2(8 * view.sx, view.sy), 110);
  draw->PopClipRect();
}

// Outline the picked slot and pixel on the game picture.
void mark_pick(const UiState& state) {
  const GameView view(state);
  const float slot = static_cast<float>(state.pick_x / 8 * 8), line = static_cast<float>(state.pick_line);
  const float px = static_cast<float>(state.pick_x);
  ImDrawList* draw = ImGui::GetBackgroundDrawList();
  draw->AddRect(view.to_screen(slot, line - 1), view.to_screen(slot + 8, line + 2),
                IM_COL32(255, 255, 255, 220));
  draw->AddRectFilled(view.to_screen(px, line), view.to_screen(px + 1, line + 1),
                      IM_COL32(255, 40, 200, 255));
}

ImVec4 kind_color(const std::string& kind) {
  if (kind == "attention")
    return ImVec4(0.35f, 0.85f, 1.0f, 1.0f);
  if (kind == "ffn")
    return ImVec4(1.0f, 0.8f, 0.3f, 1.0f);
  if (kind == "input")
    return ImVec4(0.9f, 0.9f, 0.9f, 1.0f);
  return ImVec4(0.65f, 0.65f, 0.7f, 1.0f);
}

// `knock_out`, when given, adds a button per computed value that ablates it.
void draw_dissection(const char* id, const Dissection& d,
                     const std::function<void(const std::string&)>& knock_out = {}) {
  ImGui::PushID(id);
  ImGui::TextWrapped("%s", d.title.c_str());
  ImGui::TextDisabled("%d layers x %d-wide residual stream; hardmax attention + ReGLU",
                      d.layers, d.d_model);
  std::vector<float> activity(d.activity.begin(), d.activity.end());
  ImGui::PlotHistogram("##activity", activity.data(), static_cast<int>(activity.size()), 0,
                       "residual slots written per layer", 0.0f, FLT_MAX, ImVec2(-1.0f, 60.0f));
  for (const char* kind : {"attention", "ffn", "linear", "input"}) {
    ImGui::TextColored(kind_color(kind), "%s", kind);
    ImGui::SameLine();
  }
  ImGui::NewLine();
  const int columns = knock_out ? 4 : 3;
  if (ImGui::BeginTable("steps", columns,
                        ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_BordersInnerV,
                        ImVec2(0.0f, 260.0f))) {
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("layer", ImGuiTableColumnFlags_WidthFixed, 44.0f);
    ImGui::TableSetupColumn("value");
    ImGui::TableSetupColumn("=", ImGuiTableColumnFlags_WidthFixed, 90.0f);
    if (knock_out)
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 72.0f);
    ImGui::TableHeadersRow();
    int row = 0;
    for (const auto& [layer, kind, name, detail, value] : d.steps) {
      ImGui::PushID(row++);
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      if (layer == 0)
        ImGui::TextDisabled("embed");
      else
        ImGui::Text("L%d", layer - 1);
      ImGui::TableNextColumn();
      ImGui::TextColored(kind_color(kind), "%s", name.c_str());
      if (!detail.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled("<- %s", detail.c_str());
      }
      ImGui::TableNextColumn();
      if (value == static_cast<double>(static_cast<long long>(value)) && value >= 0 && value < 65536)
        ImGui::Text(value < 256 ? "$%02llX" : "$%04llX", static_cast<long long>(value));
      else
        ImGui::Text("%.3g", value);
      if (knock_out) {
        ImGui::TableNextColumn();
        if (layer > 0 && ImGui::SmallButton("knock out"))
          knock_out(name);
      }
      ImGui::PopID();
    }
    ImGui::EndTable();
  }
  ImGui::PopID();
}

// Near black -> purple -> orange -> pale yellow, by intensity (0 = unread).
constexpr uint32_t heat_rgba(unsigned char v) {
  constexpr float stops[4][3] = {
      {0.10f, 0.05f, 0.25f}, {0.55f, 0.10f, 0.55f}, {0.95f, 0.45f, 0.10f}, {1.0f, 0.95f, 0.6f}};
  if (v == 0)
    return 0xFF0A0A0Au;
  const float t = v / 255.0f * 3.0f;
  const int i = std::min(2, static_cast<int>(t));
  const float f = t - static_cast<float>(i);
  uint32_t out = 0xFFu << 24;
  for (int c = 0; c < 3; ++c) {
    const float x = stops[i][c] + (stops[i + 1][c] - stops[i][c]) * f;
    out |= static_cast<uint32_t>(x * 255.0f) << (8 * c);
  }
  return out;
}

// Attention reads per CPU address: one texel per byte, a row per page.
void draw_read_heat(const XrayFrame& frame) {
  if (frame.read_heat.size() != 0x10000)
    return;
  static sg_image image{};
  static uint64_t texture = 0;
  static std::vector<uint32_t> rgba(0x10000);
  static std::string previous;
  static constexpr auto palette = [] {
    std::array<uint32_t, 256> colors{};
    for (size_t i = 0; i < colors.size(); ++i)
      colors[i] = heat_rgba(static_cast<unsigned char>(i));
    return colors;
  }();
  if (!texture) {
    sg_image_desc img{};
    img.usage.dynamic_update = true;
    img.width = img.height = 256;
    img.pixel_format = SG_PIXELFORMAT_RGBA8;
    img.label = "xray-read-heat";
    image = sg_make_image(&img);
    sg_view_desc view{};
    view.texture.image = image;
    sg_sampler_desc smp{};
    smp.min_filter = smp.mag_filter = SG_FILTER_NEAREST;
    texture = simgui_imtextureid_with_sampler(sg_make_view(&view), sg_make_sampler(&smp));
  }
  if (previous != frame.read_heat) {
    const auto* heat = reinterpret_cast<const unsigned char*>(frame.read_heat.data());
    for (size_t i = 0; i < rgba.size(); ++i)
      rgba[i] = palette[heat[i]];
    sg_image_data data{};
    data.mip_levels[0] = {rgba.data(), rgba.size() * sizeof(uint32_t)};
    sg_update_image(image, &data);
    previous = frame.read_heat;
  }
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  ImGui::Image(ImTextureRef(static_cast<ImTextureID>(texture)), ImVec2(256.0f, 256.0f));
  if (ImGui::IsItemHovered()) {
    const ImVec2 m = ImGui::GetMousePos();
    const int x = static_cast<int>(m.x - origin.x), y = static_cast<int>(m.y - origin.y);
    if (x >= 0 && x < 256 && y >= 0 && y < 256)
      ImGui::SetTooltip("$%04X", (y << 8) | x);
  }
}

void draw_pixel_dissection(const DebugHooks& hooks, const UiState& state) {
  if (state.pick_line < 0) {
    ImGui::TextDisabled("click any pixel on the game picture");
    return;
  }
  const auto d = hooks.pixel_dissection();
  if (!d) {
    ImGui::TextDisabled("tracing line %d (when it is next drawn)...", state.pick_line);
    return;
  }
  ImGui::TextDisabled("its tile slot through the full render graph (on a copy); the slot "
                      "draws 8 pixels and fetches the tile two slots ahead");
  static std::string ablation;
  if (!ablation.empty()) {
    ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f), "%s", ablation.c_str());
    ImGui::SameLine();
    if (ImGui::SmallButton("restore")) {
      hooks.restore();
      ablation.clear();
    }
  }
  draw_dissection("pixel", *d, [&](const std::string& name) { ablation = hooks.knock_out(name); });
}

// `fetched` receives the X-ray frame when the section drew it, so the
// game tint can reuse it.
void draw_xray(const DebugHooks& hooks, UiState& state, const XrayFrame*& fetched) {
  if (!hooks.xray_frame)
    return;
  if (state.xray)
    ImGui::SetNextItemOpen(true, ImGuiCond_Once);
  state.xray_visible = ImGui::CollapsingHeader("Transformer X-ray");
  if (!state.xray_visible)
    return;
  fetched = &hooks.xray_frame();
  const XrayFrame& frame = *fetched;
  const auto open = ImGuiTreeNodeFlags_DefaultOpen;
  if (!hooks.dissect)
    ImGui::TextDisabled("CPU views need the weights CPU (--backend weights)");
  if (hooks.dissect && ImGui::TreeNodeEx("CPU: how instructions ran", open)) {
    ImGui::TextDisabled("whole folded loops, folded shapes, or one forward each");
    draw_counts("paths", frame.cpu_paths, 8);
    if (!frame.cpu_singles.empty()) {
      ImGui::TextDisabled("busiest single-step opcodes");
      draw_counts("singles", frame.cpu_singles, 5);
    }
    ImGui::TreePop();
  }
  if (hooks.dissect && ImGui::TreeNodeEx("CPU memory: where attention reads", open)) {
    ImGui::TextDisabled("each texel one address ($0000 top left, a row per page)");
    draw_read_heat(frame);
    ImGui::SameLine();
    ImGui::BeginGroup();
    ImGui::TextDisabled("busiest pages, reads/frame");
    for (const auto& [page, n] : frame.hot_pages)
      ImGui::Text("%s  %lld", page.c_str(), n);
    ImGui::EndGroup();
    ImGui::TreePop();
  }
  if (hooks.dissect && ImGui::TreeNodeEx("CPU: dissect the next instruction")) {
    static std::optional<Dissection> last;
    if (ImGui::Button("Run it through the full model"))
      last = hooks.dissect();
    if (last)
      draw_dissection("cpu", *last);
    ImGui::TreePop();
  }
  if (ImGui::TreeNodeEx("PPU: which expert drew each tile slot", open)) {
    ImGui::Checkbox("Tint the game picture with it (stays on with the overlay closed)",
                    &state.tint_game);
    draw_expert_map(frame);
    ImGui::SameLine();
    ImGui::BeginGroup();
    draw_counts("models", frame.ppu_models, 8);
    ImGui::EndGroup();
    ImGui::TreePop();
  }
  if (hooks.pixel_dissection &&
      ImGui::TreeNodeEx("PPU: why is this pixel this color?", open)) {
    draw_pixel_dissection(hooks, state);
    ImGui::TreePop();
  }
}

void draw_window(UiState& state, const DebugHooks& hooks, const XrayFrame*& xray) {
  ImGui::SetNextWindowSize(ImVec2(560.0f, 640.0f), ImGuiCond_FirstUseEver);
  if (ImGui::Begin("debug (~)", &state.open)) {
    if (ImGui::CollapsingHeader("Stats", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Text("display: %.1f fps (%.2f ms)", ImGui::GetIO().Framerate,
                  1000.0f / ImGui::GetIO().Framerate);
      if (hooks.stats)
        for (const auto& [name, value] : hooks.stats())
          ImGui::Text("%s: %s", name.c_str(), value.c_str());
    }
    if (ImGui::CollapsingHeader("Options", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Checkbox("Overscan crop", &state.crop);
      if (hooks.options && hooks.set_option) {
        for (auto& [name, value] : hooks.options()) {
          bool checked = value;
          if (ImGui::Checkbox(name.c_str(), &checked))
            hooks.set_option(name, checked);
        }
      }
      if (hooks.step_frame) {
        ImGui::SameLine();
        if (ImGui::Button("Step frame"))
          hooks.step_frame();
      }
    }
    draw_xray(hooks, state, xray);
    draw_hex_inspector(hooks);
  }
  ImGui::End();
}

} // namespace

void setup() {
  simgui_desc_t desc{};
  desc.logger.func = slog_func;
  simgui_setup(&desc);
}

void shutdown() { simgui_shutdown(); }

bool handle_event(const sapp_event* event, UiState& state) {
  if (event->type == SAPP_EVENTTYPE_KEY_DOWN && event->key_code == SAPP_KEYCODE_GRAVE_ACCENT) {
    state.open = !state.open;
    return true;
  }
  if (!state.open)
    return false;
  if (event->type == SAPP_EVENTTYPE_KEY_DOWN && event->key_code == SAPP_KEYCODE_ESCAPE) {
    state.open = false;
    return true;
  }
  simgui_handle_event(event);
  // The game keeps playing under the overlay: keys reach it unless a
  // text field (the hex jump box) has focus, and the mouse unless it is
  // over a window, where a click picks a pixel to dissect instead.
  // Key-ups always pass, so no button sticks down when a field takes
  // focus mid-press.
  const ImGuiIO& io = ImGui::GetIO();
  if (event->type == SAPP_EVENTTYPE_MOUSE_DOWN && event->mouse_button == SAPP_MOUSEBUTTON_LEFT &&
      !io.WantCaptureMouse) {
    state.click_pending = true;
    state.click_x = event->mouse_x / sapp_dpi_scale();
    state.click_y = event->mouse_y / sapp_dpi_scale();
    return true;
  }
  if (event->type == SAPP_EVENTTYPE_KEY_UP)
    return false;
  if (event->type == SAPP_EVENTTYPE_KEY_DOWN || event->type == SAPP_EVENTTYPE_CHAR)
    return io.WantTextInput;
  return io.WantCaptureMouse;
}

void draw(UiState& state, const DebugHooks& hooks, double display_seconds) {
  // The game tint outlives the overlay: with it on, a frame is built even
  // while the window is closed, holding just the tint.
  state.xray_visible = false;
  const bool tint = state.tint_game && hooks.xray_frame;
  if (!state.open && !tint)
    return;
  frame_built = true;
  simgui_frame_desc_t frame{};
  frame.width = sapp_width();
  frame.height = sapp_height();
  frame.delta_time = std::max(display_seconds, 1e-6);
  frame.dpi_scale = sapp_dpi_scale();
  simgui_new_frame(&frame);

  if (std::exchange(state.click_pending, false) && state.game_w > 0 && hooks.trace_pixel) {
    const ImVec2 p = GameView(state).to_frame(state.click_x, state.click_y);
    const int x = static_cast<int>(p.x), y = static_cast<int>(p.y);
    if (x >= 0 && x < frame_width && y >= 0 && y < frame_height) {
      state.pick_x = x;
      state.pick_line = y;
      hooks.trace_pixel(y, x);
    }
  }
  const XrayFrame* xray = nullptr;
  if (state.open) {
    draw_window(state, hooks, xray);
    if (state.pick_line >= 0 && state.game_w > 0)
      mark_pick(state);
  }
  if (tint) {
    if (!xray)
      xray = &hooks.xray_frame();
    tint_game(*xray, state);
  }
}

void render(const UiState&) {
  if (frame_built) {
    simgui_render();
    frame_built = false;
  }
}

} // namespace transformer_nes::ui
