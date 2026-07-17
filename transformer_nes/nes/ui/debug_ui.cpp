// Debug overlay implementation; also hosts the sokol_imgui glue
// (SOKOL_IMGUI_IMPL) so the ObjC/C sokol implementation unit stays
// ImGui-free.

#include "ui/sokol_platform.h"

#include "sokol_app.h"
#include "sokol_gfx.h"
#include "sokol_log.h"

#include "imgui.h"

#define SOKOL_IMGUI_IMPL
#include "sokol_imgui.h"

#include "ui/debug_ui.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace transformer_nes::ui {

namespace {

using std::uint32_t;

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
  // While the overlay is open the keyboard belongs to it; mouse events
  // pass through only when ImGui does not want them.
  if (event->type == SAPP_EVENTTYPE_KEY_DOWN || event->type == SAPP_EVENTTYPE_KEY_UP ||
      event->type == SAPP_EVENTTYPE_CHAR)
    return true;
  return ImGui::GetIO().WantCaptureMouse;
}

void draw(UiState& state, const DebugHooks& hooks) {
  if (!state.open)
    return;
  frame_built = true;
  simgui_frame_desc_t frame{};
  frame.width = sapp_width();
  frame.height = sapp_height();
  frame.delta_time = sapp_frame_duration();
  frame.dpi_scale = sapp_dpi_scale();
  simgui_new_frame(&frame);

  ImGui::SetNextWindowSize(ImVec2(520.0f, 460.0f), ImGuiCond_FirstUseEver);
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
    draw_hex_inspector(hooks);
  }
  ImGui::End();
}

void render(const UiState&) {
  if (frame_built) {
    simgui_render();
    frame_built = false;
  }
}

} // namespace transformer_nes::ui
