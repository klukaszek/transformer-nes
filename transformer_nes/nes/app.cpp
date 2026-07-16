// Native NES app: a Sokol window over the console frame loop.  The
// window, input, and one textured quad live here; each display frame
// calls back into Python (Console.run_frame via app.py) at the NTSC
// rate and blits the returned 256x240 RGB frame.  macOS/Metal only for
// now; compiled as Objective-C++ and cached by content digest.

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>

#define SOKOL_METAL
#define SOKOL_NO_ENTRY
#include "sokol_app.h"
#include "sokol_audio.h"
#include "sokol_gfx.h"
#include "sokol_glue.h"
#include "sokol_log.h"

namespace py = pybind11;

namespace {

using std::array;
using std::string;
using std::uint32_t;
using std::uint8_t;
using std::unique_ptr;

constexpr int frame_width = 256;
constexpr int frame_height = 240;
// Overscan-cropped presentation (8-pixel border, as on a CRT).
constexpr int view_width = 240;
constexpr int view_height = 224;
// NTSC: 29,781 CPU cycles per frame at 1.789773 MHz.
constexpr double nes_frame_seconds = 29'781.0 / 1'789'772.727272727;

// Button bit i == console.BUTTONS[i].
enum Button : uint32_t {
  BTN_A = 1 << 0,
  BTN_B = 1 << 1,
  BTN_SELECT = 1 << 2,
  BTN_START = 1 << 3,
  BTN_UP = 1 << 4,
  BTN_DOWN = 1 << 5,
  BTN_LEFT = 1 << 6,
  BTN_RIGHT = 1 << 7,
};

constexpr char vertex_shader_source[] = R"(
#include <metal_stdlib>
using namespace metal;
struct vs_out { float4 pos [[position]]; float2 uv; };
vertex vs_out vs_main(uint vid [[vertex_id]]) {
    vs_out o;
    float2 uv = float2(float((vid << 1) & 2), float(vid & 2));
    o.pos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    o.uv = uv;
    return o;
}
)";

constexpr char fragment_shader_source[] = R"(
#include <metal_stdlib>
using namespace metal;
struct vs_out { float4 pos [[position]]; float2 uv; };
fragment float4 fs_main(vs_out in [[stage_in]],
                        texture2d<float> tex [[texture(0)]],
                        sampler smp [[sampler(0)]]) {
    // NTSC overscan: crop the 8-pixel border a CRT never showed (games
    // mask the left column and stream tiles through the right one).
    return tex.sample(smp, float2((8.0 + in.uv.x * 240.0) / 256.0,
                                  (8.0 + in.uv.y * 224.0) / 240.0));
}
)";

struct AppState {
  py::object frame_cb;
  py::object quit_cb;
  unique_ptr<py::error_already_set> error;
  uint32_t buttons = 0;
  double acc = 0.0;
  bool have_frame = false;
  array<uint8_t, 64 * 3> palette{};
  array<uint8_t, frame_height * frame_width * 4> rgba{};
  sg_image image{};
  sg_view view{};
  sg_sampler sampler{};
  sg_pipeline pipeline{};
};
AppState* state = nullptr;

void app_init() {
  sg_desc desc{};
  desc.environment = sglue_environment();
  desc.logger.func = slog_func;
  sg_setup(&desc);

  sg_image_desc img{};
  img.usage.dynamic_update = true;
  img.width = frame_width;
  img.height = frame_height;
  img.pixel_format = SG_PIXELFORMAT_RGBA8;
  img.label = "nes-framebuffer";
  state->image = sg_make_image(&img);

  sg_view_desc view{};
  view.texture.image = state->image;
  state->view = sg_make_view(&view);

  sg_sampler_desc smp{};
  smp.min_filter = SG_FILTER_NEAREST;
  smp.mag_filter = SG_FILTER_NEAREST;
  smp.wrap_u = SG_WRAP_CLAMP_TO_EDGE;
  smp.wrap_v = SG_WRAP_CLAMP_TO_EDGE;
  state->sampler = sg_make_sampler(&smp);

  sg_shader_desc shd{};
  shd.vertex_func.source = vertex_shader_source;
  shd.vertex_func.entry = "vs_main";
  shd.fragment_func.source = fragment_shader_source;
  shd.fragment_func.entry = "fs_main";
  shd.views[0].texture.stage = SG_SHADERSTAGE_FRAGMENT;
  shd.views[0].texture.image_type = SG_IMAGETYPE_2D;
  shd.views[0].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
  shd.views[0].texture.msl_texture_n = 0;
  shd.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
  shd.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
  shd.samplers[0].msl_sampler_n = 0;
  shd.texture_sampler_pairs[0].stage = SG_SHADERSTAGE_FRAGMENT;
  shd.texture_sampler_pairs[0].view_slot = 0;
  shd.texture_sampler_pairs[0].sampler_slot = 0;
  shd.label = "nes-blit";

  sg_pipeline_desc pip{};
  pip.shader = sg_make_shader(&shd);
  pip.label = "nes-blit";
  state->pipeline = sg_make_pipeline(&pip);
}

// One emulated frame: cb(buttons) -> (240, 256) uint8 palette indices, or None.
bool emulate_frame() {
  py::object out = state->frame_cb(state->buttons);
  if (out.is_none()) {
    sapp_request_quit();
    return false;
  }
  auto arr = py::cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(out);
  if (arr.ndim() != 2 || arr.shape(0) != frame_height || arr.shape(1) != frame_width) {
    throw std::runtime_error("frame callback must return (240, 256) uint8 indices");
  }
  const uint8_t* src = arr.data();
  uint8_t* dst = state->rgba.data();
  for (int i = 0; i < frame_width * frame_height; ++i) {
    const uint8_t* color = state->palette.data() + (src[i] & 0x3F) * 3;
    dst[4 * i + 0] = color[0];
    dst[4 * i + 1] = color[1];
    dst[4 * i + 2] = color[2];
    dst[4 * i + 3] = 0xFF;
  }
  return true;
}

void app_frame() {
  state->acc += sapp_frame_duration();
  // A stall (window drag, machine handoff) must not queue a burst.
  if (state->acc > 4 * nes_frame_seconds) {
    state->acc = 4 * nes_frame_seconds;
  }
  // Wall-clock budget per callback: when the backend cannot keep up
  // with real time, drop the backlog (the game slows down) instead of
  // saturating the main thread -- the event loop must always breathe.
  // Half the display cadence, so a 120 Hz callback keeps headroom too.
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                            std::chrono::duration<double>(0.5 * sapp_frame_duration()));
  bool fresh = false;
  while (state->acc >= nes_frame_seconds && !state->error) {
    state->acc -= nes_frame_seconds;
    try {
      if (!emulate_frame())
        break;
      fresh = true;
    } catch (py::error_already_set& e) {
      std::fprintf(stderr, "frame callback raised:\n%s\n", e.what());
      state->error = std::make_unique<py::error_already_set>(std::move(e));
      sapp_request_quit();
    }
    if (std::chrono::steady_clock::now() > deadline) {
      state->acc = 0.0;
      break;
    }
  }
  if (fresh) {
    sg_image_data data{};
    data.mip_levels[0] = {state->rgba.data(), state->rgba.size()};
    sg_update_image(state->image, &data);
    state->have_frame = true;
  }

  sg_pass pass{};
  pass.swapchain = sglue_swapchain();
  pass.action.colors[0].load_action = SG_LOADACTION_CLEAR;
  pass.action.colors[0].clear_value = {0.0f, 0.0f, 0.0f, 1.0f};
  sg_begin_pass(&pass);
  if (state->have_frame) {
    // Letterbox to the cropped frame's 256:224.
    int w = sapp_width(), h = sapp_height();
    int vw = w, vh = w * view_height / view_width;
    if (vh > h) {
      vh = h;
      vw = h * view_width / view_height;
    }
    sg_apply_viewport((w - vw) / 2, (h - vh) / 2, vw, vh, true);
    sg_apply_pipeline(state->pipeline);
    sg_bindings bind{};
    bind.views[0] = state->view;
    bind.samplers[0] = state->sampler;
    sg_apply_bindings(&bind);
    sg_draw(0, 3, 1);
  }
  sg_end_pass();
  sg_commit();
}

void app_event(const sapp_event* e) {
  if (e->type != SAPP_EVENTTYPE_KEY_DOWN && e->type != SAPP_EVENTTYPE_KEY_UP) {
    return;
  }
  if (e->type == SAPP_EVENTTYPE_KEY_DOWN && e->key_code == SAPP_KEYCODE_ESCAPE) {
    sapp_request_quit();
    return;
  }
  std::uint32_t bit = 0;
  switch (e->key_code) {
  case SAPP_KEYCODE_X:
    bit = BTN_A;
    break;
  case SAPP_KEYCODE_Z:
    bit = BTN_B;
    break;
  case SAPP_KEYCODE_TAB:
    bit = BTN_SELECT;
    break;
  case SAPP_KEYCODE_ENTER:
    bit = BTN_START;
    break;
  case SAPP_KEYCODE_UP:
    bit = BTN_UP;
    break;
  case SAPP_KEYCODE_DOWN:
    bit = BTN_DOWN;
    break;
  case SAPP_KEYCODE_LEFT:
    bit = BTN_LEFT;
    break;
  case SAPP_KEYCODE_RIGHT:
    bit = BTN_RIGHT;
    break;
  default:
    return;
  }
  if (e->type == SAPP_EVENTTYPE_KEY_DOWN) {
    state->buttons |= bit;
  } else {
    state->buttons &= ~bit;
  }
}

// Open the audio device (push mode, mono) and return the actual sample
// rate, which the console's APU must match.  Safe to call before the
// window opens -- sokol_audio is independent of sokol_app.
int audio_setup() {
  saudio_desc desc{};
  desc.sample_rate = 44100;
  desc.num_channels = 1;
  desc.logger.func = slog_func;
  saudio_setup(&desc);
  return saudio_isvalid() ? saudio_sample_rate() : 0;
}

// Push one frame's mono samples into the audio buffer; sokol's own audio
// thread drains them (saudio_push is internally locked).
void push_audio(py::array_t<float, py::array::c_style | py::array::forcecast> samples) {
  if (saudio_isvalid() && samples.size() > 0) {
    saudio_push(samples.data(), static_cast<int>(samples.size()));
  }
}

// On macOS quitting goes through [NSApp terminate:], which exits the
// process inside sapp_run -- the caller of run() never resumes.  The
// quit callback, invoked here (GIL still held, same thread), is the
// last Python to execute on every quit path.
void app_cleanup() {
  if (state && !state->quit_cb.is_none()) {
    try {
      state->quit_cb();
    } catch (py::error_already_set& e) {
      e.discard_as_unraisable("nes app quit callback");
    }
  }
  saudio_shutdown();
  sg_shutdown();
}

// Blocks until the window closes.  The callbacks run on this thread
// inside sapp_run, so the GIL held by this call covers every frame.
void run(const string& title, int scale, py::object frame_cb,
         py::array_t<uint8_t, py::array::c_style | py::array::forcecast> palette,
         py::object quit_cb) {
  if (palette.ndim() != 2 || palette.shape(0) != 64 || palette.shape(1) != 3)
    throw std::runtime_error("palette must have shape (64, 3)");
  if (scale <= 0)
    throw std::invalid_argument("window scale must be positive");
  if (!PyCallable_Check(frame_cb.ptr()))
    throw std::invalid_argument("frame callback must be callable");
  AppState st;
  st.frame_cb = frame_cb;
  st.quit_cb = quit_cb;
  std::memcpy(st.palette.data(), palette.data(), st.palette.size());
  state = &st;
  sapp_desc desc{};
  desc.init_cb = app_init;
  desc.frame_cb = app_frame;
  desc.event_cb = app_event;
  desc.cleanup_cb = app_cleanup;
  desc.width = view_width * scale;
  desc.height = view_height * scale;
  desc.window_title = title.c_str();
  desc.high_dpi = true;
  desc.swap_interval = 1;
  desc.logger.func = slog_func;
  sapp_run(&desc);
  state = nullptr;
  st.frame_cb = py::object();
  if (st.error) {
    throw *st.error;
  }
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, py::arg("title"), py::arg("scale"), py::arg("frame_cb"), py::arg("palette"),
        py::arg("quit_cb") = py::none(),
        "Open the window and drive frame_cb(buttons) until quit; on macOS "
        "the process exits inside run, with quit_cb the last Python to run.");
  m.def("audio_setup", &audio_setup,
        "Open the audio device (mono push mode); returns the sample rate.");
  m.def("push_audio", &push_audio, py::arg("samples"),
        "Push a frame's mono float samples to the audio buffer.");
}
