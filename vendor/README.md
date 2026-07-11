# vendor

Native-app dependencies, fetched at pinned revisions by
`./install_deps.sh` (pins live at the top of the script).

- `sokol/` — single-file headers (zlib): `sokol_app.h` window + input,
  `sokol_gfx.h` the framebuffer quad, `sokol_glue.h`, `sokol_log.h`,
  `sokol_audio.h` (unused until the APU exists), `sokol_imgui.h` for the
  debug overlay.
- `imgui/` — Dear ImGui (MIT), the debug-overlay dependency of
  `sokol_imgui.h`.

Consumed by `transformer_nes/nes/app.cpp` — the one implementation TU
(`SOKOL_IMPL`), compiled as Objective-C++ with the Metal backend on
macOS and digest-cached like every other native piece.
