#!/bin/sh
# Fetch the vendored native-app dependencies at pinned revisions.
# Re-run after bumping a pin; the files land next to this script.
set -eu

SOKOL_COMMIT=3743ea681fce95afd7bee41511cbe51480a046e5   # 2026-07-08
IMGUI_TAG=v1.92.8                                       # 2026-05-12
PYBIND11_TAG=v3.0.1                                     # torch's bundled version

cd "$(dirname "$0")"

sokol_raw="https://raw.githubusercontent.com/floooh/sokol/$SOKOL_COMMIT"
mkdir -p sokol
for f in sokol_app.h sokol_gfx.h sokol_glue.h sokol_log.h sokol_audio.h \
         util/sokol_imgui.h LICENSE; do
    curl -sf "$sokol_raw/$f" -o "sokol/$(basename "$f")"
    echo "sokol/$(basename "$f")"
done

imgui_raw="https://raw.githubusercontent.com/ocornut/imgui/$IMGUI_TAG"
mkdir -p imgui
for f in imgui.h imgui_internal.h imconfig.h imstb_rectpack.h \
         imstb_textedit.h imstb_truetype.h imgui.cpp imgui_draw.cpp \
         imgui_tables.cpp imgui_widgets.cpp LICENSE.txt; do
    curl -sf "$imgui_raw/$f" -o "imgui/$f"
    echo "imgui/$f"
done

# pybind11 is header-only; the Makefile compiles the NES extensions
# against this tree (pinned to torch's bundled version) and clangd
# resolves against it too.
pybind11_dir="pybind11-${PYBIND11_TAG#v}"
curl -sfL "https://github.com/pybind/pybind11/archive/refs/tags/$PYBIND11_TAG.tar.gz" | tar xz
rm -rf pybind11
mkdir -p pybind11
mv "$pybind11_dir/include" pybind11/include
mv "$pybind11_dir/LICENSE" pybind11/LICENSE
rm -rf "$pybind11_dir"
echo "pybind11/include"
