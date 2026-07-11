#!/bin/sh
# Fetch the vendored native-app dependencies at pinned revisions.
# Re-run after bumping a pin; the files land next to this script.
set -eu

SOKOL_COMMIT=3743ea681fce95afd7bee41511cbe51480a046e5   # 2026-07-08
IMGUI_TAG=v1.92.8                                       # 2026-05-12

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
         imgui_tables.cpp imgui_widgets.cpp imgui_demo.cpp LICENSE.txt; do
    curl -sf "$imgui_raw/$f" -o "imgui/$f"
    echo "imgui/$f"
done
