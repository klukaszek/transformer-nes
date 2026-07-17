// One backend selection shared by every sokol TU (app.cpp, debug_ui.cpp,
// and the C/ObjC implementation unit): Metal on Apple, desktop GL
// elsewhere.  Shader sources in app.cpp branch on the same defines.
#pragma once

#if defined(__APPLE__)
#define SOKOL_METAL
#else
#define SOKOL_GLCORE
#endif
#define SOKOL_NO_ENTRY
