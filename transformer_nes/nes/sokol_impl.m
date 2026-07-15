// Sokol's macOS implementation imports legacy Apple SDK headers that are not
// valid C++26. Keep that platform-only code in Objective-C; app.cpp remains a
// C++26 translation unit and consumes only Sokol's C declarations.

#define SOKOL_IMPL
#define SOKOL_METAL
#define SOKOL_NO_ENTRY
#include "sokol_app.h"
#include "sokol_audio.h"
#include "sokol_gfx.h"
#include "sokol_glue.h"
#include "sokol_log.h"
