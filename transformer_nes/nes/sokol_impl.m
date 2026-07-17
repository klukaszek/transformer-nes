// Sokol's macOS implementation imports legacy Apple SDK headers that are not
// valid C++26. Keep that platform-only code in Objective-C; app.cpp remains a
// C++26 translation unit and consumes only Sokol's C declarations.  On
// non-Apple platforms the Makefile compiles this same file as plain C
// and the shared platform header selects the GL backend.

#define SOKOL_IMPL
#include "ui/sokol_platform.h"
#include "sokol_app.h"
#include "sokol_audio.h"
#include "sokol_gfx.h"
#include "sokol_glue.h"
#include "sokol_log.h"
