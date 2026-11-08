# Native NES build: the two Python extension modules (hardware console,
# sokol frontend) plus the native hardware test, all into bin/.  Sources
# live under native/: console/ (devices), frontend/ (window and overlay),
# reference/ (the conventional PPU test oracle) and tests/.  frontend/app.py
# uses build.py to run `make` and import the artifacts; transformer kernel
# compilation (.cache/engine) is separate and stays torch-driven.
#
# pybind11 is the vendored copy (same version torch bundles); Python.h
# comes from the interpreter invoking make (PYTHON=... from app.py).

# Default to the project venv so a bare `make` matches what app.py
# builds (app.py passes PYTHON=sys.executable explicitly).
PYTHON ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)
UNAME := $(shell uname -s)

NATIVE := native
CPU_ABI := transformer_nes/cpu6502/native
VENDOR := vendor
BIN := bin
OBJ := $(BIN)/obj

PY_INCLUDE := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_paths()["include"])')

CXX ?= c++
CC ?= cc
CXXFLAGS := -O2 -std=c++20 -Wall -Wextra -Wpedantic -fPIC -fvisibility=hidden
NES_INCLUDES := -I$(NATIVE) -I$(NATIVE)/include -I$(CPU_ABI)
PY_INCLUDES := -I$(VENDOR)/pybind11/include -I$(PY_INCLUDE)
UI_INCLUDES := -I$(VENDOR)/sokol -I$(VENDOR)/imgui

ifeq ($(UNAME),Darwin)
LDSHARED := $(CXX) -shared -undefined dynamic_lookup
FRONTEND_LIBS := -framework Cocoa -framework QuartzCore -framework Metal \
                 -framework MetalKit -framework AudioToolbox
SOKOL_LANG := objective-c
else
LDSHARED := $(CXX) -shared
FRONTEND_LIBS := -lGL -lX11 -lXi -lXcursor -lasound -ldl -lpthread -lm
SOKOL_LANG := c
endif

HW_OBJS := $(addprefix $(OBJ)/,bindings.o bus.o ppu.o apu.o console.o tape.o)
REF_OBJ := $(BIN)/reference_obj
REF_OBJS := $(HW_OBJS:$(OBJ)/%=$(REF_OBJ)/%)
IMGUI_OBJS := $(addprefix $(OBJ)/,imgui.o imgui_draw.o imgui_tables.o imgui_widgets.o)
FRONTEND_OBJS := $(OBJ)/app.o $(OBJ)/debug_ui.o $(OBJ)/sokol_impl.o $(IMGUI_OBJS)

.PHONY: all hardware frontend reference-hardware test-native clean
all: hardware frontend
hardware: $(BIN)/nes_hardware.so
frontend: $(BIN)/nes_app.so
reference-hardware: $(BIN)/nes_hardware_reference.so

$(OBJ):
	mkdir -p $(OBJ)

$(REF_OBJ):
	mkdir -p $(REF_OBJ)

# The conventional PPU exists only as an independent test oracle.
$(REF_OBJ)/%.o: $(NATIVE)/console/%.cpp Makefile | $(REF_OBJ)
	$(CXX) $(CXXFLAGS) $(NES_INCLUDES) $(PY_INCLUDES) -DTRANSFORMER_NES_PPU_REFERENCE -MMD -MP -c $< -o $@

$(REF_OBJ)/bindings.o: $(NATIVE)/bindings.cpp Makefile | $(REF_OBJ)
	$(CXX) $(CXXFLAGS) $(NES_INCLUDES) $(PY_INCLUDES) -DTRANSFORMER_NES_PPU_REFERENCE -MMD -MP -c $< -o $@

$(OBJ)/%.o: $(NATIVE)/console/%.cpp Makefile | $(OBJ)
	$(CXX) $(CXXFLAGS) $(NES_INCLUDES) $(PY_INCLUDES) -MMD -MP -c $< -o $@

$(OBJ)/bindings.o: $(NATIVE)/bindings.cpp Makefile | $(OBJ)
	$(CXX) $(CXXFLAGS) $(NES_INCLUDES) $(PY_INCLUDES) -MMD -MP -c $< -o $@

$(OBJ)/app.o: $(NATIVE)/frontend/app.cpp Makefile | $(OBJ)
	$(CXX) $(CXXFLAGS) $(NES_INCLUDES) $(PY_INCLUDES) $(UI_INCLUDES) -MMD -MP -c $< -o $@

$(OBJ)/debug_ui.o: $(NATIVE)/frontend/debug_ui.cpp Makefile | $(OBJ)
	$(CXX) $(CXXFLAGS) $(NES_INCLUDES) $(UI_INCLUDES) -MMD -MP -c $< -o $@

# Vendored code builds warning-silent; its style is not ours to fix.
# (Plain % rule: make only applies it where the vendor source exists.)
$(OBJ)/%.o: $(VENDOR)/imgui/%.cpp Makefile | $(OBJ)
	$(CXX) -O2 -std=c++20 -fPIC -w -I$(VENDOR)/imgui -c $< -o $@

$(OBJ)/sokol_impl.o: $(NATIVE)/frontend/sokol_impl.m $(NATIVE)/frontend/sokol_platform.h Makefile | $(OBJ)
	$(CC) -std=c17 -O2 -fPIC -w -x $(SOKOL_LANG) -I$(VENDOR)/sokol -I$(NATIVE) -c $< -o $@

$(BIN)/nes_hardware.so: $(HW_OBJS)
	$(LDSHARED) $^ -o $@

$(BIN)/nes_hardware_reference.so: $(REF_OBJS)
	$(LDSHARED) $^ -o $@

$(BIN)/nes_app.so: $(FRONTEND_OBJS)
	$(LDSHARED) $^ -o $@ $(FRONTEND_LIBS)

$(BIN)/native_hardware_test: $(NATIVE)/tests/native_hardware_test.cpp $(NATIVE)/frontend/frame_clock.h Makefile \
		$(REF_OBJ)/bus.o $(REF_OBJ)/ppu.o $(REF_OBJ)/apu.o $(REF_OBJ)/console.o $(REF_OBJ)/tape.o
	$(CXX) $(CXXFLAGS) $(NES_INCLUDES) -DTRANSFORMER_NES_PPU_REFERENCE $(filter %.cpp %.o,$^) -o $@

test-native: $(BIN)/native_hardware_test
	$(BIN)/native_hardware_test

clean:
	rm -rf $(BIN)

-include $(OBJ)/*.d
-include $(REF_OBJ)/*.d
