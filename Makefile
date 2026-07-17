# Native NES build: the two Python extension modules (hardware console,
# sokol frontend) plus the native hardware test, all into bin/.  app.py
# shells out to `make` and imports the artifacts; the weights-engine
# codegen build (model/_engine_cache) is separate and stays torch-driven.
#
# pybind11 is the vendored copy (same version torch bundles); Python.h
# comes from the interpreter invoking make (PYTHON=... from app.py).

# Default to the project venv so a bare `make` matches what app.py
# builds (app.py passes PYTHON=sys.executable explicitly).
PYTHON ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)
UNAME := $(shell uname -s)

NES := transformer_nes/nes
VENDOR := vendor
BIN := bin
OBJ := $(BIN)/obj

PY_INCLUDE := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_paths()["include"])')

CXX ?= c++
CC ?= cc
CXXFLAGS := -O2 -std=c++26 -Wall -Wextra -Wpedantic -fPIC -fvisibility=hidden
NES_INCLUDES := -I$(NES) -I$(NES)/include
PY_INCLUDES := -I$(VENDOR)/pybind11/include -I$(PY_INCLUDE)
UI_INCLUDES := -I$(VENDOR)/sokol -I$(VENDOR)/imgui

ifeq ($(UNAME),Darwin)
LDSHARED := $(CXX) -shared -undefined dynamic_lookup
FRONTEND_LIBS := -framework Cocoa -framework QuartzCore -framework Metal \
                 -framework MetalKit -framework AudioToolbox
SOKOL_LANG := objective-c
else
LDSHARED := $(CXX) -shared
FRONTEND_LIBS := -lGL -lX11 -lXi -lXcursor -ldl -lpthread -lm
SOKOL_LANG := c
endif

HW_OBJS := $(addprefix $(OBJ)/,bindings.o bus.o ppu.o apu.o console.o tape.o)
IMGUI_OBJS := $(addprefix $(OBJ)/,imgui.o imgui_draw.o imgui_tables.o imgui_widgets.o)
FRONTEND_OBJS := $(OBJ)/app.o $(OBJ)/debug_ui.o $(OBJ)/sokol_impl.o $(IMGUI_OBJS)

.PHONY: all hardware frontend test-native clean
all: hardware frontend
hardware: $(BIN)/nes_hardware.so
frontend: $(BIN)/nes_app.so

$(OBJ):
	mkdir -p $(OBJ)

$(OBJ)/%.o: $(NES)/%.cpp | $(OBJ)
	$(CXX) $(CXXFLAGS) $(NES_INCLUDES) $(PY_INCLUDES) -MMD -MP -c $< -o $@

$(OBJ)/app.o: $(NES)/app.cpp | $(OBJ)
	$(CXX) $(CXXFLAGS) $(NES_INCLUDES) $(PY_INCLUDES) $(UI_INCLUDES) -MMD -MP -c $< -o $@

$(OBJ)/debug_ui.o: $(NES)/ui/debug_ui.cpp | $(OBJ)
	$(CXX) $(CXXFLAGS) $(NES_INCLUDES) $(UI_INCLUDES) -MMD -MP -c $< -o $@

# Vendored code builds warning-silent; its style is not ours to fix.
# (Plain % rule: make only applies it where the vendor source exists.)
$(OBJ)/%.o: $(VENDOR)/imgui/%.cpp | $(OBJ)
	$(CXX) -O2 -std=c++26 -fPIC -w -I$(VENDOR)/imgui -c $< -o $@

$(OBJ)/sokol_impl.o: $(NES)/sokol_impl.m $(NES)/ui/sokol_platform.h | $(OBJ)
	$(CC) -std=c17 -O2 -fPIC -w -x $(SOKOL_LANG) -I$(VENDOR)/sokol -I$(NES) -c $< -o $@

$(BIN)/nes_hardware.so: $(HW_OBJS)
	$(LDSHARED) $^ -o $@

$(BIN)/nes_app.so: $(FRONTEND_OBJS)
	$(LDSHARED) $^ -o $@ $(FRONTEND_LIBS)

$(BIN)/native_hardware_test: $(NES)/tests/native_hardware_test.cpp \
		$(OBJ)/bus.o $(OBJ)/ppu.o $(OBJ)/apu.o $(OBJ)/console.o $(OBJ)/tape.o
	$(CXX) $(CXXFLAGS) $(NES_INCLUDES) $^ -o $@

test-native: $(BIN)/native_hardware_test
	$(BIN)/native_hardware_test

clean:
	rm -rf $(BIN)

-include $(OBJ)/*.d
