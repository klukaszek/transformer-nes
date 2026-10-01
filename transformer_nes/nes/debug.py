"""Emulator-side state for the tilde debug overlay."""

from __future__ import annotations

from collections import Counter, deque

from transformer_nes.nes.xray import (
    PATHS,
    CpuXray,
    PpuXray,
    ReadHeat,
    expert_colors,
    ppu_expert_summary,
)

# The X-ray's CPU counts cover the last half second of emulated frames,
# so the bars hold still between display refreshes.
XRAY_WINDOW = 30


class DebugBridge:
    """app.cpp wraps these methods as the overlay's hooks (called from
    the frame/event callbacks with the GIL held).  The X-ray's CPU views
    need the weights CPU (`cpu_runtime`) and its pixel views the PPU
    transformer (`ppu_engine`); app.cpp wires only those that are present."""

    def __init__(self, console, backend: str, *, cpu_runtime=None, ppu_engine=None):
        self.console = console
        self.backend = backend
        self.cpu_xray = CpuXray(cpu_runtime) if cpu_runtime is not None else None
        self.ppu_xray = PpuXray(ppu_engine) if ppu_engine is not None else None
        self.expert_names = list(ppu_engine.expert_names) if ppu_engine is not None else []
        self.expert_colors = expert_colors(self.expert_names)
        self.pixel: dict | None = None
        self.pixel_x = 0
        self.cpu_frames: deque = deque(maxlen=XRAY_WINDOW)
        self.read_heat = ReadHeat()
        self.pause = False
        self.step_pending = 0
        self.mute = False
        self.no_sprite_limit = False
        self.last_ms = 0.0
        self.avg_ms = 0.0

    def note_frame(self, seconds: float):
        # Called once per emulated frame: the CPU counters are taken here,
        # not per display refresh (which can come between frames).
        if self.cpu_xray is not None:
            self.cpu_frames.append(self.cpu_xray.take_paths())
            self.read_heat.add(self.cpu_xray.take_reads())
        self.last_ms = 1e3 * seconds
        self.avg_ms = self.last_ms if not self.avg_ms else 0.95 * self.avg_ms + 0.05 * self.last_ms

    def read_mem(self, start: int, count: int) -> bytes:
        return self.console.read_ram(start, count)

    def stats(self):
        return [
            ("backend", self.backend),
            ("frame", str(self.console.frame_index)),
            ("cycles", str(self.console.cycles)),
            ("emulation", f"{self.last_ms:.2f} ms/frame (avg {self.avg_ms:.2f})"),
        ]

    def options(self):
        return [
            ("Pause emulation", self.pause),
            ("Remove sprite limit", self.no_sprite_limit),
            ("Mute audio", self.mute),
        ]

    def set_option(self, name: str, value: bool):
        if name == "Pause emulation":
            self.pause = value
        elif name == "Remove sprite limit":
            self.no_sprite_limit = value
            self.console.set_sprite_limit(not value)
        elif name == "Mute audio":
            self.mute = value

    def step_frame(self):
        self.pause = True
        self.step_pending += 1

    def xray_frame(self) -> dict:
        paths, singles = Counter(), Counter()
        for frame_paths, frame_singles in self.cpu_frames:
            paths.update(dict(frame_paths))
            singles.update(dict(frame_singles))
        expert_map = self.console.ppu_expert_map()
        return {
            "cpu_paths": [(name, paths[name]) for name in PATHS] if paths else [],
            "cpu_singles": singles.most_common(8),
            "ppu_models": ppu_expert_summary(expert_map, self.expert_names),
            "expert_map": expert_map,
            "expert_names": self.expert_names,
            "expert_colors": self.expert_colors,
            "read_heat": self.read_heat.image() if self.cpu_xray else b"",
            "hot_pages": self.read_heat.hottest() if self.cpu_xray else [],
        }

    # ── With the render graph (ppu_xray) ──

    def trace_pixel(self, line: int, x: int) -> None:
        self.pixel, self.pixel_x = None, x
        self.console.request_ppu_slot_trace(line, x // 8)

    def pixel_dissection(self) -> dict | None:
        """The traced pixel, once its slot has been drawn."""
        if self.pixel is None:
            trace = self.console.take_ppu_slot_trace()
            if trace is not None:
                self.pixel = self.ppu_xray.dissect(trace, self.pixel_x)
        return self.pixel

    def knock_out(self, name: str) -> str:
        try:
            return "knocked out: " + self.ppu_xray.knock_out(self.console, name)
        except ValueError as error:
            return str(error)

    def restore(self) -> None:
        self.ppu_xray.restore(self.console)

    # ── With the weights CPU (cpu_xray) ──

    def dissect(self) -> dict:
        return self.cpu_xray.dissect()
