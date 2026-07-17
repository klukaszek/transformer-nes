"""Emulator-side state for the tilde debug overlay."""

from __future__ import annotations


class DebugBridge:
    """app.cpp wraps these methods as the overlay's hooks (called from
    the frame/event callbacks with the GIL held)."""

    def __init__(self, console, backend: str):
        self.console = console
        self.backend = backend
        self.pause = False
        self.step_pending = 0
        self.mute = False
        self.no_sprite_limit = False
        self.last_ms = 0.0
        self.avg_ms = 0.0

    def note_frame(self, seconds: float):
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
