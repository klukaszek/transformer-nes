"""SMB3's status bar: an MMC3 IRQ split whose handler is cycle-timed.

Its top border must stay on one row frame to frame.  DMC sample fetches
that stall the CPU used to pile up over a long CPU run and land at once
past the IRQ, dropping the split a line or two on scattered frames (the
"bouncing HUD")."""

import os

import numpy as np
import pytest

from transformer_nes.nes.backends import backend_factory
from transformer_nes.nes.build import load_hardware

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ROM = os.path.join(ROOT, "transformer_nes", "nes", "examples", "smb3.nes")
TAPE = os.path.join(ROOT, "tapes", "smb3_status_bar")
WHITE = 0x30


def replay(hardware, cart, tape):
    with open(tape) as lines:
        events = [(int(p[0]), int(p[1], 16)) for p in map(str.split, lines) if len(p) >= 2]
    console = hardware.Console(cart, backend_factory("weights"), 44100.0)
    buttons, at = 0, 0
    for frame in range(events[-1][0] + 60):
        while at < len(events) and events[at][0] <= frame:
            buttons = events[at][1]
            at += 1
        console.set_buttons(buttons)
        yield console.run_frame()


def border_row(frame) -> int:
    """The status bar's white top border: first lower row mostly white."""
    rows = np.nonzero((frame[150:] == WHITE).sum(axis=1) >= 100)[0]
    return 150 + int(rows[0]) if len(rows) else -1


@pytest.mark.slow
def test_status_bar_holds_its_row():
    if not os.path.exists(ROM):
        pytest.skip("smb3.nes not present")
    hardware = load_hardware()
    rows = [border_row(f) for f in replay(hardware, hardware.Cartridge.load(ROM), TAPE)]
    blips = [
        i
        for i in range(1, len(rows) - 1)
        if rows[i] >= 0 and rows[i - 1] == rows[i + 1] != rows[i]
    ]
    assert not blips, f"status bar jumped for single frames at {blips[:10]}"
