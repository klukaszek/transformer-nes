"""Replay timing samples cover exactly the frames that actually ran."""

import math

from transformer_nes.nes.backends import backend_factory
from transformer_nes.nes.build import load_hardware


def test_tape_samples_obey_limits_and_empty_replays(tmp_path):
    hardware = load_hardware()
    prg = bytearray(0x4000)
    prg[:3] = bytes((0x4C, 0, 0x80))  # JMP $8000
    prg[-4:-2] = bytes((0, 0x80))
    image = b"NES\x1a" + bytes((1, 1)) + bytes(10) + prg + bytes(0x2000)
    console = hardware.Console(hardware.Cartridge.parse(image), backend_factory("reference"))
    tape = tmp_path / "inputs"
    tape.write_text("2 00\n")

    for limit, expected in ((2, 2), (2, 1), (2, 0)):
        result = console.run_tape(str(tape), tail_frames=0, max_frames=limit)
        assert result["frames"] == expected
        assert len(result["frame_seconds"]) == expected
        assert len(result["frame_cpu_seconds"]) in (0, expected)
        assert all(math.isfinite(v) and v >= 0 for v in result["frame_seconds"])
        assert all(math.isfinite(v) and v >= 0 for v in result["frame_cpu_seconds"])
        assert sum(result["frame_seconds"]) <= result["seconds"]
    assert console.frame_index == 3
