"""The overlay's X-ray reads the live engines without disturbing them."""

from transformer_nes.cpu6502.backends.weights.runner import Cpu6502TransformerMachine
from transformer_nes.nes.xray import PATHS, CpuXray, PpuXray


def machine_at(program: bytes) -> Cpu6502TransformerMachine:
    image = bytearray(0x10000)
    image[0x8000 : 0x8000 + len(program)] = program
    image[0xFFFC:0xFFFE] = (0x00, 0x80)
    machine = Cpu6502TransformerMachine(bcd=False, mirror_cpu_ram=True)
    machine.load_image(bytes(image), start_pc=0x8000)
    return machine


def test_counter_views_do_not_construct_debug_models():
    class EngineOnly:
        engine = None

        @property
        def lives(self):
            raise AssertionError("debug graph built before dissection")

    runtime = EngineOnly()
    runtime.engine = runtime
    cpu = CpuXray(runtime)
    ppu = PpuXray(runtime)
    assert cpu.engine is runtime and ppu.engine is runtime


def test_dissection_follows_an_instruction_through_the_layers():
    machine = machine_at(bytes([0xA9, 0x42, 0x00]))  # LDA #$42; BRK
    xray = CpuXray(machine)
    before = machine.engine.get_run_state()
    d = xray.dissect()
    assert d["title"] == "$8000  LDA #$42"
    assert len(d["activity"]) == d["layers"]
    steps = {name: (row, kind, detail, value) for row, kind, name, detail, value in d["steps"]}
    # The opcode arrives by attention over memory, the result later.
    fetch = steps["fetched_opcode"]
    assert fetch[1:] == ("attention", "mem[$8000]", 0xA9)
    assert steps["new_a"][3] == 0x42 and steps["new_a"][0] > fetch[0]
    # It ran on a copy: the machine has not moved.
    assert machine.engine.get_run_state() == before


def test_path_counters_cover_every_instruction():
    # LDX #3; loop: DEX; BNE loop; BRK
    machine = machine_at(bytes([0xA2, 0x03, 0xCA, 0xD0, 0xFD, 0x00]))
    xray = CpuXray(machine)
    out = machine.run(100, max_cycles=0, trace=False)
    paths, _ = xray.take_paths()
    assert [p for p, _ in paths] == list(PATHS)
    assert sum(n for _, n in paths) == out["steps"]
    assert sum(n for _, n in xray.take_paths()[0]) == 0  # taken and reset


def test_overlay_reuses_one_snapshot_until_the_next_emulated_frame():
    from types import SimpleNamespace

    from transformer_nes.nes.debug import DebugBridge

    maps = []
    console = SimpleNamespace(ppu_expert_map=lambda: maps.append(True) or bytes([255]) * 7680)
    bridge = DebugBridge(console, "reference")
    first = bridge.xray_frame()
    assert bridge.xray_frame() is first
    assert len(maps) == 1
    bridge.note_frame(0.003)
    assert bridge.xray_frame() is not first
    assert len(maps) == 2


def test_overlay_collects_reads_only_while_observed():
    from types import SimpleNamespace

    import numpy as np

    from transformer_nes.nes.debug import DebugBridge

    calls = []

    class Engine:
        def xray_take(self):
            calls.append("paths")
            return [1, 0, 0, 0], np.zeros(256, dtype=int), [1, 0, 0, 0]

        def xray_reads(self, enabled):
            calls.append(enabled)
            return np.ones(65536, dtype=np.uint32)

    runtime = SimpleNamespace(engine=SimpleNamespace(engine=Engine()))
    bridge = DebugBridge(None, "weights", cpu_runtime=runtime)
    bridge.note_frame(0.001)
    assert calls == [] and not bridge.cpu_frames
    bridge.set_observing(True)
    assert calls == ["paths", True]
    bridge.note_frame(0.002)
    assert len(bridge.cpu_frames) == 1 and bridge.read_heat.heat.sum() == 65536
    before = list(calls)
    bridge.set_observing(True)
    assert calls == before and len(bridge.cpu_frames) == 1
    bridge.set_observing(False)
    assert calls[-2:] == ["paths", False]
    before = list(calls)
    bridge.note_frame(0.003)
    assert calls == before and not bridge.cpu_frames
    assert not bridge.read_heat.heat.any() and bridge.last_ms == 3

