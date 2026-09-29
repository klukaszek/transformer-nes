"""The overlay's X-ray reads the live engines without disturbing them."""

from transformer_nes.cpu6502.backends.weights.runner import Cpu6502TransformerMachine
from transformer_nes.nes.xray import PATHS, CpuXray


def machine_at(program: bytes) -> Cpu6502TransformerMachine:
    image = bytearray(0x10000)
    image[0x8000 : 0x8000 + len(program)] = program
    image[0xFFFC:0xFFFE] = (0x00, 0x80)
    machine = Cpu6502TransformerMachine(bcd=False, mirror_cpu_ram=True)
    machine.load_image(bytes(image), start_pc=0x8000)
    return machine


def test_dissection_follows_an_instruction_through_the_layers():
    machine = machine_at(bytes([0xA9, 0x42, 0x00]))  # LDA #$42; BRK
    xray = CpuXray(machine)
    before = machine.engine.get_run_state()
    d = xray.dissect()
    assert d["title"] == "$8000  LDA #$42"
    assert len(d["activity"]) == d["layers"]
    steps = {name: (row, kind, detail, value) for row, kind, name, detail, value in d["steps"]}
    # The opcode arrives by attention over memory, the result later.
    fetch = steps["fetched_opcode_raw"]
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
