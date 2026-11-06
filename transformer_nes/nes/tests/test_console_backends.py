"""Reference-vs-weights lockstep through the native console: identical
frames, cycles, and RAM under scripted input, and MMC1's RMW serial-port
quirk through the engine's watched-write path."""

import itertools
import os

import numpy as np
import pytest

from transformer_nes.nes.backends import backend_factory
from transformer_nes.nes.build import load_hardware

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")

# Frame -> controller mask: Start taps at the title, then hold A+Right
# (exercises the strobe writes and the serial-read lease queue).
SCRIPT = {30: 0x08, 33: 0x00, 45: 0x08, 48: 0x00, 60: 0x81}
FRAMES = 90


def reference_channel_gains(state):
    gains = []
    for i in range(2):
        p = {field: state[f"pulse{i}_{field}"] for field in (
            "timer", "sweep_shift", "sweep_negate", "enabled", "length",
            "env_constant", "env_period", "env_decay",
        )}
        shift = p["timer"] >> p["sweep_shift"]
        target = p["timer"] - shift - (i == 0) if p["sweep_negate"] else p["timer"] + shift
        active = p["enabled"] and p["length"] > 0 and p["timer"] >= 8 and target < 2048
        gains.append((p["env_period"] if p["env_constant"] else p["env_decay"]) if active else 0)
    gains.append(int(state["triangle_enabled"] and state["triangle_length"] > 0
                     and state["triangle_linear_counter"] > 0 and state["triangle_timer"] >= 2))
    active = state["noise_enabled"] and state["noise_length"] > 0
    gains.append((state["noise_env_period"] if state["noise_env_constant"] else state["noise_env_decay"])
                 if active else 0)
    return gains


@pytest.mark.parametrize("token", ["control", "parameters", "levels", "sample", "dac", "frame", "program", "dmc", "oscillate", "tick", "noise"])
def test_audio_tokens_execute_the_constructed_weights(token):
    from transformer_nes.graph.core import InputDimension
    from transformer_nes.model.graph_engine import construct
    from transformer_nes.model.inference import ExactInference
    from transformer_nes.nes import audio, oscillators

    floating = token in ("parameters", "sample", "levels", "dac")
    engine = audio.sample_engine() if floating else audio.audio_engine(41)
    graph = {"control": audio.channel_graph, "parameters": audio.parameter_graph, "levels": audio.sample_graph,
             "sample": audio.sample_graph, "dac": audio.dac_graph,
             "frame": audio.frame_graph, "program": audio.program_graph,
             "dmc": audio.dmc_graph, "oscillate": lambda: oscillators.clock_graph(41),
             "tick": lambda: oscillators.clock_graph(1), "noise": lambda: oscillators.noise_graph(41)}[token]()
    model, _, info = construct(graph)
    eager = ExactInference(model)
    dims = {d.name: d for d in graph.all_dims}
    inputs = {d.name: d for d in graph.all_dims if isinstance(d, InputDimension)}
    rng = np.random.default_rng(53)
    ranges = {"event": 3}
    bounds = {"timer": 2048, "length": 255, "sweep_period": 8, "sweep_shift": 8,
              "sweep_divider": 8, "seq_pos": 8, "duty": 4, "env_period": 16,
              "env_divider": 16, "env_decay": 16, "linear_reload_value": 128,
              "linear_counter": 128, "lfsr": 32768}
    for prefix, fields in (("pulse0", audio.PULSE), ("pulse1", audio.PULSE),
                           ("triangle", audio.TRIANGLE), ("noise", audio.NOISE)):
        ranges.update({f"{prefix}_{f}": bounds.get(f, 2) for f in fields})
    ranges.update(triangle_seq_pos=32, seq_step=4, value=256, noise_period=4069,
                  port=24, dmc_remaining=4082, output_level=128, sample_address=65536,
                  sample_length=4082, current_address=65536, bytes_remaining=4082,
                  sample_buffer=256, shift_register=256)
    ranges.update(cycles=2 if token == "tick" else 42, triangle_period=2048,
                  triangle_counter=2048, triangle_pos=32, triangle_linear=128, noise_counter=4069)
    ranges.update(steps=22, pulse=31, triangle=16, noise=16, dmc=128, dmc_output=128)
    ranges.update(pulse0_gain=16, pulse1_gain=16, noise_gain=16)
    for i in range(2):
        ranges.update({f"pulse{i}_period": 2048, f"pulse{i}_counter": 2048, f"pulse{i}_pos": 8})
    if token in ("frame", "dmc"):
        ranges["event"] = 4
    io = engine.graphs[token]
    for _ in range(128):
        values = {name: int(rng.integers(ranges.get(name, 2))) for name in io.inputs}
        if token in ("sample", "levels"):
            values.update(previous_mix=float(rng.random()), previous_filter=float(rng.uniform(-1, 1)))
        if token == "dmc":
            values.update(bits_remaining=int(rng.integers(1, 9)),
                          period=int(rng.choice(audio.DMC_PERIODS)))
        residual = np.zeros(info.d_model)
        residual[info.slot_of[graph.one]] = 1
        for name, value in values.items():
            residual[info.slot_of[inputs[name]]] = value
        actual = eager.forward(residual)
        expected = [actual[info.slot_of[dims[name]]] if floating else round(actual[info.slot_of[dims[name]]])
                    for name in io.outputs]
        packet = [values[name] for name in io.inputs]
        np.testing.assert_allclose(engine.step(packet, engine.model_id(token)), expected, rtol=0, atol=5e-7 if floating else 0)
        if token == "parameters":
            np.testing.assert_array_equal(expected, reference_channel_gains(values))
        elif token in ("control", "frame"):
            events = audio.FRAME_EVENTS if token == "frame" else audio.EVENTS
            np.testing.assert_array_equal(
                engine.step(packet, engine.model_id(token, events[values["event"]])), expected
            )
        elif token == "program" and values["is_write"] and values["port"] in audio.WRITE_PORTS:
            np.testing.assert_array_equal(
                engine.step(packet, engine.model_id(token, f"write_{values['port']}")), expected
            )
        elif token == "dmc":
            name = (f"write_{values['port']}" if values["event"] == 0 and values["port"] in audio.DMC_WRITE_PORTS
                    else audio.DMC_EVENTS[values["event"]] if values["event"] else None)
            np.testing.assert_array_equal(engine.step(packet, engine.model_id(token, name)), expected)
        elif token == "noise":
            np.testing.assert_array_equal(
                engine.step(packet, engine.model_id(token, f"mode{values['noise_mode']}")), expected
            )
            # The single-step expert is a folded recurrence, not a host LFSR.
            packet[0] = 1
            np.testing.assert_array_equal(
                engine.step(packet, engine.model_id(token, f"tick_mode{values['noise_mode']}")),
                engine.step(packet, engine.model_id(token)),
            )


def test_neural_audio_filter_tracks_the_nonlinear_reference_and_clips():
    from transformer_nes.nes import audio

    engine = audio.sample_engine()
    x = dict.fromkeys(audio.SAMPLE_INPUTS, 0)
    x.update(pulse0_seq_pos=1, pulse1_seq_pos=1, triangle_gain=1)
    rng = np.random.default_rng(89)
    previous_mix, previous_filter = 0.0, 0.0
    clipped = False
    for i in range(24576):
        # Repeated maximum DC, silence, and varying channel levels exercise
        # steady state, both filter tails and the clipping boundaries.
        # Long decay catches both pole quantization and cancellation when
        # the mixed level and its previous value are almost equal.
        values = (15, 15, 15, 15, 127) if i < 8192 else ((0,) * 5 if i < 16384 else
                  (*rng.integers(16, size=4), int(rng.integers(128))))
        p0, p1, t, n, d = values
        x.update(pulse0_gain=p0, pulse1_gain=p1, triangle_seq_pos=15-t,
                 noise_gain=n, dmc_output=d)
        p = p0 + p1
        tnd = t / 8227 + n / 12241 + d / 22638
        mixed = (95.88 / (8128 / p + 100) if p else 0) + (159.79 / (1 / tnd + 100) if tnd else 0)
        filtered = mixed - previous_mix + 0.9995 * previous_filter
        expected = min(1, max(-1, 2 * filtered))
        out = engine.step([x[name] for name in audio.SAMPLE_INPUTS], engine.model_id("sample"))
        np.testing.assert_allclose(out, [mixed, filtered, expected], rtol=0, atol=4e-6)
        x["previous_mix"], x["previous_filter"] = out[:2]
        previous_mix, previous_filter = mixed, filtered
        clipped |= abs(expected) == 1
    assert clipped


@pytest.mark.parametrize("maximum", [1, 41, 429])
@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("parity", [0, 1])
def test_audio_oscillator_spans_match_literal_clocks_and_compose(mode, parity, maximum):
    from transformer_nes.nes import audio, oscillators

    engine = audio.audio_engine(maximum)
    rng = np.random.default_rng(67)

    def advance(x, cycles):
        clock = engine.step([cycles, *x[1:15]], engine.model_id("tick" if cycles == 1 else "oscillate"))
        steps = int(clock[8])
        state = x[16:]
        if steps:
            expert = f"{'tick_' if steps == 1 else ''}mode{mode}"
            state = engine.step([steps, mode, *state], engine.model_id("noise", expert)).tolist()
        return [*clock[:8].tolist(), *state]

    def resume(x, out):
        x = x.copy()
        x[1] = out[0]
        for i in range(2):
            x[3 + 3*i], x[4 + 3*i] = out[1 + 2*i:3 + 2*i]
        x[9], x[10], x[14] = out[5:8]
        x[16:] = out[8:]
        return x

    for cycles in sorted({0, 1, maximum, *(n for n in (2, 7, 16, 127, 214) if n <= maximum)}):
        for seed in range(32):
            x = [0, parity]
            for _ in range(2):
                x.extend((int(rng.choice([0, 1, 7, 255, 2047])), int(rng.integers(2048)), int(rng.integers(8))))
            x.extend((int(rng.choice([0, 1, 2, 2047])), int(rng.integers(2048)),
                      int(rng.integers(32)), seed & 1, (seed >> 1) & 1,
                      int(rng.choice([0, 4, 4068])), int(rng.integers(4069)), mode))
            word = int(rng.integers(32768))
            x.extend((word >> i) & 1 for i in range(15))
            assert len(x) == len(oscillators.CLOCK_INPUTS) + 16
            tick, counters, positions = parity, [x[3], x[6]], [x[4], x[7]]
            tc, tp, nc = x[9], x[10], x[14]
            for _ in range(cycles):
                if tc == 0:
                    tc = x[8]
                    if x[11] and x[12]:
                        tp = (tp + 1) & 31
                else:
                    tc -= 1
                tick = 1 - tick
                if tick:
                    for i in range(2):
                        if counters[i] == 0:
                            counters[i], positions[i] = x[2 + 3*i], (positions[i] + 1) & 7
                        else:
                            counters[i] -= 1
                    if nc == 0:
                        nc = x[13]
                        word = (word >> 1) | (((word & 1) ^ ((word >> (6 if mode else 1)) & 1)) << 14)
                    else:
                        nc -= 1
            expected = [tick, counters[0], positions[0], counters[1], positions[1], tc, tp, nc,
                        *((word >> i) & 1 for i in range(15))]
            assert advance(x, cycles) == expected
            split = cycles // 2
            assert advance(resume(x, advance(x, split)), cycles - split) == expected


def test_audio_dmc_fetch_wrap_reload_irq_and_disable():
    from transformer_nes.nes import audio

    engine = audio.audio_engine()
    state = dict.fromkeys(audio.DMC_STATE, 0)
    state.update(irq_enabled=1, period=428, output_level=125, sample_address=0xC000,
                 sample_length=1, current_address=0xFFFF, bytes_remaining=1,
                 buffer_empty=1, shift_register=1, bits_remaining=1)

    def event(kind, value=0, port=0):
        packet = [state[n] for n in audio.DMC_STATE] + [audio.DMC_EVENTS.index(kind), port, value]
        expert = f"write_{port}" if kind == "write" else kind
        out = engine.step(packet, engine.model_id("dmc", expert))
        np.testing.assert_array_equal(out, engine.step(packet, engine.model_id("dmc")))
        state.update(zip(audio.DMC_STATE, out[:14].tolist(), strict=True))
        return dict(zip(audio.DMC_OUTPUTS[14:], out[14:].tolist(), strict=True))

    effects = event("fetch", 0xAA)
    assert effects["dma_stall"] == 4
    assert (state["current_address"], state["bytes_remaining"], state["irq"]) == (0x8000, 0, 1)
    assert not effects["dma_active"]
    effects = event("clock")
    assert (state["output_level"], state["shift_register"], state["bits_remaining"]) == (127, 0xAA, 8)
    assert state["buffer_empty"] and not state["silence"]
    assert effects["reset_clock"] and effects["clock_reload"] == 427
    event("clock")
    assert (state["output_level"], state["shift_register"], state["bits_remaining"]) == (125, 0x55, 7)
    effects = event("write", 16, 21)
    assert (state["current_address"], state["bytes_remaining"], state["irq"]) == (0xC000, 1, 0)
    assert effects["fetch_due"]
    event("write", 64, 16)  # loop, IRQ disabled
    effects = event("fetch", 0x80)
    assert effects["dma_stall"] == 4
    assert (state["current_address"], state["bytes_remaining"], state["irq"]) == (0xC000, 1, 0)
    unchanged = state.copy()
    assert event("fetch", 0x11)["dma_stall"] == 0  # a full buffer cannot accept another reply
    assert state == unchanged
    effects = event("write", 0, 21)
    assert not effects["dma_active"] and not state["bytes_remaining"]
    assert state["sample_buffer"] == 0x80  # disabling does not flush the output unit


def test_audio_frame_reset_irq_and_immediate_clocks():
    from transformer_nes.nes import audio

    engine = audio.audio_engine()
    for five, inhibit, irq, step, event, value in itertools.product(
        range(2), range(2), range(2), range(4), range(4), range(256)
    ):
        clock, write, read = event == 0, event == 1, event == 2
        next_five = (value >> 7) if write else five
        next_inhibit = ((value >> 6) & 1) if write else inhibit
        reset = write or (clock and step == 3)
        next_step = 0 if reset else step + clock
        next_irq = irq or (clock and step == 3 and not five and not inhibit)
        if read or (write and next_inhibit):
            next_irq = 0
        immediate = write and next_five
        expected = [next_five, next_inhibit, next_irq, next_step, reset,
                    clock or immediate, (clock and step in (1, 3)) or immediate,
                    (7457, 14913, 22371, 37281 if next_five else 29829)[next_step]]
        packet = [five, inhibit, irq, step, event, value]
        np.testing.assert_array_equal(
            engine.step(packet, engine.model_id("frame", audio.FRAME_EVENTS[event])), expected
        )


def test_audio_joint_clock_reloads_and_distinct_pulse_negation():
    from transformer_nes.nes import audio

    engine = audio.audio_engine()
    x = dict.fromkeys(audio.INPUTS, 0)
    x["event"] = 2
    for i in range(2):
        fields = dict(timer=100, length=2, enabled=1, sweep_enable=1, sweep_period=7,
                      sweep_negate=1, sweep_shift=1, sweep_reload=1, env_start=1,
                      env_period=3, seq_pos=1)
        x.update({f"pulse{i}_{f}": v for f, v in fields.items()})
    x.update(triangle_timer=8, triangle_length=2, triangle_enabled=1,
             triangle_linear_reload=1, triangle_linear_reload_value=127,
             noise_length=2, noise_enabled=1, noise_env_loop=1, noise_env_period=3)
    out = engine.step([x[n] for n in audio.INPUTS], engine.model_id("control", "both"))
    # Pulse 1's negative sweep subtracts an extra unit.
    expected = [0, 3, 15, 1, 49, 0, 7, 0, 3, 15, 1, 50, 0, 7, 1, 0, 127, 0, 3, 15, 2]
    np.testing.assert_array_equal(out, expected)
    x.update(zip(audio.STATE_OUTPUTS, out.tolist(), strict=True))
    samples = audio.sample_engine()
    gains = samples.step([x[n] for n in audio.PARAMETER_INPUTS], samples.model_id("parameters"))
    x.update(zip(audio.PARAMETER_OUTPUTS, gains.tolist(), strict=True))
    x.update(dmc_output=0, previous_mix=0, previous_filter=0)
    np.testing.assert_array_equal(
        samples.step([x[n] for n in audio.SAMPLE_INPUTS], samples.model_id("levels")), [15, 15, 15, 15]
    )


def run_console(hardware, cart, backend, frames, script=SCRIPT):
    console = hardware.Console(cart, backend_factory(backend), 44100.0)
    frames_out = []
    mask = 0
    for i in range(frames):
        mask = script.get(i, mask)
        console.set_buttons(mask)
        frames_out.append(console.run_frame().copy())
    return console, frames_out


@pytest.mark.parametrize("rom", ["smb.nes", "legendofzelda.nes"])
def test_console_weights_matches_reference(rom):
    path = os.path.join(EXAMPLES, rom)
    if not os.path.exists(path):
        pytest.skip(f"{rom} not present")
    hardware = load_hardware()
    cart = hardware.Cartridge.load(path)
    ref, ref_frames = run_console(hardware, cart, "reference", FRAMES)
    wts, wts_frames = run_console(hardware, cart, "weights", FRAMES)
    for i, (a, b) in enumerate(zip(ref_frames, wts_frames, strict=True)):
        assert np.array_equal(a, b), f"frame {i} diverged"
    assert ref.cycles == wts.cycles
    # RAM, PRG RAM, and the mapped ROM window; MMIO ranges only mirror
    # serviced device reads, whose cadence legitimately differs.
    for lo, count in ((0x0000, 0x2000), (0x6000, 0xA000)):
        assert ref.read_ram(lo, count) == wts.read_ram(lo, count)


def mmc1_rmw_cart(hardware):
    """MMC1 cart whose reset code loads the PRG register with five INC
    $E000s: hardware clocks the ROM byte's bit (1), not the CPU's final
    value (0), so $8000 must map bank 1 afterwards."""
    code = bytes(
        [
            0xEE,
            0x00,
            0xE0,  # INC $E000  x5 (serial load via RMW)
            0xEE,
            0x00,
            0xE0,
            0xEE,
            0x00,
            0xE0,
            0xEE,
            0x00,
            0xE0,
            0xEE,
            0x00,
            0xE0,
            0xAD,
            0x00,
            0x80,  # LDA $8000  (which bank landed?)
            0x85,
            0x10,  # STA $10
            0x4C,
            0x14,
            0xC0,  # spin
        ]
    )
    bank0 = bytes(0x4000)
    bank1 = bytearray(b"\x01" * 0x4000)
    bank1[: len(code)] = code
    bank1[0x3FFC:0x3FFE] = b"\x00\xc0"  # reset -> $C000 (fixed last bank)
    header = bytes([0x4E, 0x45, 0x53, 0x1A, 2, 0, 0x10, 0]) + bytes(8)
    return hardware.Cartridge.parse(header + bank0 + bytes(bank1))


@pytest.mark.parametrize("backend", ["reference", "weights"])
def test_mmc1_rmw_register_write_uses_rom_byte(backend):
    hardware = load_hardware()
    cart = mmc1_rmw_cart(hardware)
    console, _frames = run_console(hardware, cart, backend, 2, script={})
    # Register = 0b11111 -> bank 1 at $8000, whose first byte is the
    # code's 0xEE; the final-value bug would leave bank 0 (0x00).
    assert console.read_ram(0x10, 1) == b"\xee"


def vblank_poll_cart(hardware):
    """NROM cart that polls $2002 in a loop the engine retires as a
    recurrence (LDA $2002; AND #$80; BNE), counting its passes.  At each
    vblank the read pauses and the PPU serves $80, clearing the flag; the
    lease it then sets serves $00, so the loop runs once more and exits."""
    code = bytes(
        [
            0xE6,
            0x10,  # $C000 INC $10
            0xAD,
            0x02,
            0x20,  # $C002 LDA $2002
            0x29,
            0x80,  # $C005 AND #$80
            0xD0,
            0xF9,  # $C007 BNE $C002
            0xE6,
            0x11,  # $C009 INC $11
            0x4C,
            0x00,
            0xC0,  # $C00B JMP $C000
        ]
    )
    prg = bytearray(0x4000)
    prg[: len(code)] = code
    prg[0x3FFA:0x4000] = b"\x00\xc0\x00\xc0\x00\xc0"  # NMI, reset, IRQ -> $C000
    header = bytes([0x4E, 0x45, 0x53, 0x1A, 1, 0, 0, 0]) + bytes(8)
    return hardware.Cartridge.parse(header + bytes(prg))


def test_leased_poll_recurrence_matches_reference():
    """A poll recurrence extrapolates only what single steps would read: a
    lease serving another value than the read the poll began with (served
    by the device at a pause) ends it."""
    hardware = load_hardware()
    cart = vblank_poll_cart(hardware)
    ref, _ = run_console(hardware, cart, "reference", 12, script={})
    wts, _ = run_console(hardware, cart, "weights", 12, script={})
    assert ref.cycles == wts.cycles
    assert ref.read_ram(0x10, 2) == wts.read_ram(0x10, 2)



def test_controller_reply_survives_lease_eviction_and_device_invalidation():
    hardware = load_hardware()
    ports = (0x4016, 0x4016, 0x4017, 0x2002, 0x200A, 0x2012, 0x201A, 0x2022)
    strobe = bytes.fromhex("a9018d1640a9008d1640")
    code = bytearray(strobe)
    for row in range(8):
        if row == 4:
            code.extend(strobe)
        for column, address in enumerate(ports):
            code.extend((0xAD, address & 255, address >> 8, 0x85, 0x20 + 8*row + column))
        code.extend(bytes.fromhex("a9008d0120"))  # withdraw only the PPU's leases
    code.extend(bytes.fromhex("4c00c0"))
    prg = bytearray(0x4000)
    prg[:len(code)] = code
    prg[0x3FFA:] = bytes.fromhex("00c000c000c0")
    cart = hardware.Cartridge.parse(b"NES\x1a" + bytes([1]) + bytes(11) + bytes(prg))
    ref, _ = run_console(hardware, cart, "reference", 3, script={0: 0xA5})
    actual, _ = run_console(hardware, cart, "weights", 3, script={0: 0xA5})
    assert ref.cycles == actual.cycles
    assert ref.read_ram(0x20, 64) == actual.read_ram(0x20, 64)
    replies = actual.read_ram(0x20, 64)
    for row in range(8):
        assert list(replies[8*row:8*row+2]) == [0x40 | ((0xA5 >> ((2*row+k) % 8)) & 1) for k in range(2)]


def irq_step(engine, chip, event, state, value=0):
    from transformer_nes.nes.mappers import EVENTS

    inputs = np.array([*state, EVENTS.index(event), value], dtype=np.int32)
    full = engine.engine.step(engine.model_id(chip), inputs)
    routed = engine.engine.step(engine.model_id(chip, event), inputs)
    np.testing.assert_array_equal(routed, full)
    clock_fire = (
        event == "clock"
        and state[4]
        and (routed[0] == 0 if chip == "mmc3" else state[1] and routed[0] == state[1])
    )
    assert routed[9] == int(
        bool(clock_fire or event == "disable" or (chip == "mmc5" and event in ("enable", "ack")))
    )
    return routed[:9].tolist()


@pytest.mark.parametrize("chip", [1, 4, 5, 9, 10])
def test_mapper_access_packets_execute_exact_weights(chip):
    from transformer_nes.graph.core import InputDimension
    from transformer_nes.model.graph_engine import construct
    from transformer_nes.model.inference import ExactInference
    from transformer_nes.nes import mappers

    engine = mappers.mapper_control_engine()
    graph = mappers.access_graph(chip)
    model, _, info = construct(graph)
    eager = ExactInference(model)
    dims = {d.name: d for d in graph.all_dims}
    inputs = {d.name: d for d in graph.all_dims if isinstance(d, InputDimension)}
    rng = np.random.default_rng(83)
    for _ in range(256):
        values = {n: int(rng.integers(256)) for n in mappers.ACCESS_INPUTS}
        values.update(is_write=int(rng.integers(2)), original=int(rng.choice([-1, 0, 127, 255])),
                      ram_bank=int(rng.integers(8)), writable=int(rng.integers(2)))
        for i in range(4):
            values[f"ram{i}"] = int(rng.integers(2))
            values[f"bank{i}"] = int(rng.integers(8 if values[f"ram{i}"] else 64))
        residual = np.zeros(info.d_model)
        residual[info.slot_of[graph.one]] = 1
        for name, value in values.items():
            if inputs[name] in info.slot_of:
                residual[info.slot_of[inputs[name]]] = value
        result = eager.forward(residual)
        expected = [round(result[info.slot_of[dims[n]]]) for n in mappers.ACCESS_OUTPUTS]
        packet = [values[n] for n in mappers.ACCESS_INPUTS]
        for expert in (None, "write" if values["is_write"] else "read"):
            np.testing.assert_array_equal(engine.step(packet, engine.model_id(f"access_{chip}", expert)), expected)
        np.testing.assert_array_equal(engine.step(packet, engine.model_id(f"address_{chip}")), expected[4:7])


@pytest.mark.parametrize("chip,prg,chr_", [(1, 3, 12), (4, 6, 24), (9, 6, 3), (10, 3, 3)])
def test_neural_bank_transitions_and_mapping_effects(chip, prg, chr_):
    from transformer_nes.nes.mappers import BANK_STATE, mapper_bank_engine

    engine = mapper_bank_engine(chip, prg, chr_)
    state = [0] * len(BANK_STATE[chip])

    def step(port=0, value=0, original=0, event=1):
        nonlocal state
        inputs = [*state, event, port, value, original]
        expert = "reset" if event == 0 else f"{'rmw' if event == 2 else 'write'}_{port:X}"
        out = engine.step(inputs, engine.model_id("transition", expert))
        np.testing.assert_array_equal(out, engine.step(inputs, engine.model_id("transition")))
        state = out[:-1].tolist()
        return int(out[-1])

    def layout():
        return engine.step(state, engine.model_id("layout")).tolist()

    assert step(value=3, event=0) == 7
    if chip == 1:

        def serial(port, value):
            masks = [step(port, (value >> bit) & 1) for bit in range(5)]
            assert masks[:4] == [0] * 4  # nonterminal writes need no layout forward
            return masks[4]

        assert serial(0xE000, 31) == 1
        assert layout()[:2] == [15 % prg, prg - 1]
        assert serial(0xA000, 31) == 2
        assert serial(0x8000, 31) == 7
        assert layout()[2:6] == [(31 * 4 + i) % chr_ for i in range(4)]
        state[-1] = 1  # next write commits the serial register
        assert step(0xC000, 0x80, 1, event=2) == 3
        assert state == [31, 31, 31, 16, 16]  # first write commits, second resets
    elif chip == 4:
        assert step(0x8000, 0xC6) == 3
        assert layout()[:4] == [prg - 2, 1, 0, prg - 1]
        assert step(0x8001, 255) == 1
        assert layout()[2] == 255 % prg
        assert step(0x8000, 0xC2) == 0  # register selection alone changes no map
        assert step(0x8001, 255) == 2
        assert layout()[4] == 255 % chr_
        assert step(0xA000, 0) == 4
        assert layout()[-1] == 2
    else:
        assert step(0xA000, 255) == 1
        assert layout()[0] == 15 % prg
        assert step(0xB000, 255) == 2
        assert layout()[4 if chip == 9 else 2] == 31 % chr_
        assert step(0xF000, 0) == 4
        assert layout()[-1] == 2


@pytest.mark.parametrize("chip", ["mmc3", "mmc5"])
def test_mapper_irq_clock_and_neural_deadline(chip):
    from transformer_nes.nes.mappers import mapper_control_engine

    engine = mapper_control_engine()
    for counter, target, phase, pending, enabled in itertools.product(
        (0, 1, 2, 127, 128, 239, 255), (0, 1, 2, 127, 128, 239, 255), range(2), range(2), range(2)
    ):
        state = [counter, target, phase, pending, enabled]
        if chip == "mmc3":
            result = target if phase or counter == 0 else counter - 1
            pending = int(pending or (enabled and result == 0))
            predictable = enabled
            distance = (target + 1 if result == 0 else result) if enabled else 0
            new_phase = 0
        else:
            result = counter + 1 if phase else 0
            pending = int(pending or (target and result == target))
            predictable = int(enabled and target and not pending)
            distance = max(0, target - result) if predictable else 0
            new_phase = 1
        assert irq_step(engine, chip, "clock", state) == [
            result,
            target,
            new_phase,
            pending,
            enabled,
            enabled * pending,
            0,
            predictable,
            distance,
        ]


@pytest.mark.parametrize("chip", ["mmc3", "mmc5"])
def test_mapper_irq_acknowledgement_and_frame_epoch(chip):
    from transformer_nes.nes.mappers import mapper_control_engine

    engine = mapper_control_engine()
    state = [0] * 5

    def step(event, value=0):
        nonlocal state
        out = irq_step(engine, chip, event, state, value)
        state = out[:5]
        return out

    if chip == "mmc3":
        step("enable")
        assert step("clock")[5:] == [1, 0, 1, 1]  # zero latch fires on every clock
        step("disable")
        assert step("enable")[5:] == [0, 0, 1, 1]  # no retroactive assertion
        step("target", 2)
        assert step("reload")[-1] == 3
        assert [step("clock")[-1] for _ in range(3)] == [2, 1, 3]
        assert state == [0, 2, 0, 1, 1]
    else:
        step("target", 2)
        assert step("enable", 0x7F)[-2:] == [0, 0]
        assert step("enable", 0x80)[-2:] == [1, 3]
        assert [step("clock")[-1] for _ in range(3)] == [2, 1, 0]
        assert step("ack")[5:] == [0, 0xC0, 1, 0]  # target has passed this frame
        assert step("idle")[-1] == 3
        for _ in range(3):
            step("clock")
        assert step("enable", 0)[5] == 0  # disabling preserves pending
        assert step("enable", 0xFF)[5] == 1
        assert step("ack")[6] == 0xC0
        assert step("target", 0)[-2:] == [0, 0]


def test_controller_reply_packets_match_serial_reads():
    from transformer_nes.nes.controller import EVENTS, controller_engine

    engine = controller_engine()
    full = engine.model_id("controller")
    for mask, strobe, index in itertools.product(range(256), range(2), range(9)):
        buttons = [(mask >> i) & 1 for i in range(8)]
        packet = [64 | (buttons[i] if i < 8 else 1) for i in range(index, index + 8)]
        reply = 64 | (buttons[0] if strobe else buttons[index] if index < 8 else 1)
        for event, value, count in (
            ("lease", 0, 0),
            ("read", 0, 0),
            ("consume", 0, 8),
            ("write", 0, 0),
            ("write", 255, 0),
        ):
            inputs = [strobe, index, *buttons, EVENTS.index(event), value, count]
            out = engine.step(inputs, engine.model_id("controller", event)).tolist()
            next_strobe, next_index = strobe, index
            if event == "write":
                next_strobe = value & 1
                if strobe or next_strobe:
                    next_index = 0
            elif not strobe:
                next_index = min(8, index + (1 if event == "read" else count))
            assert out == [next_strobe, next_index, reply, 0 if strobe else 8 - index, *packet]
            assert out == engine.step(inputs, full).tolist()


@pytest.mark.parametrize("backend", ["reference", "weights"])
def test_mmc5_neural_ram_protection_and_multiplier(backend):
    hardware = load_hardware()
    code = []

    def write(addr, value):
        code.extend((0xA9, value, 0x8D, addr & 255, addr >> 8))

    def read(addr, destination):
        code.extend((0xAD, addr & 255, addr >> 8, 0x85, destination))

    write(0x5114, 2)  # RAM bank 2 at $8000; RAM bank 3 at $6000
    write(0x5113, 3)
    for addr, destination in ((0x8000, 0x10), (0x6000, 0x11)):
        write(addr, 0xFF)  # locked writes must repair the CPU image
        read(addr, destination)
    write(0x5102, 2)
    write(0x5103, 1)
    write(0x8000, 0xA5)
    write(0x6000, 0x5A)
    write(0x5103, 0)
    for addr, destination in ((0x8000, 0x12), (0x6000, 0x13)):
        write(addr, 0xFF)
        read(addr, destination)
    write(0x5114, 0x82)  # ROM bank 2, then return to the saved RAM bank
    read(0x8000, 0x14)
    write(0x5114, 2)
    read(0x8000, 0x15)
    write(0x5205, 255)
    write(0x5206, 255)
    read(0x5205, 0x16)
    read(0x5206, 0x17)
    spin = 0xE000 + len(code)
    code.extend((0x4C, spin & 255, spin >> 8))
    prg = bytearray(b"".join(bytes([bank]) * 0x2000 for bank in range(4)))
    prg[0x6000 : 0x6000 + len(code)] = bytes(code)
    prg[-4:-2] = b"\x00\xe0"
    header = bytes([0x4E, 0x45, 0x53, 0x1A, 2, 0, 0x50, 0]) + bytes(8)
    cart = hardware.Cartridge.parse(header + bytes(prg))
    console, _ = run_console(hardware, cart, backend, 2, script={})
    assert console.read_ram(0x10, 8) == bytes([0, 0, 0xA5, 0x5A, 2, 0xA5, 1, 254])


@pytest.mark.parametrize("backend", ["reference", "weights"])
def test_mmc5_bank_matcher_refreshes_writable_code(backend):
    hardware = load_hardware()
    code = []

    def write(address, value):
        code.extend((0xA9, value, 0x8D, address & 255, address >> 8))

    def remap():
        write(0x5114, 1)
        write(0x5114, 0)

    write(0x5102, 2)
    write(0x5103, 1)
    write(0x5114, 0)  # Writable physical RAM at $8000.
    for offset, value in enumerate((0xCA, 0xD0, 0xFD, 0x60)):  # DEX; BNE; RTS
        write(0x8000 + offset, value)
    remap()
    code.extend((0xA2, 5, 0x20, 0, 0x80, 0x86, 0x10))
    write(0x8000, 0xE8)  # Replace DEX with INX in the same physical allocation.
    remap()
    code.extend((0xA2, 250, 0x20, 0, 0x80, 0x86, 0x11))
    spin = 0xE000 + len(code)
    code.extend((0x4C, spin & 255, spin >> 8))
    prg = bytearray(0x8000)
    prg[0x6000:0x6000 + len(code)] = bytes(code)
    prg[-4:-2] = b"\x00\xe0"
    header = bytes([0x4E, 0x45, 0x53, 0x1A, 2, 0, 0x50, 0]) + bytes(8)
    cart = hardware.Cartridge.parse(header + prg)
    matched = []
    build = backend_factory(backend)

    def factory(image, start):
        runtime, engine = build(image, start)
        if engine:
            original = runtime.region_windows

            def match(base, data):
                if base == 0x8000:
                    matched.append(data[:4])
                return original(base, data)

            runtime.region_windows = match
        return runtime, engine

    console = hardware.Console(cart, factory, 44100)
    console.run_frame()
    assert console.read_ram(0x10, 2) == b"\0\0"
    if backend == "weights":
        assert bytes((0xCA, 0xD0, 0xFD, 0x60)) in matched
        assert bytes((0xE8, 0xD0, 0xFD, 0x60)) in matched
