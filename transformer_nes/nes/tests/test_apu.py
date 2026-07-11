"""APU synthesis and register semantics (nes/apu.cpp)."""

from __future__ import annotations

import numpy as np

from transformer_nes.nes.apu import make_apu

SR = 44100.0
CPU_HZ = 1789772.727272727
FRAME_CYCLES = 29781


def pulse1_timer(freq: float) -> int:
    return round(CPU_HZ / (16 * freq)) - 1


def test_idle_apu_is_silent():
    apu = make_apu(SR)
    sig = np.asarray(apu.end_frame(FRAME_CYCLES))
    assert len(sig) in (733, 734)  # ~44100/60.1, fractional accumulator
    assert np.ptp(sig) == 0.0


def test_pulse_tone_frequency_and_centered():
    apu = make_apu(SR)
    t = pulse1_timer(440.0)
    apu.write(0x4015, 0x01)                        # enable pulse 1
    apu.write(0x4000, 0x9F)                        # duty 50%, constant vol 15
    apu.write(0x4002, t & 0xFF)
    apu.write(0x4003, ((t >> 8) & 7) | (1 << 3))   # timer high + length load

    sig = np.concatenate([np.asarray(apu.end_frame(FRAME_CYCLES)) for _ in range(30)])
    crossings = int(np.sum((sig[:-1] < 0) & (sig[1:] >= 0)))
    freq = crossings / (len(sig) / SR)
    assert abs(freq - 440.0) < 5.0          # integer-quantized timer, ~exact
    assert abs(float(sig.mean())) < 0.05    # DC blocker centers the output
    assert np.ptp(sig) > 0.1                # audibly loud


def test_length_counter_silences_note():
    apu = make_apu(SR)
    t = pulse1_timer(440.0)
    apu.write(0x4015, 0x01)
    apu.write(0x4000, 0x9F)                 # halt=0 -> length counts down
    apu.write(0x4002, t & 0xFF)
    apu.write(0x4003, (3 << 3))             # LENGTH_TABLE[3] = 2 half-frames
    first = np.ptp(np.asarray(apu.end_frame(FRAME_CYCLES)))
    # Silence, then let the high-pass tail decay (~3 frames) before checking.
    for _ in range(6):
        settled = np.ptp(np.asarray(apu.end_frame(FRAME_CYCLES)))
    assert first > 0.1        # sounds at first
    assert settled < 0.01     # length counter expired -> no oscillation left


def test_length_halt_sustains():
    apu = make_apu(SR)
    t = pulse1_timer(440.0)
    apu.write(0x4015, 0x01)
    apu.write(0x4000, 0xBF)                 # halt=1 (bit 5) -> length frozen
    apu.write(0x4002, t & 0xFF)
    apu.write(0x4003, (3 << 3))
    apu.end_frame(FRAME_CYCLES)
    later = np.ptp(np.asarray(apu.end_frame(FRAME_CYCLES)))
    assert later > 0.1        # still sounding many frames later


def test_disable_via_4015_clears_length():
    apu = make_apu(SR)
    t = pulse1_timer(440.0)
    apu.write(0x4015, 0x01)
    apu.write(0x4000, 0xBF)
    apu.write(0x4002, t & 0xFF)
    apu.write(0x4003, (1 << 3))
    assert np.ptp(np.asarray(apu.end_frame(FRAME_CYCLES))) > 0.1
    apu.write(0x4015, 0x00)                 # disable all channels
    for _ in range(6):                      # let the high-pass tail decay
        settled = np.ptp(np.asarray(apu.end_frame(FRAME_CYCLES)))
    assert settled < 0.01


def test_status_read_reports_length_and_clears_frame_irq():
    apu = make_apu(SR)
    apu.write(0x4017, 0x00)                 # 4-step mode, IRQ enabled
    apu.write(0x4015, 0x01)
    apu.write(0x4000, 0xBF)
    apu.write(0x4003, (1 << 3))             # load pulse-1 length
    assert apu.read(0x4015) & 0x01          # pulse 1 length active
    # Run a full sequence so the frame IRQ flag sets, then confirm read-clear.
    for _ in range(2):
        apu.end_frame(FRAME_CYCLES)
    assert apu.irq_pending
    status = apu.read(0x4015)
    assert status & 0x40                    # frame IRQ bit was set
    assert not apu.irq_pending              # reading $4015 cleared it


def test_irq_inhibit():
    apu = make_apu(SR)
    apu.write(0x4017, 0x40)                 # inhibit frame IRQ
    for _ in range(3):
        apu.end_frame(FRAME_CYCLES)
    assert not apu.irq_pending
