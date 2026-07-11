"""APU synthesis and register semantics (nes/apu.cpp)."""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

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


def test_dmc_fetches_sample_and_generates_output():
    apu = make_apu(SR)
    reads = []

    def read_cpu(addr: int) -> int:
        reads.append(addr)
        return 0xFF

    apu.write(0x4010, 0x0F)                 # fastest rate, no loop/IRQ
    apu.write(0x4011, 0x20)                 # starting DAC level
    apu.write(0x4012, 0x00)                 # sample starts at $C000
    apu.write(0x4013, 0x00)                 # one byte
    apu.write(0x4015, 0x10)                 # enable/restart DMC
    sig = np.asarray(apu.end_frame(1000, read_cpu))

    assert reads == [0xC000]
    assert apu.take_dma_stall() == 4
    assert apu.take_dma_stall() == 0         # stalls drain exactly once
    assert np.ptp(sig) > 0.001               # $FF raises the 7-bit DAC
    assert not (apu.read(0x4015) & 0x10)     # no bytes remain


def test_dmc_irq_and_disable_acknowledgement():
    apu = make_apu(SR)

    def read_cpu(_addr: int) -> int:
        return 0x00

    apu.write(0x4010, 0x8F)                 # DMC IRQ enabled
    apu.write(0x4012, 0x00)
    apu.write(0x4013, 0x00)
    apu.write(0x4015, 0x10)
    apu.end_frame(1, read_cpu)              # the one-byte DMA completes

    assert apu.irq_pending
    assert apu.read(0x4015) & 0x80
    assert apu.irq_pending                  # status reads do not clear DMC IRQ
    apu.write(0x4015, 0x00)
    assert not apu.irq_pending              # $4015 write acknowledges it


def test_dmc_loop_restarts_without_irq():
    apu = make_apu(SR)
    reads = []

    def read_cpu(addr: int) -> int:
        reads.append(addr)
        return 0xAA

    apu.write(0x4010, 0xCF)                 # loop + IRQ-enable (loop wins)
    apu.write(0x4012, 0x01)                 # $C040
    apu.write(0x4013, 0x00)                 # one byte per loop
    apu.write(0x4015, 0x10)
    apu.end_frame(2000, read_cpu)

    assert len(reads) >= 2
    assert set(reads) == {0xC040}
    assert not apu.irq_pending
    assert apu.read(0x4015) & 0x10          # loop remains active


def test_dmc_reader_is_released_when_callback_raises():
    class BrokenReader:
        def __call__(self, _addr: int) -> int:
            raise RuntimeError("sample read failed")

    apu = make_apu(SR)
    apu.write(0x4012, 0)
    apu.write(0x4013, 0)
    apu.write(0x4015, 0x10)
    reader = BrokenReader()
    ref = weakref.ref(reader)
    with pytest.raises(RuntimeError, match="sample read failed"):
        apu.end_frame(1, reader)
    del reader
    gc.collect()
    assert ref() is None
