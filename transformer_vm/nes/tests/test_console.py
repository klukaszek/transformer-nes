"""Console integration tests: frame loop, vblank NMI, DMA, and input.

A hand-built NROM cart exercises the whole system deterministically
(palette + CHR RAM + nametable writes through $2006/$2007, OAM DMA from
the NMI handler, sprite-0 overlap); the kernel-driven console and the
reference-driven console must produce identical frames, NMI counts, and
cycle clocks.  The real nestest ROM then boots through its reset vector
and runs its full official-opcode suite interactively via the controller.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from transformer_vm.nes.cartridge import Cartridge
from transformer_vm.nes.console import FRAME_CYCLES, Console

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
NESTEST_ROM = os.path.join(EXAMPLES_DIR, "nestest.nes")


def _test_cart() -> Cartridge:
    """PRG that draws two bg tiles and a sprite-0 overlapping the first."""
    code = bytearray()

    def emit(*bs: int) -> None:
        code.extend(bs)

    def ppuaddr(hi: int, lo: int) -> None:
        emit(0xA9, hi, 0x8D, 0x06, 0x20, 0xA9, lo, 0x8D, 0x06, 0x20)

    def data(value: int) -> None:
        emit(0xA9, value, 0x8D, 0x07, 0x20)

    org = 0x8000
    emit(0xA2, 0xFF, 0x9A)                    # LDX #$FF; TXS
    emit(0xA9, 0x00, 0x8D, 0x00, 0x20)        # PPUCTRL = 0 (NMI off during init)
    ppuaddr(0x3F, 0x00)                       # bg palette: backdrop + 3 colors
    data(0x0F)
    data(0x20)
    data(0x21)
    data(0x22)
    ppuaddr(0x3F, 0x11)                       # sprite palette color 1
    data(0x16)
    ppuaddr(0x00, 0x10)                       # CHR RAM tile 1: solid color 1
    emit(0xA2, 0x08, 0xA9, 0xFF)              # LDX #8; LDA #$FF
    emit(0x8D, 0x07, 0x20, 0xCA, 0xD0, 0xFA)  # loop: STA $2007; DEX; BNE loop
    ppuaddr(0x20, 0x41)                       # nametable row 2, cols 1-2
    data(0x01)
    data(0x01)                                # autoincrement lands at $2042
    for offset, value in ((0x00, 15), (0x01, 1), (0x02, 0), (0x03, 8)):
        emit(0xA9, value, 0x8D, offset, 0x02)  # sprite 0 in the $0200 page
    emit(0xA9, 0x00, 0x8D, 0x05, 0x20, 0x8D, 0x05, 0x20)  # PPUSCROLL 0,0
    emit(0xA9, 0x1E, 0x8D, 0x01, 0x20)        # PPUMASK: bg + sprites
    emit(0xA9, 0x80, 0x8D, 0x00, 0x20)        # PPUCTRL: NMI on
    main = org + len(code)
    emit(0xE6, 0x10, 0x4C, main & 0xFF, main >> 8)  # main: INC $10; JMP main
    nmi = org + len(code)
    emit(0xE6, 0x11)                          # INC $11 (NMI counter)
    emit(0xAD, 0x02, 0x20)                    # LDA $2002 (ack)
    emit(0xA9, 0x02, 0x8D, 0x14, 0x40)        # OAM DMA from $0200
    emit(0x40)                                # RTI

    prg = bytearray(0x4000)
    prg[:len(code)] = code
    prg[0x3FFA:0x3FFC] = bytes((nmi & 0xFF, nmi >> 8))
    prg[0x3FFC:0x3FFE] = bytes((0x00, 0x80))
    prg[0x3FFE:0x4000] = bytes((0x00, 0x80))
    return Cartridge(prg=bytes(prg), chr=b"", mirroring="vertical", mapper=0)


def _ram(console: Console, addr: int) -> int:
    if console.backend == "kernel":
        return console._rt.read_ram(addr)
    return console.bus.ram[addr]


def _run_frames(backend: str, frames: int):
    console = Console(_test_cart(), backend=backend)
    rendered = [console.run_frame() for _ in range(frames)]
    return console, rendered


def _check_final_frame(console: Console, frame) -> None:
    assert frame[0, 0] == 0x0F                    # backdrop
    assert frame[18, 18] == 0x20                  # bg tile at row 2, col 2
    assert frame[18, 10] == 0x16                  # sprite 0 over the col-1 tile
    assert console._sprite0_scanline == 16        # overlap begins at scanline 16
    assert bytes(console.ppu.oam[0:4]) == bytes((15, 1, 0, 8))


@pytest.mark.slow
def test_kernel_console_runs_the_test_cart():
    console, rendered = _run_frames("kernel", 4)
    _check_final_frame(console, rendered[-1])
    assert _ram(console, 0x11) == 4               # one NMI per frame
    assert _ram(console, 0x10) > 0                # main loop ran
    assert console.cycles >= 4 * FRAME_CYCLES
    assert not console.halted


@pytest.mark.slow
def test_reference_console_matches_kernel_console():
    kernel, kernel_frames = _run_frames("kernel", 4)
    ref, ref_frames = _run_frames("reference", 4)
    for i, (a, b) in enumerate(zip(kernel_frames, ref_frames, strict=True)):
        assert np.array_equal(a, b), f"frame {i} differs"
    assert kernel.cycles == ref.cycles
    assert _ram(kernel, 0x10) == _ram(ref, 0x10)
    assert _ram(kernel, 0x11) == _ram(ref, 0x11) == 4
    assert bytes(kernel.ppu.oam) == bytes(ref.ppu.oam)
    assert bytes(kernel.ppu.nametables) == bytes(ref.ppu.nametables)
    _check_final_frame(ref, ref_frames[-1])


@pytest.mark.slow
def test_graph_console_matches_reference_console():
    graph, graph_frames = _run_frames("graph", 2)
    ref, ref_frames = _run_frames("reference", 2)
    for i, (a, b) in enumerate(zip(graph_frames, ref_frames, strict=True)):
        assert np.array_equal(a, b), f"frame {i} differs"
    assert graph.cycles == ref.cycles
    assert _ram(graph, 0x11) == _ram(ref, 0x11) == 2


@pytest.mark.slow
def test_checkpoint_is_invisible():
    """Snapshot-and-rebuild must not change a single pixel, byte, or cycle."""
    plain = Console(_test_cart(), backend="kernel", checkpoint_tokens=0)
    plain_frames = [plain.run_frame() for _ in range(6)]

    ck = Console(_test_cart(), backend="kernel", checkpoint_tokens=1)  # every frame
    ck_frames = [ck.run_frame() for _ in range(6)]

    assert ck.checkpoint_count == 6
    for i, (a, b) in enumerate(zip(plain_frames, ck_frames, strict=True)):
        assert np.array_equal(a, b), f"frame {i} differs"
    assert plain.cycles == ck.cycles
    assert _ram(plain, 0x10) == _ram(ck, 0x10)
    assert _ram(plain, 0x11) == _ram(ck, 0x11)
    assert bytes(plain.ppu.oam) == bytes(ck.ppu.oam)
    # The log actually shrank back to the RAM reseed each frame.
    assert ck.bus.log_tokens < plain.bus.log_tokens


@pytest.mark.slow
def test_checkpoint_is_invisible_on_graph_backend():
    ref = Console(_test_cart(), backend="reference")
    ref_frames = [ref.run_frame() for _ in range(2)]

    graph = Console(_test_cart(), backend="graph", checkpoint_tokens=1)
    graph_frames = [graph.run_frame() for _ in range(2)]

    assert graph.checkpoint_count == 2
    for i, (a, b) in enumerate(zip(ref_frames, graph_frames, strict=True)):
        assert np.array_equal(a, b), f"frame {i} differs"
    assert ref.cycles == graph.cycles
    assert _ram(ref, 0x11) == _ram(graph, 0x11) == 2


@pytest.mark.slow
def test_backend_handoff_is_invisible():
    """A mid-game handoff to another CPU backend changes nothing observable."""
    straight = Console(_test_cart(), backend="reference")
    straight_frames = [straight.run_frame() for _ in range(4)]

    mixed = Console(_test_cart(), backend="kernel")
    mixed_frames = [mixed.run_frame() for _ in range(2)]
    mixed.handoff("reference")
    mixed_frames += [mixed.run_frame() for _ in range(2)]

    for i, (a, b) in enumerate(zip(straight_frames, mixed_frames, strict=True)):
        assert np.array_equal(a, b), f"frame {i} differs"
    assert straight.cycles == mixed.cycles
    assert _ram(straight, 0x11) == _ram(mixed, 0x11) == 4


def _nametable_text(console: Console) -> str:
    nt = console.ppu.nametables[:0x3C0]
    return "\n".join(
        "".join(chr(b) if 0x20 <= b < 0x7F else " " for b in nt[r * 32:(r + 1) * 32])
        for r in range(30)
    )


@pytest.mark.slow
def test_nestest_boots_and_passes_interactively():
    console = Console(NESTEST_ROM, backend="kernel")
    console.run(5)
    menu = _nametable_text(console)
    assert "Run all tests" in menu
    assert "Start: run test" in menu

    console.controller.set_buttons(start=True)
    console.run(2)
    console.controller.set_buttons(start=False)
    console.run(30)
    results = _nametable_text(console)
    assert "OK Run all tests" in results
    assert "OK Absolute,X tests" in results
    assert "Er" not in results  # nestest prints error codes as "Er ##h"
    assert not console.halted
