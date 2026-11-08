"""The PPU render graph: folded experts against the full graph, and
graph-rendered frames against the native PPU."""

import os

import numpy as np
import pytest
from carts import (
    absolute,
    cart,
    console_frames,
    last_bank_cart,
    latch_cart,
    load_reference_hardware,
    mmc5_cart,
    register_cart,
)

from transformer_nes.build import FIXTURES_DIR, load_nes_extension
from transformer_nes.nes.backends import backend_factory
from transformer_nes.ppu2c02 import render
from transformer_nes.ppu2c02.machine import ppu_engine

EXAMPLES = str(FIXTURES_DIR / "roms")


def random_slot(rng, pins: dict) -> list[int]:
    """SLOT inputs drawn at random where an expert's pins hold."""
    first, last = pins.get("first_slot"), pins.get("last_slot")
    slot = 0 if first else 31 if last else int(rng.choice([*range(1, 31), 32, 33]))
    mask = int(rng.integers(256))
    for bit in range(8):
        if f"maskbit{bit}" in pins:
            mask = (mask & ~(1 << bit)) | (pins[f"maskbit{bit}"] << bit)
    ctrl = int(rng.integers(256))
    if "ctrlbit4" in pins:
        ctrl = (ctrl & ~0x10) | (pins["ctrlbit4"] << 4)
    fine_x = int(rng.integers(8))
    if "fine_x_bit0" in pins:
        fine_x = sum(pins[f"fine_x_bit{b}"] << b for b in range(3))
    # Background pixels: 0 (transparent) or palette * 4 + value 1-3.
    bg = [
        0 if rng.random() < 0.3 else int(rng.integers(4)) * 4 + int(rng.integers(1, 4))
        for _ in range(16)
    ]
    sprites = [0] * 8
    if "sprite_pixel0" not in pins:
        sprites = [
            0
            if rng.random() < 0.4
            else 16 + int(rng.integers(16)) | int(rng.integers(2)) << 6 | int(rng.integers(2)) << 7
            for _ in range(8)
        ]
    v = [int(rng.integers(32)), int(rng.integers(32)), int(rng.integers(4)), int(rng.integers(8))]
    head = [1, slot, int(rng.integers(2)), fine_x, ctrl, mask]
    pages = [int(rng.integers(4)) for _ in range(4)]
    latches = [int(rng.integers(2)), int(rng.integers(2)), int(rng.integers(2))]
    if "latch" in pins:  # carts without CHR latches
        latches = [0, 0, 0]
    mmc5 = [
        int(rng.integers(2)) if pins.get("rendering") == 0 else 0,
        int(rng.integers(4)),
    ]  # extended mode pinned off; chr_upper
    return head + pages + latches + mmc5 + v + bg + sprites


def test_experts_agree_with_the_full_graph():
    engine = ppu_engine()
    rng = np.random.default_rng(7)
    for lo, hi in ((0, 0x3000), (0x5000, 0x7000)):
        for addr in range(lo, hi):
            engine.write(addr, int(rng.integers(256)))
    for addr in range(0x3F00, 0x3F20):
        engine.write(addr, int(rng.integers(64)))
    for model, name in enumerate(engine.expert_names, start=1):
        for _ in range(300):
            inputs = random_slot(rng, render.EXPERTS[name])
            full, folded = engine.step(inputs, 0), engine.step(inputs, model)
            projected = engine.step(inputs, engine.model_id("render_status", name))
            assert np.array_equal(projected, folded[8:])
            if "latch" in render.EXPERTS[name]:  # new latches unused without them
                full, folded = full[:-2], folded[:-2]
            assert np.array_equal(full, folded), (name, inputs, full, folded)


def whole_ppu(cart, count):
    """Frames through the production console's mandatory transformer PPU."""
    return console_frames(load_nes_extension("hardware"), cart, count)


def test_dma_wrap_mirrors_parity_and_destination_after_render_catchup():
    from transformer_nes.cpu6502.reference import Cpu6502, instruction_cycles
    from transformer_nes.cpu6502.spec import opcode_for

    # The first request ends at CPU cycle 89 (dot 265), crossing dot 257
    # after OAMADDR was set to $FF at cycle 83. Rendering resets it to zero.
    code = [0xA9, 0x18] + absolute(0x8D, 0x2001) + [0xEA] * 34 + [0x24, 0]
    code += [0xA9, 0xFF] + absolute(0x8D, 0x2003)
    code += [0xA9, 0xB0] + absolute(0x8D, 0x4014)
    code += [0xA9, 0] + absolute(0x8D, 0x2001)

    def capture(index):
        nonlocal code
        # DMA preserves OAMADDR; its current byte should be source byte 0.
        code += absolute(0xAD, 0x2004) + absolute(0x8D, 0x7F00 + index)
        body = [0x8A] + absolute(0x8D, 0x2003) + absolute(0xAD, 0x2004)
        body += absolute(0x9D, 0x6000 + 256 * index) + [0xE8]
        code += [0xA2, 0] + body + [0xD0, (-len(body) - 2) & 255]

    rom_page = bytes((29 * i + 7) & 255 for i in range(256))
    expected = [rom_page]
    capture(0)
    # Seed $0200 and transfer it through all four internal-RAM mirrors.
    body = [0x8A, 0x49, 0xA5] + absolute(0x9D, 0x0200) + [0xE8]
    code += [0xA2, 0] + body + [0xD0, (-len(body) - 2) & 255]
    ram_page = bytes(i ^ 0xA5 for i in range(256))
    for index, (page, address) in enumerate(((2, 1), (10, 255), (18, 127), (26, 0)), 1):
        code += [0xA9, address] + absolute(0x8D, 0x2003)
        code += [0xA9, page] + absolute(0x8D, 0x4014)
        expected.append(ram_page[-address:] + ram_page[:-address] if address else ram_page)
        capture(index)
    code += [0x00]  # BRK to vector zero: the harness's halt convention
    prg = bytearray(0x4000)
    prg[:len(code)] = bytes(code)
    prg[0x3000:0x3100] = rom_page
    prg[-4:-2] = bytes((0, 0x80))
    image = cart(0, prg, bytes(0x2000))

    # Independent instruction-cycle oracle; MMIO data never affects a branch.
    cpu = Cpu6502(memory=bytearray(0x8000) + prg * 2, bcd_enabled=False, mirror_cpu_ram=True)
    cpu.pc = 0x8000
    cycles, parities = 0, set()
    while not cpu.halted:
        op = opcode_for(cpu.read_memory(cpu.pc))
        cycles += instruction_cycles(cpu, op, cpu.pc, cpu.x, cpu.y, cpu.p)
        if cpu.read_memory(cpu.pc) == 0x8D and cpu.memory[cpu.pc + 1:cpu.pc + 3] == b'\x14\x40':
            parities.add(cycles & 1)
            cycles += 513 + (cycles & 1)
        cpu.step()
    assert parities == {0, 1}
    for hardware in (load_reference_hardware(), load_nes_extension("hardware")):
        for backend in ("reference", "weights"):
            console = hardware.Console(hardware.Cartridge.parse(image), backend_factory(backend), 44100)
            for _ in range(10):
                console.run_frame()
                if console.halted:
                    break
            assert console.halted and console.cycles == cycles
            assert console.read_ram(0x6000, 256 * len(expected)) == b''.join(expected)
            assert console.read_ram(0x7F00, len(expected)) == bytes((rom_page[0], *[ram_page[0]] * 4))


@pytest.mark.parametrize(("mask_bits", "vertical"), [(0x18, False), (0x18, True), (0x1F, False)])
def test_graph_renders_the_synthetic_cart(mask_bits, vertical):
    """Every register: rendering toggles, scroll and address writes
    mid-frame, $2007 across CHR, nametables and palette, OAM, sprites,
    both mirrorings; with 0x1F also greyscale and left-edge clipping
    (the full render graph)."""
    hardware = load_reference_hardware()
    cart = register_cart(hardware, vertical=vertical, mask_bits=mask_bits)
    assert np.array_equal(console_frames(hardware, cart, 6), whole_ppu(cart, 6))


@pytest.mark.parametrize("mapper", [9, 10])
def test_graph_renders_latch_mappers(mapper):
    """MMC2/MMC4: the render graph switches CHR through the latches its
    own fetches set (and the sprite fetches between lines)."""
    hardware = load_reference_hardware()
    cart = latch_cart(hardware, mapper=mapper)
    assert np.array_equal(console_frames(hardware, cart, 6), whole_ppu(cart, 6))


@pytest.mark.parametrize("exram_mode", [0, 1, 2])
def test_graph_renders_mmc5(exram_mode):
    """MMC5: nametable maps over CIRAM, ExRAM and fill mode, both CHR
    sets, and (mode 1) extended attributes from the whole CHR ROM."""
    hardware = load_reference_hardware()
    cart = mmc5_cart(hardware, exram_mode=exram_mode)
    assert np.array_equal(console_frames(hardware, cart, 6), whole_ppu(cart, 6))


@pytest.mark.parametrize("mapper", [0, 9, 10, 5])
def test_unlimited_sprites_preserve_hardware_bus_effects(mapper):
    reference, production = load_reference_hardware(), load_nes_extension("hardware")
    setup = [0xA9, 0] + absolute(0x8D, 0x2001) + absolute(0x8D, 0x2003)
    for i in range(64):
        for value in (40, 0xFD + i % 2, i % 4, i * 4):
            setup += [0xA9, value] + absolute(0x8D, 0x2004)
    # Distinct FD/FE windows expose accidental latch flips by debug rows.
    if mapper in (9, 10):
        for k, address in enumerate((0xB000, 0xC000, 0xD000, 0xE000)):
            setup += [0xA9, k] + absolute(0x8D, address)
    setup += [0xA9, 0x1E] + absolute(0x8D, 0x2001) + [0xD0, 0xFE]  # render, then spin
    image = last_bank_cart(mapper, [], [], 0x10000, after_setup=setup)
    cart = reference.Cartridge.parse(image)
    limited = console_frames(reference, cart, 6)
    unlimited = console_frames(reference, cart, 6, lambda c: c.set_sprite_limit(False))
    actual = console_frames(production, cart, 6, lambda c: c.set_sprite_limit(False))
    assert not np.array_equal(limited, unlimited)
    assert np.array_equal(unlimited, actual)


def test_production_console_requires_the_transformer_ppu():
    hardware = load_nes_extension("hardware")
    console = hardware.Console(
        register_cart(hardware, vertical=False), backend_factory("weights"), 44100
    )
    assert console.ppu_engine is not None
    with pytest.raises(ValueError, match="transformer engine is required"):
        console.set_ppu_engine(None)


@pytest.mark.slow
@pytest.mark.parametrize(
    "rom",
    [
        "smb.nes",
        "legendofzelda.nes",
        "smb3.nes",
        "megaman6.nes",
        "punchout.nes",
        "castlevania3.nes",
    ],
)
def test_graph_renders_games(rom):
    """Background, compositing and the register file all as transformer
    weights: frames identical to the native PPU."""
    path = os.path.join(EXAMPLES, rom)
    if not os.path.exists(path):
        pytest.skip(f"{rom} not present")
    hardware = load_reference_hardware()
    cart = hardware.Cartridge.load(path)
    assert np.array_equal(console_frames(hardware, cart, 300), whole_ppu(cart, 300))


def test_slot_trace_explains_the_frame():
    """The X-ray's pixel dissection: a slot traced through the full model
    (on a copy of the engine) reports the color the frame shows."""
    from transformer_nes.frontend.xray import PpuXray

    hardware = load_nes_extension("hardware")
    console = hardware.Console(latch_cart(hardware, mapper=9), backend_factory("weights"), 44100.0)
    engine = ppu_engine()
    console.set_ppu_engine(engine)
    xray = PpuXray(engine)
    for _ in range(3):
        console.run_frame()
    for line, x in ((40, 37), (120, 130), (200, 251)):
        console.request_ppu_slot_trace(line, x // 8)
        frame = console.run_frame()
        trace = console.take_ppu_slot_trace()
        assert trace is not None and (trace["line"], trace["slot"]) == (line, x // 8)
        d = xray.dissect(trace, x)
        assert d["title"].endswith(f"color ${frame[line][x]:02X}"), d["title"]
        assert any(
            kind == "attention" and "palette" in detail for _, kind, _, detail, _ in d["steps"]
        )
    assert console.take_ppu_slot_trace() is None  # one trace per request


def test_knock_out_breaks_rendering_until_restored():
    """Ablation: the full model alone renders identically; forcing its
    palette-selection neuron to 0 changes the picture; restoring brings it back."""
    from transformer_nes.frontend.xray import PpuXray

    hardware = load_reference_hardware()
    cart = latch_cart(hardware, mapper=9)
    native = console_frames(hardware, cart, 8)
    console = hardware.Console(cart, backend_factory("weights"), 44100.0)
    engine = ppu_engine()
    console.set_ppu_engine(engine)
    xray = PpuXray(engine)
    out = []
    for i in range(8):
        if i == 2:
            console.ppu_force_full_render = True
        if i == 4:
            xray.knock_out(console, "palette_select+")
        if i == 6:
            xray.restore(console)
        console.set_buttons(0)
        out.append(console.run_frame().copy())
    assert np.array_equal(native[:4], out[:4])
    assert not np.array_equal(native[4:6], out[4:6])
    # The first frame after restoring still shows the two tiles prefetched
    # for its line 0 while ablated; from the next frame on, it is gone.
    assert np.array_equal(native[7:], out[7:])


@pytest.mark.parametrize("chr_full", [0, 0x20000], ids=["mapped-chr", "mmc5-chr"])
def test_blank_expert_preserves_palette_and_disables_fetch_effects(chr_full):
    engine = ppu_engine(chr_full)
    blank = engine.model_id("render", "blank")
    rng = np.random.default_rng(61)
    for address in range(0x3F00, 0x3F20):
        engine.write(address, int(rng.integers(64)))
    for case in range(1024):
        inputs = random_slot(rng, render.EXPERTS["blank"])
        inputs[1] = case % 34  # includes clipping, vertical wrap and prefetch slots
        inputs[5] = 0xE0 | (case & 1)  # rendering off, both greyscale states
        address = (0x3F00, 0x2000, 0x7F00)[case % 3] + case % 32
        start = render.INPUTS.index("v_cx")
        inputs[start : start + 4] = [
            address & 31,
            (address >> 5) & 31,
            (address >> 10) & 3,
            (address >> 12) & 7,
        ]
        full, folded = engine.step(inputs, 0), engine.step(inputs, blank)
        np.testing.assert_array_equal(full, folded)
        assert folded[render.OUTPUTS.index("fetched")] == 0
        assert not folded[8:16].any()  # inactive tile-effect payload
