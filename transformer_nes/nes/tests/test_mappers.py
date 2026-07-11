"""Mapper register protocols (nes/mappers.py) against synthetic carts."""

from __future__ import annotations

import pytest

from transformer_nes.nes.cartridge import Cartridge
from transformer_nes.nes.mappers import BANK8, BANK16, make_mapper


def prg_of(n_banks: int, bank_size: int) -> bytes:
    """PRG where every byte of bank i reads i."""
    return b"".join(bytes([i]) * bank_size for i in range(n_banks))


def wired(mapper):
    """Wire the mapper to a 64 KiB memory via its own initial image."""
    memory = bytearray(mapper.cpu_image())

    def poke(addr, value):
        memory[addr] = value
    mapper.poke = poke
    mapper.on_mirroring = lambda mode: events.append(mode)
    events = []
    return memory, events


def mmc1_serial(mapper, addr: int, value: int) -> None:
    for i in range(5):
        mapper.write(addr, (value >> i) & 1)


def test_mmc1_initial_image_fixes_last_bank():
    cart = Cartridge(prg=prg_of(16, BANK16), chr=b"", mirroring="horizontal", mapper=1)
    mapper = make_mapper(cart)
    image = mapper.cpu_image()
    assert image[0x8000] == 0 and image[0xC000] == 15
    assert mapper.prg_banks() == (0, 15)


def test_mmc1_prg_switch_and_repair():
    cart = Cartridge(prg=prg_of(16, BANK16), chr=b"", mirroring="horizontal", mapper=1)
    mapper = make_mapper(cart)
    memory, _ = wired(mapper)
    mmc1_serial(mapper, 0xE000, 5)  # PRG reg = bank 5
    assert mapper.prg_banks() == (5, 15)
    assert memory[0x8000:0xC000] == bytes([5]) * BANK16   # bank injected
    assert memory[0xC000:0x10000] == bytes([15]) * BANK16  # fixed bank intact
    assert memory[0xE000] == 15  # register write repaired to ROM byte


def test_mmc1_control_modes_and_mirroring():
    cart = Cartridge(prg=prg_of(16, BANK16), chr=b"", mirroring="horizontal", mapper=1)
    mapper = make_mapper(cart)
    memory, events = wired(mapper)
    mmc1_serial(mapper, 0x8000, 0x0E)  # PRG mode 3, mirroring 2 (vertical)
    assert events == [2]
    mmc1_serial(mapper, 0xE000, 3)
    mmc1_serial(mapper, 0x8000, 0x06)  # mode 1: 32K switch, bank pair (2,3)
    assert mapper.prg_banks() == (2, 3)
    assert memory[0x8000] == 2 and memory[0xC000] == 3
    mmc1_serial(mapper, 0x8000, 0x0A)  # mode 2: first bank fixed at $8000
    assert mapper.prg_banks() == (0, 3)


def test_mmc1_reset_bit():
    cart = Cartridge(prg=prg_of(16, BANK16), chr=b"", mirroring="horizontal", mapper=1)
    mapper = make_mapper(cart)
    wired(mapper)
    mapper.write(0xE000, 1)     # partial serial write...
    mapper.write(0x8000, 0x80)  # ...reset discards it, restores mode 3
    assert mapper.shift == 0x10
    assert (mapper.control & 0x0C) == 0x0C
    mmc1_serial(mapper, 0xE000, 7)  # protocol realigned
    assert mapper.prg_banks() == (7, 15)


def test_mmc3_bank_modes():
    cart = Cartridge(prg=prg_of(32, BANK8), chr=bytes(0x20000),
                     mirroring="horizontal", mapper=4)
    mapper = make_mapper(cart)
    memory, _ = wired(mapper)
    mapper.set_chr_banks = lambda banks: chr_sets.append(banks)
    chr_sets = []
    mapper.write(0x8000, 6)      # select R6
    mapper.write(0x8001, 12)     # R6 = bank 12
    assert mapper.prg_banks() == (12, 1, 30, 31)
    assert memory[0x8000] == 12
    mapper.write(0x8000, 0x46)   # PRG mode 1: R6 moves to $C000
    assert mapper.prg_banks() == (30, 1, 12, 31)
    assert memory[0x8000] == 30 and memory[0xC000] == 12
    assert memory[0xE000:0x10000] == bytes([31]) * BANK8


def test_mmc3_chr_arrangement_and_mirroring():
    cart = Cartridge(prg=prg_of(4, BANK8), chr=bytes(0x20000),
                     mirroring="horizontal", mapper=4)
    mapper = make_mapper(cart)
    _, events = wired(mapper)
    chr_sets = []
    mapper.set_chr_banks = lambda banks: chr_sets.append(banks)
    mapper.write(0x8000, 0)      # select R0
    mapper.write(0x8001, 9)      # 2K banks force even: 8,9
    assert chr_sets[-1] == [8, 9, 2, 3, 4, 5, 6, 7]
    mapper.write(0x8000, 0x80)   # A12 inversion: 1K slots first
    assert chr_sets[-1] == [4, 5, 6, 7, 8, 9, 2, 3]
    mapper.write(0xA000, 0)
    assert events[-1] == 2       # vertical
    mapper.write(0xA000, 1)
    assert events[-1] == 3       # horizontal


def test_mmc1_rejects_chr_rom():
    cart = Cartridge(prg=prg_of(2, BANK16), chr=bytes(0x2000),
                     mirroring="horizontal", mapper=1)
    with pytest.raises(NotImplementedError):
        make_mapper(cart)


def test_mmc2_prg_switch_fixed_banks_and_repair():
    cart = Cartridge(prg=prg_of(16, BANK8), chr=bytes(0x20000),
                     mirroring="horizontal", mapper=9)
    mapper = make_mapper(cart)
    memory, _ = wired(mapper)
    assert mapper.prg_banks() == (0, 13, 14, 15)
    mapper.write(0xA123, 7)
    assert mapper.prg_banks() == (7, 13, 14, 15)
    assert memory[0x8000:0xA000] == bytes([7]) * BANK8
    assert memory[0xA000] == 13 and memory[0xE000] == 15
    assert memory[0xA123] == 13


def test_mmc2_chr_registers_and_mirroring():
    cart = Cartridge(prg=prg_of(4, BANK8), chr=bytes(0x20000),
                     mirroring="horizontal", mapper=9)
    mapper = make_mapper(cart)
    _, events = wired(mapper)
    chr_sets = []
    mapper.set_mmc2_banks = lambda banks: chr_sets.append(banks)
    for addr, value in zip((0xB000, 0xC000, 0xD000, 0xE000),
                           (3, 5, 7, 9), strict=True):
        mapper.write(addr, value)
    assert mapper.mmc2_banks() == [3, 5, 7, 9]
    assert chr_sets[-1] == [3, 5, 7, 9]
    mapper.write(0xF000, 0)
    mapper.write(0xF000, 1)
    assert events[-2:] == [2, 3]


def test_mmc4_prg_chr_and_mirroring():
    cart = Cartridge(prg=prg_of(16, BANK16), chr=bytes(0x20000),
                     mirroring="horizontal", mapper=10)
    mapper = make_mapper(cart)
    memory, events = wired(mapper)
    chr_sets = []
    mapper.set_mmc4_banks = lambda banks: chr_sets.append(banks)
    assert mapper.prg_banks() == (0, 15)
    mapper.write(0xA123, 6)
    assert mapper.prg_banks() == (6, 15)
    assert memory[0x8000:0xC000] == bytes([6]) * BANK16
    for addr, value in zip((0xB000, 0xC000, 0xD000, 0xE000),
                           (2, 4, 6, 8), strict=True):
        mapper.write(addr, value)
    assert chr_sets[-1] == [2, 4, 6, 8]
    mapper.write(0xF000, 0)
    mapper.write(0xF000, 1)
    assert events[-2:] == [2, 3]
