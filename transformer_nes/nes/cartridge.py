"""iNES cartridge loader.

Parses the header and raw PRG/CHR data; the CPU-visible image and any
banking live in `nes/mappers.py` (NROM covers the launch-era library and
the test ROMs; MMC1/MMC2/MMC3 add banked cartridge families). A
CHR size of zero means the cart has 8 KiB of CHR RAM.
"""

from __future__ import annotations

from dataclasses import dataclass

MIRROR_HORIZONTAL = "horizontal"
MIRROR_VERTICAL = "vertical"


@dataclass(frozen=True)
class Cartridge:
    prg: bytes                 # 16 KiB banks (mapper-dependent count)
    chr: bytes                 # CHR ROM, or b"" for 8 KiB CHR RAM
    mirroring: str             # nametable arrangement from the header
    mapper: int
    battery: bool = False      # battery-backed PRG RAM at $6000-$7FFF

    @classmethod
    def load(cls, path: str) -> Cartridge:
        with open(path, "rb") as f:
            data = f.read()
        return cls.parse(data)

    @classmethod
    def parse(cls, data: bytes) -> Cartridge:
        if data[:4] != b"NES\x1a":
            raise ValueError("not an iNES file")
        prg_banks, chr_banks, flags6, flags7 = data[4], data[5], data[6], data[7]
        mapper = (flags7 & 0xF0) | (flags6 >> 4)
        offset = 16 + (512 if flags6 & 0x04 else 0)  # skip trainer if present
        prg_size = prg_banks * 0x4000
        chr_size = chr_banks * 0x2000
        prg = data[offset:offset + prg_size]
        chr_rom = data[offset + prg_size:offset + prg_size + chr_size]
        if len(prg) != prg_size or len(chr_rom) != chr_size:
            raise ValueError("truncated iNES file")
        mirroring = MIRROR_VERTICAL if flags6 & 0x01 else MIRROR_HORIZONTAL
        return cls(prg=prg, chr=chr_rom, mirroring=mirroring, mapper=mapper,
                   battery=bool(flags6 & 0x02))
