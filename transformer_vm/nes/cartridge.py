"""iNES cartridge loader (mapper 0 / NROM).

NROM covers the launch-era library (Donkey Kong, Super Mario Bros, and the
test ROMs this project verifies against): 16 or 32 KiB of PRG ROM at
$8000-$FFFF (16 KiB carts mirror it at $C000) and 8 KiB of CHR at PPU
$0000-$1FFF.  A CHR size of zero means the cart has 8 KiB of CHR RAM.
"""

from __future__ import annotations

from dataclasses import dataclass

MIRROR_HORIZONTAL = "horizontal"
MIRROR_VERTICAL = "vertical"


@dataclass(frozen=True)
class Cartridge:
    prg: bytes                 # 16 KiB or 32 KiB
    chr: bytes                 # 8 KiB CHR ROM, or b"" for CHR RAM
    mirroring: str             # nametable arrangement
    mapper: int

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
        if mapper != 0:
            raise NotImplementedError(f"mapper {mapper} not supported (NROM only)")
        if prg_banks not in (1, 2):
            raise ValueError(f"NROM expects 1-2 PRG banks, got {prg_banks}")
        offset = 16 + (512 if flags6 & 0x04 else 0)  # skip trainer if present
        prg_size = prg_banks * 0x4000
        chr_size = chr_banks * 0x2000
        prg = data[offset:offset + prg_size]
        chr_rom = data[offset + prg_size:offset + prg_size + chr_size]
        if len(prg) != prg_size or len(chr_rom) != chr_size:
            raise ValueError("truncated iNES file")
        mirroring = MIRROR_VERTICAL if flags6 & 0x01 else MIRROR_HORIZONTAL
        return cls(prg=prg, chr=chr_rom, mirroring=mirroring, mapper=mapper)

    def cpu_image(self) -> bytes:
        """The 64 KiB CPU address space with PRG mapped at $8000-$FFFF."""
        image = bytearray(0x10000)
        image[0x8000:0x8000 + len(self.prg)] = self.prg
        if len(self.prg) == 0x4000:  # NROM-128: mirror at $C000
            image[0xC000:0x10000] = self.prg
        return bytes(image)

    @property
    def reset_vector(self) -> int:
        image = self.cpu_image()
        return image[0xFFFC] | (image[0xFFFD] << 8)
