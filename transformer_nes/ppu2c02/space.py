"""The PPU memory space the graphs read, and its addressing idioms.

Physical layout of the WRITE-token log (4 KiB attention groups):

    $0000-$1FFF   CHR window: the 8 KiB the pattern tables map now (under
                  MMC2/MMC4, the FD-latch banks)
    $2000-$2FFF   nametable sources, $400 each: CIRAM page 0, CIRAM page
                  1, MMC5 ExRAM (as the PPU sees it), MMC5 fill mode
    $3F00-$3F1F   palette RAM (mirrors resolved: $3F10 -> $3F00, ...)
    $4000-$40FF   OAM
    $5000-$6FFF   CHR window under the FE latches (MMC2/MMC4)
    $10000-       the whole CHR ROM (MMC5 extended attributes)

A mapper's nametable arrangement is the source each logical nametable
reads (MMC5's $5105 in general; plain mirroring as its special cases),
carried on the token as four page numbers.  The harness writes every
byte of a region before a graph reads it, so reads are dense
(graph/ops.py:grouped_read).
"""

from __future__ import annotations

from transformer_nes.graph import core as graph
from transformer_nes.graph.core import Expression, ExprLike, persist, stepglu
from transformer_nes.graph.ops import (
    both,
    either,
    equals,
    gate,
    grouped_read,
    memory_site,
    read_bits,
)

CHR = (0, 1)
PAGES = (2,)
PALETTE = (3,)
OAM = (4,)
CHR_ALT = (5, 6)
CHR_ALT_BASE = 0x5000
CHR_FULL_BASE = 0x10000
BASE_GROUPS = 7  # $0000-$6FFF


def memory_size(chr_full: int = 0) -> int:
    """Bytes of graph memory, with `chr_full` bytes of whole-CHR region."""
    return CHR_FULL_BASE + chr_full if chr_full else BASE_GROUPS * 0x1000


def chr_full_groups(chr_full: int) -> tuple[int, ...]:
    return tuple(range(CHR_FULL_BASE >> 12, (CHR_FULL_BASE + chr_full) >> 12))


class Memory:
    """A graph's WRITE-token inputs, and reads over them."""

    def __init__(self, is_write, addr, value, *, groups: int = BASE_GROUPS):
        self.is_write, self.addr, self.value = is_write, addr, value
        self.groups = groups

    def read(self, address: ExprLike, groups: tuple[int, ...], name: str) -> Expression:
        """The byte at `address` (within `groups`), as an annotated read
        site: structural folds serve it from the engine's memory."""
        value = grouped_read(
            address,
            self.value,
            self.addr,
            self.is_write,
            group_count=self.groups,
            groups=groups,
            dense=True,
        )
        return memory_site(persist(value, name=name), address, self.value, self.addr, self.is_write)

    @staticmethod
    def bits(read: Expression, name: str) -> list[Expression]:
        """A read's bits, MSB first, served with the read in folds."""
        return read_bits(read, name, wide=False)


def nametable_page(nt: ExprLike, pages: list[ExprLike]) -> Expression:
    """The source page of logical nametable `nt` (0-3), from the token's
    four page numbers."""
    one = graph.current_graph().one
    ge = [stepglu(one, nt - k) for k in (1, 2, 3)]
    onehot = [1 - ge[0], ge[0] - ge[1], ge[1] - ge[2], ge[2]]
    return sum((gate(h, p) for h, p in zip(onehot, pages, strict=True)), Expression())


def palette_offset(index: ExprLike) -> Expression:
    """Palette index 0-31 with $10/$14/$18/$1C folded onto $00/$04/..."""
    one = graph.current_graph().one
    b4 = stepglu(one, index - 16)
    b3 = stepglu(one, index - 16 * b4 - 8)
    b2 = stepglu(one, index - 16 * b4 - 8 * b3 - 4)
    low2 = index - 16 * b4 - 8 * b3 - 4 * b2
    return index - 16 * both(b4, equals(low2, 0))


def in_palette_region(v: dict[str, ExprLike]) -> Expression:
    """v & $3FFF >= $3F00, from loopy fields: fine y 3 or 7, nametable 3,
    coarse y >= 24."""
    one = graph.current_graph().one
    fy_low3 = either(equals(v["fy"], 3), equals(v["fy"], 7))
    return both(both(fy_low3, equals(v["nt"], 3)), stepglu(one, v["cy"] - 24))
