"""The overlay's X-ray: the transformers at work while a game runs.

Everything is read from the live engines:

- CPU paths: how each frame's instructions were retired -- fixed polling
  and counter shapes,
  per-opcode experts, interrupt experts, or the full union model.
- CPU memory: attention reads per address, as a decaying heatmap.
- Instruction dissection: the next instruction through the full CPU model
  on a copy of the engine, the residual stream read after every layer and
  labeled with the named values the graph computes there.
- PPU experts: which folded render model drew each 8-pixel slot.
- Pixel dissection: one tile slot through the full render model, likewise.
- Ablation: a named render value forced to a constant in every slot.
"""

from __future__ import annotations

import re
from functools import cached_property

import numpy as np

from transformer_nes.cpu6502.spec import OPCODES
from transformer_nes.graph.core import (
    InputDimension,
    LookUpDimension,
    MemoryReadDimension,
    PersistDimension,
    ReGLUDimension,
)
from transformer_nes.ppu2c02.render import EXPERTS

PATHS = ("shape expert", "interrupt expert", "opcode expert", "full model")
# Auto-generated names carry no meaning to a reader; only named values are
# listed (the anonymous ones still count toward each layer's activity).
_ANONYMOUS = re.compile(r"^(reglu|persist|lookup|dim|memory_read)_\d+")
# Values whose meaning is the whole instruction's input or result,
# shown even when the graph computes them early.
_HEADLINE = (
    "in_pc_lo",
    "in_pc_hi",
    "in_a",
    "in_x",
    "in_y",
    "in_s",
    "fetched_opcode",
    "fetched_operand_lo",
    "fetched_operand_hi",
    "ea",
    "new_a",
    "new_x",
    "new_y",
    "new_s",
    "next_pc",
)


def dimension_kind(dim) -> str:
    """Where a value comes from: the embedding, an attention read (also
    through persists that only carry one along), a ReGLU neuron, or a
    linear combination of earlier values."""
    if isinstance(dim, InputDimension):
        return "input"
    if isinstance(dim, (LookUpDimension, MemoryReadDimension)):
        return "attention"
    if isinstance(dim, ReGLUDimension):
        return "ffn"
    if getattr(dim, "memory_read", None) is not None:
        return "attention"  # a memory read (graph/ops.py:memory_site)
    if isinstance(dim, PersistDimension):
        terms = [t for t in dim.expr.terms if not isinstance(t, InputDimension)]
        if len(terms) == 1 and dimension_kind(terms[0]) == "attention":
            return "attention"
    return "linear"


def disassemble(memory: bytes, pc: int) -> str:
    op = OPCODES.get(memory[pc])
    if op is None:
        return f"${pc:04X}  .byte ${memory[pc]:02X}"
    operand = memory[pc + 1 : pc + op.length]
    value = int.from_bytes(operand, "little") if operand else 0
    forms = {
        "imm": f"#${value:02X}",
        "zpg": f"${value:02X}",
        "zpx": f"${value:02X},X",
        "zpy": f"${value:02X},Y",
        "abs": f"${value:04X}",
        "abx": f"${value:04X},X",
        "aby": f"${value:04X},Y",
        "ind": f"(${value:04X})",
        "idx": f"(${value:02X},X)",
        "idy": f"(${value:02X}),Y",
        "acc": "A",
        "imp": "",
        "rel": f"${(pc + 2 + (value - 256 if value > 127 else value)) & 0xFFFF:04X}",
    }
    return f"${pc:04X}  {op.mnemonic} {forms.get(op.mode, '')}".rstrip()


def memory_origin(dim):
    if isinstance(dim, MemoryReadDimension):
        return dim, dim.memory_read.address
    site = getattr(dim, "memory_read", None)
    if site is not None:
        return dim, site["address"]
    if isinstance(dim, PersistDimension):
        sources = [d for d in dim.expr.terms if not isinstance(d, InputDimension)]
        if len(sources) == 1:
            return memory_origin(sources[0])
    return None


def layer_steps(named, snap, f_bits, slot_of, read_label, headline=()) -> dict:
    """The named values of one traced forward, each at the row (0 the
    embedding, L + 1 after layer L) that produced it: (row, kind, name,
    detail, value), where a memory read's detail names the address its
    query asked for (read_label(address)); plus each layer's count of
    residual slots written."""
    values = snap.astype(np.float64) / (1 << f_bits)

    def evaluate(expr, row):
        total = 0.0
        for term, coeff in expr.terms.items():
            if term not in slot_of:
                return None
            total += coeff * values[row, slot_of[term]]
        return total

    steps = []
    first_row = {dim: first for dim, _slot, first, _end in named}
    for dim, slot, first, end in named:
        value = values[min(end, len(values)) - 1, slot]
        if value == 0.0 and dim.name not in headline:
            continue  # a decode line that stayed off
        detail = ""
        site = memory_origin(dim)
        if site is not None and first > 0:
            source, expression = site
            address = evaluate(expression, first_row.get(source, first) - 1)
            if address is not None:
                detail = read_label(int(round(address)))
        steps.append((int(first), dimension_kind(dim), dim.name, detail, float(value)))
    steps.sort(key=lambda s: (s[0], s[1] != "attention", s[2]))
    return {
        "layers": len(values) - 1,
        "d_model": values.shape[1],
        "steps": steps,
        "activity": [int(c) for c in (np.diff(snap, axis=0) != 0).sum(axis=1)],
    }


class CpuXray:
    """X-ray of the weights CPU (cpu6502/backends/weights/runner.py)."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.engine = runtime.engine.engine

    @cached_property
    def named(self):
        # Counters need only the running engine. Materialize the full graph
        # and its debug labels when a dissection actually asks for them.
        return [life for life in self.runtime.lives if not _ANONYMOUS.match(life[0].name)]

    def take_paths(self) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
        """Instructions per path since the last call, and the opcodes that
        ran singly (opcode experts or the full model), busiest first."""
        paths, opcodes, _dispatches = self.engine.xray_take()
        busiest = np.argsort(opcodes)[::-1][:8]
        top = [
            (
                f"{OPCODES[int(b)].mnemonic} {OPCODES[int(b)].mode}"
                if int(b) in OPCODES
                else f"${int(b):02X}",
                int(opcodes[b]),
            )
            for b in busiest
            if opcodes[b]
        ]
        return list(zip(PATHS, map(int, paths), strict=True)), top

    def take_reads(self) -> np.ndarray:
        """Attention reads per CPU address since the last call (counting
        stays on from the first call)."""
        return self.engine.xray_reads(True)

    def dissect(self) -> dict:
        """The next instruction, layer by layer through the full model."""
        snap, f_bits = self.runtime.trace_union()
        memory = self.runtime.engine.dump_ram()
        pc = int(self.runtime.engine.get_run_state()["pc"])
        out = layer_steps(
            self.named,
            snap,
            f_bits,
            self.runtime.slot_of,
            lambda address: f"mem[${address & 0xFFFF:04X}]",
            _HEADLINE,
        )
        return {"title": disassemble(memory, pc), **out}


def ppu_region(address: int) -> str:
    """A render-graph memory address by what it holds (ppu2c02/space.py)."""
    if address >= 0x10000:
        return f"CHR ROM ${address - 0x10000:05X}"
    if address < 0x2000:
        return f"pattern ${address:04X}"
    if address < 0x3000:
        page, offset = (address - 0x2000) >> 10, address & 0x3FF
        what = "attribute" if offset >= 0x3C0 else "tile"
        return f"nametable page {page} +${offset:03X} ({what})"
    if 0x3F00 <= address < 0x3F20:
        return f"palette ${address:04X}"
    if 0x4000 <= address < 0x4100:
        return f"OAM ${address - 0x4000:02X}"
    if 0x5000 <= address < 0x7000:
        return f"pattern (FE latch) ${address - 0x5000:04X}"
    return f"${address:04X}"


class PpuXray:
    """X-ray of the render graph (ppu2c02/render.py): one tile slot traced
    through its full model on a copy of the running engine."""

    def __init__(self, ppu_engine):
        self.engine = ppu_engine  # its model 0 is the full render graph

    @cached_property
    def named(self):
        return [life for life in self.engine.lives if not _ANONYMOUS.match(life[0].name)]

    @cached_property
    def life_of(self):
        return {life[0].name: life for life in self.named}

    def knock_out(self, console, name: str, value: int = 0) -> str:
        """Ablate a named value of the render graph: force it to `value`
        right after the layer that computes it, in every slot drawn from
        now on (all through the full model, where it lives)."""
        life = self.life_of.get(name)
        if life is None or life[2] == 0:
            raise ValueError(f"{name} is not a computed value of the render graph")
        _dim, slot, first, _end = life
        self.engine.engine.ablate(slot, first, value)
        console.ppu_force_full_render = True
        return f"{name} forced to {value} after layer {first - 1}"

    def restore(self, console) -> None:
        self.engine.engine.ablate()
        console.ppu_force_full_render = False

    def dissect(self, trace: dict, x: int) -> dict:
        engine = self.engine
        snap = np.asarray(trace["residual"]).reshape(engine.n_layers + 1, engine.d_model)
        out = layer_steps(self.named, snap, engine.f_bits, engine.slot_of, ppu_region)
        model = trace["model"]
        drawn_by = "the full graph" if model == 0 else engine.expert_names[model - 1]
        pixel_slot = self.life_of[f"pixel{x % 8}"][1]
        color = int(snap[-1, pixel_slot]) >> engine.f_bits
        title = (
            f"pixel ({x}, {trace['line']}): slot {trace['slot']}, drawn by {drawn_by}; "
            f"color ${color:02X}"
        )
        return {"title": title, **out}


class ReadHeat:
    """The memory heatmap: each emulated frame's reads per address, decayed
    so the picture follows the game over about a second."""

    DECAY = 0.9

    def __init__(self):
        self.heat = np.zeros(0x10000)

    def add(self, reads: np.ndarray) -> None:
        self.heat *= self.DECAY
        self.heat += reads

    def image(self) -> bytes:
        """256 x 256 intensities (row = page, column = byte), log scaled."""
        scaled = np.log1p(self.heat)
        top = scaled.max()
        if top > 0:
            scaled *= 255.0 / top
        return scaled.astype(np.uint8).tobytes()

    def hottest(self, n: int = 6) -> list[tuple[str, int]]:
        """The busiest 256-byte pages, as ("$XX00", reads per frame)."""
        pages = self.heat.reshape(256, 256).sum(axis=1) * (1 - self.DECAY)
        order = np.argsort(pages)[::-1][:n]
        return [(f"${int(p):02X}00", int(pages[p])) for p in order if pages[p] >= 1]


UNRENDERED = 255  # Ppu::unrendered_slot: no transformer output yet


def rgba(r: int, g: int, b: int) -> int:
    return r | g << 8 | b << 16 | 0xFF << 24


def expert_colors(names: list[str]) -> list[int]:
    """An RGBA color per render model id (0 the full graph, k the k-th
    expert, UNRENDERED), from what each expert's pins fix: blue shades per
    fine x for the plain background experts (lighter with table 1),
    orange with sprites, green at the line's edge slots; the variants
    that keep the CHR-latch logic lean purple."""
    colors = [rgba(230, 70, 60)] * 256  # the full graph
    colors[UNRENDERED] = rgba(70, 70, 76)
    for model, name in enumerate(names, start=1):
        pins = EXPERTS[name]
        r, g, b = 250, 160, 40  # sprites: nothing below pins sprite pixels off
        if pins.get("first_slot") or pins.get("last_slot"):
            r, g, b = 90, 200, 110
        elif "fine_x_bit0" in pins:
            fine_x = sum(pins[f"fine_x_bit{k}"] << k for k in range(3))
            r, g, b = 40 + 30 * pins["ctrlbit4"], 90 + 18 * fine_x, 230
        if "latch" not in pins:
            r, g, b = (r + 190) // 2, g * 2 // 3, (b + 230) // 2
        colors[model] = rgba(r, g, b)
    return colors


def ppu_expert_summary(expert_map: bytes, names: list[str]) -> list[tuple[str, int]]:
    """Slots per render expert in the last frame (0 = the full graph,
    255 = no output yet), busiest first."""
    counts = np.bincount(np.frombuffer(expert_map, np.uint8), minlength=256)
    label = {0: "full graph", UNRENDERED: "no output yet"} | {i + 1: n for i, n in enumerate(names)}
    order = np.argsort(counts)[::-1]
    return [(label.get(int(i), f"model {int(i)}"), int(counts[i])) for i in order if counts[i]]
