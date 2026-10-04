"""The PPU register graph: its compiled engine is the exact transformer
forward, and each register's folded expert computes what the full graph
does.  The console runs with it in test_render.py (whole_ppu)."""

import numpy as np

from transformer_nes.graph.core import InputDimension
from transformer_nes.model.graph_engine import CompiledGraph, TokenGraph, construct
from transformer_nes.model.inference import ExactInference
from transformer_nes.ppu2c02 import registers, space
from transformer_nes.ppu2c02.machine import ppu_engine, register_graph


def random_access(rng, reg=None, is_write=None) -> dict[str, int]:
    """An ACCESS token over every register and register-file state."""
    ranges = {
        "reg": 8, "value": 256, "is_write": 2, "status": 256, "rendering": 2,
        "chr_writable": 2, "latch_lo": 2, "latch_hi": 2, "fine_x": 8, "w": 2, "ctrl": 256,
        "mask": 256, "oam_addr": 256, "read_buffer": 256,
        **{f"nt_page{k}": 4 for k in range(4)},
        **{f"{r}_{f}": n for r in "tv" for f, n in (("cx", 32), ("cy", 32), ("nt", 4), ("fy", 8))},
    }  # fmt: skip
    access = {"is_access": 1, **{name: int(rng.integers(n)) for name, n in ranges.items()}}
    access.update({f"bank{s}": 0 for s in range(8)})  # MMC5 builds only
    if reg is not None:
        access.update(reg=reg, is_write=is_write)
    return access


def random_writes(rng, write, count: int) -> None:
    for _ in range(count):
        region = rng.choice([0, 0x2000, 0x3F00, 0x4000, 0x5000])
        size = {0: 0x2000, 0x2000: 0x1000, 0x3F00: 0x20, 0x4000: 0x100, 0x5000: 0x2000}[region]
        write(int(region + rng.integers(size)), int(rng.integers(256)))


def test_compiled_engine_is_the_exact_forward():
    """The native engine against model/inference.py's float64 forward of
    the same weights, on random memory writes and accesses."""
    graph = register_graph()
    model, _rows, info = construct(graph)
    reference = ExactInference(model, info.d_model)
    token = TokenGraph(graph, registers.INPUTS, registers.OUTPUTS)
    engine = CompiledGraph(
        lambda: {"access": token}, tag="ppu_registers",
        sources=("transformer_nes.ppu2c02.machine",), memory_size=space.memory_size()
    )
    dims = {d.name: d for d in graph.all_dims}
    inputs = {d.name: d for d in graph.all_dims if isinstance(d, InputDimension)}

    def forward(values: dict[str, int]) -> np.ndarray:
        x = np.zeros(info.d_model)
        x[info.slot_of[graph.one]] = 1.0
        for name, value in values.items():
            if name in inputs:
                x[info.slot_of[inputs[name]]] = value
        return reference.forward(x)

    def write(addr: int, value: int) -> None:
        engine.write(addr, value)
        forward({"is_mem_write": 1, "mem_addr": addr, "mem_value": value})

    rng = np.random.default_rng(3)
    # Every byte first, as the console seeds it: reads assume dense memory.
    for addr in range(space.memory_size()):
        write(addr, int(rng.integers(256)))
    for _ in range(300):
        random_writes(rng, write, 4)
        access = random_access(rng)
        compiled = engine.step([access[name] for name in registers.INPUTS])
        residual = forward(access)
        expected = [round(residual[info.slot_of[dims[name]]]) for name in registers.OUTPUTS]
        assert list(compiled) == expected, access


def test_register_experts_agree_with_the_full_graph():
    engine = ppu_engine()
    rng = np.random.default_rng(11)
    for addr in range(0x7000):
        engine.write(addr, int(rng.integers(256)))
    full = engine.model_id("access")
    for is_write, models in enumerate(engine.register_dispatch):
        for reg, model in enumerate(models):
            for _ in range(200):
                random_writes(rng, engine.write, 2)
                access = random_access(rng, reg, is_write)
                inputs = [access[name] for name in registers.INPUTS]
                assert np.array_equal(engine.step(inputs, full), engine.step(inputs, model)), access
