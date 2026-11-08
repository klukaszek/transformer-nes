"""The NES model: its eager PyTorch forward is exactly what the compiled
library runs, it saves and loads exactly, and its specializations fold."""

import numpy as np
import pytest
import torch

from transformer_nes.model.backend import model_digest
from transformer_nes.model.construct import TokenGraph, construct_chip
from transformer_nes.nes import controller
from transformer_nes.nes.model import NES, compiled


@pytest.fixture(scope="module")
def nes():
    return NES.load()


def test_eager_model_is_the_compiled_model(nes):
    """Every circuit of every chip, eager against native, on random inputs
    in and well outside each circuit's domain: equal outputs, or both refuse
    a fractional readout."""
    native = compiled()
    rng = np.random.default_rng(7)
    for name, chip in nes.chips.items():
        engine = native.chip(name)
        memory = chip.memory()
        for token, spec, _circuit, _outputs in chip.programs():
            model = engine.model_id(token, spec)
            n_in = len(chip.io(token)[0])
            for _ in range(200):
                if chip.dtype == torch.float32:
                    # Real values, signed zeros and integers: the float order must be exact.
                    inputs = rng.uniform(-2, 20, n_in).astype(np.float32)
                    kind = rng.integers(4, size=n_in)
                    inputs[kind == 1] = -0.0
                    inputs[kind == 2] = rng.integers(16, size=int((kind == 2).sum()))
                else:
                    inputs = rng.integers(-8, 300, n_in, dtype=np.int32)
                try:
                    eager = nes(name, token, torch.from_numpy(inputs), memory, spec).numpy()
                except ValueError:
                    with pytest.raises(RuntimeError, match="fractional"):
                        engine.step(inputs, model)
                    continue
                # Bitwise: equal values with equal signs of zero.
                np.testing.assert_array_equal(
                    engine.step(inputs, model).view(np.uint32 if chip.dtype == torch.float32 else np.int32),
                    eager.view(np.uint32 if chip.dtype == torch.float32 else np.int32),
                    err_msg=f"{name}/{token}/{spec}",
                )


def test_saved_model_loads_exactly(nes, tmp_path):
    nes.save(tmp_path)
    loaded = NES.from_directory(tmp_path)
    assert model_digest(dict(loaded.chips)) == model_digest(dict(nes.chips))
    # Every stored weight is a nonzero coefficient.
    for chip in loaded.chips.values():
        for _t, _s, circuit in chip.circuits():
            for layer in circuit.layers:
                for matrix in layer.matrices():
                    assert getattr(layer, f"{matrix}_coef").all()


def test_specializations_must_pin_foldable_values():
    """Pins name persist dims: an input pin would fold nothing and copy the circuit."""
    token = TokenGraph(controller.controller_graph(), controller.INPUTS, controller.OUTPUTS, {"read": {"event": 1}})
    with pytest.raises(ValueError, match="name no persist dim"):
        construct_chip({"controller": token}, memory_size=1)


def test_outputs_must_name_one_value():
    """A readout named like another value of its circuit would read one of the two."""
    from transformer_nes.graph.core import InputDimension, construction, current_graph, persist

    @construction
    def graph():
        x = InputDimension("x")
        inner = persist(x + 1, name="y")
        y = persist(2 * inner, name="y")
        return current_graph().capture({"T": x.as_expr()}, {"READOUT": 1e-6 * y})

    with pytest.raises(ValueError, match="name more than one value"):
        construct_chip({"t": TokenGraph(graph(), ("x",), ("y",))}, memory_size=1)


def test_specializations_that_fold_alike_share_one_circuit():
    """Pins a projection no longer depends on fold to the same circuit: it is
    stored once and both names run it."""
    from transformer_nes.graph.core import InputDimension, construction, current_graph, persist
    from transformer_nes.graph.ops import select

    @construction
    def graph():
        x, a, b = (InputDimension(n) for n in "xab")
        flag_a, flag_b = persist(a, name="flag_a"), persist(b, name="flag_b")
        u = persist(select(flag_a, x, 2 * x), name="u")
        v = persist(select(flag_b, x + 1, x + 2), name="v")
        return current_graph().capture({"T": x + a + b}, {"READOUT": 1e-6 * (u + v)})

    token = TokenGraph(graph(), ("x", "a", "b"), ("u", "v"), {"b0": {"flag_b": 0}, "b1": {"flag_b": 1}})
    chip = construct_chip({"t": token, "u": token.project(("u",))}, memory_size=1)
    assert chip.tokens["t"].same == {}
    # Projected to u, both pins are moot: each specialization is the full circuit.
    assert chip.tokens["u"].same == {"b0": None, "b1": None}
    assert len(chip.circuits()) == 3 + 1
    for a in range(2):
        inputs = torch.tensor([5, a, 0], dtype=torch.int32)
        assert chip("u", inputs, chip.memory(), "b1").tolist() == [5 if a else 10]
