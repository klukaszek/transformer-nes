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
        for token, spec, _circuit in chip.circuits():
            model = engine.model_id(token, spec)
            n_in = len(chip.tokens[token].inputs)
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
