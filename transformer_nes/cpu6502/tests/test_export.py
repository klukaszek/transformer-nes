"""Smoke-test the self-describing standard-ML export artifact."""

import json

import pytest


@pytest.mark.slow
def test_exported_weights_are_self_describing(tmp_path):
    from safetensors.numpy import load_file

    from transformer_nes.model.export import export_model

    meta = export_model(str(tmp_path))
    tensors = load_file(str(tmp_path / "weights.safetensors"))
    on_disk_meta = json.loads((tmp_path / "machine.json").read_text())

    assert meta == on_disk_meta
    assert {"tok", "head"} <= tensors.keys()
    assert tensors["tok"].shape[1] == meta["d_model"]
    assert tensors["head"].shape[1] == meta["d_model"]
    assert len(meta["layer_heads"]) == meta["n_layers"]
    assert "byte_value" in meta["input_slots"]
