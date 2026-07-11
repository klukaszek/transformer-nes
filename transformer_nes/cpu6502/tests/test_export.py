"""Receipt for the standard-ML artifact: the exported safetensors weights,
executed by the standalone PyTorch runner (experiments/torch_reference --
no transformer_nes imports), run the 6502 lockstep-exact with the oracle."""

import importlib.util
import sys
from pathlib import Path

import pytest

from transformer_nes.cpu6502 import reference
from transformer_nes.cpu6502.weights_runner import VERIFY_START, verify_program

_RUNNER = (
    Path(__file__).resolve().parents[3]
    / "experiments" / "torch_reference" / "torch_runner.py"
)


@pytest.mark.slow
def test_exported_weights_run_in_torch(tmp_path):
    from transformer_nes.model.export import export_model

    export_model(str(tmp_path))
    assert (tmp_path / "weights.safetensors").exists()

    spec = importlib.util.spec_from_file_location("torch_runner", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["torch_runner"] = mod
    spec.loader.exec_module(mod)
    # Independence: the runner must not import the project.
    assert not any(name.startswith("transformer_nes.model.engine")
                   for name in getattr(mod, "__dict__", {}))

    m = mod.TorchMachine(str(tmp_path))
    image = verify_program()
    m.boot_image(image, start_pc=VERIFY_START)
    ref = reference.Cpu6502(memory=bytearray(image), bcd_enabled=False)
    ref.pc = VERIFY_START
    steps = 0
    for i in range(40):
        row = m.step()
        r = ref.step()
        s = r.state
        got = (row["pc"], row["a"], row["x"], row["y"], row["s"], row["p"],
               row["ir"], row["oplo"], row["ophi"], row["ea"], row["data"],
               set(row["writes"]))
        want = (s.pc, s.a, s.x, s.y, s.s, s.p, s.ir, s.operand_lo,
                s.operand_hi, s.ea, s.data,
                {(w.addr, w.value) for w in r.writes})
        assert got == want, f"step {i} diverged"
        steps += 1
        if m.halted:
            break
    assert steps >= 30
