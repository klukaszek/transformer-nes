"""Smoke-test the self-describing standard-ML export artifact."""

import json

import pytest


@pytest.mark.slow
def test_exported_weights_are_self_describing(tmp_path):
    from safetensors.numpy import load_file

    from transformer_nes.cpu6502.backends.weights.export import export_model

    meta = export_model(str(tmp_path))
    tensors = load_file(str(tmp_path / "weights.safetensors"))
    on_disk_meta = json.loads((tmp_path / "machine.json").read_text())

    assert meta == on_disk_meta
    assert {"tok", "head"} <= tensors.keys()
    assert tensors["tok"].shape[1] == meta["d_model"]
    assert tensors["head"].shape[1] == meta["d_model"]
    assert len(meta["layer_heads"]) == meta["n_layers"]
    assert "byte_value" in meta["input_slots"]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    from transformer_nes.cpu6502.backends.weights.export import export_model

    out = tmp_path_factory.mktemp("checkpoint")
    export_model(str(out))
    return str(out)


@pytest.mark.parametrize("runner", ["checkpoint_runner", "expert_runner"])
def test_checkpoint_runners_stand_alone(runner):
    """The runners are the claim made checkable: they may import nothing
    from this repository, only torch and safetensors."""
    import ast
    import importlib

    module = importlib.import_module(f"transformer_nes.cpu6502.backends.weights.{runner}")
    with open(module.__file__) as f:
        tree = ast.parse(f.read())
    modules = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(m.startswith("transformer_nes") for m in modules), modules


def test_vanilla_pytorch_runs_the_checkpoint(checkpoint):
    """The exported weights in a plain PyTorch transformer, step for step
    against the reference core: registers, cycles (page crossings and
    taken branches included), and the BRK halt."""
    from transformer_nes.cpu6502.backends import reference
    from transformer_nes.cpu6502.backends.weights.checkpoint_runner import Cpu, Transformer

    code = bytes([
        0xA2, 0x05,        # $8000 LDX #5
        0xBD, 0xFE, 0x80,  # $8002 LDA $80FE,X  (crosses a page)
        0x9D, 0x00, 0x02,  # $8005 STA $0200,X
        0x48, 0x68,        # $8008 PHA; PLA
        0x69, 0x37,        # $800A ADC #$37
        0xCA,              # $800C DEX
        0xD0, 0xF3,        # $800D BNE $8002
        0x20, 0x20, 0x80,  # $800F JSR $8020
        0x00,              # $8012 BRK (IRQ vector 0: halt)
    ])
    subroutine = bytes([0xA0, 0x03, 0x88, 0xD0, 0xFD, 0x60])  # LDY #3; DEY; BNE; RTS
    image = bytearray(0x10000)
    image[0x8000 : 0x8000 + len(code)] = code
    image[0x8020 : 0x8020 + len(subroutine)] = subroutine

    cpu = Cpu(Transformer(checkpoint), bytes(image), 0x8000)
    ref = reference.Cpu6502(memory=bytearray(image), bcd_enabled=False)
    ref.pc = 0x8000
    for step in range(200):
        op = reference.opcode_for(ref.read_memory(ref.pc))
        expected = reference.instruction_cycles(ref, op, ref.pc, ref.x, ref.y, ref.p)
        row = ref.step()
        assert cpu.step() == expected, f"step {step}: cycles"
        st, want = cpu.state, row.state
        assert (st["a"], st["x"], st["y"], st["s"], st["p"]) == (
            want.a, want.x, want.y, want.s, want.p
        ), f"step {step}"
        assert st["pc"] == ref.pc and bool(st["halted"]) == ref.halted, f"step {step}"
        if ref.halted:
            break
    # Memory round-trips through the model's own writes: PHA then PLA
    # each pass, and JSR/RTS through the stack.
    assert ref.halted and step > 20


def test_sparse_memory_and_repeated_writes(checkpoint):
    """Missing addresses read zero, and each address keeps its latest write."""
    from transformer_nes.cpu6502.backends.weights.checkpoint_runner import Cpu, Transformer

    code = bytes([0xAD, 3, 0, 0xAD, 6, 0, 0xAD, 6, 0, 0xAD, 0xFE, 0xFF])
    image = bytearray(65536)
    image[0x8000:0x8000 + len(code)] = code
    image[6] = 255
    cpu = Cpu(Transformer(checkpoint), image, 0x8000)
    cpu.write(0, 0)
    assert cpu.step() == 4 and cpu.state["a"] == 0
    assert cpu.step() == 4 and cpu.state["a"] == 255
    cpu.write(6, 17)
    assert cpu.step() == 4 and cpu.state["a"] == 17
    assert cpu.step() == 4 and cpu.state["a"] == 0



@pytest.mark.slow
def test_checkpoint_matches_the_nestest_golden_log(checkpoint):
    from transformer_nes.cpu6502.backends.weights.checkpoint_runner import Transformer, nestest
    from transformer_nes.cpu6502.examples.nestest import NESTEST_LOG, NESTEST_ROM

    assert nestest(Transformer(checkpoint), NESTEST_ROM, NESTEST_LOG, steps=300) == 300


@pytest.mark.parametrize("attention", [False, True], ids=["indexed", "attention"])
def test_routed_experts_pass_nestest(tmp_path, attention):
    """The folded per-opcode experts as a mixture of small PyTorch models
    routed by the byte at PC: all of nestest against the golden log,
    registers and cycles -- with memory read by index, and by softmax
    attention over all of it, which the index must equal."""
    from transformer_nes.cpu6502.backends.weights.expert_runner import nestest
    from transformer_nes.cpu6502.backends.weights.export import export_experts
    from transformer_nes.cpu6502.examples.nestest import NESTEST_LOG, NESTEST_ROM

    assert export_experts(str(tmp_path))["execution_abi"] == 3
    assert nestest(str(tmp_path), NESTEST_ROM, NESTEST_LOG, attention=attention) == 8991


def test_old_expert_control_abi_is_rejected_before_loading_weights(tmp_path):
    from transformer_nes.cpu6502.backends.weights.expert_runner import Cpu

    (tmp_path / "experts.json").write_text(json.dumps({"experts": []}))
    with pytest.raises(ValueError, match="export a new checkpoint"):
        Cpu(str(tmp_path), bytes(65536), 0x8000)
