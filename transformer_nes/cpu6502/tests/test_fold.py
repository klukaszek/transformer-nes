"""Structural receipts for :mod:`transformer_nes.graph.fold`.

These tests cover graph construction, folding, liveness, and the execution
ABI consumed by the weights compiler. Runtime correctness is covered by the
reference/weights lockstep suites; this module only checks construction facts.
"""

import pytest

from transformer_nes.cpu6502.backends.weights.engine import generate_source
from transformer_nes.cpu6502.backends.weights.execution import (
    SEQUENCE_ROOT_DIMS,
    SEQUENCE_STATE_DIMS,
    project_sequence_execution,
)
from transformer_nes.cpu6502.backends.weights.experts import (
    build_opcode_execution_experts,
    build_shape_experts,
    schedule_and_build,
)
from transformer_nes.cpu6502.backends.weights.folds import (
    SEQUENCE_FEEDBACK,
    SequenceFolder,
    opcode_pins,
    sequence_pins,
)
from transformer_nes.cpu6502.backends.weights.shapes import decode_ops, last_a_bit_step
from transformer_nes.cpu6502.graph import RUNTIME_READ_DIMS, build
from transformer_nes.cpu6502.spec import FLAG_C, OPCODES
from transformer_nes.graph.core import (
    MEMORY_GROUP_FUSED,
    InputDimension,
    LookUpDimension,
    MemoryReadDimension,
    PersistDimension,
    ReGLUDimension,
    construction,
    current_graph,
)
from transformer_nes.model.schedule import asap_schedule


@pytest.fixture(scope="module")
@construction
def union_pg():
    input_tokens, output_tokens = build(bcd=False)
    return current_graph().capture(input_tokens, output_tokens)


@pytest.fixture(scope="module")
def folder(union_pg):
    return SequenceFolder(union_pg, root_names=RUNTIME_READ_DIMS)


def depth(fold_result):
    pg = fold_result.program_graph
    return asap_schedule(pg)["num_layers"]


def graph_counts(pg):
    reglus = persists = terms = 0
    lookups = set()
    for dim in pg.all_dims:
        if isinstance(dim, ReGLUDimension):
            reglus += 1
            terms += len(dim.a_expr.terms) + len(dim.b_expr.terms)
        elif isinstance(dim, PersistDimension):
            persists += 1
            terms += len(dim.expr.terms)
        elif isinstance(dim, LookUpDimension) and dim.lookup not in lookups:
            lookups.add(dim.lookup)
            terms += sum(
                len(expr.terms)
                for expr in (dim.lookup.query_exprs + dim.lookup.key_exprs + dim.lookup.value_exprs)
            )
        elif isinstance(dim, MemoryReadDimension) and dim.memory_read not in lookups:
            lookups.add(dim.memory_read)
            terms += len(dim.memory_read.address.terms)
    return reglus, persists, len(lookups), terms


@pytest.fixture(scope="module")
def union_fold(folder):
    return folder.fold({}, label="union")


def test_unpinned_fold_preserves_lookup_state_and_depth(union_pg, union_fold):
    assert union_fold.insert_side_hazards == ()
    assert depth(union_fold) <= asap_schedule(union_pg)["num_layers"]


@pytest.mark.parametrize("byte", [0x4C, 0xAD, 0x29, 0xF0])
def test_opcode_fold_is_materially_smaller(folder, union_fold, byte):
    # Isolate the benefit of pinning the decode one-hots first.
    r = folder.fold(opcode_pins(byte, OPCODES, pin_fetched_byte=False), label=f"{byte:02X}")
    base = union_fold
    # The shared-state contract: decode pins never reach insert-side exprs.
    assert r.insert_side_hazards == ()
    # Every runtime readout resolves by name (kept or re-materialized).
    for name in RUNTIME_READ_DIMS:
        assert name in r.dim_of_name, f"readout {name} missing from fold"
    assert graph_counts(r.program_graph)[3] <= graph_counts(base.program_graph)[3] * 0.30
    assert depth(r) < depth(base) * 0.5

    # The default fold additionally pins the fetched byte (the router
    # predicate ram[pc] == byte): the PC fetch read and decode residue fold
    # away, so the graph must get strictly smaller, never larger.
    rb = folder.fold(opcode_pins(byte, OPCODES), label=f"{byte:02X}+byte")
    assert rb.insert_side_hazards == ()
    assert graph_counts(rb.program_graph)[3] < graph_counts(r.program_graph)[3]
    assert (
        graph_counts(rb.program_graph)[2] < graph_counts(r.program_graph)[2]
    )  # the PC read's group lookups died


# ── Sequence fold: the poll trio and the JMP spin as one step ────────

POLL_HEAD = 0x0400
SPIN_HEAD = 0x0408


def test_sequence_execution_projection_keeps_only_final_state(folder):
    """The execution model is built from the licensed fold after removing
    the per-instruction trace receipt. A write-free three-instruction poll
    keeps exactly one final architectural receipt."""
    code = bytes([0xAD, 0x00, 0x07, 0x29, 0x40, 0xF0, 0xF9])
    pin_list = sequence_pins(POLL_HEAD, code, OPCODES)
    # Projection consumes individual flag roots, which need not remain live
    # under the packed new_p readout after affine circuit simplification.
    proof = SequenceFolder(folder.pg, root_names=(*RUNTIME_READ_DIMS, *SEQUENCE_ROOT_DIMS))
    r = proof.fold_execution_sequence(
        pin_list,
        SEQUENCE_FEEDBACK,
        label="projected_poll",
    )
    proof_dims = len(r.program_graph.all_dims)
    execution = project_sequence_execution(
        r,
        len(pin_list),
    )

    assert execution.memory_writes == ()
    classified = {output.name for output in execution.outputs}
    assert classified == set(SEQUENCE_STATE_DIMS)
    assert len(execution.program_graph.all_dims) < proof_dims * 0.76
    assert len(r.program_graph.all_dims) == proof_dims
    for k in range(len(pin_list) - 1):
        for name in (
            "current_pc",
            "fetched_opcode",
            "fetched_operand_lo",
            "fetched_operand_hi",
            "ea",
            "data_out",
            "next_pc",
            "is_store",
            "is_store2",
            "is_store3",
        ):
            assert f"{name}__s{k}" not in execution.dim_of_name


def test_sequence_execution_projection_keeps_only_the_pinned_state_dependencies(folder):
    """Pinned control flow retains at most the carry-flag identity circuit."""
    import numpy as np

    from transformer_nes.model.inference import ExactInference
    pins = (
        sequence_pins(
            SPIN_HEAD,
            bytes([0x4C, 0x08, 0x04]),
            OPCODES,
        )
        * 8
    )
    structural = SequenceFolder(
        folder.pg,
        root_names=SEQUENCE_ROOT_DIMS,
        structural_reads=True,
    )
    execution = structural.fold_execution_sequence(
        pins,
        SEQUENCE_FEEDBACK,
        label="execution_spin",
        inline_linear=True,
    )
    execution_graph = project_sequence_execution(
        execution,
        len(pins),
    )
    model, _rows, execution_info = schedule_and_build(execution_graph)

    live = [output for output in execution_graph.outputs if output.source == "model"]
    assert {output.name for output in live} <= {"p_c"}
    assert {
        dim.name
        for dim in execution_graph.program_graph.all_dims
        if isinstance(dim, InputDimension)
    } <= {"one", "in_p_c"}
    # The existing circuit retains C as a model output. Check the numerical
    # identity rather than assuming the folder classifies it as preservation.
    eager = ExactInference(model)
    for carry in (0, 1):
        values = np.zeros(execution_info.d_model)
        for dim, slot in execution_info.slot_of.items():
            if isinstance(dim, InputDimension):
                values[slot] = 1 if dim.name == "one" else carry
        residual = eager.forward(values)
        for output in live:
            assert residual[execution_info.slot_of[output.dimension]] == carry


def test_structural_execution_uses_external_memory_reads(folder):
    code = bytes([0xAD, 0x00, 0x07, 0x29, 0x40, 0xF0, 0xF9])
    pins = sequence_pins(POLL_HEAD, code, OPCODES)
    structural = SequenceFolder(
        folder.pg,
        root_names=SEQUENCE_ROOT_DIMS,
        structural_reads=True,
    )
    proof = structural.fold_execution_sequence(
        pins,
        SEQUENCE_FEEDBACK,
        label="external_poll",
        inline_linear=True,
    )
    execution = project_sequence_execution(
        proof,
        len(pins),
    )
    model, _rows, _info = schedule_and_build(execution)

    assert execution.program_graph.all_memory_reads
    assert all(
        dim.memory_read in execution.program_graph.all_memory_reads
        for dim in execution.program_graph.all_dims
        if isinstance(dim, MemoryReadDimension)
    )
    assert not {"byte_value", "addr_lo", "addr_hi", "is_memory_write"} & {
        dim.name for dim in execution.program_graph.all_dims if isinstance(dim, InputDimension)
    }
    fused = [
        outputs
        for groups, outputs_by_head in zip(
            model.head_memory_groups,
            model.head_memory_outputs,
            strict=True,
        )
        for group, outputs in zip(groups, outputs_by_head, strict=True)
        if group == MEMORY_GROUP_FUSED
    ]
    assert fused
    assert fused[0] == ("value", "address")


def test_execution_ir_retains_ordered_writes(folder):
    pins = sequence_pins(SPIN_HEAD, bytes([0x85, 0x10]), OPCODES)
    structural = SequenceFolder(
        folder.pg,
        root_names=SEQUENCE_ROOT_DIMS,
        structural_reads=True,
    )
    proof = structural.fold_execution_sequence(
        pins,
        SEQUENCE_FEEDBACK,
        label="ordered_store",
        inline_linear=True,
    )
    execution = project_sequence_execution(
        proof,
        len(pins),
    )

    assert len(execution.memory_writes) == 1
    write = execution.memory_writes[0]
    assert (write.step, write.order) == (0, 0)
    assert write.address in execution.program_graph.all_dims
    assert write.value in execution.program_graph.all_dims
    assert write.original is None


@construction
def test_external_memory_read_exposes_only_consumed_bits():
    input_tokens, output_tokens = build(bcd=False, wide_bits=True)
    pg = current_graph().capture(input_tokens, output_tokens)
    code = bytes([0x6D, 0x00, 0x07])  # ADC $0700 consumes only operand bit 7.
    pins = sequence_pins(SPIN_HEAD, code, OPCODES)
    ops = decode_ops(code)
    folder = SequenceFolder(
        pg,
        root_names=SEQUENCE_ROOT_DIMS,
        structural_reads=True,
    )
    proof = folder.fold_execution_sequence(
        pins,
        SEQUENCE_FEEDBACK,
        label="selective_bits",
        inline_linear=True,
        bit_feedback_until=last_a_bit_step(ops),
    )
    execution = project_sequence_execution(
        proof,
        len(pins),
    )

    assert [read.output_kinds for read in execution.program_graph.all_memory_reads] == [
        ("value", "address", "bit7"),
    ]


def test_execution_proof_materializes_only_license_facts(folder):
    code = bytes([0xAD, 0x00, 0x07, 0x29, 0x40, 0xF0, 0xF9])
    pins = sequence_pins(POLL_HEAD, code, OPCODES)
    ops = decode_ops(code)
    structural = SequenceFolder(
        folder.pg,
        root_names=SEQUENCE_ROOT_DIMS,
        structural_reads=True,
    )
    facts = structural.fold_execution_sequence(
        pins,
        SEQUENCE_FEEDBACK,
        label="proof_facts",
        inline_linear=True,
        bit_feedback_until=last_a_bit_step(ops),
    )
    for k in range(len(pins) - 1):
        assert f"next_pc__s{k}" in facts.dim_of_name
        assert f"new_halted__s{k}" in facts.dim_of_name
        assert f"fetched_opcode__s{k}" not in facts.dim_of_name
        assert f"data_out__s{k}" not in facts.dim_of_name

    execution = project_sequence_execution(
        facts,
        len(pins),
    )
    for k in range(len(pins) - 1):
        assert f"next_pc__s{k}" not in execution.dim_of_name
        assert f"new_halted__s{k}" not in execution.dim_of_name


def test_shape_absolute_store_is_rejected(union_pg):
    """Symbolic store shapes cannot prove disjointness from their own code."""
    shape = ("lda_sta_abs", bytes.fromhex("a9008d0000"), 1)
    with pytest.raises(RuntimeError, match="store shapes are not licensed"):
        build_shape_experts(union_pg, [shape])


def test_shape_execution_abi_contains_only_live_state(union_pg):
    shape = ("dex_bne", bytes.fromhex("cad0fd"), 1)
    experts, _meta = build_shape_experts(union_pg, [shape])
    spec = experts[0]["step_spec"]

    # The PC feeds the closing branch's target and page-cross timing.
    assert set(spec["inputs"]) == {"x", "pc_lo", "pc_hi"}
    assert [mask for _slot, mask in spec["p_flags"]] == [FLAG_C]
    assert {"p_z", "p_n"} <= set(spec["sequence_final"]["model"])
    assert "new_pc" in spec["sequence_final"]["model"]
    assert "sequence_control" not in spec
    # DEX's 2 cycles fold; the closing BNE's are the model's (taken or
    # not, across a page or not), read out after the forward.
    assert experts[0]["cycles_prefix"] == 2 and experts[0]["spin"] == 0
    assert spec["sequence_cycles"]["constant"] == 2
    assert len(spec["sequence_cycles"]["model"]) == 1


def test_compact_opcode_experts_cover_stack_and_control(union_pg):
    experts = build_opcode_execution_experts(
        union_pg,
        [0xCA, 0xA5, 0x0A, 0x85, 0x87, 0x48, 0x4C, 0x20, 0x60, 0x00],
    )
    assert [expert["byte"] for expert in experts] == [
        0xCA,
        0xA5,
        0x0A,
        0x85,
        0x87,
        0x48,
        0x4C,
        0x20,
        0x60,
        0x00,
    ]
    assert all(expert["kind"] == "exec_opcode" for expert in experts)
    dex = experts[0]
    assert set(dex["step_spec"]["inputs"]) == {"x", "pc_lo", "pc_hi"}
    assert dex["step_spec"]["sequence_writes"] == [[]]
    sta = experts[3]
    assert len(sta["step_spec"]["sequence_writes"][0]) == 1
    sax = experts[4]
    assert len(sax["step_spec"]["sequence_writes"][0]) == 1
    assert experts[5]["step_spec"]["sequence_writes"][0]
    assert "new_pc" in experts[6]["step_spec"]["sequence_final"]["model"]
    assert len(experts[7]["step_spec"]["sequence_writes"][0]) == 2
    assert "new_pc" in experts[8]["step_spec"]["sequence_final"]["model"]
    brk = experts[9]
    assert len(brk["step_spec"]["sequence_writes"][0]) == 3
    assert brk["step_spec"]["sequence_halt"] is not None

    legacy = dict(experts[0])
    legacy.pop("kind")
    with pytest.raises(RuntimeError, match="unsupported execution contract"):
        generate_source(
            experts[0]["model"],
            experts[0]["step_spec"],
            experts=[legacy],
        )


def test_shape_interior_branch_is_rejected(union_pg):
    """Shape dispatch has no speculative rollback ABI."""
    plain = ("bne_lda", bytes.fromhex("d000a900"), 1)
    with pytest.raises(RuntimeError, match="interior next_pc"):
        build_shape_experts(union_pg, [plain])


def test_completed_compilation_does_not_retain_the_source_graph():
    import gc
    import weakref

    from transformer_nes.cpu6502.backends.weights.experts import sequence_folder
    from transformer_nes.cpu6502.graph import Cpu6502Graph

    program = Cpu6502Graph(bcd=False, wide_bits=True).program_graph()
    reference = weakref.ref(program)
    folder = sequence_folder(program)
    del program, folder
    gc.collect()
    assert reference() is None
