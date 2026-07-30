"""Structural receipts for :mod:`transformer_nes.graph.fold`.

These tests cover graph construction, folding, liveness, and the execution
ABI consumed by the weights compiler. Runtime correctness is covered by the
reference/weights lockstep suites; this module only checks construction facts.
"""

import pytest

from transformer_nes.cpu6502.backends import reference
from transformer_nes.cpu6502.backends.weights.experts import (
    SEQUENCE_ROOT_DIMS,
    SEQUENCE_STATE_DIMS,
    build_opcode_execution_experts,
    build_shape_experts,
    decode_ops,
    last_a_bit_step,
    pinned_sequence_control,
    project_sequence_execution,
    schedule_and_build,
    semantic_recurrence,
    sparse_forward_work,
)
from transformer_nes.cpu6502.construction import RUNTIME_READ_DIMS
from transformer_nes.cpu6502.interpreter import build
from transformer_nes.cpu6502.spec import OPCODES
from transformer_nes.graph.core import (
    MEMORY_GROUP_FUSED,
    InputDimension,
    LookUpDimension,
    MemoryReadDimension,
    PersistDimension,
    ProgramGraph,
    ReGLUDimension,
    reset_graph,
)
from transformer_nes.graph.fold import (
    SEQUENCE_FEEDBACK,
    GraphFolder,
    opcode_pins,
    sequence_pins,
)
from transformer_nes.model.engine_gen import generate_source
from transformer_nes.model.schedule import build_graph, min_layers


@pytest.fixture(scope="module")
def union_pg():
    reset_graph()
    input_tokens, output_tokens = build(bcd=False)
    return ProgramGraph(input_tokens, output_tokens)


@pytest.fixture(scope="module")
def folder(union_pg):
    return GraphFolder(union_pg, root_names=RUNTIME_READ_DIMS)


def depth(fold_result):
    pg = fold_result.program_graph
    g = build_graph(pg.all_dims, pg.all_lookups)
    return min_layers(g["ops"], g["op_deps"]) if g["ops"] else 0


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
                for expr in (
                    dim.lookup.query_exprs_2d
                    + dim.lookup.key_exprs_2d
                    + dim.lookup.value_exprs
                )
            )
        elif isinstance(dim, MemoryReadDimension) and dim.memory_read not in lookups:
            lookups.add(dim.memory_read)
            terms += len(dim.memory_read.address.terms)
    return reglus, persists, len(lookups), terms


@pytest.fixture(scope="module")
def union_fold(folder):
    return folder.fold({}, label="union")


def test_unpinned_fold_metrics_match_materialized_graph(union_pg, union_fold):
    r = union_fold
    assert r.insert_side_hazards == ()
    assert r.folded_roots == ()
    assert (r.n_reglus, r.n_persists, r.n_lookups, r.n_terms) == graph_counts(
        r.program_graph
    )
    graph = build_graph(union_pg.all_dims, union_pg.all_lookups)
    assert depth(r) <= min_layers(graph["ops"], graph["op_deps"])


@pytest.mark.parametrize("byte", [0x4C, 0xAD, 0x29, 0xF0])
def test_opcode_fold_is_materially_smaller(folder, union_fold, byte):
    # Isolate the benefit of pinning the decode one-hots first.
    r = folder.fold(opcode_pins(byte, OPCODES, pin_fetched_byte=False),
                    label=f"{byte:02X}")
    base = union_fold
    # The shared-state contract: decode pins never reach insert-side exprs.
    assert r.insert_side_hazards == ()
    # Every runtime readout resolves by name (kept or re-materialized).
    for name in RUNTIME_READ_DIMS:
        assert name in r.dim_of_name, f"readout {name} missing from fold"
    assert r.n_terms <= base.n_terms * 0.30
    assert depth(r) < depth(base) * 0.5

    # The default fold additionally pins the fetched byte (the router
    # predicate ram[pc] == byte): the PC fetch read and decode residue fold
    # away, so the graph must get strictly smaller, never larger.
    rb = folder.fold(opcode_pins(byte, OPCODES), label=f"{byte:02X}+byte")
    assert rb.insert_side_hazards == ()
    assert rb.n_terms < r.n_terms
    assert rb.n_lookups < r.n_lookups  # the PC read's group lookups died


# ── Sequence fold: the poll trio and the JMP spin as one step ────────

POLL_HEAD = 0x0400
SPIN_HEAD = 0x0408


def test_sequence_execution_projection_keeps_only_final_state(folder):
    """The execution model is built from the licensed fold after removing
    the per-instruction trace receipt. A write-free three-instruction poll
    keeps exactly one final architectural receipt."""
    code = bytes([0xAD, 0x00, 0x07, 0x29, 0x40, 0xF0, 0xF9])
    pin_list = sequence_pins(POLL_HEAD, code, OPCODES)
    r = folder.fold_sequence(
        pin_list, SEQUENCE_FEEDBACK, label="projected_poll",
    )
    proof_dims = len(r.program_graph.all_dims)
    execution = project_sequence_execution(
        r, len(pin_list), pinned_sequence_control(r, pin_list, decode_ops(code)),
    )

    assert execution.memory_writes == ()
    classified = {output.name for output in execution.outputs}
    assert classified == set(SEQUENCE_STATE_DIMS)
    assert len(execution.program_graph.all_dims) < proof_dims * 0.76
    assert len(r.program_graph.all_dims) == proof_dims
    for k in range(len(pin_list) - 1):
        for name in (
            "current_pc", "fetched_opcode", "fetched_operand_lo",
            "fetched_operand_hi", "ea", "data_out", "next_pc",
            "is_store", "is_store2", "is_store3",
        ):
            assert f"{name}__s{k}" not in execution.dim_of_name

def test_sequence_execution_projection_eliminates_deterministic_model(folder):
    """A licensed deterministic region needs no transformer operations."""
    pins = sequence_pins(
        SPIN_HEAD, bytes([0x4C, 0x08, 0x04]), OPCODES,
    ) * 8
    structural = GraphFolder(
        folder.pg, root_names=SEQUENCE_ROOT_DIMS, structural_reads=True,
    )
    execution = structural.fold_sequence(
        pins, SEQUENCE_FEEDBACK, label="execution_spin", inline_linear=True,
    )
    execution_graph = project_sequence_execution(
        execution, len(pins),
        pinned_sequence_control(execution, pins, decode_ops(bytes([0x4C, 0x08, 0x04])) * 8),
    )
    _model, _tokens, _rows, execution_info = schedule_and_build(execution_graph)

    assert not any(
        output.source == "model" for output in execution_graph.outputs
    )
    assert execution_info.n_layers == 0
    assert {
        dim.name
        for dim in execution_graph.program_graph.all_dims
        if isinstance(dim, InputDimension)
    } == {"one"}


def test_structural_execution_uses_external_memory_reads(folder):
    code = bytes([0xAD, 0x00, 0x07, 0x29, 0x40, 0xF0, 0xF9])
    pins = sequence_pins(POLL_HEAD, code, OPCODES)
    structural = GraphFolder(
        folder.pg, root_names=SEQUENCE_ROOT_DIMS, structural_reads=True,
    )
    proof = structural.fold_sequence(
        pins, SEQUENCE_FEEDBACK, label="external_poll", inline_linear=True,
    )
    execution = project_sequence_execution(
        proof, len(pins), pinned_sequence_control(proof, pins, decode_ops(code)),
    )
    model, _tokens, _rows, _info = schedule_and_build(execution)

    assert execution.memory_reads
    assert all(
        dim.memory_read in execution.memory_reads
        for dim in execution.program_graph.all_dims
        if isinstance(dim, MemoryReadDimension)
    )
    assert not {
        "byte_value", "addr_lo", "addr_hi", "is_memory_write"
    } & {field.name for field in execution.state_fields}
    fused = [
        outputs
        for groups, outputs_by_head in zip(
            model.head_memory_groups, model.head_memory_outputs, strict=True,
        )
        for group, outputs in zip(groups, outputs_by_head, strict=True)
        if group == MEMORY_GROUP_FUSED
    ]
    assert fused
    assert fused[0] == ("value", "address")


def test_execution_ir_retains_ordered_writes(folder):
    pins = sequence_pins(SPIN_HEAD, bytes([0x85, 0x10]), OPCODES)
    structural = GraphFolder(
        folder.pg, root_names=SEQUENCE_ROOT_DIMS, structural_reads=True,
    )
    proof = structural.fold_sequence(
        pins, SEQUENCE_FEEDBACK, label="ordered_store", inline_linear=True,
    )
    execution = project_sequence_execution(
        proof, len(pins),
        pinned_sequence_control(proof, pins, decode_ops(bytes([0x85, 0x10]))),
    )

    assert len(execution.memory_writes) == 1
    write = execution.memory_writes[0]
    assert (write.step, write.order) == (0, 0)
    assert write.address in execution.program_graph.all_dims
    assert write.value in execution.program_graph.all_dims


def test_external_memory_read_exposes_only_consumed_bits():
    reset_graph()
    input_tokens, output_tokens = build(bcd=False, wide_bits=True)
    pg = ProgramGraph(input_tokens, output_tokens)
    code = bytes([0x6D, 0x00, 0x07])  # ADC $0700 consumes only operand bit 7.
    pins = sequence_pins(SPIN_HEAD, code, OPCODES)
    ops = decode_ops(code)
    folder = GraphFolder(
        pg, root_names=SEQUENCE_ROOT_DIMS, structural_reads=True,
    )
    proof = folder.fold_sequence(
        pins, SEQUENCE_FEEDBACK, label="selective_bits", inline_linear=True,
        bit_feedback_until=last_a_bit_step(ops),
    )
    execution = project_sequence_execution(
        proof, len(pins), pinned_sequence_control(proof, pins, ops),
    )

    assert [read.output_kinds for read in execution.memory_reads] == [
        ("value", "address", "bit7"),
    ]


def test_execution_proof_materializes_only_license_facts(folder):
    code = bytes([0xAD, 0x00, 0x07, 0x29, 0x40, 0xF0, 0xF9])
    pins = sequence_pins(POLL_HEAD, code, OPCODES)
    ops = decode_ops(code)
    structural = GraphFolder(
        folder.pg, root_names=SEQUENCE_ROOT_DIMS, structural_reads=True,
    )
    full = structural.fold_sequence(
        pins, SEQUENCE_FEEDBACK, label="full_proof", inline_linear=True,
        bit_feedback_until=last_a_bit_step(ops),
    )
    facts = structural.fold_execution_sequence(
        pins, SEQUENCE_FEEDBACK, label="proof_facts", inline_linear=True,
        bit_feedback_until=last_a_bit_step(ops),
    )
    for k in range(len(pins) - 1):
        assert f"next_pc__s{k}" in facts.dim_of_name
        assert f"new_halted__s{k}" in facts.dim_of_name
        assert f"fetched_opcode__s{k}" not in facts.dim_of_name
        assert f"data_out__s{k}" not in facts.dim_of_name
    assert len(facts.program_graph.all_dims) < len(full.program_graph.all_dims)

    full_execution = project_sequence_execution(
        full, len(pins), pinned_sequence_control(full, pins, ops),
    )
    fact_execution = project_sequence_execution(
        facts, len(pins), pinned_sequence_control(facts, pins, ops),
    )
    full_model, _t1, _r1, full_info = schedule_and_build(full_execution)
    fact_model, _t2, _r2, fact_info = schedule_and_build(fact_execution)
    assert (fact_info.n_layers, fact_info.d_model) == (
        full_info.n_layers, full_info.d_model,
    )
    assert sparse_forward_work(fact_model, fact_info.n_layers) == (
        sparse_forward_work(full_model, full_info.n_layers)
    )


# ── Canonical shape fold: symbolic pc + operands, one fold per SHAPE ──

def test_shape_absolute_store_is_rejected(union_pg):
    """Symbolic store shapes cannot prove disjointness from their own code."""
    shape = ("lda_sta_abs", bytes.fromhex("a9008d0000"), 1)
    with pytest.raises(RuntimeError, match="not provably below"):
        build_shape_experts(union_pg, [shape])


def test_shape_execution_abi_contains_only_live_state(union_pg):
    shape = ("dex_bne", bytes.fromhex("cad0fd"), 1)
    experts, _meta = build_shape_experts(union_pg, [shape])
    spec = experts[0]["step_spec"]

    assert set(spec["inputs"]) == {"x"}
    assert spec["p_flags"] == []
    assert {"p_z", "p_n"} <= set(spec["sequence_final"]["model"])
    assert spec["sequence_control"] == {
        "predicate": None,
        "fallthrough": 3,
        "target": 0,
        "target_dimension": None,
        "relative_to_head": True,
        "relative_operand_offset": 2,
        "absolute_operand_offset": None,
        "flag_mask": reference.FLAG_Z,
        "flag_value": False,
    }
    recurrence = experts[0]["recurrence"]
    assert (
        recurrence.kind, recurrence.register, recurrence.delta,
        recurrence.closure_operand_offset, recurrence.closure_operand_value,
    ) == ("affine_until", "x", -1, 2, 0xFD)


def test_compact_opcode_experts_cover_stack_and_control(union_pg):
    experts = build_opcode_execution_experts(
        union_pg,
        [0xCA, 0xA5, 0x0A, 0x85, 0x87, 0x48, 0x4C, 0x20, 0x60, 0x00],
    )
    assert [expert["byte"] for expert in experts] == [
        0xCA, 0xA5, 0x0A, 0x85, 0x87, 0x48, 0x4C, 0x20, 0x60, 0x00,
    ]
    assert all(expert["kind"] == "exec_opcode" for expert in experts)
    dex = experts[0]
    assert set(dex["step_spec"]["inputs"]) == {"x"}
    assert dex["step_spec"]["sequence_writes"] == [[]]
    sta = experts[3]
    assert len(sta["step_spec"]["sequence_writes"][0]) == 1
    sax = experts[4]
    assert len(sax["step_spec"]["sequence_writes"][0]) == 1
    assert experts[5]["step_spec"]["sequence_writes"][0]
    assert experts[6]["step_spec"]["sequence_control"]["absolute_operand_offset"] == 1
    assert len(experts[7]["step_spec"]["sequence_writes"][0]) == 2
    assert experts[8]["step_spec"]["sequence_control"]["target_dimension"] is not None
    brk = experts[9]
    assert len(brk["step_spec"]["sequence_writes"][0]) == 3
    assert brk["step_spec"]["sequence_halt"] is not None

    legacy = dict(experts[0])
    legacy.pop("kind")
    with pytest.raises(RuntimeError, match="compact opcodes, shapes or interrupts"):
        generate_source(
            experts[0]["model"], experts[0]["step_spec"], experts=[legacy],
        )


@pytest.mark.parametrize(
    ("code", "register", "delta"),
    (("cad0fd", "x", -1), ("e8d0fd", "x", 1),
     ("88d0fd", "y", -1), ("c8d0fd", "y", 1)),
)
def test_affine_recurrences_are_isa_semantic(code, register, delta):
    ops = decode_ops(bytes.fromhex(code))
    recurrence = semantic_recurrence(ops, [0, 1])

    assert recurrence.kind == "affine_until"
    assert (recurrence.register, recurrence.delta) == (register, delta)
    assert recurrence.steps_per_iteration == 2
    assert recurrence.cycles_per_iteration == 4
    assert recurrence.cycles_prefix == 2
    assert recurrence.fallthrough_offset == 3


def test_poll_recurrence_is_isa_semantic():
    ops = decode_ops(bytes.fromhex("ad00072940f0f9"))
    recurrence = semantic_recurrence(ops, [0, 3, 5])

    assert recurrence.kind == "invariant_poll"
    assert recurrence.steps_per_iteration == 3
    assert recurrence.cycles_per_iteration == 8
    assert recurrence.cycles_prefix == 6
    assert recurrence.closure_operand_offset == 6
    assert recurrence.closure_operand_value == 0xF9


def test_shape_interior_branch_is_rejected(union_pg):
    """Shape dispatch has no speculative rollback ABI."""
    plain = ("bne_lda", bytes.fromhex("d000a900"), 1)
    with pytest.raises(RuntimeError, match="interior next_pc"):
        build_shape_experts(union_pg, [plain])
