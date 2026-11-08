"""Structural receipts for :mod:`transformer_nes.graph.fold`.

These tests cover graph construction, folding, liveness, and the execution
ABI consumed by the weights compiler. Runtime correctness is covered by the
reference/weights lockstep suites; this module only checks construction facts.
"""

import pytest

from transformer_nes.cpu6502.execution import (
    SEQUENCE_ROOT_DIMS,
    STATE_FIELDS,
    project_sequence_execution,
)
from transformer_nes.cpu6502.experts import (
    build_opcode_execution_experts,
    build_shape_experts,
    schedule_and_build,
)
from transformer_nes.cpu6502.folds import (
    SEQUENCE_FEEDBACK,
    SequenceFolder,
    opcode_pins,
    sequence_shape_pins,
)
from transformer_nes.cpu6502.graph import RUNTIME_READ_DIMS, build
from transformer_nes.cpu6502.shapes import last_a_bit_step
from transformer_nes.cpu6502.spec import OPCODES
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


def test_sequence_execution_projection_keeps_only_final_state(folder):
    """The execution model is built from the licensed fold after removing
    the per-instruction trace receipt. A write-free three-instruction poll
    keeps exactly one final architectural receipt."""
    code = bytes([0xAD, 0x00, 0x07, 0x29, 0x40, 0xF0, 0xF9])
    pin_list, offsets = sequence_shape_pins(code, OPCODES)
    # Projection consumes individual flag roots, which need not remain live
    # under the packed new_p readout after affine circuit simplification.
    proof = SequenceFolder(folder.pg, root_names=(*RUNTIME_READ_DIMS, *SEQUENCE_ROOT_DIMS))
    r = proof.fold_execution_sequence(
        pin_list,
        SEQUENCE_FEEDBACK,
        label="projected_poll",
        pc_offsets=offsets,
    )
    proof_dims = len(r.program_graph.all_dims)
    execution = project_sequence_execution(
        r,
        len(pin_list),
    )

    assert execution.memory_writes == ()
    classified = {output.name for output in execution.outputs}
    assert classified == {name for name, _final, _input in STATE_FIELDS}
    assert len(execution.program_graph.all_dims) < proof_dims * 0.76
    assert len(r.program_graph.all_dims) == proof_dims
    # Interior trace receipts are gone; the symbolic pc chain and effective
    # address that later steps compute from remain.
    for k in range(len(pin_list) - 1):
        for name in (
            "fetched_opcode",
            "fetched_operand_lo",
            "fetched_operand_hi",
            "data_out",
            "next_pc",
            "is_store",
            "is_store2",
            "is_store3",
        ):
            assert f"{name}__s{k}" not in execution.dim_of_name


def test_structural_execution_uses_external_memory_reads(folder):
    code = bytes([0xAD, 0x00, 0x07, 0x29, 0x40, 0xF0, 0xF9])
    pins, offsets = sequence_shape_pins(code, OPCODES)
    structural = SequenceFolder(
        folder.pg,
        root_names=SEQUENCE_ROOT_DIMS,
        structural_reads=True,
    )
    proof = structural.fold_execution_sequence(
        pins,
        SEQUENCE_FEEDBACK,
        label="external_poll",
        pc_offsets=offsets,
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
    pins, offsets = sequence_shape_pins(bytes([0x85, 0x10]), OPCODES)
    structural = SequenceFolder(
        folder.pg,
        root_names=SEQUENCE_ROOT_DIMS,
        structural_reads=True,
    )
    proof = structural.fold_execution_sequence(
        pins,
        SEQUENCE_FEEDBACK,
        label="ordered_store",
        pc_offsets=offsets,
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
    pins, offsets = sequence_shape_pins(code, OPCODES)
    ops = [OPCODES[code[o]] for o in offsets]
    folder = SequenceFolder(
        pg,
        root_names=SEQUENCE_ROOT_DIMS,
        structural_reads=True,
    )
    proof = folder.fold_execution_sequence(
        pins,
        SEQUENCE_FEEDBACK,
        label="selective_bits",
        pc_offsets=offsets,
        inline_linear=True,
        bit_feedback_until=last_a_bit_step(ops),
    )
    execution = project_sequence_execution(
        proof,
        len(pins),
    )

    # The symbolic shape reads its operand bytes (no bits), then the operand,
    # of which only bit 7 is served.
    assert [read.output_kinds for read in execution.program_graph.all_memory_reads] == [
        ("value", "address"), ("value", "address"), ("value", "address", "bit7"),
    ]


def test_execution_proof_materializes_only_license_facts(folder):
    code = bytes([0xAD, 0x00, 0x07, 0x29, 0x40, 0xF0, 0xF9])
    pins, offsets = sequence_shape_pins(code, OPCODES)
    ops = [OPCODES[code[o]] for o in offsets]
    structural = SequenceFolder(
        folder.pg,
        root_names=SEQUENCE_ROOT_DIMS,
        structural_reads=True,
    )
    facts = structural.fold_execution_sequence(
        pins,
        SEQUENCE_FEEDBACK,
        label="proof_facts",
        pc_offsets=offsets,
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


def reads(execution) -> set[str]:
    """The inputs an execution's circuit reads."""
    graph = execution.program_graph
    return {d.name for d in graph.all_dims if isinstance(d, InputDimension) and d is not graph.one}


def computed(execution) -> set[str]:
    """The architectural outputs its circuit computes (not folded or kept)."""
    return {out.name for out in execution.outputs if out.source == "model"}


def test_shape_execution_reads_only_live_state(union_pg):
    shape = ("dex_bne", bytes.fromhex("cad0fd"), 1)
    experts, _meta = build_shape_experts(union_pg, [shape])
    execution = experts[0]["execution"]

    # The PC feeds the closing branch's target and page-cross timing; the
    # carry is P's only flag the window reads.
    assert reads(execution) == {"in_x", "in_pc_lo", "in_pc_hi", "in_p_c"}
    assert {"p_z", "p_n", "new_pc"} <= computed(execution)
    # DEX's 2 cycles fold; the closing BNE's are the circuit's (taken or
    # not, across a page or not).
    assert experts[0]["cycles_prefix"] == 2 and experts[0]["spin"] == 0
    assert [c for c in execution.cycles if isinstance(c, int)] == [2]
    assert sum(not isinstance(c, int) for c in execution.cycles) == 1


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
    execution = [expert["execution"] for expert in experts]
    dex, sta, sax, pha, jmp, jsr, rts, brk = (execution[i] for i in (0, 3, 4, 5, 6, 7, 8, 9))
    assert {n for n in reads(dex) if not n.startswith("in_p_")} == {"in_x", "in_pc_lo", "in_pc_hi"}
    assert not dex.memory_writes
    assert len(sta.memory_writes) == len(sax.memory_writes) == 1
    assert pha.memory_writes
    assert "new_pc" in computed(jmp) and "new_pc" in computed(rts)
    assert len(jsr.memory_writes) == 2
    assert len(brk.memory_writes) == 3 and brk.halt_dimension is not None


def test_shape_interior_branch_is_rejected(union_pg):
    """Shape dispatch has no speculative rollback ABI."""
    plain = ("bne_lda", bytes.fromhex("d000a900"), 1)
    with pytest.raises(RuntimeError, match="interior next_pc"):
        build_shape_experts(union_pg, [plain])


def test_completed_compilation_does_not_retain_the_source_graph():
    import gc
    import weakref

    from transformer_nes.cpu6502.experts import sequence_folder
    from transformer_nes.cpu6502.graph import Cpu6502Graph

    program = Cpu6502Graph(bcd=False, wide_bits=True).program_graph()
    reference = weakref.ref(program)
    folder = sequence_folder(program)
    del program, folder
    gc.collect()
    assert reference() is None
