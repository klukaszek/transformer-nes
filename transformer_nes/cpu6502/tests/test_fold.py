"""Receipts for graph/fold.py: rebuilt (folded) per-opcode graphs.

Three layers of evidence, cheapest first:

1. The unpinned fold reports metrics matching its materialized graph and does
   not deepen the union graph (it may prune dead construction nodes).
2. Folded per-opcode graphs are materially smaller than the union graph, and
   the shared-state contract holds: no lookup's insert-side expressions can
   reach a decode pin.
3. The semantic receipt, mirroring the engine's routed-dispatch contract:
   the union graph (via the direct ProgramEvaluator) drives the token
   stream and is verified against the reference oracle on the architectural
   fields (PC/A/X/Y/S/P + write set -- the same comparison the klaus
   lockstep makes; ir/operand/ea/data are graph-convention trace context).
   Each folded expert graph then replays the identical token stream and
   must reproduce the union's readouts *exactly* on every step whose
   fetched opcode matches its pin.

The materialized fold carries dependencies through inlined gate=1 folds, so
its depth and term counts are measured directly from the graph under test.
"""

import pytest

from transformer_nes.cpu6502.backends import reference
from transformer_nes.cpu6502.backends.graph import RUNTIME_READ_DIMS
from transformer_nes.cpu6502.backends.weights.experts import (
    SEQUENCE_PRESERVE_INPUTS,
    SEQUENCE_ROOT_DIMS,
    SEQUENCE_STATE_DIMS,
    build_shape_experts,
    decode_ops,
    last_a_bit_step,
    pinned_sequence_control,
    project_sequence_execution,
    schedule_and_build,
    semantic_recurrence,
    sparse_forward_work,
)
from transformer_nes.cpu6502.backends.weights.runner import VERIFY_START, verify_program
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
from transformer_nes.graph.eval import ProgramEvaluator
from transformer_nes.graph.fold import (
    SEQUENCE_FEEDBACK,
    GraphFolder,
    opcode_pins,
    sequence_pins,
    sequence_shape_pins,
)
from transformer_nes.scheduler.milp import build_graph, min_layers

# Opcodes exercised by weights_runner.verify_program, chosen to cover the
# distinct machinery: zero-page store (write channel), zero-page load
# (memory read), branch, jump, JSR (two stack writes), immediate ALU+flags.
SHADOW_OPCODES = (0x85, 0xA5, 0xD0, 0x4C, 0x20, 0x69)

SHADOW_STEPS = 40


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


# ── Semantic shadow receipt ──────────────────────────────────────────


def boot_tokens(image):
    tokens = [("boot", {
        "start_pc_lo": VERIFY_START & 0xFF,
        "start_pc_hi": (VERIFY_START >> 8) & 0xFF,
        "is_start": 1,
    })]
    for addr, b in enumerate(image):
        if b:
            tokens.append(("boot", {
                "is_memory_write": 1, "byte_value": b,
                "addr_lo": addr & 0xFF, "addr_hi": (addr >> 8) & 0xFF,
            }))
    return tokens


def read_all(ev, v):
    return {name: int(round(ev.read(v, name))) for name in RUNTIME_READ_DIMS}


def writes_of(reads):
    writes = set()
    for store, addr, value in (
        ("is_store", "write_addr_out", "write_value_out"),
        ("is_store2", "write_addr2_out", "write_value2_out"),
        ("is_store3", "write_addr3_out", "write_value3_out"),
    ):
        if reads[store] & 0xFF:
            writes.add((reads[addr] & 0xFFFF, reads[value] & 0xFF))
    return writes


@pytest.fixture(scope="module")
def union_run(union_pg):
    """Drive the union graph over the verify program with the direct
    evaluator, checking the architectural fields against the oracle at every
    step (this is also the receipt for ProgramEvaluator itself).  Returns
    the token stream plus the union's per-step readouts to replay experts
    against."""
    ev = ProgramEvaluator(union_pg)
    image = verify_program()
    tokens = boot_tokens(image)
    for _, inputs in tokens:
        ev.step(inputs)

    ref = reference.Cpu6502(memory=bytearray(image), bcd_enabled=False)
    ref.pc = VERIFY_START
    st = {
        "pc": VERIFY_START, "a": 0, "x": 0, "y": 0, "s": 0xFD,
        "p": reference.FLAG_U | reference.FLAG_I,
        "ir": 0, "oplo": 0, "ophi": 0, "ea": 0, "data": 0,
    }
    state_expr = union_pg.output_tokens["STATE"]
    halt_expr = union_pg.output_tokens["HALT"]

    steps = []
    for _ in range(SHADOW_STEPS):
        p = st["p"]
        inputs = {
            "is_state": 1,
            "in_pc_lo": st["pc"] & 0xFF, "in_pc_hi": (st["pc"] >> 8) & 0xFF,
            "in_a": st["a"], "in_x": st["x"], "in_y": st["y"], "in_s": st["s"],
            "in_p_c": 1 if p & reference.FLAG_C else 0,
            "in_p_z": 1 if p & reference.FLAG_Z else 0,
            "in_p_i": 1 if p & reference.FLAG_I else 0,
            "in_p_d": 1 if p & reference.FLAG_D else 0,
            "in_p_b": 1 if p & reference.FLAG_B else 0,
            "in_p_u": 1 if p & reference.FLAG_U else 0,
            "in_p_v": 1 if p & reference.FLAG_V else 0,
            "in_p_n": 1 if p & reference.FLAG_N else 0,
            "in_ir": st["ir"], "in_oplo": st["oplo"], "in_ophi": st["ophi"],
            "in_ea_lo": st["ea"] & 0xFF, "in_ea_hi": (st["ea"] >> 8) & 0xFF,
            "in_data": st["data"], "in_halted": 0, "in_nmi": 0,
        }
        v = ev.step(inputs)
        reads = read_all(ev, v)
        writes = writes_of(reads)

        # The union graph must match the oracle on the architectural fields
        # (same contract as klaus.lockstep_arrays_vs_reference).
        row = ref.step()
        s = row.state
        got = (reads["current_pc"] & 0xFFFF, reads["new_a"] & 0xFF,
               reads["new_x"] & 0xFF, reads["new_y"] & 0xFF,
               reads["new_s"] & 0xFF, reads["new_p"] & 0xFF)
        assert got == (s.pc, s.a, s.x, s.y, s.s, s.p)
        # Trace context follows the reference convention exactly (2026-07-09).
        context = (reads["fetched_opcode"] & 0xFF,
                   reads["fetched_operand_lo"] & 0xFF,
                   reads["fetched_operand_hi"] & 0xFF,
                   reads["ea"] & 0xFFFF, reads["data_out"] & 0xFF)
        assert context == (s.ir, s.operand_lo, s.operand_hi, s.ea, s.data)
        assert writes == {(w.addr, w.value) for w in row.writes}
        assert state_expr.evaluate(v) > halt_expr.evaluate(v)

        tokens.append(("state", inputs))
        steps.append({"reads": reads, "opcode": reads["fetched_opcode"] & 0xFF})
        for addr, value in sorted(writes):
            wtok = {
                "is_memory_write": 1, "byte_value": value,
                "addr_lo": addr & 0xFF, "addr_hi": (addr >> 8) & 0xFF,
            }
            ev.step(wtok)
            tokens.append(("boot", wtok))

        st.update(
            pc=reads["next_pc"] & 0xFFFF, a=reads["new_a"] & 0xFF,
            x=reads["new_x"] & 0xFF, y=reads["new_y"] & 0xFF,
            s=reads["new_s"] & 0xFF, p=reads["new_p"] & 0xFF,
            ir=reads["fetched_opcode"] & 0xFF,
            oplo=reads["fetched_operand_lo"] & 0xFF,
            ophi=reads["fetched_operand_hi"] & 0xFF,
            ea=reads["ea"] & 0xFFFF, data=reads["data_out"] & 0xFF,
        )
    return {"tokens": tokens, "steps": steps}


# ── Sequence fold: the poll trio and the JMP spin as one step ────────

POLL_HEAD = 0x0400
SPIN_HEAD = 0x0408


def poll_program():
    """LDA $0700 / AND #$40 / BEQ -7 poll loop, then NOP and a JMP spin."""
    image = bytearray(0x10000)
    image[POLL_HEAD:POLL_HEAD + 11] = bytes([
        0xAD, 0x00, 0x07,  # LDA $0700
        0x29, 0x40,        # AND #$40
        0xF0, 0xF9,        # BEQ $0400
        0xEA,              # NOP
        0x4C, 0x08, 0x04,  # JMP $0408 (self)
    ])
    return image


@pytest.fixture(scope="module")
def poll_run(union_pg):
    """Drive the union over the poll program (three loop iterations, a flag
    write flipping the poll, then the fall-through into the JMP spin)."""
    ev = ProgramEvaluator(union_pg)
    image = poll_program()
    tokens = [("boot", {
        "start_pc_lo": POLL_HEAD & 0xFF, "start_pc_hi": POLL_HEAD >> 8,
        "is_start": 1,
    })]
    for addr, b in enumerate(image):
        if b:
            tokens.append(("boot", {
                "is_memory_write": 1, "byte_value": b,
                "addr_lo": addr & 0xFF, "addr_hi": (addr >> 8) & 0xFF,
            }))
    for _, inputs in tokens:
        ev.step(inputs)

    st = {
        "pc": POLL_HEAD, "a": 0, "x": 0, "y": 0, "s": 0xFD,
        "p": reference.FLAG_U | reference.FLAG_I,
        "ir": 0, "oplo": 0, "ophi": 0, "ea": 0, "data": 0,
    }
    steps = []
    for i in range(24):
        if i == 9:  # after three full loop iterations: raise the flag
            wtok = {"is_memory_write": 1, "byte_value": 0x40,
                    "addr_lo": 0x00, "addr_hi": 0x07}
            ev.step(wtok)
            tokens.append(("boot", wtok))
        p = st["p"]
        inputs = {
            "is_state": 1,
            "in_pc_lo": st["pc"] & 0xFF, "in_pc_hi": (st["pc"] >> 8) & 0xFF,
            "in_a": st["a"], "in_x": st["x"], "in_y": st["y"], "in_s": st["s"],
            "in_p_c": 1 if p & reference.FLAG_C else 0,
            "in_p_z": 1 if p & reference.FLAG_Z else 0,
            "in_p_i": 1 if p & reference.FLAG_I else 0,
            "in_p_d": 1 if p & reference.FLAG_D else 0,
            "in_p_b": 1 if p & reference.FLAG_B else 0,
            "in_p_u": 1 if p & reference.FLAG_U else 0,
            "in_p_v": 1 if p & reference.FLAG_V else 0,
            "in_p_n": 1 if p & reference.FLAG_N else 0,
            "in_ir": st["ir"], "in_oplo": st["oplo"], "in_ophi": st["ophi"],
            "in_ea_lo": st["ea"] & 0xFF, "in_ea_hi": (st["ea"] >> 8) & 0xFF,
            "in_data": st["data"], "in_halted": 0, "in_nmi": 0,
        }
        v = ev.step(inputs)
        reads = read_all(ev, v)
        assert not writes_of(reads), "the poll program must be write-free"
        tokens.append(("state", inputs))
        steps.append({"reads": reads, "pc": st["pc"]})
        st.update(
            pc=reads["next_pc"] & 0xFFFF, a=reads["new_a"] & 0xFF,
            x=reads["new_x"] & 0xFF, y=reads["new_y"] & 0xFF,
            s=reads["new_s"] & 0xFF, p=reads["new_p"] & 0xFF,
            ir=reads["fetched_opcode"] & 0xFF,
            oplo=reads["fetched_operand_lo"] & 0xFF,
            ophi=reads["fetched_operand_hi"] & 0xFF,
            ea=reads["ea"] & 0xFFFF, data=reads["data_out"] & 0xFF,
        )
    # The stream must contain poll iterations, the fall-through, and spins.
    assert sum(s["pc"] == POLL_HEAD for s in steps) == 4
    assert sum(s["pc"] == SPIN_HEAD for s in steps) >= 3
    return {"tokens": tokens, "steps": steps}


@pytest.mark.parametrize(
    ("head", "code", "repeat"),
    [
        (POLL_HEAD, bytes([0xAD, 0x00, 0x07, 0x29, 0x40, 0xF0, 0xF9]), 1),
        (SPIN_HEAD, bytes([0x4C, 0x08, 0x04]), 3),
    ],
    ids=["poll-trio", "jmp-spin-x3"],
)
@pytest.mark.parametrize(
    "inline_linear", [False, True], ids=["materialized", "linear-inlined"],
)
def test_sequence_fold_shadows_union(
    folder, poll_run, head, code, repeat, inline_linear,
):
    """One sequence forward must reproduce the union's K consecutive step
    readouts exactly whenever the router would dispatch it (pc == head; the
    window bytes are pinned).  Intermediate STATE tokens are not fed to the
    sequence graph -- exactly the engine's one-forward-per-window contract."""
    pin_list = sequence_pins(head, code, OPCODES) * repeat
    n_steps = len(pin_list)
    r = folder.fold_sequence(pin_list, SEQUENCE_FEEDBACK,
                             label=f"seq@{head:04X}",
                             inline_linear=inline_linear)
    assert r.n_steps == n_steps
    assert not r.insert_side_hazards
    ev = ProgramEvaluator(r.program_graph)
    state_expr = r.program_graph.output_tokens["STATE"]
    halt_expr = r.program_graph.output_tokens["HALT"]

    steps = poll_run["steps"]
    idx = 0  # union step index for state tokens
    skip = 0
    windows = 0
    for kind, inputs in poll_run["tokens"]:
        if kind != "state":
            ev.step(inputs)
            continue
        if skip:
            skip -= 1
            idx += 1
            continue
        pc = inputs["in_pc_lo"] | (inputs["in_pc_hi"] << 8)
        dispatch = pc == head and idx + n_steps <= len(steps)
        if not dispatch:
            ev.step(inputs)
            idx += 1
            continue
        v = ev.step(inputs)
        for k in range(n_steps):
            expected = steps[idx + k]["reads"]
            got = {
                name: int(round(ev.read(v, f"{name}__s{k}")))
                for name in RUNTIME_READ_DIMS
            }
            for name in RUNTIME_READ_DIMS:
                mask = 0xFFFF if name in ("current_pc", "next_pc", "ea") else 0xFF
                assert got[name] & mask == expected[name] & mask, (
                    f"step {k} ({name}) diverged at union step {idx + k}"
                )
        assert state_expr.evaluate(v) > halt_expr.evaluate(v)
        windows += 1
        skip = n_steps - 1
        idx += 1
    assert windows >= 2, "the stream never dispatched the sequence"


def test_sequence_execution_projection_keeps_only_final_state(folder, poll_run):
    """The execution model is built from the licensed fold after removing
    the per-instruction trace receipt.  A write-free three-instruction poll
    keeps exactly one final architectural receipt and still agrees with the
    union on that state at every dispatch."""
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

    ev = ProgramEvaluator(execution.program_graph)
    steps = poll_run["steps"]
    idx = 0
    skip = 0
    windows = 0
    for kind, inputs in poll_run["tokens"]:
        if kind != "state":
            ev.step(inputs)
            continue
        if skip:
            skip -= 1
            idx += 1
            continue
        pc = inputs["in_pc_lo"] | (inputs["in_pc_hi"] << 8)
        if pc != POLL_HEAD or idx + len(pin_list) > len(steps):
            ev.step(inputs)
            idx += 1
            continue
        values = ev.step(inputs)
        expected = steps[idx + len(pin_list) - 1]["reads"]
        for output in execution.outputs:
            name = output.name
            mask = 0xFFFF if name in ("next_pc", "ea") else 0xFF
            if output.source == "model":
                value = ev.read(values, output.dimension.name)
            elif output.source == "constant":
                value = output.constant
            else:
                value = sum(
                    (1 if input_name == "one" else inputs[input_name]) * coeff
                    for input_name, coeff in SEQUENCE_PRESERVE_INPUTS[name]
                )
            assert int(round(value)) & mask == expected[name] & mask
        control = execution.control
        taken = (
            False if control.predicate is None
            else bool(round(values[control.predicate]))
        )
        successor = control.target if taken else control.fallthrough
        assert successor == expected["next_pc"]
        windows += 1
        skip = len(pin_list) - 1
        idx += 1
    assert windows >= 2


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
        folder.pg, root_names=RUNTIME_READ_DIMS, structural_reads=True,
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


@pytest.mark.parametrize("byte", SHADOW_OPCODES)
@pytest.mark.parametrize(
    "inline_linear", [False, True], ids=["materialized", "linear-inlined"],
)
def test_folded_expert_shadows_union(folder, union_run, byte, inline_linear):
    """Replay the union's token stream through the folded expert graph; on
    every step whose fetched opcode matches the pin, all runtime readouts
    must agree exactly -- the engine's routed-dispatch contract."""
    r = folder.fold(
        opcode_pins(byte, OPCODES), label=f"{byte:02X}",
        inline_linear=inline_linear,
    )
    ev = ProgramEvaluator(r.program_graph)
    state_expr = r.program_graph.output_tokens["STATE"]
    halt_expr = r.program_graph.output_tokens["HALT"]

    step_iter = iter(union_run["steps"])
    matched = 0
    for kind, inputs in union_run["tokens"]:
        v = ev.step(inputs)
        if kind != "state":
            continue
        expected = next(step_iter)
        if expected["opcode"] != byte:
            continue
        reads = read_all(ev, v)
        assert reads == expected["reads"], f"routed step diverged for {byte:02X}"
        assert state_expr.evaluate(v) > halt_expr.evaluate(v)
        matched += 1
    assert matched > 0, f"opcode {byte:02X} never executed in the prefix"


# ── Canonical shape fold: symbolic pc + operands, one fold per SHAPE ──

SHAPE_HEADS = {
    "dex-bne": (0x0480, 0x04FF),   # the second window crosses a page
    "poll-trio": (0x0400, 0x0520),  # different poll address AND mask
}


def shape_program():
    """Two instantiations of each shape: the DEX/BNE delay at an in-page
    head and at $04FF (its BNE sits at $0500 -- the pc carry gadget fires),
    then two poll trios with different operands."""
    image = bytearray(0x10000)
    image[0x047E:0x0483] = bytes([0xA2, 0x03,          # LDX #$03
                                  0xCA, 0xD0, 0xFD])   # DEX / BNE -3
    image[0x0483:0x0488] = bytes([0xA2, 0x02,          # LDX #$02
                                  0x4C, 0xFF, 0x04])   # JMP $04FF
    image[0x04FF:0x0505] = bytes([0xCA, 0xD0, 0xFD,    # DEX / BNE -3 (cross)
                                  0x4C, 0x00, 0x04])   # JMP $0400
    image[0x0400:0x040A] = bytes([0xAD, 0x00, 0x07,    # LDA $0700
                                  0x29, 0x40,          # AND #$40
                                  0xF0, 0xF9,          # BEQ -7
                                  0x4C, 0x20, 0x05])   # JMP $0520
    image[0x0520:0x052A] = bytes([0xAD, 0x03, 0x07,    # LDA $0703
                                  0x29, 0x80,          # AND #$80
                                  0xF0, 0xF9,          # BEQ -7
                                  0x4C, 0x27, 0x05])   # JMP self
    return image


@pytest.fixture(scope="module")
def shape_run(union_pg):
    """Union walk over the shape program: both delay loops (taken + exit),
    both polls (three spins, a flag flip, the fall-through)."""
    ev = ProgramEvaluator(union_pg)
    image = shape_program()
    start = 0x047E
    tokens = [("boot", {"start_pc_lo": start & 0xFF, "start_pc_hi": start >> 8,
                        "is_start": 1})]
    for addr, b in enumerate(image):
        if b:
            tokens.append(("boot", {"is_memory_write": 1, "byte_value": b,
                                    "addr_lo": addr & 0xFF, "addr_hi": addr >> 8}))
    for _, inputs in tokens:
        ev.step(inputs)

    ref = reference.Cpu6502(memory=bytearray(image), bcd_enabled=False)
    ref.pc = start
    st = {"pc": start, "a": 0, "x": 0, "y": 0, "s": 0xFD,
          "p": reference.FLAG_U | reference.FLAG_I,
          "ir": 0, "oplo": 0, "ophi": 0, "ea": 0, "data": 0}
    steps = []
    for i in range(52):
        for flip_i, addr, value in ((23, 0x0700, 0x40), (37, 0x0703, 0x80)):
            if i == flip_i:
                w = {"is_memory_write": 1, "byte_value": value,
                     "addr_lo": addr & 0xFF, "addr_hi": addr >> 8}
                ev.step(w)
                tokens.append(("boot", w))
                ref.memory[addr] = value
        p = st["p"]
        inputs = {
            "is_state": 1,
            "in_pc_lo": st["pc"] & 0xFF, "in_pc_hi": (st["pc"] >> 8) & 0xFF,
            "in_a": st["a"], "in_x": st["x"], "in_y": st["y"], "in_s": st["s"],
            "in_p_c": 1 if p & reference.FLAG_C else 0,
            "in_p_z": 1 if p & reference.FLAG_Z else 0,
            "in_p_i": 1 if p & reference.FLAG_I else 0,
            "in_p_d": 1 if p & reference.FLAG_D else 0,
            "in_p_b": 1 if p & reference.FLAG_B else 0,
            "in_p_u": 1 if p & reference.FLAG_U else 0,
            "in_p_v": 1 if p & reference.FLAG_V else 0,
            "in_p_n": 1 if p & reference.FLAG_N else 0,
            "in_ir": st["ir"], "in_oplo": st["oplo"], "in_ophi": st["ophi"],
            "in_ea_lo": st["ea"] & 0xFF, "in_ea_hi": (st["ea"] >> 8) & 0xFF,
            "in_data": st["data"], "in_halted": 0, "in_nmi": 0,
        }
        v = ev.step(inputs)
        reads = read_all(ev, v)
        row = ref.step()
        s = row.state
        got = (reads["current_pc"] & 0xFFFF, reads["new_a"] & 0xFF,
               reads["new_x"] & 0xFF, reads["new_p"] & 0xFF)
        assert got == (s.pc, s.a, s.x, s.p), f"union vs oracle at step {i}"
        tokens.append(("state", inputs))
        steps.append({"reads": reads, "pc": st["pc"]})
        st.update(pc=reads["next_pc"] & 0xFFFF, a=reads["new_a"] & 0xFF,
                  x=reads["new_x"] & 0xFF, y=reads["new_y"] & 0xFF,
                  s=reads["new_s"] & 0xFF, p=reads["new_p"] & 0xFF,
                  ir=reads["fetched_opcode"] & 0xFF,
                  oplo=reads["fetched_operand_lo"] & 0xFF,
                  ophi=reads["fetched_operand_hi"] & 0xFF,
                  ea=reads["ea"] & 0xFFFF, data=reads["data_out"] & 0xFF)
    for heads in SHAPE_HEADS.values():
        for head in heads:
            assert sum(s["pc"] == head for s in steps) >= 2, hex(head)
    return {"tokens": tokens, "steps": steps}


@pytest.mark.parametrize(
    ("shape_id", "code"),
    [
        ("dex-bne", bytes([0xCA, 0xD0, 0xFD])),
        ("poll-trio", bytes([0xAD, 0, 0, 0x29, 0, 0xF0, 0])),
    ],
    ids=["dex-bne", "poll-trio"],
)
@pytest.mark.parametrize(
    "inline_linear", [False, True], ids=["materialized", "linear-inlined"],
)
def test_shape_fold_shadows_union(
    folder, shape_run, shape_id, code, inline_linear,
):
    """ONE fold of a canonical shape (pc and every operand byte symbolic --
    only the opcode bytes are pinned) must reproduce the union's K-step
    readouts at BOTH instantiations, page-cross and both branch outcomes
    included.  This is the zero-per-ROM-compile contract: the same weights
    serve every (head, operands) the router matches."""
    heads = SHAPE_HEADS[shape_id]
    pin_list, offsets = sequence_shape_pins(code, OPCODES)
    n_steps = len(pin_list)
    r = folder.fold_sequence(pin_list, SEQUENCE_FEEDBACK,
                             label=f"shape_{shape_id}", pc_offsets=offsets,
                             inline_linear=inline_linear)
    assert r.n_steps == n_steps
    assert not r.insert_side_hazards
    ev = ProgramEvaluator(r.program_graph)

    steps = shape_run["steps"]
    idx = 0
    skip = 0
    dispatched = dict.fromkeys(heads, 0)
    for kind, inputs in shape_run["tokens"]:
        if kind != "state":
            ev.step(inputs)
            continue
        if skip:
            skip -= 1
            idx += 1
            continue
        pc = inputs["in_pc_lo"] | (inputs["in_pc_hi"] << 8)
        if pc not in dispatched or idx + n_steps > len(steps):
            ev.step(inputs)
            idx += 1
            continue
        v = ev.step(inputs)
        for k in range(n_steps):
            expected = steps[idx + k]["reads"]
            for name in RUNTIME_READ_DIMS:
                mask = 0xFFFF if name in ("current_pc", "next_pc", "ea") else 0xFF
                got = int(round(ev.read(v, f"{name}__s{k}"))) & mask
                assert got == expected[name] & mask, (
                    f"head {pc:04X} step {k} ({name}) diverged"
                )
        dispatched[pc] += 1
        skip = n_steps - 1
        idx += 1
    for head, n in dispatched.items():
        assert n >= 2, f"head {head:04X} never dispatched twice"


def test_shape_absolute_store_requires_runtime_guard(union_pg):
    """A symbolic absolute store is admitted only with the narrow router
    license and reports the live operand offset the runtime must guard."""
    shape = ("lda_sta_abs", bytes.fromhex("a9008d0000"), 1)
    with pytest.raises(RuntimeError, match="not provably below"):
        build_shape_experts(
            union_pg, [shape], inline_linear=True, allow_stores=True,
        )
    experts, _meta = build_shape_experts(
        union_pg, [shape], inline_linear=True, allow_stores=True,
        allow_runtime_store_guard=True,
    )
    assert experts[0]["has_stores"]
    assert experts[0]["store_bounds"] is None
    assert experts[0]["store_guard_offset"] == 3


def test_shape_execution_abi_contains_only_live_state(union_pg):
    shape = ("dex_bne", bytes.fromhex("cad0fd"), 1)
    experts, _meta = build_shape_experts(union_pg, [shape])
    spec = experts[0]["step_spec"]

    assert set(spec["inputs"]) == {"pc_lo", "pc_hi", "x"}
    assert {mask for _slot, mask in spec["p_flags"]} == {
        reference.FLAG_C,
        reference.FLAG_I,
        reference.FLAG_D,
        reference.FLAG_B,
        reference.FLAG_V,
    }
    assert spec["sequence_control"] == {
        "predicate": spec["sequence_control"]["predicate"],
        "fallthrough": 3,
        "target": 0,
        "relative_to_head": True,
        "relative_operand_offset": 2,
    }
    assert spec["sequence_control"]["predicate"] is not None
    recurrence = experts[0]["recurrence"]
    assert (
        recurrence.kind, recurrence.register, recurrence.delta,
        recurrence.closure_operand_offset, recurrence.closure_operand_value,
    ) == ("affine_until", "x", -1, 2, 0xFD)


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
    assert recurrence.fallthrough_offset == 3


def test_shape_interior_branch_requires_explicit_speculation(union_pg):
    """A symbolic interior branch records the fall-through PC offset only
    when the router opts into exact prefix rollback."""
    plain = ("bne_lda", bytes.fromhex("d000a900"), 1)
    with pytest.raises(RuntimeError, match="interior next_pc"):
        build_shape_experts(union_pg, [plain], inline_linear=True)

    speculative = (*plain, True)
    experts, _meta = build_shape_experts(
        union_pg, [speculative], inline_linear=True,
    )
    assert experts[0]["spec_offsets"] == [2, -1]
