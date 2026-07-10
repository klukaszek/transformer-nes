"""Receipts for graph/fold.py: rebuilt (folded) per-opcode graphs.

Three layers of evidence, cheapest first:

1. The unpinned fold reproduces the union graph op-for-op and layer-for-layer
   (the moe_probe self-validation, strengthened to a materialized graph).
2. Folded per-opcode graphs shrink the way the probe measured (~0.2x terms),
   and the shared-state contract holds: no lookup's insert-side expressions
   can reach a decode pin.
3. The semantic receipt, mirroring the engine's routed-dispatch contract:
   the union graph (via the direct ProgramEvaluator) drives the token
   stream and is verified against the reference oracle on the architectural
   fields (PC/A/X/Y/S/P + write set -- the same comparison the klaus
   lockstep makes; ir/operand/ea/data are graph-convention trace context).
   Each folded expert graph then replays the identical token stream and
   must reproduce the union's readouts *exactly* on every step whose
   fetched opcode matches its pin.

Note on depth: the probe's per-opcode depths (9/13/22) under-measured --
it kept original expressions over pruned dim sets, so a dependency through
an inlined gate=1 fold was dropped as a dangling edge.  The materialized
fold carries the true dependency, so its depth is moderately larger (e.g.
JMP abs: 17 layers vs the probe's 10, against the union's 71).  Terms (the
load-bound cost) match the probe.
"""

import json
from pathlib import Path

import pytest

from transformer_vm.cpu6502 import reference
from transformer_vm.cpu6502.evaluator import _RUNTIME_READ_DIMS
from transformer_vm.cpu6502.interpreter import build
from transformer_vm.cpu6502.spec import OPCODES
from transformer_vm.cpu6502.weights_runner import VERIFY_START, verify_program
from transformer_vm.graph.core import ProgramGraph, reset_graph
from transformer_vm.graph.eval import ProgramEvaluator
from transformer_vm.graph.fold import GraphFolder, opcode_pins
from transformer_vm.scheduler.milp import _build_graph, _min_layers

_PROBE_JSON = (
    Path(__file__).resolve().parents[3]
    / "experiments" / "moe_probe" / "moe_probe_results.json"
)

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
    return GraphFolder(union_pg, root_names=_RUNTIME_READ_DIMS)


@pytest.fixture(scope="module")
def probe_results():
    if not _PROBE_JSON.exists():
        pytest.skip("moe_probe results not checked in")
    with open(_PROBE_JSON) as f:
        return json.load(f)


def _depth(fold_result):
    pg = fold_result.program_graph
    g = _build_graph(pg.all_dims, pg.all_lookups, pg.inv_log_pos)
    return _min_layers(g["ops"], g["op_deps"]) if g["ops"] else 0


def test_unpinned_fold_reproduces_union(folder, probe_results):
    r = folder.fold({}, label="union")
    base = probe_results["union"]
    assert r.insert_side_hazards == ()
    assert r.folded_roots == ()
    assert r.n_reglus == base["reglus"]
    assert r.n_persists == base["persists"]
    assert r.n_lookups == base["lookups"]
    assert r.n_terms == base["terms"]
    assert _depth(r) == base["depth"]


@pytest.mark.parametrize("byte", [0x4C, 0xAD, 0x29, 0xF0])
def test_opcode_fold_matches_probe(folder, probe_results, byte):
    # pin_fetched_byte=False: the probe pinned only the decode one-hots,
    # so that is the apples-to-apples configuration.
    r = folder.fold(opcode_pins(byte, OPCODES, pin_fetched_byte=False),
                    label=f"{byte:02X}")
    base = probe_results["union"]
    probe = probe_results["per_opcode"][f"{byte:02X}"]
    # The shared-state contract: decode pins never reach insert-side exprs.
    assert r.insert_side_hazards == ()
    # Every runtime readout resolves by name (kept or re-materialized).
    for name in _RUNTIME_READ_DIMS:
        assert name in r.dim_of_name, f"readout {name} missing from fold"
    # The fold roots folded-away readouts back in as persists (the probe
    # dropped them), so counts sit slightly above the probe's; the ratio to
    # the union is what the construction is for.
    assert probe["terms"] * 0.95 <= r.n_terms <= probe["terms"] * 1.10
    assert r.n_terms <= base["terms"] * 0.30
    # True (materialized) depth; see the module docstring for why this runs
    # deeper than the probe's optimistic number.
    assert _depth(r) < base["depth"] * 0.5

    # The default fold additionally pins the fetched byte (the router
    # predicate ram[pc] == byte): the PC fetch read and decode residue fold
    # away, so the graph must get strictly smaller, never larger.
    rb = folder.fold(opcode_pins(byte, OPCODES), label=f"{byte:02X}+byte")
    assert rb.insert_side_hazards == ()
    assert rb.n_terms < r.n_terms
    assert rb.n_lookups < r.n_lookups  # the PC read's group lookups died


# ── Semantic shadow receipt ──────────────────────────────────────────


def _boot_tokens(image):
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


def _read_all(ev, v):
    return {name: int(round(ev.read(v, name))) for name in _RUNTIME_READ_DIMS}


def _writes_of(reads):
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
    tokens = _boot_tokens(image)
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
        reads = _read_all(ev, v)
        writes = _writes_of(reads)

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


@pytest.mark.parametrize("byte", SHADOW_OPCODES)
def test_folded_expert_shadows_union(folder, union_run, byte):
    """Replay the union's token stream through the folded expert graph; on
    every step whose fetched opcode matches the pin, all runtime readouts
    must agree exactly -- the engine's routed-dispatch contract."""
    r = folder.fold(opcode_pins(byte, OPCODES), label=f"{byte:02X}")
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
        reads = _read_all(ev, v)
        assert reads == expected["reads"], f"routed step diverged for {byte:02X}"
        assert state_expr.evaluate(v) > halt_expr.evaluate(v)
        matched += 1
    assert matched > 0, f"opcode {byte:02X} never executed in the prefix"
