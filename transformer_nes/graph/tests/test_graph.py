"""Graph construction isolation, partial evaluation and exact integer circuits."""

import asyncio
import gc
import weakref

import numpy as np
import pytest

from transformer_nes.graph import core
from transformer_nes.graph.core import (
    Expression,
    GraphContext,
    InputDimension,
    LookUp,
    PersistDimension,
    ReGLUDimension,
    construction,
    current_graph,
    persist,
    project_program_graph,
    reglu,
)
from transformer_nes.graph.evaluate import ABSENT, FOLDED, VALUE, enumerate_values, exact_bounds
from transformer_nes.graph.fold import GraphFolder
from transformer_nes.graph.ops import extract_bits, extract_bits_wide, match_zero, wrap8, wrapped_nz


def test_interleaved_contexts_keep_ids_constants_and_interning_local():
    left, right = GraphContext(), GraphContext()
    with left.activate():
        x = InputDimension("x")
        first = reglu(x, 1)
        snapshot = left.capture({"X": x.as_expr()}, {"OUT": first})
    with right.activate():
        y = InputDimension("y")
        other = reglu(y, 1)
        # Pure arithmetic on an existing value keeps its original constant.
        assert (x + 3).terms == {x: 1, left.one: 3}
        before = len(right.dims)
        with pytest.raises(ValueError, match="different graphs"):
            persist(x + 3)
        assert len(right.dims) == before
        with pytest.raises(ValueError, match="different graphs"):
            _ = x + y
        with pytest.raises(ValueError, match="different graphs"):
            Expression({x: 1, y: 1})
    with left.activate():
        assert reglu(x, 1).terms == first.terms
        assert len(left.dims) == len(snapshot.all_dims)
        later = persist(first + 1)
        assert next(iter(later.terms)).id == len(snapshot.all_dims)
    assert x.id == y.id == 1
    assert next(iter(first.terms)).id == next(iter(other.terms)).id == 2
    assert len(snapshot.all_dims) == 3
    assert snapshot.one is left.one


def test_folding_restores_the_surrounding_builder():
    with GraphContext().activate() as source:
        x = InputDimension("x")
        out = persist(reglu(x, 1) + 2, name="out")
        original = source.capture({"X": x.as_expr()}, {"OUT": out})
    folder = GraphFolder(original)
    with GraphContext().activate() as surrounding:
        y = InputDimension("y")
        before = tuple(surrounding.dims)
        folded = folder.fold({})
        assert current_graph() is surrounding
        assert tuple(surrounding.dims) == before
        following = persist(y + 1)
        caller = surrounding.capture({"Y": y.as_expr()}, {"OUT": following})
    assert folded.program_graph.one is not original.one
    assert caller.one is not original.one
    assert all(d.one is original.one for d in original.all_dims)
    assert all(d.one is folded.program_graph.one for d in folded.program_graph.all_dims)


def test_nested_builder_restores_context_after_failure():
    @construction
    def failing():
        InputDimension("temporary")
        raise LookupError("failed build")

    with GraphContext().activate() as outer:
        with pytest.raises(LookupError, match="failed build"):
            failing()
        assert current_graph() is outer
        assert outer.dims == [outer.one]
    with pytest.raises(RuntimeError, match="active GraphContext"):
        InputDimension("unowned")


def test_projection_does_not_retain_builder_caches_or_dead_nodes():
    def build():
        with GraphContext().activate() as context:
            x = InputDimension("x")
            out = persist(2 * x, name="out")
            dead = reglu(x + 7, 1)
            full = context.capture({"X": x.as_expr()}, {"OUT": out})
            projected = project_program_graph(full, full.output_tokens)
            return projected, weakref.ref(context), weakref.ref(next(iter(dead.terms)))

    graph, builder, dead = build()
    gc.collect()
    assert builder() is None
    assert dead() is None
    assert [d.name for d in graph.all_dims] == ["one", "x", "out"]
    assert graph.one.one is graph.one


def test_cpu_capture_survives_other_constructions_and_folds():
    from transformer_nes.cpu6502.construction import Cpu6502Graph
    from transformer_nes.ppu2c02.machine import register_graph

    cpu = Cpu6502Graph(bcd=False, wide_bits=True)
    original = cpu.program_graph()
    ppu = register_graph()
    GraphFolder(ppu).fold({})
    captured = cpu.program_graph()
    assert captured.all_dims == original.all_dims
    assert captured.all_lookups == original.all_lookups
    assert captured.one is cpu.one
    assert captured.one is not ppu.one
    assert captured.output_tokens["READOUT"].terms == original.output_tokens["READOUT"].terms


def test_async_builders_keep_their_active_context_across_yields():
    async def build(name):
        with GraphContext().activate() as context:
            x = InputDimension(name)
            await asyncio.sleep(0)
            assert current_graph() is context
            out = persist(x + 1)
            await asyncio.sleep(0)
            return context.capture({name: x.as_expr()}, {"OUT": out})

    async def interleave():
        return await asyncio.gather(build("x"), build("y"))

    left, right = asyncio.run(interleave())
    assert left.one is not right.one
    assert [d.id for d in left.all_dims] == [0, 1, 2]
    assert [d.id for d in right.all_dims] == [0, 1, 2]
    assert all(d.one is left.one for d in left.all_dims)
    assert all(d.one is right.one for d in right.all_dims)


@pytest.fixture
@construction
def circuit():
    x = InputDimension("x")
    gate = PersistDimension(x.as_expr(), name="gate")
    value = PersistDimension(3 * x + 2, name="value")
    neuron = ReGLUDimension(value.as_expr(), gate.as_expr(), name="neuron")
    out = PersistDimension(neuron.as_expr(), name="out")
    pg = core.current_graph().capture({"VALUE": x.as_expr()}, {"READOUT": out.as_expr()})
    return GraphFolder(pg), x, gate, value, neuron, out


def test_nonpositive_gate_eliminates_the_computation(circuit):
    folder, _x, _gate, _value, _neuron, out = circuit
    evaluation = folder.evaluate({"gate": -2}, inline_linear=True)
    assert evaluation.val[out].terms == {}


def test_positive_gate_substitutes_a_linear_expression(circuit):
    folder, x, _gate, _value, _neuron, out = circuit
    evaluation = folder.evaluate({"gate": 2}, inline_linear=True)
    assert evaluation.val[out].terms == {x: 6, folder.pg.one: 4}


def test_frozen_dimension_overrides_a_pin(circuit):
    folder, _x, gate, _value, neuron, _out = circuit
    evaluation = folder.evaluate({"gate": -2}, frozen={"gate"}, inline_linear=True)
    assert evaluation.status(gate) == ABSENT
    assert evaluation.status(neuron) == FOLDED


def test_preserved_dimension_stays_materialized(circuit):
    folder, _x, _gate, value, _neuron, out = circuit
    evaluation = folder.evaluate({"gate": 2}, inline_linear=True, preserve={value})
    assert evaluation.status(value) == FOLDED
    assert evaluation.val[out].terms == {value: 2}


@construction
def test_exact_byte_bounds_preserve_wrapping_correlations():
    x = InputDimension("x")
    result = PersistDimension(wrap8(x + 248))

    def bounds(d):
        return (1, 1) if d is core.current_graph().one else (0, 255)

    assert exact_bounds(result, bounds) == (0, 255)


@construction
def test_exact_bounds_leave_large_domains_and_memory_unknown():
    x, y = InputDimension("x"), InputDimension("y")
    result = PersistDimension(x + y)
    assert exact_bounds(result, lambda _d: (0, 255)) is None
    read = core.MemoryRead(x.as_expr())
    assert exact_bounds(read.dims[0], lambda _d: (0, 255)) is None


@pytest.mark.parametrize("coefficient", [0.5, 2**40])
@construction
def test_exact_bounds_refuse_fractional_or_overflowing_arithmetic(coefficient):
    x = InputDimension("x")
    result = PersistDimension(coefficient * x)
    assert exact_bounds(result, lambda _d: (0, 255)) is None


@pytest.mark.parametrize("right", [False, True])
@construction
def test_enumerated_memory_bits_share_their_byte_domain(right):
    carry = InputDimension("carry")
    read = core.MemoryRead(0x10, bits=["bit0"])
    byte, _address, low = read.dims
    result = (byte - low) * 0.5 + 128 * carry if right else wrap8(2 * byte + carry)

    def domains(d):
        return (0, 255) if d is byte else (1, 1) if d is core.current_graph().one else (0, 1)

    values, inputs = enumerate_values(result, domains, max_combinations=512)
    expected = (
        (inputs[byte] >> 1) | (inputs[carry] << 7)
        if right
        else ((2 * inputs[byte] + inputs[carry]) & 255)
    )
    assert (values == expected).all()
    assert values.size == 512


@construction
def test_enumeration_keeps_distinct_reads_independent_and_addresses_opaque():
    left, right = core.MemoryRead(0x10), core.MemoryRead(0x10)
    assert enumerate_values(left.dims[0] - right.dims[0], lambda _d: (0, 255)) is None
    assert enumerate_values(left.dims[1].as_expr(), lambda _d: (0, 255)) is None


@pytest.mark.parametrize("kind", ["coefficient", "product", "overflow"])
@construction
def test_enumeration_refuses_unrepresentable_fixed_point_values(kind):
    x = InputDimension("x")
    half = PersistDimension(x * 0.5)
    result = {
        "coefficient": x * 0.25,
        "product": ReGLUDimension(half.as_expr(), half.as_expr()).as_expr(),
        "overflow": x * (2**30),
    }[kind]
    assert enumerate_values(result, lambda _d: (0, 1)) is None


@pytest.mark.parametrize("structural", [False, True])
@construction
def test_query_inlining_preserves_key_and_value_structure(structural):
    x = InputDimension("x")
    query = PersistDimension(2 * x, name="query")
    key = PersistDimension(3 * x, name="key")
    value = PersistDimension(4 * x, name="value")
    lookup = LookUp(
        [value.as_expr()],
        [query.as_expr(), core.current_graph().one.as_expr()],
        [key.as_expr(), core.current_graph().one.as_expr()],
    )
    pg = core.current_graph().capture({"TOKEN": x.as_expr()}, {"READOUT": lookup.dims[0].as_expr()})
    evaluation = GraphFolder(pg, structural_reads=structural).evaluate({}, inline_linear=True)
    assert evaluation.status(query) == (VALUE if structural else FOLDED)
    assert evaluation.status(key) == FOLDED
    assert evaluation.status(value) == FOLDED
    _, _, lookups = evaluation
    assert lookups[lookup]["query"][0].terms == ({x: 2} if structural else {query: 1})


def circuit_values(input_dim, inputs):
    """Evaluate elementwise graph primitives with the runtime's int32 arithmetic."""
    values = {core.current_graph().one: np.ones_like(inputs), input_dim: inputs}

    def expression(expr):
        result = np.zeros_like(inputs)
        for dim, coefficient in expr.terms.items():
            assert coefficient == int(coefficient)
            result += int(coefficient) * values[dim]
        return result

    for dim in core.current_graph().dims:
        if isinstance(dim, ReGLUDimension):
            values[dim] = expression(dim.a_expr) * np.maximum(expression(dim.b_expr), 0)
        elif isinstance(dim, PersistDimension):
            values[dim] = expression(dim.expr)
    return expression


@construction
def test_zero_mask_does_not_square_a_full_address_difference():
    difference = InputDimension("difference")
    mask = match_zero(difference)
    domain = np.arange(-65535, 65536, dtype=np.int32)
    evaluate = circuit_values(difference, domain)
    np.testing.assert_array_equal(evaluate(mask), domain == 0)


@pytest.mark.parametrize("extract", [extract_bits, extract_bits_wide])
@construction
def test_byte_bit_extraction_is_exact_for_every_byte(extract):
    value = InputDimension("byte")
    bits = extract(value, "result")
    domain = np.arange(256, dtype=np.int32)
    evaluate = circuit_values(value, domain)
    for index, bit in enumerate(bits):
        np.testing.assert_array_equal(evaluate(bit), (domain >> (7 - index)) & 1)


@pytest.mark.parametrize("lo,hi", [(-256, 511), (-1, 254), (1, 256), (0, 511)])
@construction
def test_parallel_wrapped_flags_cover_the_full_integer_domain(lo, hi):
    raw = InputDimension("raw")
    negative, zero = wrapped_nz(raw, lo=lo, hi=hi)
    domain = np.arange(lo, hi + 1, dtype=np.int32)
    evaluate = circuit_values(raw, domain)
    np.testing.assert_array_equal(evaluate(negative), (domain & 255) >= 128)
    np.testing.assert_array_equal(evaluate(zero), (domain & 255) == 0)


@construction
def test_lookup_rejects_unsupported_ties_at_construction():
    with pytest.raises(ValueError, match="only the LATEST tie-break"):
        core.LookUp(
            [],
            [core.current_graph().one.as_expr()] * 2,
            [core.current_graph().one.as_expr()] * 2,
            tie_break="average",
        )
