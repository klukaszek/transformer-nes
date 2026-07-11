"""Contracts for attention structures shared by both generated backends."""

from __future__ import annotations

from transformer_nes.model.classify import const_query, memory_query, register_query

ONE = 7


def test_const_query_accepts_only_the_constant_slot():
    assert const_query([], ONE) == 0.0
    assert const_query([(3.5, ONE)], ONE) == 3.5
    assert const_query([(0.0, 99), (3.5, ONE)], ONE) == 3.5
    assert const_query([(3.5, 99)], ONE) is None
    assert const_query([(1.0, ONE), (2.0, ONE)], ONE) is None


def test_memory_query_normalizes_a_positive_score_scale():
    assert memory_query(
        [(512.0, 10), (-2.0, 11)],
        [(2.0, ONE)],
        one_id=ONE,
    ) == [(256, 10), (-1, 11)]


def test_memory_query_rejects_unsafe_score_forms():
    valid_qx = [(4.0, 10)]
    assert memory_query(valid_qx, [], one_id=ONE) is None
    assert memory_query(valid_qx, [(-1.0, ONE)], one_id=ONE) is None
    assert memory_query([(1.5, 10)], [(1.0, ONE)], one_id=ONE) is None
    assert memory_query([(float(2**24), 10)], [(1.0, ONE)], one_id=ONE) is None
    assert memory_query([(float("inf"), 10)], [(1.0, ONE)], one_id=ONE) is None


def test_register_query_requires_constant_queries_and_latest_ties():
    qx = [(2.0, ONE)]
    qy = [(-3.0, ONE)]
    assert register_query(qx, qy, one_id=ONE, latest=True) == (2.0, -3.0)
    assert register_query(qx, qy, one_id=ONE, latest=False) is None
    assert register_query([(2.0, 99)], qy, one_id=ONE, latest=True) is None
