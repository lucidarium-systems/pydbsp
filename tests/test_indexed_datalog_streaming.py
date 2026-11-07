"""Shared persistent driver: compare every batch with fresh stratified evaluation."""

import pytest

from pydbsp.datalog_stratified import (
    DatalogStratifiedComposition,
    IndexedDatalog,
    IndexedDatalogWithNegation,
)
from pydbsp.zset import ZSet, ZSetAddition


def drive(cls, episodes):
    driver = cls()
    group = ZSetAddition()
    rules, facts = ZSet({}), ZSet({})
    models = []
    evaluator = None
    for rule_delta, fact_delta in episodes:
        driver.push_rules(ZSet(rule_delta))
        driver.push_facts(ZSet(fact_delta))
        rules = group.add(rules, ZSet(rule_delta))
        facts = group.add(facts, ZSet(fact_delta))
        reference = DatalogStratifiedComposition()
        reference.push_rules(rules)
        reference.push_facts(facts)
        got = driver.saturate()
        assert got == reference.saturate()
        if evaluator is not None:
            assert driver._e is evaluator  # retaining the circuit is part of the streaming contract
        evaluator = driver._e
        models.append(got)
    return models


@pytest.mark.parametrize("cls", [IndexedDatalog, IndexedDatalogWithNegation])
def test_positive_updates_share_driver(cls):
    seed = (("path", ("?X", "?Y")), ("edge", ("?X", "?Y")))
    trans = (("path", ("?X", "?Z")), ("edge", ("?X", "?Y")), ("path", ("?Y", "?Z")))
    models = drive(cls, [
        ({seed: 1, trans: 1}, {}),
        ({}, {("edge", (0, 1)): 1, ("edge", (1, 2)): 1, ("edge", (2, 3)): 1}),
        ({}, {}),
        ({}, {("edge", (2, 3)): -1}),
        ({trans: -1}, {}),
        ({trans: 1}, {("edge", (2, 3)): 1}),
    ])
    assert models[1] == models[-1]
    assert ("path", (0, 2)) not in models[4].inner


def test_late_blocker_and_multiple_negation_levels():
    rules = {
        (("s1", ("?X",)), ("source", ("?X",))): 1,
        (("s2", ("?X",)), ("s1", ("?X",))): 1,
        (("s3", ("?X",)), ("s2", ("?X",))): 1,
        (("blocked", ("?X",)), ("s3", ("?X",))): 1,
        (("b", ("?X",)), ("probe", ("?X",)), ("!blocked", ("?X",))): 1,
        (("a", ("?X",)), ("probe", ("?X",)), ("!b", ("?X",))): 1,
    }
    models = drive(IndexedDatalogWithNegation, [
        (rules, {("probe", (0,)): 1}),
        ({}, {("source", (0,)): 1}),
        ({}, {}),
        ({}, {("source", (0,)): -1}),
        ({}, {("source", (0,)): 1}),
    ])
    assert models[0] == models[3]
    assert models[1] == models[2] == models[4]
    assert ("a", (0,)) in models[1].inner


def test_rule_updates_change_negation_dependencies():
    blocker = (("mark", ("?X",)), ("special", ("?X",)))
    rules = {
        (("b", ("?X",)), ("probe", ("?X",)), ("!mark", ("?X",))): 1,
        (("a", ("?X",)), ("probe", ("?X",)), ("!b", ("?X",))): 1,
    }
    models = drive(IndexedDatalogWithNegation, [
        (rules, {("probe", (0,)): 1, ("special", (0,)): 1}),
        ({blocker: 1}, {}),
        ({blocker: -1}, {}),
    ])
    assert models[0] == models[2]
    assert ("a", (0,)) in models[1].inner


@pytest.mark.parametrize("cls", [IndexedDatalog, IndexedDatalogWithNegation])
def test_circular_support_retracts(cls):
    rules = {
        (("p", ("?X",)), ("e", ("?X",))): 1,
        (("q", ("?X",)), ("p", ("?X",))): 1,
        (("p", ("?X",)), ("q", ("?X",))): 1,
    }
    models = drive(cls, [(rules, {("e", (0,)): 1}), ({}, {("e", (0,)): -1}), ({}, {})])
    assert models[1] == models[2] == ZSet({})


@pytest.mark.parametrize("cls", [IndexedDatalog, IndexedDatalogWithNegation])
def test_limit_raises_instead_of_returning_partial_model(cls):
    driver = cls(max_inner=1)
    driver.push_facts(ZSet({("e", (0,)): 1}))
    with pytest.raises(RuntimeError, match="did not converge"):
        driver.saturate()
    with pytest.raises(RuntimeError, match="previous batch"):
        driver.saturate()
