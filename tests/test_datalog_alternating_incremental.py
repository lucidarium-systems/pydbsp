"""IncrementalDatalogAlternating: the multi-batch (t1-incremental) driver of
the alternating-fixpoint 3-D circuit.

Every episode tick is TRIPLE-GATED: the incremental driver's model must equal
a fresh single-batch DatalogAlternating recompute on the cumulative input AND
a DatalogStratifiedComposition recompute. The cases pin the multi-batch shapes
that matter:
  - split arrival (rules at t0, triggering facts at t1),
  - UN-blocking (retract a fact -> a blocker un-derives -> negation-suppressed
    facts must RE-derive, including a positive cycle re-establishing),
  - A->B->A blocker flips (exact round-trip),
  - rule DELETION (negative-weight rule delta),
  - restratification via rule delta (the stratum structure changes across
    batches; no levels exist anywhere, so this needs zero special-casing),
  - pure positive growth and no-op ticks.
"""

from pydbsp.datalog_stratified import (
    DatalogAlternating,
    DatalogStratifiedComposition,
    IncrementalDatalogAlternating,
)
from pydbsp.zset import ZSet


def _merge(cum: dict, delta: dict) -> None:
    for k, w in delta.items():
        cum[k] = cum.get(k, 0) + w
        if cum[k] == 0:
            del cum[k]


def _drive(episodes: list[tuple[dict, dict]], **kwargs) -> list[set]:
    """Drive the incremental evaluator through (rule_delta, fact_delta)
    episodes; gate every tick against both recompute drivers."""
    inc = IncrementalDatalogAlternating(**kwargs)
    cum_rules: dict = {}
    cum_facts: dict = {}
    models: list[set] = []
    for t, (rule_delta, fact_delta) in enumerate(episodes):
        if rule_delta:
            inc.push_rules(ZSet(rule_delta))
        if fact_delta:
            inc.push_facts(ZSet(fact_delta))
        _merge(cum_rules, rule_delta)
        _merge(cum_facts, fact_delta)
        got = {a for a, w in inc.saturate().inner.items() if w > 0}
        for cls in (DatalogAlternating, DatalogStratifiedComposition):
            ref = cls()
            ref.push_rules(ZSet(cum_rules))
            ref.push_facts(ZSet(cum_facts))
            exp = {a for a, w in ref.saturate().inner.items() if w > 0}
            assert got == exp, (
                f"tick {t} vs {cls.__name__}: extra={sorted(got - exp)} missing={sorted(exp - got)}"
            )
        models.append(got)
    return models


GHOST_RULES = {
    (("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))): 1,
    (("reach", ("?X", "?Z")), ("edge", ("?X", "?Y")), ("reach", ("?Y", "?Z"))): 1,
    (("mark", ("?X",)), ("probe", ("?X",)), ("reach", ("?X", 3))): 1,
    (("b", ("?X",)), ("probe", ("?X",)), ("!mark", ("?X",))): 1,
    (("p", ("?X",)), ("b", ("?X",))): 1,
    (("q", ("?X",)), ("p", ("?X",))): 1,
    (("p", ("?X",)), ("q", ("?X",))): 1,
}
GHOST_EDB = {("edge", (0, 1)): 1, ("edge", (1, 2)): 1, ("edge", (2, 3)): 1, ("probe", (0,)): 1}


def test_ghost_split_arrival() -> None:
    # rules + edges first; the probe that triggers the negation chain arrives a batch later
    edges_only = {k: w for k, w in GHOST_EDB.items() if k[0] == "edge"}
    m0, m1 = _drive([(GHOST_RULES, edges_only), ({}, {("probe", (0,)): 1})])
    assert ("mark", (0,)) not in m0 and ("b", (0,)) not in m0
    assert ("mark", (0,)) in m1 and ("b", (0,)) not in m1 and ("p", (0,)) not in m1


def test_unblocking_recovers_suppressed_cycle() -> None:
    # retract edge(2,3): mark un-derives, b RE-derives, the p/q cycle re-establishes
    m0, m1 = _drive([(GHOST_RULES, GHOST_EDB), ({}, {("edge", (2, 3)): -1})])
    assert ("mark", (0,)) in m0 and ("b", (0,)) not in m0
    assert ("mark", (0,)) not in m1
    assert {("b", (0,)), ("p", (0,)), ("q", (0,))} <= m1


def test_blocker_flip_aba_roundtrip() -> None:
    rules = {
        (("mark", ("?X",)), ("special", ("?X",))): 1,
        (("b", ("?X",)), ("probe", ("?X",)), ("!mark", ("?X",))): 1,
        (("a", ("?X",)), ("probe", ("?X",)), ("!b", ("?X",))): 1,
    }
    m0, m1, m2 = _drive([
        (rules, {("probe", (0,)): 1}),
        ({}, {("special", (0,)): 1}),
        ({}, {("special", (0,)): -1}),
    ])
    assert ("b", (0,)) in m0 and ("a", (0,)) not in m0
    assert ("b", (0,)) not in m1 and ("a", (0,)) in m1 and ("mark", (0,)) in m1
    assert m2 == m0  # exact round-trip


def test_rule_deletion() -> None:
    mark_rule = {(("mark", ("?X",)), ("probe", ("?X",))): 1}
    rules = {**mark_rule, (("b", ("?X",)), ("probe", ("?X",)), ("!mark", ("?X",))): 1}
    m0, m1 = _drive([
        (rules, {("probe", (0,)): 1}),
        ({k: -1 for k in mark_rule}, {}),  # delete the mark rule
    ])
    assert ("mark", (0,)) in m0 and ("b", (0,)) not in m0
    assert ("mark", (0,)) not in m1 and ("b", (0,)) in m1


def test_restratification_via_rule_delta() -> None:
    # t0: p level 0, q level 1. t1's rule delta makes p negatively depend on a
    # NEW pred r, raising p to level 1 and q to level 2 — the stratification
    # changes shape across batches; the driver computes no levels, ever.
    t0_rules = {
        (("p", ("?X",)), ("e0", ("?X",))): 1,
        (("q", ("?X",)), ("probe", ("?X",)), ("!p", ("?X",))): 1,
    }
    t1_rules = {
        (("p", ("?X",)), ("probe", ("?X",)), ("!r", ("?X",))): 1,
        (("r", ("?X",)), ("e1", ("?X",))): 1,
    }
    m0, m1 = _drive([
        (t0_rules, {("e0", (0,)): 1, ("probe", (0,)): 1, ("probe", (1,)): 1}),
        (t1_rules, {("e1", (1,)): 1}),
    ])
    assert ("q", (1,)) in m0 and ("q", (0,)) not in m0
    # after: p(1) blocked by r(1), p(0) fires both ways -> q(1) stays, q(0) stays out
    assert ("p", (0,)) in m1 and ("p", (1,)) not in m1
    assert ("q", (1,)) in m1 and ("q", (0,)) not in m1


def test_positive_growth() -> None:
    rules = {
        (("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))): 1,
        (("reach", ("?X", "?Z")), ("edge", ("?X", "?Y")), ("reach", ("?Y", "?Z"))): 1,
    }
    m0, m1, m2 = _drive([
        (rules, {("edge", (0, 1)): 1}),
        ({}, {("edge", (1, 2)): 1}),
        ({}, {("edge", (2, 3)): 1}),
    ])
    assert ("reach", (0, 1)) in m0 and ("reach", (0, 2)) not in m0
    assert ("reach", (0, 2)) in m1
    assert {("reach", (0, 3)), ("reach", (1, 3)), ("reach", (2, 3))} <= m2


def test_noop_tick_is_stable() -> None:
    rules = {(("b", ("?X",)), ("probe", ("?X",)), ("!mark", ("?X",))): 1}
    m0, m1 = _drive([(rules, {("probe", (0,)): 1}), ({}, {})])
    assert m0 == m1


def test_compact_equivalence() -> None:
    # GC per tick must not change any model
    episodes = [
        (GHOST_RULES, GHOST_EDB),
        ({}, {("edge", (2, 3)): -1}),
        ({}, {("edge", (2, 3)): 1}),
    ]
    assert _drive(episodes) == _drive(episodes, compact_each_tick=True)
