"""The NON-indexed 2-D negation body
(:class:`pydbsp.relational_operators.IncrementalDatalogWithNegationBody`, driven by
:class:`pydbsp.datalog_stratified.DatalogWithNegation`) must compute the SAME minimal
model as the indexed 2-D body (driven by :class:`IndexedDatalogWithNegation`) on every
single-stratum negation program. It is a direct de-index of the indexed body: plain
``pred``-based joins, no ``LiftIndex`` / sort-merge index."""

from __future__ import annotations

import pytest

from pydbsp.datalog_stratified import DatalogWithNegation, IndexedDatalogWithNegation
from pydbsp.zset import ZSet

# Single-stratum negation programs (copied from tests/test_3d_datalog_negation.py).
PARITY_CASES: list[tuple[str, ZSet, ZSet]] = [
    (
        "anti-join-no-blocker",
        ZSet({(("alive", ("?X",)), ("person", ("?X",)), ("!dead", ("?X",))): 1}),
        ZSet({("person", ("alice",)): 1, ("person", ("bob",)): 1}),
    ),
    (
        "anti-join-with-blocker",
        ZSet({(("alive", ("?X",)), ("person", ("?X",)), ("!dead", ("?X",))): 1}),
        ZSet({("person", ("alice",)): 1, ("person", ("bob",)): 1, ("dead", ("bob",)): 1}),
    ),
    (
        "positive-recursion-then-anti-join",
        ZSet(
            {
                (("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))): 1,
                (("reach", ("?X", "?Z")), ("edge", ("?X", "?Y")), ("reach", ("?Y", "?Z"))): 1,
                (("only-reaches-self", ("?X",)), ("node", ("?X",)), ("!reach", ("?X", "?Y"))): 1,
            }
        ),
        ZSet(
            {
                ("edge", (0, 1)): 1,
                ("edge", (1, 2)): 1,
                ("node", (0,)): 1,
                ("node", (1,)): 1,
                ("node", (2,)): 1,
            }
        ),
    ),
    (
        "two-layer-anti-join",
        ZSet(
            {
                (("mid", ("?X",)), ("bot", ("?X",)), ("!excluded", ("?X",))): 1,
                (("top", ("?X",)), ("mid", ("?X",)), ("!blocked", ("?X",))): 1,
            }
        ),
        ZSet(
            {
                ("bot", (0,)): 1,
                ("bot", (1,)): 1,
                ("bot", (2,)): 1,
                ("excluded", (2,)): 1,
                ("blocked", (1,)): 1,
            }
        ),
    ),
    (
        "no-negation-at-all",
        ZSet(
            {
                (("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))): 1,
                (("reach", ("?X", "?Z")), ("edge", ("?X", "?Y")), ("reach", ("?Y", "?Z"))): 1,
            }
        ),
        ZSet({("edge", (0, 1)): 1, ("edge", (1, 2)): 1, ("edge", (2, 0)): 1}),
    ),
]


def _run(driver_cls, rules, facts):
    s = driver_cls()
    s.push_rules(rules)
    s.push_facts(facts)
    return s.saturate()


@pytest.mark.parametrize("name,rules,facts", PARITY_CASES, ids=[c[0] for c in PARITY_CASES])
def test_nonindexed_2d_negation_matches_indexed(name, rules: ZSet, facts: ZSet) -> None:
    got = _run(DatalogWithNegation, rules, facts)
    want = _run(IndexedDatalogWithNegation, rules, facts)
    assert got.inner == want.inner, f"{name}: non-indexed {got.inner} != indexed {want.inner}"
