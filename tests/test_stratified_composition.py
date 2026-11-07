"""Per-stratum composition driver (:class:`DatalogStratifiedComposition`).

Two obligations:

1. PARITY — on the single-stratum negation suite it must agree with the
   (independently validated) indexed 2-D driver.
2. MULTI-STRATUM CORRECTNESS — on negation depth >= 2 and on
   retraction-meets-cycle programs it must produce the stratified (perfect)
   model. Expected models below were verified against Soufflé; the
   guard-based :class:`IncrementalDatalogStratified` computes several of them
   WRONG (its cursor scheduling misses stratum-axis diffs below the shared
   inner cursor), so they are hardcoded rather than compared to a driver.
"""

from __future__ import annotations

import pytest

from pydbsp.datalog_stratified import (
    DatalogStratifiedComposition,
    IndexedDatalogWithNegation,
    stratum_levels,
)
from pydbsp.zset import ZSet

from tests.test_nonindexed_2d_negation import PARITY_CASES


def _run(driver_cls, rules: ZSet, facts: ZSet) -> ZSet:
    s = driver_cls()
    s.push_rules(rules)
    s.push_facts(facts)
    return s.saturate()


@pytest.mark.parametrize("name,rules,facts", PARITY_CASES, ids=[c[0] for c in PARITY_CASES])
def test_parity_with_indexed_driver(name, rules: ZSet, facts: ZSet) -> None:
    got = _run(DatalogStratifiedComposition, rules, facts)
    want = _run(IndexedDatalogWithNegation, rules, facts)
    assert got.inner == want.inner, f"{name}: {got.inner} != {want.inner}"


# --- multi-stratum cases (Soufflé-verified expected models) -----------------

REACH_RULES = {
    (("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))): 1,
    (("reach", ("?X", "?Z")), ("edge", ("?X", "?Y")), ("reach", ("?Y", "?Z"))): 1,
}
CHAIN_EDB = {("edge", (0, 1)): 1, ("edge", (1, 2)): 1, ("edge", (2, 3)): 1, ("probe", (0,)): 1}
REACH_CLOSURE = {("reach", p): 1 for p in [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]}

STRATIFIED_CASES = [
    (
        # IncrementalDatalogStratified: WRONG (loses a(0))
        "depth-2-chain",
        {
            (("mark", ("?X",)), ("probe", ("?X",))): 1,
            (("b", ("?X",)), ("probe", ("?X",)), ("!mark", ("?X",))): 1,
            (("a", ("?X",)), ("probe", ("?X",)), ("!b", ("?X",))): 1,
        },
        {("probe", (0,)): 1},
        {("probe", (0,)): 1, ("mark", (0,)): 1, ("a", (0,)): 1},
    ),
    (
        "depth-2-chain-mark-absent",
        {
            (("b", ("?X",)), ("probe", ("?X",)), ("!mark", ("?X",))): 1,
            (("a", ("?X",)), ("probe", ("?X",)), ("!b", ("?X",))): 1,
        },
        {("probe", (0,)): 1},
        {("probe", (0,)): 1, ("b", (0,)): 1},
    ),
    (
        # IncrementalDatalogStratified: WRONG (loses a(0))
        "depth-3-chain",
        {
            (("mark", ("?X",)), ("probe", ("?X",))): 1,
            (("b", ("?X",)), ("probe", ("?X",)), ("!mark", ("?X",))): 1,
            (("a", ("?X",)), ("probe", ("?X",)), ("!b", ("?X",))): 1,
            (("c", ("?X",)), ("probe", ("?X",)), ("!a", ("?X",))): 1,
        },
        {("probe", (0,)): 1},
        {("probe", (0,)): 1, ("mark", (0,)): 1, ("a", (0,)): 1},
    ),
    (
        # IncrementalDatalogStratified: WRONG (loses a(0) AND stratum-1 b(1))
        "depth-2-mixed-fire-and-block",
        {
            (("mark", ("?X",)), ("special", ("?X",))): 1,
            (("b", ("?X",)), ("probe", ("?X",)), ("!mark", ("?X",))): 1,
            (("a", ("?X",)), ("probe", ("?X",)), ("!b", ("?X",))): 1,
        },
        {("probe", (0,)): 1, ("probe", (1,)): 1, ("special", (0,)): 1},
        {
            ("probe", (0,)): 1, ("probe", (1,)): 1, ("special", (0,)): 1,
            ("mark", (0,)): 1, ("b", (1,)): 1, ("a", (0,)): 1,
        },
    ),
    (
        # 2-D drivers: WRONG (ghost p/q island survives b's retraction).
        "ghost-cycle-late-blocker",
        {
            **REACH_RULES,
            (("mark", ("?X",)), ("probe", ("?X",)), ("reach", ("?X", 3))): 1,
            (("b", ("?X",)), ("probe", ("?X",)), ("!mark", ("?X",))): 1,
            (("p", ("?X",)), ("b", ("?X",))): 1,
            (("q", ("?X",)), ("p", ("?X",))): 1,
            (("p", ("?X",)), ("q", ("?X",))): 1,
        },
        CHAIN_EDB,
        {**CHAIN_EDB, **REACH_CLOSURE, ("mark", (0,)): 1},
    ),
    (
        # IncrementalDatalogStratified: WRONG (loses a(0))
        "depth-2-over-recursion",
        {
            **REACH_RULES,
            (("mark", ("?X",)), ("probe", ("?X",)), ("reach", ("?X", 3))): 1,
            (("b", ("?X",)), ("probe", ("?X",)), ("!mark", ("?X",))): 1,
            (("a", ("?X",)), ("probe", ("?X",)), ("!b", ("?X",))): 1,
        },
        CHAIN_EDB,
        {**CHAIN_EDB, **REACH_CLOSURE, ("mark", (0,)): 1, ("a", (0,)): 1},
    ),
]


@pytest.mark.parametrize(
    "name,rules,facts,expected", STRATIFIED_CASES, ids=[c[0] for c in STRATIFIED_CASES]
)
def test_multi_stratum_perfect_model(name, rules, facts, expected) -> None:
    got = _run(DatalogStratifiedComposition, ZSet(rules), ZSet(facts))
    assert got.inner == expected, f"{name}: {got.inner} != {expected}"


def test_stratum_levels() -> None:
    lv = stratum_levels(ZSet(STRATIFIED_CASES[2][1]))  # depth-3-chain
    assert lv == {"mark": 0, "b": 1, "a": 2, "c": 3}
