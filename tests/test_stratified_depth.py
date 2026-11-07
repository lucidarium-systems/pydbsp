"""Stratification deeper than two levels.

``IncrementalDatalogStratified`` walks stratum ``s + 1`` only when it reads a
non-empty ``level_next`` inside stratum ``s``. Because the ``stratum_ready``
guards share one inner axis, stratum ``s`` starts at the depth where ``s - 1``
finished, while that announcement lands at the low inner ticks below it. The
driver therefore used to stop after stratum 1 and drop every higher stratum
silently. These programs all need three or more strata.
"""

import pytest

from pydbsp.datalog_stratified import IncrementalDatalogStratified
from pydbsp.zset import ZSet


def derive(rules, edb):
    e = IncrementalDatalogStratified()
    e.push_rules(ZSet({r: 1 for r in rules}))
    e.push_facts(ZSet({f: 1 for f in edb}))
    return {f for f, w in e.saturate().inner.items() if w > 0}


def test_three_level_chain():
    rules = [
        (("a", ("?X",)), ("e", ("?X",))),
        (("b", ("?X",)), ("n", ("?X",)), ("!a", ("?X",))),
        (("c", ("?X",)), ("n", ("?X",)), ("!b", ("?X",))),
    ]
    edb = [("e", (1,)), ("n", (1,)), ("n", (2,))]
    assert derive(rules, edb) == set(edb) | {("a", (1,)), ("b", (2,)), ("c", (1,))}


def test_mixed_fire_and_block_across_three_strata():
    rules = [
        (("mark", ("?X",)), ("probe", ("?X",)), ("src", ("?X",))),
        (("b", ("?X",)), ("probe", ("?X",)), ("!mark", ("?X",))),
        (("a", ("?X",)), ("probe", ("?X",)), ("!b", ("?X",))),
    ]
    edb = [("probe", (0,)), ("probe", (1,)), ("src", (1,))]
    assert derive(rules, edb) == set(edb) | {("mark", (1,)), ("b", (0,)), ("a", (1,))}


def test_derived_blockage_over_recursion():
    """The reach / blocked / output program: recursion at stratum 0,
    derived blockage at 1, negated again at 2."""
    rules = [
        (("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))),
        (("reach", ("?X", "?Z")), ("edge", ("?X", "?Y")), ("reach", ("?Y", "?Z"))),
        (("clear", ("?Y",)), ("mk", ("?Y",))),
        (("blocked", ("?Y",)), ("node", ("?Y",)), ("!clear", ("?Y",))),
        (("output", ("?X", "?Y")), ("reach", ("?X", "?Y")), ("!blocked", ("?Y",))),
    ]
    edb = [("edge", (1, 2)), ("edge", (2, 3)), ("edge", (3, 4)),
           ("node", (2,)), ("node", (3,)), ("node", (4,)), ("mk", (4,))]
    got = derive(rules, edb)
    assert {f[1] for f in got if f[0] == "output"} == {(1, 4), (2, 4), (3, 4)}
    assert {f[1] for f in got if f[0] == "blocked"} == {(2,), (3,)}


@pytest.mark.parametrize("depth", [3, 4, 5, 6])
def test_negation_chain_of_depth(depth):
    rules = [(("p0", ("?X",)), ("e", ("?X",)))]
    for i in range(1, depth):
        rules.append(((f"p{i}", ("?X",)), ("n", ("?X",)), (f"!p{i - 1}", ("?X",))))
    edb = [("e", (1,))] + [("n", (c,)) for c in range(3)]
    got = derive(rules, edb)
    # p_i holds exactly where p_{i-1} does not, so every level is non-empty
    for i in range(depth):
        assert any(f[0] == f"p{i}" for f in got), f"stratum {i} produced nothing"
