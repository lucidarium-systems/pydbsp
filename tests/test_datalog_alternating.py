"""DatalogAlternating: the alternating-fixpoint stratified evaluator as ONE
physical 3-D circuit. Expected models below are Soufflé-adjudicated (see
tools/oracle/souffle_oracle.py); every case also cross-checks against
DatalogStratifiedComposition, the per-stratum reference driver.

The cases pin exactly the shapes that killed earlier designs:
  - depth-2/depth-3 negation chains (guard-based 3-D driver: wrong),
  - ghost cycle, late blocker + positive cycle downstream (2-D drivers:
    ghosts survive; here the retraction must flow through the alternation
    feedback as negative weights and unwind the cycle),
  - late blocker over recursion (the freeze: negation reads z⁻¹ on the
    alternation axis, never in-flight facts),
  - positive-only program (alternation must quiesce after one extra tick).
"""

from pydbsp.datalog_stratified import DatalogAlternating, DatalogStratifiedComposition
from pydbsp.zset import ZSet


def _run(cls, rules: ZSet, facts: ZSet) -> set:
    s = cls()
    s.push_rules(rules)
    s.push_facts(facts)
    return {a for a, w in s.saturate().inner.items() if w > 0}


def _check(rules: ZSet, facts: ZSet, expected: set) -> None:
    got = _run(DatalogAlternating, rules, facts)
    assert got == expected, f"vs expected: extra={got - expected} missing={expected - got}"
    ref = _run(DatalogStratifiedComposition, rules, facts)
    assert got == ref, f"vs composition: extra={got - ref} missing={ref - got}"


def test_depth_2_chain() -> None:
    # mark fires -> b blocked -> a fires
    rules = ZSet({
        (("mark", ("?X",)), ("probe", ("?X",))): 1,
        (("b", ("?X",)), ("probe", ("?X",)), ("!mark", ("?X",))): 1,
        (("a", ("?X",)), ("probe", ("?X",)), ("!b", ("?X",))): 1,
    })
    facts = ZSet({("probe", (0,)): 1})
    _check(rules, facts, {("probe", (0,)), ("mark", (0,)), ("a", (0,))})


def test_depth_3_chain() -> None:
    # mark -> b blocked -> a fires -> c blocked
    rules = ZSet({
        (("mark", ("?X",)), ("probe", ("?X",))): 1,
        (("b", ("?X",)), ("probe", ("?X",)), ("!mark", ("?X",))): 1,
        (("a", ("?X",)), ("probe", ("?X",)), ("!b", ("?X",))): 1,
        (("c", ("?X",)), ("probe", ("?X",)), ("!a", ("?X",))): 1,
    })
    facts = ZSet({("probe", (0,)): 1})
    _check(rules, facts, {("probe", (0,)), ("mark", (0,)), ("a", (0,))})


def test_depth_2_mixed_per_constant() -> None:
    # per-constant fire/block: 0 marked (b blocked, a fires), 1 unmarked (b fires, a blocked)
    rules = ZSet({
        (("mark", ("?X",)), ("special", ("?X",))): 1,
        (("b", ("?X",)), ("probe", ("?X",)), ("!mark", ("?X",))): 1,
        (("a", ("?X",)), ("probe", ("?X",)), ("!b", ("?X",))): 1,
    })
    facts = ZSet({("probe", (0,)): 1, ("probe", (1,)): 1, ("special", (0,)): 1})
    _check(rules, facts, {
        ("probe", (0,)), ("probe", (1,)), ("special", (0,)),
        ("mark", (0,)), ("a", (0,)), ("b", (1,)),
    })


def test_ghost_cycle_unwound() -> None:
    # mark(0) arrives late (needs reach(0,3), 3 inner steps); the transient
    # ghost b(0) has already fed the p/q positive cycle. The alternation must
    # retract b AND unwind the cycle with negative weights.
    rules = ZSet({
        (("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))): 1,
        (("reach", ("?X", "?Z")), ("edge", ("?X", "?Y")), ("reach", ("?Y", "?Z"))): 1,
        (("mark", ("?X",)), ("probe", ("?X",)), ("reach", ("?X", 3))): 1,
        (("b", ("?X",)), ("probe", ("?X",)), ("!mark", ("?X",))): 1,
        (("p", ("?X",)), ("b", ("?X",))): 1,
        (("q", ("?X",)), ("p", ("?X",))): 1,
        (("p", ("?X",)), ("q", ("?X",))): 1,
    })
    facts = ZSet({("edge", (0, 1)): 1, ("edge", (1, 2)): 1, ("edge", (2, 3)): 1, ("probe", (0,)): 1})
    _check(rules, facts, {
        ("edge", (0, 1)), ("edge", (1, 2)), ("edge", (2, 3)), ("probe", (0,)),
        ("reach", (0, 1)), ("reach", (0, 2)), ("reach", (0, 3)),
        ("reach", (1, 2)), ("reach", (1, 3)), ("reach", (2, 3)),
        ("mark", (0,)),
    })


def test_depth_2_over_recursion() -> None:
    # late mark over recursion, then a second negation level on top
    rules = ZSet({
        (("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))): 1,
        (("reach", ("?X", "?Z")), ("edge", ("?X", "?Y")), ("reach", ("?Y", "?Z"))): 1,
        (("mark", ("?X",)), ("probe", ("?X",)), ("reach", ("?X", 3))): 1,
        (("b", ("?X",)), ("probe", ("?X",)), ("!mark", ("?X",))): 1,
        (("a", ("?X",)), ("probe", ("?X",)), ("!b", ("?X",))): 1,
    })
    facts = ZSet({("edge", (0, 1)): 1, ("edge", (1, 2)): 1, ("edge", (2, 3)): 1, ("probe", (0,)): 1})
    _check(rules, facts, {
        ("edge", (0, 1)), ("edge", (1, 2)), ("edge", (2, 3)), ("probe", (0,)),
        ("reach", (0, 1)), ("reach", (0, 2)), ("reach", (0, 3)),
        ("reach", (1, 2)), ("reach", (1, 3)), ("reach", (2, 3)),
        ("mark", (0,)), ("a", (0,)),
    })


def test_positive_only_quiesces() -> None:
    rules = ZSet({
        (("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))): 1,
        (("reach", ("?X", "?Z")), ("edge", ("?X", "?Y")), ("reach", ("?Y", "?Z"))): 1,
    })
    facts = ZSet({("edge", (0, 1)): 1, ("edge", (1, 2)): 1})
    _check(rules, facts, {
        ("edge", (0, 1)), ("edge", (1, 2)),
        ("reach", (0, 1)), ("reach", (1, 2)), ("reach", (0, 2)),
    })


def test_empty_program() -> None:
    _check(ZSet({}), ZSet({("e", (0,)): 1}), {("e", (0,))})
