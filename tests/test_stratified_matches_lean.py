"""``IncrementalDatalogStratified`` against V3/StratifiedCircuit.lean.

The Lean circuit is

    stratSpec = integral ; prepInject ; feedback2 stratumBody

with ``placedRules S k`` the rules of level ``k`` at stratum cell ``k`` (the
keyed introduction) and ``eCell = delta0 E [k] + z^-1 cand [k]`` (the stratum
delay, axis 1). Its content theorems are

    stratSpec_computes       cell (t, k) = stratifiedModel S E k
    stratSpec_computes_full  cell (t, k) = stratifiedModel S E maxLevel, k >= maxLevel
    incStratCircuit_computes the same after integrating the incremental form

``cand`` below is that recurrence, and is the oracle for all of it. Two
properties separate "agrees with the Lean" from "IS the Lean circuit":

* every stratum cell must expose its own stage, not just the final answer
  (``stratSpec_computes`` is a statement about cell ``k``);
* the stratum lift must be POINTWISE -- three lifts, two incrementalizations
  -- so no ``stratum_ready`` control fact may exist, and nothing may
  integrate or delay along axis 1 except the stratum delay itself.
"""

import pytest

from tests import souffle_oracle

from pydbsp.datalog_stratified import (
    DatalogStratifiedComposition,
    IncrementalDatalogStratified,
)
from pydbsp.zset import ZSet


# --------------------------------------------------------------------------
# the Lean recurrence, as oracle
# --------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not souffle_oracle.available, reason="souffle unavailable (needs souffle or nix)"
)


def levels(rules):
    heads = {r[0][0] for r in rules}
    lvl = {h: 0 for h in heads}
    for _ in range(len(heads) + 2):
        changed = False
        for r in rules:
            h = r[0][0]
            for atom in r[1:]:
                p = atom[0].lstrip("!")
                if p not in lvl:
                    continue
                want = lvl[p] + (1 if atom[0].startswith("!") else 0)
                if want > lvl[h]:
                    lvl[h], changed = want, True
        if not changed:
            break
    return lvl


def cand(rules, edb, k):
    """``stratifiedModel S E k``, as Souffle computes it.

    NOT a transcription of the Lean. The Lean pins the spec but is
    ``noncomputable``, so the semantic oracle is an independent engine; stage
    ``k`` is the model of the sub-program of levels <= k, which is exactly the
    Lean's ``cand`` unfolded. ``levels`` above is the placement, and a wrong
    placement shows up here as a disagreement with Souffle rather than being
    assumed away.
    """
    lvl = levels(rules)
    sub = [r for r in rules if lvl[r[0][0]] <= k]
    return souffle_oracle.model(sub, edb)


def observed(zset):
    return souffle_oracle.as_souffle_shape(
        {f for f, w in zset.inner.items() if w > 0})


# --------------------------------------------------------------------------
# programs
# --------------------------------------------------------------------------
REACH_BLOCKED_OUTPUT = ([
    (("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))),
    (("reach", ("?X", "?Z")), ("edge", ("?X", "?Y")), ("reach", ("?Y", "?Z"))),
    (("clear", ("?Y",)), ("mk", ("?Y",))),
    (("blocked", ("?Y",)), ("node", ("?Y",)), ("!clear", ("?Y",))),
    (("output", ("?X", "?Y")), ("reach", ("?X", "?Y")), ("!blocked", ("?Y",))),
], [("edge", (1, 2)), ("edge", (2, 3)), ("edge", (3, 4)),
    ("node", (2,)), ("node", (3,)), ("node", (4,)), ("mk", (4,))])


def chain(depth):
    rules = [(("p0", ("?X",)), ("e", ("?X",)))]
    for i in range(1, depth):
        rules.append(((f"p{i}", ("?X",)), ("n", ("?X",)), (f"!p{i - 1}", ("?X",))))
    return rules, [("e", (1,))] + [("n", (c,)) for c in range(3)]


def drive(cls, rules, edb):
    d = cls()
    d.push_rules(ZSet({r: 1 for r in rules}))
    d.push_facts(ZSet({f: 1 for f in edb}))
    return observed(d.saturate())


# --------------------------------------------------------------------------
# semantics: it must compute the stratified model
# --------------------------------------------------------------------------
@pytest.mark.parametrize("depth", [2, 3, 4, 5, 6, 7])
def test_computes_stratified_model_at_every_depth(depth):
    rules, edb = chain(depth)
    assert drive(IncrementalDatalogStratified, rules, edb) == cand(rules, edb, depth)


def test_computes_stratified_model_on_reach_blocked_output():
    rules, edb = REACH_BLOCKED_OUTPUT
    assert drive(IncrementalDatalogStratified, rules, edb) == cand(rules, edb, 2)


def test_agrees_with_per_stratum_composition():
    """The composition exists to show N 2-axis circuits reproduce the 3-D one,
    so the two must not disagree."""
    for depth in range(2, 7):
        rules, edb = chain(depth)
        assert drive(IncrementalDatalogStratified, rules, edb) == drive(
            DatalogStratifiedComposition, rules, edb
        )


# --------------------------------------------------------------------------
# stratSpec_computes: cell k must BE the k-th stage
# --------------------------------------------------------------------------
def test_stage_at_cell_k_is_the_kth_stratified_stage():
    rules, edb = REACH_BLOCKED_OUTPUT
    d = IncrementalDatalogStratified()
    d.push_rules(ZSet({r: 1 for r in rules}))
    d.push_facts(ZSet({f: 1 for f in edb}))
    d.saturate()
    for k in range(3):
        got = observed(d.stage(k))
        assert got == cand(rules, edb, k), f"stratum cell {k} is not stage {k}"


def test_stages_stabilise_at_and_above_max_level():
    """stratSpec_computes_full: every cell at or above maxLevel repeats the
    full stratified model."""
    rules, edb = REACH_BLOCKED_OUTPUT
    d = IncrementalDatalogStratified()
    d.push_rules(ZSet({r: 1 for r in rules}))
    d.push_facts(ZSet({f: 1 for f in edb}))
    full = d.saturate()
    for k in (2, 3, 4):
        assert observed(d.stage(k)) == observed(full)


# --------------------------------------------------------------------------
# structure: the stratum lift is pointwise, the guards are gone
# --------------------------------------------------------------------------
def test_no_stratum_ready_control_facts_reach_the_circuit():
    """The Lean orders strata by WIRING. A control fact in the fact store is
    the guard design, not this one."""
    rules, edb = REACH_BLOCKED_OUTPUT
    d = IncrementalDatalogStratified()
    d.push_rules(ZSet({r: 1 for r in rules}))
    d.push_facts(ZSet({f: 1 for f in edb}))
    out = d.saturate()
    assert not any(f[0] == "stratum_ready" for f in out.inner)
    for k in range(3):
        assert not any(f[0] == "stratum_ready" for f in d.stage(k).inner)


def test_stratum_axis_carries_only_the_stratum_delay():
    """Pointwise lift: nothing integrates along axis 1 except the stratum
    delay, so the join is the 4-term form at inner_axis=2, not the 8-term."""
    from pydbsp import compute

    rules, edb = REACH_BLOCKED_OUTPUT
    d = IncrementalDatalogStratified()
    d.push_rules(ZSet({r: 1 for r in rules}))
    d.push_facts(ZSet({f: 1 for f in edb}))
    d.saturate()
    accumulating = tuple(
        c for name in ("Accumulate", "Prev", "Integrate", "Delay")
        if (c := getattr(compute, name, None)) is not None
    )
    axis1 = [r for r in d._e.circuit.compute_rules
             if getattr(r, "axis", None) == 1 and isinstance(r, accumulating)]
    assert not axis1, (
        f"axis 1 carries {len(axis1)} integrate/delay nodes; the stratum lift "
        f"must be pointwise")


# --------------------------------------------------------------------------
# incStratCircuit_computes: incremental across batches
# --------------------------------------------------------------------------
def test_incremental_across_batches_with_retraction():
    rules, edb = REACH_BLOCKED_OUTPUT
    d = IncrementalDatalogStratified()
    d.push_rules(ZSet({r: 1 for r in rules}))
    d.push_facts(ZSet({f: 1 for f in edb}))
    assert observed(d.saturate()) == cand(rules, edb, 2)
    d.push_facts(ZSet({("mk", (4,)): -1}))
    edb2 = [f for f in edb if f != ("mk", (4,))]
    assert observed(d.saturate()) == cand(rules, edb2, 2)
