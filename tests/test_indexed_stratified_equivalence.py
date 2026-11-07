"""IndexedIncrementalDatalogStratified ≡ IncrementalDatalogStratified.

The indexed driver swaps the non-indexed negation body for the sort-merge-indexed
one (`IndexedIncrementalDatalogWithNegationBody` at `inner_axis=2`, built on
`IndexedDeltaLiftedDeltaLiftedJoin`). It must produce BYTE-IDENTICAL results —
only the join cost changes. This battery runs positive recursion, negation, multi-level
stratified negation, and a realistic reaching-defs program through BOTH and asserts the
full saturated fact set is identical.
"""

from __future__ import annotations

import pytest

from pydbsp.datalog_stratified import (
    IncrementalDatalogStratified,
    IndexedDatalogWithNegation,
    IndexedIncrementalDatalogStratified,
)
from pydbsp.zset import ZSet


def _rules(*rs):
    return ZSet({r: 1 for r in rs})


def _facts(*fs):
    return ZSet({f: 1 for f in fs})


def _run(cls, program, edb):
    s = cls()
    s.push_rules(program)
    s.push_facts(edb)
    return {f: m for f, m in s.saturate().inner.items() if m != 0}


# ── program battery ──────────────────────────────────────────────────────────

_TC = (
    _rules(
        (("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))),
        (("reach", ("?X", "?Z")), ("edge", ("?X", "?Y")), ("reach", ("?Y", "?Z"))),
    ),
    _facts(("edge", (0, 1)), ("edge", (1, 2)), ("edge", (2, 3)), ("edge", (3, 0))),  # a cycle
)

_SAME_GEN = (
    _rules(
        (("sg", ("?X", "?X")), ("node", ("?X",))),
        (("sg", ("?X", "?Y")), ("par", ("?X", "?A")), ("sg", ("?A", "?B")), ("par", ("?Y", "?B"))),
    ),
    _facts(
        ("node", (1,)), ("node", (2,)), ("node", (3,)), ("node", (4,)),
        ("par", (1, 3)), ("par", (2, 3)), ("par", (3, 4)),
    ),
)

_NEG = (
    _rules(
        (("alive", ("?X",)), ("person", ("?X",)), ("!dead", ("?X",))),
        (("dead", ("?X",)), ("killed", ("?X",))),
    ),
    _facts(("person", (1,)), ("person", (2,)), ("person", (3,)), ("killed", (2,))),
)

_STRAT2 = (  # two negation strata: c depends on !b, b depends on !a
    _rules(
        (("b", ("?X",)), ("u", ("?X",)), ("!a", ("?X",))),
        (("c", ("?X",)), ("u", ("?X",)), ("!b", ("?X",))),
    ),
    _facts(("u", (1,)), ("u", (2,)), ("u", (3,)), ("a", (1,)), ("a", (3,))),
)

# reaching-defs (the tibia pcode use case): positive core + transparent complement
_REACH = (
    _rules(
        (("reach_out", ("?D", "?L", "?B")), ("def_", ("?D", "?L", "?B"))),
        (("reach_in", ("?D", "?L", "?B2")), ("reach_out", ("?D", "?L", "?B1")), ("cfg", ("?B1", "?B2"))),
        (("reach_out", ("?D", "?L", "?B")), ("reach_in", ("?D", "?L", "?B")), ("transparent", ("?L", "?B"))),
    ),
    _facts(
        ("def_", ("D0", "L", "B0")),
        ("cfg", ("B0", "B1")), ("cfg", ("B1", "B2")), ("cfg", ("B2", "B1")), ("cfg", ("B1", "B3")),
        ("transparent", ("L", "B1")), ("transparent", ("L", "B2")), ("transparent", ("L", "B3")),
    ),
)

_DIFF_NEG = (  # negation deriving a mismatch relation (stratified !)
    _rules(
        (("mismatch", ("missing", "?D")), ("expected", ("?D",)), ("!got", ("?D",))),
        (("mismatch", ("extra", "?D")), ("got", ("?D",)), ("!expected", ("?D",))),
        (("got", ("?D",)), ("raw", ("?D",))),
    ),
    _facts(("expected", ("a",)), ("expected", ("b",)), ("raw", ("b",)), ("raw", ("c",))),
)

_BATTERY = {
    "transitive_closure": _TC,
    "same_generation": _SAME_GEN,
    "negation": _NEG,
    "two_strata": _STRAT2,
    "reaching_defs": _REACH,
    "diff_negation": _DIFF_NEG,
}


@pytest.mark.parametrize("name", list(_BATTERY))
def test_indexed_equals_nonindexed(name):
    program, edb = _BATTERY[name]
    plain = _run(IncrementalDatalogStratified, program, edb)
    indexed = _run(IndexedIncrementalDatalogStratified, program, edb)
    assert indexed == plain, (
        name,
        sorted(set(plain) - set(indexed)),  # only in plain
        sorted(set(indexed) - set(plain)),  # only in indexed
    )


@pytest.mark.parametrize("name", list(_BATTERY))
def test_2d_negation_driver_equals_stratified(name):
    """The single-batch 2-D indexed-negation driver (no stratum axis) ≡ the 3-D
    stratified driver across the battery — its incremental negation RETRACTS when a
    negated predicate grows, so it converges to the stratified answer (incl. the
    genuine two-stratum program). ~2.4x faster on large dense EDBs."""
    program, edb = _BATTERY[name]
    plain = _run(IncrementalDatalogStratified, program, edb)
    two_d = _run(IndexedDatalogWithNegation, program, edb)
    assert two_d == plain, (
        name,
        sorted(set(plain) - set(two_d)),
        sorted(set(two_d) - set(plain)),
    )


def test_battery_is_nontrivial():
    # guard: each program must actually derive something (else the equality is vacuous)
    for name, (program, edb) in _BATTERY.items():
        assert _run(IncrementalDatalogStratified, program, edb), f"{name} derived nothing"
