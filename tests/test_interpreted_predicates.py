"""Interpreted (built-in) predicates — lazily-evaluated INFINITE relations.

`lt`/`le`/`gt`/`ge`/`neq` denote infinite relations (`{(a,b):a<b}`, …) that can never
be a finite-support ZSet of facts. They are evaluated as GUARDS on bindings already
produced by finite positive atoms (range-restriction / safety) and routed to a
stateless select in the body — the infinite relation is never materialized.

These prove: the guard fires and is order-sensitive; the SAME answer comes out of both
the 2-D indexed and the 3-D stratified driver; an interpreted guard composes with
negation (the `dominated`/min-keeps-base scheme is `lt` under a `!`); a custom guard can
be registered; and an UNSAFE rule (an interpreted arg unbound by a preceding positive
atom) is rejected at `push_rules`.
"""

from __future__ import annotations

import pytest

from pydbsp.datalog import Skolem, register_interpreted
from pydbsp.datalog_stratified import (
    DatalogStratifiedComposition,
    DatalogWithNegation,
    IncrementalDatalogStratified,
    IndexedDatalogWithNegation,
)
from pydbsp.zset import ZSet

DRIVERS = [
    IndexedDatalogWithNegation,
    DatalogWithNegation,
    IncrementalDatalogStratified,
    DatalogStratifiedComposition,
]


def _rules(*rs) -> ZSet:
    return ZSet({r: 1 for r in rs})


def _run(driver_cls, rules: ZSet, edb: dict) -> ZSet:
    s = driver_cls()
    s.push_rules(rules)
    s.push_facts(ZSet(edb))
    return s.saturate()


def _rel(out: ZSet, pred: str) -> set:
    return {a[1] for a in out.inner if a[0] == pred and out.inner[a] > 0}


# The dominated/min-keeps-base scheme: `lt` reproduces "lowest key keeps the bare name"
# (the imperative `_build_addr_name` choice) — `lt` under a `!`, the use that motivated this.
_ADDR_NAME_RULES = _rules(
    (("dominated", ("?A",)), ("fn", ("?A", "?N")), ("fn", ("?B", "?N")), ("lt", ("?B", "?A"))),
    (("addr_name", ("?A", "?N")), ("fn", ("?A", "?N")), ("!dominated", ("?A",))),
    (("addr_name", ("?A", Skolem("aname", ("?N", "?A")))), ("fn", ("?A", "?N")), ("dominated", ("?A",))),
)
_ADDR_NAME_EDB = {("fn", (0x100, "vector")): 1, ("fn", (0x200, "vector")): 1, ("fn", (0x300, "foo")): 1}


@pytest.mark.parametrize("driver", DRIVERS)
def test_lt_reproduces_min_keeps_bare_name(driver) -> None:
    out = _run(driver, _ADDR_NAME_RULES, _ADDR_NAME_EDB)
    names = {a[0]: a[1] for a in _rel(out, "addr_name")}
    assert names[0x100] == "vector"  # the min addr keeps the bare name
    assert isinstance(names[0x200], Skolem)  # the larger collider is suffixed (injective)
    assert names[0x300] == "foo"  # a unique name is untouched
    assert {a[0] for a in _rel(out, "dominated")} == {0x200}


@pytest.mark.parametrize("driver", DRIVERS)
def test_lt_is_order_sensitive(driver) -> None:
    rules = _rules((("below", ("?X",)), ("v", ("?X",)), ("v", ("?Y",)), ("lt", ("?Y", "?X"))))
    out = _run(driver, rules, {("v", (1,)): 1, ("v", (2,)): 1, ("v", (3,)): 1})
    assert {a[0] for a in _rel(out, "below")} == {2, 3}  # 1 has nothing below it


@pytest.mark.parametrize("driver", DRIVERS)
def test_neq_and_le_compose(driver) -> None:
    rules = _rules(
        (("pair", ("?X", "?Y")), ("v", ("?X",)), ("v", ("?Y",)), ("neq", ("?X", "?Y")), ("le", ("?X", "?Y")))
    )
    out = _run(driver, rules, {("v", (1,)): 1, ("v", (2,)): 1})
    assert _rel(out, "pair") == {(1, 2)}  # strict <: neq ∧ le


@pytest.mark.parametrize("driver", DRIVERS)
def test_interpreted_predicate_emits_no_relation(driver) -> None:
    """An interpreted atom is a guard, not a relation — it produces no output facts and
    creates no stratification dependency that would alter the rest of the program."""
    out = _run(driver, _ADDR_NAME_RULES, _ADDR_NAME_EDB)
    assert not _rel(out, "lt")  # `lt` never appears as a derived relation


def test_register_custom_interpreted_predicate() -> None:
    register_interpreted("divides", lambda a, b: b % a == 0)
    rules = _rules((("mult", ("?A", "?B")), ("v", ("?A",)), ("v", ("?B",)), ("divides", ("?A", "?B"))))
    out = _run(IndexedDatalogWithNegation, rules, {("v", (2,)): 1, ("v", (3,)): 1, ("v", (6,)): 1})
    assert _rel(out, "mult") == {(2, 2), (2, 6), (3, 3), (3, 6), (6, 6)}


@pytest.mark.parametrize("driver", DRIVERS)
def test_safety_rejects_unbound_interpreted_arg(driver) -> None:
    bad = _rules((("p", ("?A",)), ("fn", ("?A", "?N")), ("lt", ("?A", "?B"))))  # ?B never bound
    with pytest.raises(ValueError, match="unsafe interpreted atom"):
        driver().push_rules(bad)
