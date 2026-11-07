"""Streaming regressions for per-stratum initialization and convergence."""

import pytest

from pydbsp.datalog_stratified import (
    IncrementalDatalogStratified,
    IndexedIncrementalDatalogStratified,
)
from pydbsp.zset import ZSet

DRIVERS = [IncrementalDatalogStratified, IndexedIncrementalDatalogStratified]
BASE = (("a", ("?X",)), ("e", ("?X",)))
NEG = (("b", ("?X",)), ("n", ("?X",)), ("!a", ("?X",)))
NEXT = (("c", ("?X",)), ("n", ("?X",)), ("!b", ("?X",)))
FACTS = {("e", (1,)): 1, ("n", (1,)): 1, ("n", (2,)): 1}


@pytest.mark.parametrize("driver", DRIVERS)
def test_new_strata_after_initial_batch_and_restoration(driver):
    d = driver(max_inner=32)
    d.push_rules(ZSet({BASE: 1}))
    d.push_facts(ZSet(FACTS))
    base = FACTS | {("a", (1,)): 1}
    assert d.saturate().inner == base
    d.push_rules(ZSet({NEG: 1}))
    middle = base | {("b", (2,)): 1}
    assert d.saturate().inner == middle
    d.push_rules(ZSet({NEXT: 1}))
    full = middle | {("c", (1,)): 1}
    assert d.saturate().inner == full
    assert d.stage(0).inner == base
    assert d.stage(1).inner == middle
    assert d.stage(2).inner == full
    # Shrink the active stratum range, change its input, then restore it.
    d.push_rules(ZSet({NEG: -1, NEXT: -1}))
    assert d.saturate().inner == base
    d.push_facts(ZSet({("e", (1,)): -1}))
    reduced = {("n", (1,)): 1, ("n", (2,)): 1}
    assert d.saturate().inner == reduced
    d.push_rules(ZSet({NEG: 1, NEXT: 1}))
    assert d.saturate().inner == reduced | {("b", (1,)): 1, ("b", (2,)): 1}


@pytest.mark.parametrize("driver", DRIVERS)
@pytest.mark.parametrize("populated", [False, True])
def test_empty_batches_do_not_exhaust_inner_limit(driver, populated):
    d = driver(max_inner=16)
    if populated:
        d.push_rules(ZSet({BASE: 1, NEG: 1, NEXT: 1}))
        d.push_facts(ZSet(FACTS))
    expected = d.saturate().inner.copy()
    for _ in range(40):
        assert d.saturate().inner == expected
    # The driver must still handle a real change after the empty batches.
    d.push_facts(ZSet({("e", (2,)): 1}))
    if populated:
        expected = FACTS | {("e", (2,)): 1, ("a", (1,)): 1,
                            ("a", (2,)): 1, ("c", (1,)): 1, ("c", (2,)): 1}
    else:
        expected = {("e", (2,)): 1}
    assert d.saturate().inner == expected
