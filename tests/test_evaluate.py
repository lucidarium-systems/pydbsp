"""Tests for ``pydbsp.evaluate`` — the steppable view over
(Circuit, Storage)."""

from __future__ import annotations

from typing import cast

import pytest

from pydbsp.circuit import Circuit
from pydbsp.compute import ComputeCtx
from pydbsp.evaluate import Evaluator
from pydbsp import progress as prg
from pydbsp.operator import (
    Delay,
    Differentiate,
    Input,
    Integrate,
    Lift1,
    Lift2,
    LiftIntegrate,
)
from pydbsp.progress import Time
from pydbsp.storage import DictStorage
from pydbsp.core import AbelianGroupOperation, Antichain, dbsp_time


# ---- Test fixtures ---------------------------------------------------------


class IntGroup(AbelianGroupOperation[int]):
    def add(self, a: int, b: int) -> int:
        return a + b

    def neg(self, a: int) -> int:
        return -a

    def identity(self) -> int:
        return 0


INT_GROUP = IntGroup()


def fresh() -> Evaluator[Time]:
    return Evaluator[Time](
        circuit=Circuit[Time](),
        storage=DictStorage(),
        ctx=ComputeCtx(lattice=dbsp_time(1)),
        group=INT_GROUP,
    )


# ============================================================================
# Push / read on inputs
# ============================================================================


def test_push_then_read_returns_value() -> None:
    e = fresh()
    x = Input[int](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    e.push(x, 42)
    assert e.read(x, (0,)) == 42


def test_read_unpushed_input_returns_group_identity() -> None:
    e = fresh()
    x = Input[int](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    assert e.read(x, (5,)) == 0  # group.identity()


def test_pushes_advance_frontier_sequentially() -> None:
    e = fresh()
    x = Input[int](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    e.push(x, 10)
    e.push(x, 20)
    e.push(x, 30)
    assert e.read(x, (0,)) == 10
    assert e.read(x, (1,)) == 20
    assert e.read(x, (2,)) == 30


def test_push_rejects_non_input() -> None:
    e = fresh()
    x = Input[int](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    y = Lift1[int, int](f=lambda v: v * 2).connect(e.circuit, (x,))
    with pytest.raises(ValueError, match="not an input"):
        e.push(y, 99)


# ============================================================================
# Read drives compute through derived nodes
# ============================================================================


def test_read_lift1_invokes_compute() -> None:
    e = fresh()
    x = Input[int](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    y = Lift1[int, int](f=lambda v: v * 2).connect(e.circuit, (x,))
    e.push(x, 5)
    assert e.read(y, (0,)) == 10


def test_read_lift2_combines_two_inputs() -> None:
    e = fresh()
    a = Input[int](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    b = Input[int](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    z = Lift2[int, int, int](op=INT_GROUP.add).connect(e.circuit, (a, b))
    e.push(a, 7)
    e.push(b, 35)
    assert e.read(z, (0,)) == 42


def test_read_delay_returns_predecessor() -> None:
    e = fresh()
    x = Input[int](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    d = Delay[int](group=INT_GROUP).connect(e.circuit, (x,))
    e.push(x, 10)
    e.push(x, 20)
    # Delay at (1,) reads x at (0,) = 10.
    assert e.read(d, (1,)) == 10
    # Delay at (0,) reads at predecessor (none) → identity.
    assert e.read(d, (0,)) == 0


# ============================================================================
# Feedback recurrence: Integrate, Differentiate
# ============================================================================


def test_integrate_running_sum() -> None:
    e = fresh()
    x = Input[int](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    s = Integrate[int](group=INT_GROUP).connect(e.circuit, (x,))
    for v in [1, 2, 3, 4]:
        e.push(x, v)
    assert e.read(s, (0,)) == 1
    assert e.read(s, (1,)) == 3
    assert e.read(s, (2,)) == 6
    assert e.read(s, (3,)) == 10


def test_differentiate_recovers_pushed_deltas() -> None:
    """∂ ∘ ∫ = id: differentiating a running sum recovers the
    original input deltas."""
    e = fresh()
    x = Input[int](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    s = Integrate[int](group=INT_GROUP).connect(e.circuit, (x,))
    d = Differentiate[int](group=INT_GROUP).connect(e.circuit, (s,))
    for v in [1, 2, 3, 4]:
        e.push(x, v)
    assert e.read(d, (0,)) == 1
    assert e.read(d, (1,)) == 2
    assert e.read(d, (2,)) == 3
    assert e.read(d, (3,)) == 4


# ============================================================================
# Storage as cache for derived compute results
# ============================================================================


def test_storage_caches_derived_compute_results() -> None:
    """Storage doubles as the memo for derived nodes: the first
    ``read(derived, t)`` invokes ``compute`` and writes the result
    back into storage; the second ``read`` finds it there and skips
    the compute call. No separate memo dict — it's all one storage."""
    e = fresh()
    x = Input[int](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    y = Lift1[int, int](f=lambda v: v * 2).connect(e.circuit, (x,))
    e.push(x, 5)
    before = e.storage.size()
    e.read(y, (0,))
    after_first = e.storage.size()
    e.read(y, (0,))
    after_second = e.storage.size()
    assert after_first == before + 1  # x is already stored from push; y added.
    assert after_second == after_first  # cache hit, no new write.  # cache hit, no new entry.


# ============================================================================
# Frontiers + compact
# ============================================================================


def test_frontiers_reflect_pushed_state() -> None:
    e = fresh()
    x = Input[int](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    s = Integrate[int](group=INT_GROUP).connect(e.circuit, (x,))
    e.push(x, 1)
    e.push(x, 2)
    e.push(x, 3)
    fs = e.frontiers()
    assert fs[x].elements == [(2,)]  # frontier is the max-pushed
    assert fs[s].elements == [(2,)]  # strict-feedback theorem


def test_compact_evicts_dominated_entries() -> None:
    """After compact, storage for the input drops entries the
    Integrate has consumed, and Integrate keeps only its retained
    slot (retreat)."""
    e = fresh()
    x = Input[int](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    s = Integrate[int](group=INT_GROUP).connect(e.circuit, (x,))
    for v in [1, 2, 3, 4]:
        e.push(x, v)
    # Compute all integrate values so they're memoised.
    for t in range(4):
        e.read(s, (t,))
    # Storage: x has 4 entries, s has 4 entries → 8 total.
    assert e.storage.size() == 8

    # Compact at cursor (3,) on the Integrate.
    cursor: Antichain[Time] = Antichain(dbsp_time(1), cast(list[Time], [(3,)]))
    e.compact(cursors={s: cursor})

    # v2.1.2: the dead sits strictly below what at-or-after reads can
    # recompute through. x keeps (3,) (the delta a recompute of s at
    # (3,) reads), and s keeps the boundary memo (2,) plus the cursor
    # cell (3,): the snapshot-plus-deltas optimum.
    assert e.storage.times(x) == [(3,)]
    assert e.storage.times(s) == [(2,), (3,)]


def test_compact_returns_eviction_count() -> None:
    e = fresh()
    x = Input[int](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    s = Integrate[int](group=INT_GROUP).connect(e.circuit, (x,))
    for v in [10, 20, 30]:
        e.push(x, v)
    for t in range(3):
        e.read(s, (t,))
    cursor: Antichain[Time] = Antichain(dbsp_time(1), cast(list[Time], [(2,)]))
    n = e.compact(cursors={s: cursor})
    # x: dead = {(1,)} → entries (0,), (1,) → 2 evicted; (2,) stays for
    # boundary recomputes. s: dead = {(0,)} → 1 evicted.
    assert n == 3


def test_compact_no_args_uses_frontiers_1d() -> None:
    """``compact()`` with no cursors uses each node's propagated
    frontier — the monotone-forward default. After pushing 3 inputs and
    reading the Integrate at every tick, no-arg compact gives the same
    eviction as passing the frontier explicitly."""
    e = fresh()
    x = Input[int](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    s = Integrate[int](group=INT_GROUP).connect(e.circuit, (x,))
    for v in [10, 20, 30]:
        e.push(x, v)
    for t in range(3):
        e.read(s, (t,))
    assert e.storage.size() == 6  # 3 input + 3 integrate

    n = e.compact()  # no cursors

    # Input frontier {(2,)} → dead {(1,)} → cells (0,), (1,) evicted,
    # (2,) survives for boundary recomputes. Integrate frontier {(2,)}
    # → dead {(0,)} → cell (0,) evicted, boundary memo (1,) and the
    # frontier cell (2,) survive.
    assert e.storage.times(x) == [(2,)]
    assert e.storage.times(s) == [(1,), (2,)]
    assert n == 3


def test_compact_no_args_carries_omega_through_lift_intro() -> None:
    """The kafi pattern: a 1-D Input lifted to 2-D via
    ``LiftStreamIntroduction``. The lift's progress shape
    (``AxisIntroduction``) plants ``OMEGA`` on the introduced axis in
    its forward rule, so downstream 2-D nodes' frontiers naturally
    carry the inner-axis OMEGA. No-arg compact picks this up — no
    OMEGA literals in user code."""
    from pydbsp.operator import LiftStreamIntroduction

    e = Evaluator[Time](
        circuit=Circuit[Time](),
        storage=DictStorage(),
        ctx=ComputeCtx(lattice=dbsp_time(2)),
        group=INT_GROUP,
    )
    x = Input[int](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    x_2d = LiftStreamIntroduction[int](group=INT_GROUP).connect(e.circuit, (x,))
    cum_2d = Integrate[int](group=INT_GROUP).connect(e.circuit, (x_2d,))

    for v in [10, 20, 30]:
        e.push(x, v)
        # Read cum at (tick, 0) — natural monotone-forward read.
    for t in range(3):
        assert e.read(cum_2d, (t, 0)) == sum([10, 20, 30][: t + 1])

    # x_2d's forward frontier = (max_pushed_outer, OMEGA) — the lift
    # planted OMEGA on inner. cum_2d's frontier inherits it via
    # Feedback's forward (= input frontier).
    fr = e.frontiers()
    from pydbsp.core import OMEGA

    assert fr[x_2d].elements == [(2, OMEGA)]
    assert fr[cum_2d].elements == [(2, OMEGA)]

    e.compact()
    # Without OMEGA on inner the inner-axis machinery would leak (the
    # well-known meet-poisoning via retreat-to-empty). With OMEGA the
    # 2-D Integrate column collapses to the boundary slice at outer 1
    # plus the frontier cell, and the 1-D input keeps its last delta.
    assert e.storage.times(cum_2d) == [(1, 0), (2, 0)]
    assert e.storage.times(x) == [(2,)]


def test_compact_no_args_preserves_correctness_after_eviction() -> None:
    """After no-arg compact, future reads still produce correct
    values — eviction is bounded by the strict-feedback theorem so
    every retained cell is reachable via recompute."""
    e = fresh()
    x = Input[int](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    s = Integrate[int](group=INT_GROUP).connect(e.circuit, (x,))
    for v in [1, 2, 3]:
        e.push(x, v)
    for t in range(3):
        e.read(s, (t,))

    e.compact()  # frontier-driven

    # Push one more value, read; the Integrate at (3,) must still be
    # 1+2+3+4 = 10 — needs s(2,) cached.
    e.push(x, 4)
    assert e.read(s, (3,)) == 10


def test_compact_explicit_cursors_still_work() -> None:
    """Backward-compat: passing explicit cursors disables the
    frontier-driven default and uses what the caller supplies."""
    e = fresh()
    x = Input[int](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    s = Integrate[int](group=INT_GROUP).connect(e.circuit, (x,))
    for v in [1, 2, 3, 4]:
        e.push(x, v)
    for t in range(4):
        e.read(s, (t,))

    # Explicit cursor at (1,) — more conservative than the frontier (3,).
    cursor: Antichain[Time] = Antichain(dbsp_time(1), cast(list[Time], [(1,)]))
    e.compact(cursors={s: cursor})

    # Cursor (1,): x dead = {(0,)} → only (0,) evicted. s dead:
    # dead_below({1}) = {(0,)} met with the self-retreat, which hits
    # the axis bottom → empty antichain → annihilates → s keeps all.
    assert e.storage.times(x) == [(1,), (2,), (3,)]
    assert e.storage.times(s) == [(0,), (1,), (2,), (3,)]


def test_compact_no_args_correct_across_multi_outer_streaming() -> None:
    """A 2-D Integrate fed by both a 1-D-then-lifted input AND a
    same-axis stateful read across multiple outer ticks. The 1-D-input
    path's frontier carries ``OMEGA`` on the introduced axis; the
    stateful path doesn't. The forward-propagated frontiers MEET at the
    join, settling on the (finite-inner) state side — which keeps
    enough cells for the next outer tick's recompute.

    Regression test: a hand-coded ``(outer, OMEGA)`` cursor over-evicts
    here (claims inner-axis fully settled across all outers) and breaks
    future recomputes; no-arg compact is sound because its cursors are
    derived from the actual frontiers."""
    from pydbsp.operator import LiftStreamIntroduction, Lift2

    e = Evaluator[Time](
        circuit=Circuit[Time](),
        storage=DictStorage(),
        ctx=ComputeCtx(lattice=dbsp_time(2)),
        group=INT_GROUP,
    )
    x = Input[int](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    state = Input[int](frontier=Antichain(dbsp_time(2))).connect(e.circuit, ())
    x_2d = LiftStreamIntroduction[int](group=INT_GROUP).connect(e.circuit, (x,))
    combined = Lift2[int, int, int](op=lambda a, b: a + b).connect(e.circuit, (x_2d, state))
    cum = Integrate[int](group=INT_GROUP).connect(e.circuit, (combined,))

    expected_cum = 0
    for outer_tick in range(5):
        e.push(x, outer_tick + 1)
        # Push 'state' at this outer tick's inner=0 — emulating the
        # outer-only streaming usage (no inner advance).
        e.push(state, 10, t=(outer_tick, 0))
        expected_cum += (outer_tick + 1) + 10
        assert e.read(cum, (outer_tick, 0)) == expected_cum
        e.compact()  # no-arg per tick

    # After 5 outer ticks the cumulative at (4, 0) is still correct.
    assert e.read(cum, (4, 0)) == 1 + 2 + 3 + 4 + 5 + 10 * 5


# ============================================================================
# Evaluator.latest — read at the frontier top
# ============================================================================


def _fresh_2d() -> Evaluator[Time]:
    """2-D evaluator over ``int`` for the lifted ``latest`` tests."""
    return Evaluator[Time](
        circuit=Circuit[Time](),
        storage=DictStorage(),
        ctx=ComputeCtx(lattice=dbsp_time(2)),
        group=INT_GROUP,
    )


def test_latest_on_1d_input_returns_last_pushed_value() -> None:
    """Push ``1, 2, 3`` to a 1-D Input. ``latest(s)`` reads at the
    highest tick in the settled frontier — here ``(2,)`` — and
    returns the value pushed there."""
    e = fresh()
    s = Input[int](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    for v in [1, 2, 3]:
        e.push(s, v)
    assert e.latest(s) == 3


def test_latest_on_integrate_returns_running_sum_at_top() -> None:
    """Wire ``Integrate(s, axis=0)``; ``latest`` on the integrate
    returns the running sum at the highest tick — i.e., Σ over all
    pushed values."""
    e = fresh()
    s = Input[int](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    cum = Integrate[int](group=INT_GROUP).connect(e.circuit, (s,))
    for v in [1, 2, 3]:
        e.push(s, v)
    assert e.latest(cum) == 6


def test_latest_on_unpushed_input_returns_identity() -> None:
    """No pushes → empty frontier → ``latest`` returns
    ``group.identity()``."""
    e = fresh()
    s = Input[int](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    assert e.latest(s) == 0


def test_latest_2d_returns_freshest_known_value() -> None:
    """A 2-D source with multiple non-trivial inner ticks at outer = 0,
    populated directly so the frontier is bounded (finite, not ω).
    ``latest(cum)`` returns the running sum at the lattice-maximal
    frontier element — here ``(0, 2)`` — giving ``1 + 2 + 3 = 6``."""
    e = _fresh_2d()
    s = Input[int](frontier=Antichain(dbsp_time(2))).connect(e.circuit, ())

    e.storage.write(s, (0, 0), 1)
    e.storage.write(s, (0, 1), 2)
    e.storage.write(s, (0, 2), 3)
    cast(prg.Input[Time], e.circuit.progress_rules[s]).frontier.insert(cast(Time, (0, 2)))

    cum = LiftIntegrate[int](group=INT_GROUP).connect(e.circuit, (s,))

    assert e.latest(cum) == 6
