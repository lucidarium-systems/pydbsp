"""Tests for ``pydbsp.operator`` — each DBSP operator's
connection into a circuit, and the resulting progress structure."""

from __future__ import annotations

from typing import cast

import pytest

from pydbsp import compute as cpt
from pydbsp import progress as prg
from pydbsp.circuit import Circuit
from pydbsp.operator import (
    Delay,
    Differentiate,
    Input,
    Integrate,
    Lift1,
    Lift2,
    LiftStreamIntroduction,
)
from pydbsp.progress import (
    NodeId,
    Time,
    propagate_backward,
    propagate_forward,
)
from pydbsp.core import (
    AbelianGroupOperation,
    Antichain,
    Time1D,
    dbsp_time,
)


# ---- Test fixtures ---------------------------------------------------------


class IntGroup(AbelianGroupOperation[int]):
    def add(self, a: int, b: int) -> int:
        return a + b

    def neg(self, a: int) -> int:
        return -a

    def identity(self) -> int:
        return 0


INT_GROUP = IntGroup()


def fresh() -> Circuit[Time]:
    """A fresh broad-typed circuit — operators target ``Time = tuple[int,
    ...]`` so arity-changing shapes wire as ordinary nodes."""
    return Circuit[Time]()


# ============================================================================
# Primitive operators — single-node connection
# ============================================================================


def test_input_connects_as_one_input_node() -> None:
    b = fresh()
    f: Antichain[Time] = cast(Antichain[Time], Antichain(Time1D, [(5,)]))
    out = Input[int](frontier=f).connect(b, ())
    assert out == 0
    assert isinstance(b.progress_rules[out], prg.Input)
    assert isinstance(b.compute_rules[out], cpt.Get)


def test_input_rejects_inputs_argument() -> None:
    b = fresh()
    with pytest.raises(ValueError, match="no inputs"):
        Input[int](frontier=Antichain(dbsp_time(1))).connect(b, (0,))


def test_lift1_connects_as_identity_plus_map() -> None:
    b = fresh()
    x = Input[int](frontier=Antichain(dbsp_time(1))).connect(b, ())
    out = Lift1[int, int](f=lambda v: v * 2).connect(b, (x,))
    assert out == 1
    assert isinstance(b.progress_rules[out], prg.Identity)
    assert isinstance(b.compute_rules[out], cpt.Map)


def test_lift2_connects_as_meet_plus_zipwith() -> None:
    b = fresh()
    a = Input[int](frontier=Antichain(dbsp_time(1))).connect(b, ())
    c = Input[int](frontier=Antichain(dbsp_time(1))).connect(b, ())
    out = Lift2[int, int, int](op=lambda x, y: x + y).connect(b, (a, c))
    assert out == 2
    assert isinstance(b.progress_rules[out], prg.Meet)
    assert isinstance(b.compute_rules[out], cpt.ZipWith)


def test_delay_connects_as_axisshift_plus_prev() -> None:
    b = fresh()
    x = Input[int](frontier=Antichain(dbsp_time(1))).connect(b, ())
    out = Delay[int](group=INT_GROUP).connect(b, (x,))
    assert out == 1
    assert isinstance(b.progress_rules[out], prg.AxisShift)
    assert isinstance(b.compute_rules[out], cpt.Prev)


def test_integrate_connects_as_feedback_plus_sum() -> None:
    """Integrate's progress shape is Feedback with self_id = own id."""
    b = fresh()
    x = Input[int](frontier=Antichain(dbsp_time(1))).connect(b, ())
    out = Integrate[int](group=INT_GROUP).connect(b, (x,))
    assert out == 1
    fb = b.progress_rules[out]
    assert isinstance(fb, prg.Feedback)
    assert fb.self_id == out  # invariant verified by Circuit.add
    assert isinstance(b.compute_rules[out], cpt.Sum)


def test_time_axis_introduction_connects_as_axisintroduction_plus_constant() -> None:
    b = fresh()
    x = Input[int](frontier=Antichain(dbsp_time(1))).connect(b, ())
    out = LiftStreamIntroduction[int](group=INT_GROUP).connect(b, (x,))
    assert isinstance(b.progress_rules[out], prg.AxisIntroduction)
    assert isinstance(b.compute_rules[out], cpt.Constant)


# ============================================================================
# Composite — Differentiate connects as THREE primitive nodes
# ============================================================================


def test_differentiate_connects_as_three_nodes() -> None:
    """``Differentiate`` is composed from Delay + Lift1(neg) + Lift2(add).
    The circuit grows by exactly 3 from one Differentiate call."""
    b = fresh()
    x = Input[int](frontier=Antichain(dbsp_time(1))).connect(b, ())
    before = len(b.progress_rules)
    out = Differentiate[int](group=INT_GROUP).connect(b, (x,))
    after = len(b.progress_rules)
    assert after - before == 3
    # Output is the last node added — the Lift2(add).
    assert out == after - 1
    # Verify the shape stack: delay (AxisShift), then Lift1 (Identity),
    # then the Lift2 (Meet) which is the output.
    assert isinstance(b.progress_rules[out - 2], prg.AxisShift)
    assert isinstance(b.progress_rules[out - 1], prg.Identity)
    assert isinstance(b.progress_rules[out], prg.Meet)


# ============================================================================
# Forward / backward propagation on operator-built circuits
# ============================================================================


def test_lift1_propagation_forward_and_backward() -> None:
    """Lift1's progress is Identity — forward and backward both pass
    the cursor through unchanged."""
    b = fresh()
    F: Antichain[Time] = cast(Antichain[Time], Antichain(Time1D, [(5,)]))
    x = Input[int](frontier=F).connect(b, ())
    y = Lift1[int, int](f=lambda v: v * 2).connect(b, (x,))
    dag = b.progress_rules
    fwd = propagate_forward(dag)
    assert fwd[y].elements == [(5,)]
    deads = propagate_backward(dag, cursors={y: fwd[y]})
    # v2.1.2: the seed is dead_below of the cursor, passed through.
    assert deads[x].elements == [(4,)]


def test_lift2_progress_meet_at_fanin() -> None:
    b = fresh()
    F_big: Antichain[Time] = cast(Antichain[Time], Antichain(Time1D, [(5,)]))
    F_small: Antichain[Time] = cast(Antichain[Time], Antichain(Time1D, [(3,)]))
    a = Input[int](frontier=F_big).connect(b, ())
    c = Input[int](frontier=F_small).connect(b, ())
    out = Lift2[int, int, int](op=lambda x, y: x + y).connect(b, (a, c))
    fwd = propagate_forward(b.progress_rules)
    assert fwd[out].elements == [(3,)]  # the slower input gates


def test_delay_progress_shifts_and_seeds() -> None:
    """Delay (AxisShift) seeds at ⊥ for empty input."""
    b = fresh()
    empty: Antichain[Time] = cast(Antichain[Time], Antichain(Time1D))
    x = Input[int](frontier=empty).connect(b, ())
    d = Delay[int](group=INT_GROUP).connect(b, (x,))
    fwd = propagate_forward(b.progress_rules)
    assert fwd[d].elements == [(0,)]


def test_integrate_progress_strict_feedback() -> None:
    """Integrate's forward = input's frontier (strict-feedback theorem);
    backward retains exactly one self-slot (``retreat``)."""
    b = fresh()
    F: Antichain[Time] = cast(Antichain[Time], Antichain(Time1D, [(5,)]))
    x = Input[int](frontier=F).connect(b, ())
    s = Integrate[int](group=INT_GROUP).connect(b, (x,))
    dag = b.progress_rules
    fwd = propagate_forward(dag)
    assert fwd[s].elements == [(5,)]
    deads = propagate_backward(dag, cursors={s: fwd[s]})
    # v2.1.2: input keeps its (5,) delta for boundary recomputes, and
    # the self-edge keeps the boundary memo (4,) plus the cursor (5,).
    assert deads[x].elements == [(4,)]
    assert deads[s].elements == [(3,)]


def test_differentiate_progress_emerges_as_retreat_via_composition() -> None:
    """``Differentiate``'s backward rule on its input is ``retreat`` —
    the two-path fan-in (direct Lift2 edge meets Delay's retreat path)
    produces this. The composition (Delay+Lift1+Lift2) is exactly what
    makes the algebra work; we don't need a dedicated progress shape."""
    b = fresh()
    F: Antichain[Time] = cast(Antichain[Time], Antichain(Time1D, [(5,)]))
    x = Input[int](frontier=F).connect(b, ())
    out = Differentiate[int](group=INT_GROUP).connect(b, (x,))
    dag = b.progress_rules
    fwd = propagate_forward(dag)
    assert fwd[out].elements == [(5,)]  # forward unchanged
    deads = propagate_backward(dag, cursors={out: fwd[out]})
    # Input is reached via two paths: direct (dead_below = {(4,)}) and
    # the Delay path (its retreat, {(3,)}). The walker meets them and
    # the retreat path wins.
    assert deads[x].elements == [(3,)]


def test_time_axis_introduction_progress_lifts_arity() -> None:
    b = fresh()
    F: Antichain[Time] = cast(Antichain[Time], Antichain(Time1D, [(5,)]))
    x = Input[int](frontier=F).connect(b, ())
    out = LiftStreamIntroduction[int](group=INT_GROUP).connect(b, (x,))
    fwd = propagate_forward(b.progress_rules)
    # Insert ω at axis 1: {(5,)} → {(5, ω)}.
    from pydbsp.core import OMEGA

    assert fwd[out].elements == [(5, OMEGA)]


# ============================================================================
# End-to-end: composed pipeline
# ============================================================================


def test_pipeline_progress_advances_over_input_ticks() -> None:
    """Build once via operators, re-snapshot at successive input
    frontiers, observe progress advancing through every stage:

        x ──┬─ Delay ─┐
            │         ├─ Lift2(add) ─ Integrate ─ out
        y ──┴─────────┘
    """

    def build(
        x_f: Antichain[Time], y_f: Antichain[Time]
    ) -> tuple[
        list[prg.ProgressRule[Time]],
        NodeId,
        NodeId,
        NodeId,
        NodeId,
        NodeId,
    ]:
        b = fresh()
        x = Input[int](frontier=x_f).connect(b, ())
        y = Input[int](frontier=y_f).connect(b, ())
        dx = Delay[int](group=INT_GROUP).connect(b, (x,))
        m = Lift2[int, int, int](op=INT_GROUP.add).connect(b, (dx, y))
        s = Integrate[int](group=INT_GROUP).connect(b, (m,))
        return b.progress_rules, x, y, dx, m, s

    empty: Antichain[Time] = cast(Antichain[Time], Antichain(Time1D))

    # Tick 0: both empty — Delay's ⊥ seed gives {(0,)}, meet with empty
    # yields empty.
    dag, x, y, dx, m, s = build(empty, empty)
    fwd = propagate_forward(dag)
    assert fwd[dx].elements == [(0,)]
    assert fwd[m].elements == []
    assert fwd[s].elements == []

    # Tick at (5,) on both. Delay → {(6,)}; meet → {(5,)}; integrate → {(5,)}.
    f5: Antichain[Time] = cast(Antichain[Time], Antichain(Time1D, [(5,)]))
    dag, x, y, dx, m, s = build(f5, f5)
    fwd = propagate_forward(dag)
    assert fwd[dx].elements == [(6,)]
    assert fwd[m].elements == [(5,)]
    assert fwd[s].elements == [(5,)]

    # Backward GC at the integrate's cursor.
    deads = propagate_backward(dag, cursors={s: fwd[s]})
    assert deads[s].elements == [(3,)]  # boundary memo + cursor retained
    assert deads[m].elements == [(4,)]
    assert deads[dx].elements == [(4,)]
    assert deads[y].elements == [(4,)]
    assert deads[x].elements == [(3,)]  # retreat through Delay
