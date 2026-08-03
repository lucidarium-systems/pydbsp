"""Tests for ``pydbsp.progress`` — the antichain algebra of frontier
propagation.

Two layers:

1. Antichain primitives over ``DBSPTime`` — ``shift``, ``retreat``,
   ``drop_axis``, ``insert_axis``.
2. ProgressRule shapes — per-node ``forward``/``backward`` rules and
   propagation walks.
"""

from __future__ import annotations

import pytest

from typing import cast

from pydbsp.progress import (
    AxisIntroduction,
    AxisShift,
    Feedback,
    Identity,
    Input,
    Meet,
    ProgressRule,
    drop_axis,
    insert_axis,
    propagate_backward,
    propagate_forward,
    retreat,
    shift,
)
from pydbsp.core import (
    OMEGA,
    Antichain,
    BoundedBelowLattice,
    DBSPTime,
    Time1D,
    Time2D,
)


# ============================================================================
# Antichain primitives
# ============================================================================


def test_shift_increments_each_element_on_axis() -> None:
    A = Antichain(Time1D, [(3,)])
    assert shift(A, axis=0).elements == [(4,)]


def test_shift_empty_stays_empty() -> None:
    A: Antichain = Antichain(Time1D)
    assert shift(A, axis=0).elements == []


def test_shift_universal_is_fixed_point() -> None:
    universal = Antichain.universal(Time1D)
    assert shift(universal, axis=0).is_universal is True


def test_shift_2d_axis_independent() -> None:
    A = Antichain(Time2D, [(3, 5)])
    assert shift(A, axis=0).elements == [(4, 5)]
    assert shift(A, axis=1).elements == [(3, 6)]


def test_retreat_decrements_each_element_on_axis() -> None:
    A = Antichain(Time1D, [(3,)])
    assert retreat(A, axis=0).elements == [(2,)]


def test_retreat_drops_elements_at_axis_bottom() -> None:
    A = Antichain(Time2D, [(0, 5), (3, 2)])
    # (0, 5) has no predecessor on axis 0 → drop; (3, 2) → (2, 2).
    assert retreat(A, axis=0).elements == [(2, 2)]


def test_retreat_universal_is_fixed_point() -> None:
    universal = Antichain.universal(Time1D)
    assert retreat(universal, axis=0).is_universal is True


def test_retreat_preserves_omega() -> None:
    # OMEGA is typed ``float`` (math.inf) but represents the ω top of
    # NaturalChain — the codebase pretends it's an int (see
    # ``NaturalChain.top``); cast here to match Time1D's tuple[int].
    A = Antichain(Time1D, [cast(tuple[int], (OMEGA,))])
    assert retreat(A, axis=0).elements == [(OMEGA,)]


def test_drop_axis_projects_to_lower_arity() -> None:
    A = Antichain(Time2D, [(5, 3)])
    out = drop_axis(A, axis=1)
    assert isinstance(out.lattice, DBSPTime)
    assert out.lattice.nestedness == 1
    assert out.elements == [(5,)]


def test_drop_axis_universal_stays_universal() -> None:
    universal = Antichain.universal(Time2D)
    out = drop_axis(universal, axis=0)
    assert out.is_universal is True
    assert isinstance(out.lattice, DBSPTime)
    assert out.lattice.nestedness == 1


def test_insert_axis_lifts_to_higher_arity() -> None:
    A = Antichain(Time1D, [(5,)])
    out = insert_axis(A, axis=1, value=OMEGA)
    assert isinstance(out.lattice, DBSPTime)
    assert out.lattice.nestedness == 2
    assert out.elements == [(5, OMEGA)]


def test_insert_axis_universal_stays_universal() -> None:
    universal = Antichain.universal(Time1D)
    out = insert_axis(universal, axis=0, value=OMEGA)
    assert out.is_universal is True
    assert isinstance(out.lattice, DBSPTime)
    assert out.lattice.nestedness == 2


def test_primitives_reject_non_dbsp_lattice() -> None:
    """A generic ``BoundedBelowLattice`` antichain has no axes; the
    primitives must refuse it. The FakeLattice satisfies the
    ``T: Time`` bound (tuple-typed) so static checks pass — but it
    isn't a ``DBSPTime``, so the runtime ``_dbsp_lattice`` guard fires
    and raises ``TypeError``."""

    class FakeLattice(BoundedBelowLattice[tuple[int, ...]]):
        def join(self, a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
            return a

        def meet(self, a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
            return a

        def bottom(self) -> tuple[int, ...]:
            return (0,)

        def top(self) -> tuple[int, ...]:
            return (0,)

    A: Antichain[tuple[int, ...]] = Antichain(FakeLattice(), [(0,)])
    with pytest.raises(TypeError, match="DBSPTime"):
        shift(A, axis=0)
    with pytest.raises(TypeError, match="DBSPTime"):
        retreat(A, axis=0)
    with pytest.raises(TypeError, match="DBSPTime"):
        drop_axis(A, axis=0)
    with pytest.raises(TypeError, match="DBSPTime"):
        insert_axis(A, axis=0, value=0)


# ============================================================================
# ProgressRule shapes — forward + backward
# ============================================================================


def test_input_has_no_inputs_and_returns_frontier_in_forward() -> None:
    F = Antichain(Time1D, [(5,)])
    input = Input(frontier=F)
    assert input.inputs == ()
    assert input.forward({}) == F
    assert input.backward(Antichain(Time1D, [(7,)])) == {}


def test_identity_forward_returns_input_frontier() -> None:
    op = Identity(input=0)
    F = Antichain(Time1D, [(5,)])
    assert op.forward({0: F}).elements == [(5,)]


def test_identity_backward_returns_singleton() -> None:
    op = Identity(input=0)
    F = Antichain(Time1D, [(5,)])
    assert op.backward(F) == {0: F}


def test_meet_forward_takes_pairwise_meet() -> None:
    op = Meet(_inputs=(0, 1))
    F1 = Antichain(Time1D, [(5,)])
    F2 = Antichain(Time1D, [(3,)])
    assert op.forward({0: F1, 1: F2}).elements == [(3,)]


def test_meet_backward_distributes_to_all_inputs() -> None:
    op = Meet(_inputs=(0, 1))
    F = Antichain(Time1D, [(5,)])
    inputs = op.backward(F)
    assert set(inputs.keys()) == {0, 1}
    assert all(v.elements == [(5,)] for v in inputs.values())


def test_axis_shift_forward_shifts_nonempty() -> None:
    op = AxisShift[tuple[int]](input=0, axis=0)
    assert op.forward({0: Antichain(Time1D, [(3,)])}).elements == [(4,)]


def test_axis_shift_forward_seeds_bottom_for_empty_input() -> None:
    op = AxisShift[tuple[int]](input=0, axis=0)
    assert op.forward({0: Antichain(Time1D)}).elements == [(0,)]


def test_axis_shift_backward_retreats() -> None:
    op = AxisShift[tuple[int]](input=0, axis=0)
    assert op.backward(Antichain(Time1D, [(5,)]))[0].elements == [(4,)]


def test_axis_introduction_forward_inserts_omega() -> None:
    op = AxisIntroduction[tuple[int]](input=0, axis=1)
    F = Antichain(Time1D, [(5,)])
    out = op.forward({0: F})
    assert isinstance(out.lattice, DBSPTime)
    assert out.lattice.nestedness == 2
    assert out.elements == [(5, OMEGA)]


def test_axis_introduction_backward_drops_axis() -> None:
    """v2.1.2: only dead elements settled to OMEGA on the introduced
    axis project down — a finite coordinate there leaves cells that
    higher introduced-axis reads still serve."""
    op = AxisIntroduction[tuple[int, int]](input=0, axis=1)
    finite = Antichain(Time2D, [(5, 3)])
    out = op.backward(finite)[0]
    assert isinstance(out.lattice, DBSPTime)
    assert out.lattice.nestedness == 1
    assert out.elements == []

    settled = Antichain(Time2D, [(5, OMEGA)])
    out = op.backward(settled)[0]
    assert out.elements == [(5,)]


def test_feedback_forward_is_input_passthrough() -> None:
    op = Feedback[tuple[int]](input=0, self_id=1, axis=0)
    F = Antichain(Time1D, [(5,)])
    assert op.forward({0: F}).elements == [(5,)]


def test_feedback_backward_returns_input_and_self_retreat() -> None:
    op = Feedback[tuple[int]](input=0, self_id=1, axis=0)
    F = Antichain(Time1D, [(5,)])
    out = op.backward(F)
    assert out[0].elements == [(5,)]
    assert out[1].elements == [(4,)]


# ============================================================================
# Forward propagation
# ============================================================================


def test_forward_through_identity_chain() -> None:
    F = Antichain(Time1D, [(5,)])
    nodes: list[ProgressRule[tuple[int]]] = [
        Input(frontier=F),
        Identity(input=0),
        Identity(input=1),
    ]
    fwd = propagate_forward(nodes)
    assert fwd[0] == F
    assert fwd[1] == F
    assert fwd[2] == F


def test_forward_meet_at_fanin_takes_smaller() -> None:
    F_big = Antichain(Time1D, [(5,)])
    F_small = Antichain(Time1D, [(3,)])
    nodes: list[ProgressRule[tuple[int]]] = [
        Input(frontier=F_big),
        Input(frontier=F_small),
        Meet(_inputs=(0, 1)),
    ]
    fwd = propagate_forward(nodes)
    assert fwd[2].elements == [(3,)]


def test_forward_axis_shift_seeds_bottom_for_empty_input() -> None:
    """Delay's settled-frontier rule: empty input still gets the ⊥
    seed at the output."""
    nodes: list[ProgressRule[tuple[int]]] = [
        Input(frontier=Antichain(Time1D)),
        AxisShift(input=0, axis=0),
    ]
    fwd = propagate_forward(nodes)
    assert fwd[1].elements == [(0,)]


def test_forward_integrate_equals_input() -> None:
    """Strict-feedback theorem: integrate.forward = input.forward."""
    F = Antichain(Time1D, [(5,)])
    nodes: list[ProgressRule[tuple[int]]] = [
        Input(frontier=F),
        Feedback(input=0, self_id=1, axis=0),
    ]
    fwd = propagate_forward(nodes)
    assert fwd[1] == fwd[0] == F


# ============================================================================
# Backward propagation
# ============================================================================


def test_backward_through_identity() -> None:
    F = Antichain(Time1D, [(5,)])
    nodes: list[ProgressRule[tuple[int]]] = [
        Input(frontier=Antichain(Time1D)),
        Identity(input=0),
    ]
    deads = propagate_backward(nodes, cursors={1: F})
    # v2.1.2: the seed is dead_below(F) — strictly below the cursor,
    # so the at-cursor cell stays readable.
    assert deads[0].elements == [(4,)]


def test_backward_meet_at_fanin_takes_smaller() -> None:
    """Two consumers with different cursors meet at the shared input
    (intersection of down-sets = the smaller antichain)."""
    nodes: list[ProgressRule[tuple[int]]] = [
        Input(frontier=Antichain(Time1D)),  # 0: shared input
        Identity(input=0),  # 1: consumer A
        Identity(input=0),  # 2: consumer B
    ]
    deads = propagate_backward(
        nodes,
        cursors={
            1: Antichain(Time1D, [(5,)]),
            2: Antichain(Time1D, [(3,)]),
        },
    )
    # dead_below of the two cursors is {(4,)} and {(2,)}; their meet is
    # the smaller: consumer B may still resume at (3,).
    assert deads[0].elements == [(2,)]


def test_backward_integrate_keeps_one_self_slot() -> None:
    """Integrate's self-edge: ``deads[self] = retreat(F)`` — one slot
    retained, rest GC-able. Matches production steady-state."""
    F = Antichain(Time1D, [(5,)])
    nodes: list[ProgressRule[tuple[int]]] = [
        Input(frontier=Antichain(Time1D)),
        Feedback(input=0, self_id=1, axis=0),
    ]
    deads = propagate_backward(nodes, cursors={1: F})
    # v2.1.2: seed = dead_below({5}) = {4}; the input keeps its cell at
    # (5,) for boundary recomputes, and the self-edge keeps the memo at
    # (4,) plus the cursor cell (5,): the snapshot-plus-deltas optimum.
    assert deads[0].elements == [(4,)]
    assert deads[1].elements == [(3,)]


def test_backward_nested_integrate_keeps_one_slot_at_each_level() -> None:
    """Integrate(Integrate(x)) — every feedback node keeps one slot."""
    F = Antichain(Time1D, [(5,)])
    nodes: list[ProgressRule[tuple[int]]] = [
        Input(frontier=Antichain(Time1D)),  # 0: x
        Feedback(input=0, self_id=1, axis=0),  # 1: inner
        Feedback(input=1, self_id=2, axis=0),  # 2: outer
    ]
    deads = propagate_backward(nodes, cursors={2: F})
    assert deads[0].elements == [(4,)]  # input keeps the (5,) cell
    assert deads[1].elements == [(3,)]  # inner boundary memo retained
    assert deads[2].elements == [(3,)]  # outer boundary memo retained


def test_backward_multi_root_meets_at_shared_ancestor() -> None:
    nodes: list[ProgressRule[tuple[int]]] = [
        Input(frontier=Antichain(Time1D)),  # 0: shared
        Identity(input=0),  # 1: root A
        Identity(input=0),  # 2: root B
    ]
    deads = propagate_backward(
        nodes,
        cursors={
            1: Antichain(Time1D, [(7,)]),
            2: Antichain(Time1D, [(2,)]),
        },
    )
    # meet of dead_below({7}) = {6} and dead_below({2}) = {1}.
    assert deads[0].elements == [(1,)]


def test_backward_skips_nodes_unreached_by_any_cursor() -> None:
    nodes: list[ProgressRule[tuple[int]]] = [
        Input(frontier=Antichain(Time1D)),  # 0
        Identity(input=0),  # 1
        Identity(input=0),  # 2 — not cursored
    ]
    deads = propagate_backward(
        nodes,
        cursors={1: Antichain(Time1D, [(5,)])},
    )
    assert 2 not in deads
    assert deads[0].elements == [(4,)]
    assert deads[1].elements == [(4,)]
