"""Compute. Value rules per node, separate from progress tracking.

While :mod:`pydbsp.progress` describes WHERE in time each node is
settled (frontier algebra), this module describes WHAT each node
computes at a given timestamp (value algebra). The two layers align
on ``NodeId``: node ``i`` has both a :class:`ProgressRule` and a
:class:`ComputeRule`.

A compute rule is a **pure function** of timestamp + pre-bound input
readers + context. It carries no caching state. Memoisation, slot
tables, and scheduling belong to a separate evaluator layer.

**Closed vs. non-closed operations.** Some operators are closed in a
single value type ``V`` (``Delay``, ``Integrate``, ``Differentiate``,
``CausalAggregate``. They need a group to provide identity / add /
neg). Others map between different types (``Map`` from ``V_in`` to
``V_out``, ``ZipWith`` from ``V_l × V_r`` to ``V_out``). Closed shapes
carry their own :class:`AbelianGroupOperation` reference. The shared
:class:`ComputeCtx` only holds the lattice (V-independent).

**Wiring is the evaluator's job.** A compute rule does *not* know about
``NodeId``\\s. It consumes a tuple of pre-bound readers in the same
order as the progress layer declares its inputs. The evaluator
constructs the readers (one per upstream node) and threads them
through the rule.

**Primitives vs. compositions.** This module declares value rules for
the algebraic primitives:

* :class:`Get`. Atomic read at ``t``. Source nodes use this directly
  and every other shape composes it as its building block.
* :class:`Apply`. Generic N-ary apply: ``f`` over all readers at ``t``. Readers may have heterogeneous types.
* :class:`Map`, :class:`ZipWith`. Typed 1-ary / 2-ary specialisations
  with separate input/output type parameters.
* :class:`Prev`. Read the single input at the axis predecessor. Group
  identity at the axis bottom (closed in ``V``).
* :class:`Constant`. Place the input value at ``t_k = 0`` on a new
  axis, group identity elsewhere. Output arity is the input's plus one
  (closed in ``V``).
* :class:`Foldl`. Causal left fold: ``self(t) = op(input(t),
  self(predecessor(t, axis)))`` (closed in ``V``).
* :class:`Sum`. Specialisation of ``Foldl`` with ``op = group.add``:
  the running-sum recurrence ``self(t) = add(input(t), self(t-1))``.
* :class:`Diff`. Composition of ``ZipWith(add)``, ``Map(neg)``, and
  ``Prev`` (closed in ``V``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydbsp.core import AbelianGroupOperation, DBSPTime

from pydbsp.progress import Time


@dataclass(frozen=True)
class ComputeCtx:
    """Context threaded through every :meth:`ComputeRule.compute`
    call. Holds the time lattice (for axis predecessor / successor).

    The group lives on each shape that needs it (closed-V shapes like
    :class:`Prev`), keeping this context V-agnostic so a single
    instance serves a DAG containing heterogeneously-typed nodes."""

    lattice: DBSPTime[Time]


@runtime_checkable
class ComputeRule[V_out](Protocol):
    """The value rule for one node. Pure function of timestamp +
    pre-bound input readers + context.

    Parametrised by the **output** type ``V_out``. Input readers may
    have heterogeneous types (``Apply`` variants change types). Each
    concrete shape declares its own ``reads`` signature."""

    def compute(
        self,
        t: Time,
        reads: tuple[Callable[[Time], Any], ...],
        ctx: ComputeCtx,
    ) -> V_out: ...


# ---- Compute shapes --------------------------------------------------------


@dataclass(frozen=True)
class Get[V]:
    """The atomic value-algebra primitive: **read** at ``t``. Source
    nodes use this directly. The evaluator binds a single reader to
    storage, so ``reads = (storage_reader,)`` and ``compute`` returns
    that read. The storage reader yields the group identity for
    timestamps that have not been pushed.

    Other shapes compose ``Get`` as their building block. Every "value at ``t``" in the compute layer is structurally an
    ``Get`` call.
    """

    def compute(
        self,
        t: Time,
        reads: tuple[Callable[[Time], V]],
        ctx: ComputeCtx,
    ) -> V:
        (read,) = reads
        return read(t)


@dataclass(frozen=True)
class Apply[V_out]:
    """Generic N-ary lift: apply ``f`` to the values of all readers at
    ``t``. The arity is implicit in ``f``'s signature and the number
    of readers the evaluator binds. **Not closed**. Readers may have
    heterogeneous types. Only the output type ``V_out`` is fixed.

    Each reader's value at ``t`` is obtained via an
    :class:`Get` call.

    Use :class:`Map` / :class:`ZipWith` for the typed
    1-ary / 2-ary cases. They tighten ``f``'s signature and ``reads``
    to fixed-length tuples.
    """

    f: Callable[..., V_out]

    def compute(
        self,
        t: Time,
        reads: tuple[Callable[[Time], Any], ...],
        ctx: ComputeCtx,
    ) -> V_out:
        return self.f(*(r(t) for r in reads))


@dataclass(frozen=True)
class Map[V_in, V_out]:
    """``Map(s, f)``. 1-ary apply, possibly **non-closed**:
    ``f: V_in → V_out``. ``compute(t) = f(read(t))``.

    Composed on :class:`Get`: the single reader produces a
    ``V_in``, then ``f`` maps it to ``V_out``.
    """

    f: Callable[[V_in], V_out]

    def compute(
        self,
        t: Time,
        reads: tuple[Callable[[Time], V_in]],
        ctx: ComputeCtx,
    ) -> V_out:
        (read,) = reads
        return self.f(read(t))


@dataclass(frozen=True)
class ZipWith[V_l, V_r, V_out]:
    """``ZipWith(l, r, op)``. 2-ary apply, possibly **non-closed**:
    ``op: V_l × V_r → V_out``. ``compute(t) = op(read_l(t),
    read_r(t))``.

    Composed on :class:`Get`: each reader produces its own
    typed value (``V_l``, ``V_r``). ``op`` combines them into ``V_out``.
    """

    op: Callable[[V_l, V_r], V_out]

    def compute(
        self,
        t: Time,
        reads: tuple[Callable[[Time], V_l], Callable[[Time], V_r]],
        ctx: ComputeCtx,
    ) -> V_out:
        read_l, read_r = reads
        return self.op(read_l(t), read_r(t))


@dataclass(frozen=True)
class Prev[V]:
    """``Delay(s, axis=i)``. Strict shift on ``axis``. Reads the
    single input at ``predecessor(t, axis)``. Returns
    ``group.identity()`` at the axis bottom (no predecessor). Closed
    in ``V``. Carries its own group reference.
    """

    axis: int
    group: AbelianGroupOperation[V]

    def compute(
        self,
        t: Time,
        reads: tuple[Callable[[Time], V]],
        ctx: ComputeCtx,
    ) -> V:
        (read,) = reads
        chain = ctx.lattice.factors[self.axis]
        pred = chain.predecessor(t[self.axis])
        if pred is None:
            return self.group.identity()
        axis = self.axis
        return read(t[:axis] + (pred,) + t[axis + 1 :])


@dataclass(frozen=True)
class Constant[V]:
    """``intro_k(s)``. Δ_k impulse: place the input value at ``t_k =
    0`` and ``group.identity()`` everywhere else on the new axis.
    Input has arity ``n``. Output has arity ``n+1``. Closed in ``V``. Carries its own group reference for the identity fallback."""

    axis: int
    group: AbelianGroupOperation[V]

    def compute(
        self,
        t: Time,
        reads: tuple[Callable[[Time], V]],
        ctx: ComputeCtx,
    ) -> V:
        if t[self.axis] != 0:
            return self.group.identity()
        (read,) = reads
        axis = self.axis
        return read(t[:axis] + t[axis + 1 :])


@dataclass(frozen=True)
class Foldl[V]:
    """Causal aggregate: ``self(t) = op(input(t), self(predecessor(t,
    axis)))``. At the axis bottom the self-edge collapses to
    ``group.identity()`` (via :class:`Prev`) and the recurrence
    terminates with ``op(input(t_axis=0), identity)``.

    The composition is ``ZipWith(op)(input, Delay(self, axis))``; the
    embedded :class:`Prev` is built once at rule-construction time
    (``compute`` runs once per uncached cell — no per-call helper
    objects). Closed in ``V``. Carries its own group for the embedded
    :class:`Prev`.
    """

    axis: int
    op: Callable[[V, V], V]
    group: AbelianGroupOperation[V]
    _delay: Prev[V] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_delay", Prev(axis=self.axis, group=self.group))

    def compute(
        self,
        t: Time,
        reads: tuple[Callable[[Time], V], Callable[[Time], V]],
        ctx: ComputeCtx,
    ) -> V:
        read_input, read_self = reads
        return self.op(read_input(t), self._delay.compute(t, (read_self,), ctx))


@dataclass(frozen=True)
class Sum[V]:
    """``Integrate(s, axis=i)``. Running sum on ``axis``. Composed as
    :class:`Foldl` with ``op = group.add``.

    Reads: ``(input, self)``. Input plus the self-edge for the strict
    feedback cycle.
    """

    axis: int
    group: AbelianGroupOperation[V]
    _foldl: Foldl[V] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_foldl", Foldl(axis=self.axis, op=self.group.add, group=self.group))

    def compute(
        self,
        t: Time,
        reads: tuple[Callable[[Time], V], Callable[[Time], V]],
        ctx: ComputeCtx,
    ) -> V:
        return self._foldl.compute(t, reads, ctx)


@dataclass(frozen=True)
class Diff[V]:
    """``Differentiate(s, axis=i)(t) = s(t) - s(predecessor(t, i))``.

    Composition: ``ZipWith(add)`` over the input read and
    ``Map(neg) ∘ Prev(input, axis)``. The single input reader is
    consumed twice. Once directly, once through the Delay+neg path.
    The embedded :class:`Prev` is built once at rule-construction
    time; ``compute`` applies ``add``/``neg`` directly.

    Reads: ``(input,)``. No self-edge.
    """

    axis: int
    group: AbelianGroupOperation[V]
    _delay: Prev[V] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_delay", Prev(axis=self.axis, group=self.group))

    def compute(
        self,
        t: Time,
        reads: tuple[Callable[[Time], V]],
        ctx: ComputeCtx,
    ) -> V:
        (read,) = reads
        return self.group.add(read(t), self.group.neg(self._delay.compute(t, (read,), ctx)))
