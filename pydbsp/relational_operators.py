"""Relational operators. Doubly-incremental bilinear joins and
distincts on Z-sets, ported from pydbsp's stream/zset operators.

These are **composite** :class:`Operator` instances: each ``connect``
call materialises a sub-DAG of primitives (``Integrate``, ``Delay``,
``Lift2``, ``Differentiate``) according to the canonical DBSP
identities. The Z-set value functions (``join``, ``H``) are re-used
verbatim from :mod:`pydbsp.zset.functions`.

Operators here parameterise themselves on the **outer** and **inner**
axes of a 2- or 3-axis flat product lattice, matching the original
:func:`pydbsp.stream.zset.operators.bilinear.DLDJoin` /
:func:`pydbsp.stream.zset.operators.binary.DLDDistinct` signatures.

**Single-Lift / doubly-Lift (pointwise) variants**. Apply the value
function pointwise. Non-incremental. ``Lift1`` / ``Lift2`` is
arity-agnostic, so the ``LiftLift*`` variants are documentation
synonyms of the ``Lift*`` versions for 2-D stream-of-streams usage:

* :class:`LiftSelect`  / :class:`LiftLiftSelect`   . ``↑σ_p`` / ``↑↑σ_p``.
* :class:`LiftProject` / :class:`LiftLiftProject`  . ``↑π_f`` / ``↑↑π_f``.
* :class:`LiftJoin`    / :class:`LiftLiftJoin`     . ``↑J`` / ``↑↑J``.
* :class:`LiftH`       / :class:`LiftLiftH`        . ``↑H`` / ``↑↑H``
  (the value-level "distinct" primitive, lifted pointwise).

**Doubly-incremental variants**. Wire Integrate/Delay/Differentiate
together to maintain the incremental view of a non-linear operator on
a 2-axis lattice:

* :class:`DeltaLiftedDeltaLiftedJoin` . ``J(z⁻¹ᵒIᵒa, z⁻¹ⁱIⁱb) +
  J(IᵒIⁱa, b) + J(Iⁱa, z⁻¹ᵒIᵒb) + J(a, z⁻¹ⁱIᵒIⁱb)``. 4-term
  bilinear join.
* :class:`DeltaLiftedDeltaLiftedDistinct` . ``Dᵒ(H(z⁻¹ⁱIⁱIᵒs, Iᵒs))``. H-based incremental distinct on a 2-axis lattice.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydbsp import datalog as dlg
from pydbsp.zset import ZSet, ZSetAddition
from pydbsp.zset.functions.bilinear import join
from pydbsp.zset.functions.binary import H
from pydbsp.zset.functions.linear import project, select

from pydbsp.circuit import Circuit
from pydbsp.operator import (
    Delay,
    Differentiate,
    Integrate,
    Lift1,
    Lift2,
    LiftDelay,
    LiftIntegrate,
    LiftStreamIntroduction,
    Operator,
)
from pydbsp.progress import NodeId, Time


@dataclass(frozen=True)
class LiftSelect[V](Operator):
    """``σ_p(s)``. Pointwise filter on a Z-set stream. Linear in the
    Z-set group structure, so a single :class:`Lift1` is automatically
    doubly-incremental: the linear identity ``select(I(s))(t) =
    I(select(s))(t)`` means no additional Integrate/Differentiate
    wiring is needed.

    ``inputs = (diff_stream,)``."""

    pred: Callable[[V], bool]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        (s,) = inputs
        pred = self.pred
        return Lift1[ZSet[V], ZSet[V]](f=lambda z: select(z, pred)).connect(circuit, (s,))


@dataclass(frozen=True)
class LiftProject[A, B](Operator):
    """``π_f(s)``. Pointwise element transform on a Z-set stream.
    Maps each ``ZSet[A]`` to a ``ZSet[B]`` by applying ``f`` to every
    element (collisions are summed via the group's ``add``). Linear,
    so a single :class:`Lift1` suffices.

    ``inputs = (diff_stream,)``."""

    f: Callable[[A], B]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        (s,) = inputs
        f = self.f
        return Lift1[ZSet[A], ZSet[B]](f=lambda z: project(z, f)).connect(circuit, (s,))


@dataclass(frozen=True)
class LiftJoin[A, B, C](Operator):
    """``↑J``. Pointwise lift of the value-level ``join`` to a Z-set
    stream. A single :class:`Lift2` of the join function is the whole
    story. **Not** auto-doubly-incremental like :class:`LiftSelect` /
    :class:`LiftProject` (join is bilinear, not linear). Reach for
    :class:`DeltaLiftedDeltaLiftedJoin` when you need the incremental
    version.

    ``inputs = (s_a, s_b)``."""

    pred: Callable[[A, B], bool]
    proj: Callable[[A, B], C]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        (s_a, s_b) = inputs
        pred = self.pred
        proj = self.proj
        return Lift2[ZSet[A], ZSet[B], ZSet[C]](op=lambda za, zb: join(za, zb, pred, proj)).connect(circuit, (s_a, s_b))


@dataclass(frozen=True)
class LiftH[V](Operator):
    """``↑H``. Pointwise lift of the threshold-crossing function
    ``H(i, d)`` to a Z-set stream. ``H`` is the value-level "distinct"
    primitive. Given the integrated state ``i`` and a delta ``d``,
    it returns the per-element ±1 indicators for elements crossing
    the positivity threshold.

    Not to be confused with :class:`DeltaLiftedDeltaLiftedDistinct`,
    which wires ``H`` together with the necessary
    Integrate/Delay/Differentiate machinery for the full
    doubly-incremental distinct.

    ``inputs = (i_stream, d_stream)``."""

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        (i, d) = inputs
        return Lift2[ZSet[V], ZSet[V], ZSet[V]](op=H).connect(circuit, (i, d))


# ---- Doubly-lifted variants -----------------------------------------------
#
# Same implementations as the single-Lift versions — ``Lift1`` / ``Lift2``
# are arity-agnostic, so pointwise lift to a 2-D stream-of-streams uses
# the exact same circuit nodes as the 1-D case. The distinct class
# names document the intended stream shape (1-D vs 2-D).
#
# **Not to be confused with the ``DeltaLifted*`` variants**, which add
# Integrate/Delay/Differentiate machinery for incremental views.


@dataclass(frozen=True)
class LiftLiftSelect[V](LiftSelect[V]):
    """``↑↑σ_p``. Pointwise lift to a 2-D stream-of-streams."""


@dataclass(frozen=True)
class LiftLiftProject[A, B](LiftProject[A, B]):
    """``↑↑π_f``. Pointwise lift to a 2-D stream-of-streams."""


@dataclass(frozen=True)
class LiftLiftJoin[A, B, C](LiftJoin[A, B, C]):
    """``↑↑J``. Pointwise lift to a 2-D stream-of-streams."""


@dataclass(frozen=True)
class LiftLiftH[V](LiftH[V]):
    """``↑↑H``. Pointwise lift to a 2-D stream-of-streams."""


@dataclass(frozen=True)
class DeltaLiftedDeltaLiftedJoin[A, B, C](Operator):
    """4-term doubly-incremental bilinear join on a 2-axis flat
    product lattice (outer axis 0, inner axis 1):

        J(z⁻¹ᵒ Iᵒ a,   z⁻¹ⁱ Iⁱ b)
      + J(Iᵒ Iⁱ a,      b)
      + J(Iⁱ a,          z⁻¹ᵒ Iᵒ b)
      + J(a,             z⁻¹ⁱ Iᵒ Iⁱ b)

    Connects as 17 nodes total. 10 Integrate/Delay primitives via the
    convention overloads, 4 :class:`LiftLiftJoin` joins, 3 sums.

    ``inputs = (diff_a, diff_b)``. ``group_a`` / ``group_b`` are the
    Z-set groups for the two inputs (needed by the Integrate / Delay
    primitives' identity fallbacks). ``out_group`` is the group for
    the joined output's Z-set.
    """

    pred: Callable[[A, B], bool]
    proj: Callable[[A, B], C]
    group_a: ZSetAddition[A]
    group_b: ZSetAddition[B]
    out_group: ZSetAddition[C]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        if len(inputs) != 2:
            raise ValueError(f"DeltaLiftedDeltaLiftedJoin takes 2 inputs; got {len(inputs)}")
        diff_a, diff_b = inputs

        # Per-axis integrate / delay on each operand, via the
        # DBSP-convention overloads (outer = axis 0, inner = axis 1).
        int_a_o = Integrate[ZSet[A]](group=self.group_a).connect(circuit, (diff_a,))
        del_int_a_o = Delay[ZSet[A]](group=self.group_a).connect(circuit, (int_a_o,))

        int_b_o = Integrate[ZSet[B]](group=self.group_b).connect(circuit, (diff_b,))
        del_int_b_o = Delay[ZSet[B]](group=self.group_b).connect(circuit, (int_b_o,))

        int_a_i = LiftIntegrate[ZSet[A]](group=self.group_a).connect(circuit, (diff_a,))
        int_b_i = LiftIntegrate[ZSet[B]](group=self.group_b).connect(circuit, (diff_b,))

        int_a_oi = Integrate[ZSet[A]](group=self.group_a).connect(circuit, (int_a_i,))
        int_b_oi = Integrate[ZSet[B]](group=self.group_b).connect(circuit, (int_b_i,))

        del_int_b_i = LiftDelay[ZSet[B]](group=self.group_b).connect(circuit, (int_b_i,))
        del_int_b_oi = LiftDelay[ZSet[B]](group=self.group_b).connect(circuit, (int_b_oi,))

        # Re-use the pointwise lifted join for each of the four bilinear
        # terms — same shape, different operand paths.
        lifted_join = LiftLiftJoin[A, B, C](pred=self.pred, proj=self.proj)
        j1 = lifted_join.connect(circuit, (del_int_a_o, del_int_b_i))
        j2 = lifted_join.connect(circuit, (int_a_oi, diff_b))
        j3 = lifted_join.connect(circuit, (int_a_i, del_int_b_o))
        j4 = lifted_join.connect(circuit, (diff_a, del_int_b_oi))

        # Sum the four terms via two-arg Lift2s.
        add = self.out_group.add
        sum_12 = Lift2[ZSet[C], ZSet[C], ZSet[C]](op=add).connect(circuit, (j1, j2))
        sum_34 = Lift2[ZSet[C], ZSet[C], ZSet[C]](op=add).connect(circuit, (j3, j4))
        return Lift2[ZSet[C], ZSet[C], ZSet[C]](op=add).connect(circuit, (sum_12, sum_34))


@dataclass(frozen=True)
class DeltaLiftedDeltaLiftedDistinct[V](Operator):
    """``Dᵒ(H(z⁻¹ⁱ Iⁱ Iᵒ s, Iᵒ s))``. Doubly-incremental distinct on
    a 2-axis flat product lattice (outer axis 0, inner axis 1).
    Connects as 4 primitive nodes (via the convention overloads) plus
    a :class:`LiftLiftH` pointwise H lift, plus a ``Differentiate``
    composite (3 more nodes) . ~9 total.

    ``inputs = (diff_stream,)``. ``inner_group`` is the Z-set group
    for the stream's element type."""

    inner_group: ZSetAddition[V]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        (diff_stream,) = inputs

        integrated = Integrate[ZSet[V]](group=self.inner_group).connect(circuit, (diff_stream,))
        int_int = LiftIntegrate[ZSet[V]](group=self.inner_group).connect(circuit, (integrated,))
        del_int_int = LiftDelay[ZSet[V]](group=self.inner_group).connect(circuit, (int_int,))
        # Re-use the pointwise lifted H — one Lift2 of the value-level
        # threshold-crossing function.
        h = LiftLiftH[V]().connect(circuit, (del_int_int, integrated))
        return Differentiate[ZSet[V]](group=self.inner_group).connect(circuit, (h,))


@dataclass(frozen=True)
class IncrementalDatalogWithNegationBody(Operator):
    """Body of a negation-aware Datalog interpreter on the **2-D**
    ``(outer, inner)`` lattice — the non-indexed counterpart of
    :class:`pydbsp.indexed_relational_operators.IndexedIncrementalDatalogWithNegationBody`,
    obtained by switching every indexed join to the plain
    :class:`DeltaLiftedDeltaLiftedJoin` (with an explicit ``pred``) and
    dropping every ``LiftIndex`` / ``ext_dir`` / ``jorder`` index step.
    Equivalently, the 2-D reduction of the 3-D
    :class:`pydbsp.datalog_stratified.IncrementalDatalogBodyWithNegation`.

    Anti-rewrite bifurcation (C5 Def 4.8):
    ``final_product = product + proj - anti_product + interp_proj`` — the
    negated-atom rewrites survive iff their instantiated atom is ABSENT
    from the facts (``proj`` minus ``anti_product``).

    ``inputs = (edb_1d, program_2d, state_facts_2d, state_rewrites_2d,
    seed_1d)``. Output at each ``(outer, inner)`` cell is the pair
    ``(facts_diff, rewrites_diff)``.

    SPLIT NEGATION PORT (optional 6th input ``neg_ref_1d``): when given,
    the negation test (``anti_product``) reads THIS input — imported into
    the inner clock via ``LiftStreamIntroduction`` (δ₀, integrated inside
    the join), hence FROZEN along the inner fixpoint — instead of the
    moving in-loop facts wire. Positive derivation is untouched. This is
    the semipositive circuit with the negation reference decoupled from
    the seed: the building block of the alternating-fixpoint stratified
    evaluator (:class:`pydbsp.datalog_stratified.DatalogAlternating`).
    With 5 inputs, behavior is exactly as before (negation reads the
    in-loop facts; correct for single-stratum negation only)."""

    fact_group: ZSetAddition[dlg.Fact]
    rule_group: ZSetAddition[dlg.Rule]
    rewrite_group: ZSetAddition[dlg.ProvenanceIndexedRewrite]
    signal_group: ZSetAddition[dlg.Signal]
    dir_group: ZSetAddition[dlg.Direction]
    gatekeep_group: ZSetAddition[dlg.AtomWithSourceRewriteAndProvenance]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        if len(inputs) not in (5, 6):
            raise ValueError(
                f"IncrementalDatalogWithNegationBody takes 5 inputs "
                f"(edb, program, state_facts, state_rewrites, seed) or 6 "
                f"(…, neg_ref); got {len(inputs)}"
            )
        edb_1d, program_2d, state_facts_2d, state_rewrites_2d, seed_1d = inputs[:5]
        neg_ref_1d = inputs[5] if len(inputs) == 6 else None

        edb_2d = LiftStreamIntroduction[ZSet[dlg.Fact]](group=self.fact_group).connect(circuit, (edb_1d,))
        seed_2d = LiftStreamIntroduction[ZSet[dlg.ProvenanceIndexedRewrite]](group=self.rewrite_group).connect(
            circuit, (seed_1d,)
        )

        delayed_facts = LiftDelay[ZSet[dlg.Fact]](group=self.fact_group).connect(circuit, (state_facts_2d,))
        delayed_rewrites = LiftDelay[ZSet[dlg.ProvenanceIndexedRewrite]](group=self.rewrite_group).connect(
            circuit, (state_rewrites_2d,)
        )

        # Negation reference: the frozen split port when provided (imported via
        # δ₀ = LiftStreamIntroduction; the anti-join's internal integral holds it
        # constant along the inner clock), else the moving in-loop facts.
        neg_ref_wire = (
            LiftStreamIntroduction[ZSet[dlg.Fact]](group=self.fact_group).connect(circuit, (neg_ref_1d,))
            if neg_ref_1d is not None
            else delayed_facts
        )

        sig_2d = Lift1[ZSet[dlg.Rule], ZSet[dlg.Signal]](f=dlg.sig).connect(circuit, (program_2d,))
        dir_2d = Lift1[ZSet[dlg.Rule], ZSet[dlg.Direction]](f=dlg.dir).connect(circuit, (program_2d,))

        gatekeep = DeltaLiftedDeltaLiftedJoin[
            dlg.ProvenanceIndexedRewrite,
            dlg.Direction,
            dlg.AtomWithSourceRewriteAndProvenance,
        ](
            pred=lambda left, right: left[0] == right[0],
            proj=lambda left, right: (right[1], right[2], left[1]),
            group_a=self.rewrite_group,
            group_b=self.dir_group,
            out_group=self.gatekeep_group,
        ).connect(circuit, (delayed_rewrites, dir_2d))

        positive_atoms = Lift1[
            ZSet[dlg.AtomWithSourceRewriteAndProvenance],
            ZSet[dlg.AtomWithSourceRewriteAndProvenance],
        ](
            f=lambda z: ZSet(
                {
                    gk: w
                    for gk, w in z.inner.items()
                    if gk[1] is None or ("!" not in gk[1][0] and gk[1][0] not in dlg.INTERPRETED)
                }
            ),
        ).connect(circuit, (gatekeep,))
        negative_atoms = Lift1[
            ZSet[dlg.AtomWithSourceRewriteAndProvenance],
            ZSet[dlg.AtomWithSourceRewriteAndProvenance],
        ](
            f=lambda z: ZSet({gk: w for gk, w in z.inner.items() if not (gk[1] is None or "!" not in gk[1][0])}),
        ).connect(circuit, (gatekeep,))
        interp_proj = Lift1[
            ZSet[dlg.AtomWithSourceRewriteAndProvenance],
            ZSet[dlg.ProvenanceIndexedRewrite],
        ](
            f=lambda z: ZSet(
                {
                    (gk[0], gk[2]): w
                    for gk, w in z.inner.items()
                    if gk[1] is not None and gk[1][0] in dlg.INTERPRETED and dlg.eval_interpreted(gk[1], gk[2])
                }
            ),
        ).connect(circuit, (gatekeep,))

        product = DeltaLiftedDeltaLiftedJoin[
            dlg.AtomWithSourceRewriteAndProvenance,
            dlg.Fact,
            dlg.ProvenanceIndexedRewrite,
        ](
            pred=lambda left, right: (
                left[1] is None or (left[1][0] == right[0] and dlg.unify(left[2].apply(left[1]), right) is not None)
            ),
            proj=dlg.rewrite_product_projection,
            group_a=self.gatekeep_group,
            group_b=self.fact_group,
            out_group=self.rewrite_group,
        ).connect(circuit, (positive_atoms, delayed_facts))

        proj = Lift1[
            ZSet[dlg.AtomWithSourceRewriteAndProvenance],
            ZSet[dlg.ProvenanceIndexedRewrite],
        ](
            f=lambda z: ZSet({(gk[0], gk[2]): w for gk, w in z.inner.items()}),
        ).connect(circuit, (negative_atoms,))

        anti_product = DeltaLiftedDeltaLiftedJoin[
            dlg.AtomWithSourceRewriteAndProvenance,
            dlg.Fact,
            dlg.ProvenanceIndexedRewrite,
        ](
            pred=lambda left, right: (
                left[1] is None
                or (left[1][0].strip("!") == right[0] and dlg.unify(left[2].apply(left[1]), right) is not None)
            ),
            proj=lambda left, _right: (left[0], left[2]),
            group_a=self.gatekeep_group,
            group_b=self.fact_group,
            out_group=self.rewrite_group,
        ).connect(circuit, (negative_atoms, neg_ref_wire))

        # final_product = product + proj - anti_product + interp_proj
        product_plus_proj = Lift2[
            ZSet[dlg.ProvenanceIndexedRewrite],
            ZSet[dlg.ProvenanceIndexedRewrite],
            ZSet[dlg.ProvenanceIndexedRewrite],
        ](op=self.rewrite_group.add).connect(circuit, (product, proj))
        neg_anti = Lift1[
            ZSet[dlg.ProvenanceIndexedRewrite],
            ZSet[dlg.ProvenanceIndexedRewrite],
        ](f=self.rewrite_group.neg).connect(circuit, (anti_product,))
        final_product_neg = Lift2[
            ZSet[dlg.ProvenanceIndexedRewrite],
            ZSet[dlg.ProvenanceIndexedRewrite],
            ZSet[dlg.ProvenanceIndexedRewrite],
        ](op=self.rewrite_group.add).connect(circuit, (product_plus_proj, neg_anti))
        final_product = Lift2[
            ZSet[dlg.ProvenanceIndexedRewrite],
            ZSet[dlg.ProvenanceIndexedRewrite],
            ZSet[dlg.ProvenanceIndexedRewrite],
        ](op=self.rewrite_group.add).connect(circuit, (final_product_neg, interp_proj))

        ground = DeltaLiftedDeltaLiftedJoin[
            dlg.ProvenanceIndexedRewrite,
            dlg.Signal,
            dlg.Fact,
        ](
            pred=lambda left, right: left[0] == right[0],
            proj=lambda left, right: left[1].apply(right[1]),
            group_a=self.rewrite_group,
            group_b=self.signal_group,
            out_group=self.fact_group,
        ).connect(circuit, (final_product, sig_2d))

        ground_plus_edb = Lift2[ZSet[dlg.Fact], ZSet[dlg.Fact], ZSet[dlg.Fact]](op=self.fact_group.add).connect(
            circuit, (ground, edb_2d)
        )
        next_facts = DeltaLiftedDeltaLiftedDistinct[dlg.Fact](inner_group=self.fact_group).connect(
            circuit, (ground_plus_edb,)
        )

        final_plus_seed = Lift2[
            ZSet[dlg.ProvenanceIndexedRewrite],
            ZSet[dlg.ProvenanceIndexedRewrite],
            ZSet[dlg.ProvenanceIndexedRewrite],
        ](op=self.rewrite_group.add).connect(circuit, (final_product, seed_2d))
        next_rewrites = DeltaLiftedDeltaLiftedDistinct[dlg.ProvenanceIndexedRewrite](
            inner_group=self.rewrite_group
        ).connect(circuit, (final_plus_seed,))

        return Lift2[
            ZSet[dlg.Fact],
            ZSet[dlg.ProvenanceIndexedRewrite],
            tuple[ZSet[dlg.Fact], ZSet[dlg.ProvenanceIndexedRewrite]],
        ](op=lambda a, b: (a, b)).connect(circuit, (next_facts, next_rewrites))
