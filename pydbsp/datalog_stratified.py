"""Stratified-negation Datalog: 3-D primitives, body operator, and drivers.

The 3-D primitives are the ``(outer, ·, inner)`` analog of the 2-D
bilinear / distinct operators in :mod:`pydbsp.relational_operators`:

* :class:`DeltaLiftedDeltaLiftedDeltaLiftedJoin` — 8-term bilinear
  join (the 3-D extension of the 2-D 4-term bilinear).
* :class:`DeltaLiftedDeltaLiftedDeltaLiftedDistinct` — 3-D form of
  ``Differentiate(H(z⁻¹ I s, I s))``.

The ``Lift1`` / ``Lift2`` value-layer wrappers are arity-agnostic and
carry over unchanged. :class:`IncrementalDatalogBodyWithNegation` is the
negation-aware body built from them; the alternating drivers run it.

Drivers:

* :class:`IndexedDatalog`, :class:`IndexedDatalogWithNegation`,
  :class:`DatalogWithNegation` — 2-D; with negation, correct on programs
  tight below negation.
* :class:`DatalogStratifiedComposition` — one 2-D fixpoint per stratum.
* :class:`IncrementalDatalogStratified` and its indexed subclass —
  ``stratSpec`` of ``V3/StratifiedCircuit.lean``: the 2-D body at
  ``inner_axis=2``, lifted pointwise over the stratum axis.
* :class:`DatalogAlternating`, :class:`IncrementalDatalogAlternating` —
  the alternating fixpoint as one 3-D circuit."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydbsp import datalog as dlg
from pydbsp.circuit import Circuit
from pydbsp.compute import ComputeCtx
from pydbsp.core import Antichain, dbsp_time
from pydbsp.evaluate import Evaluator
from pydbsp.indexed_relational_operators import (
    IndexedIncrementalDatalogBody,
    IndexedIncrementalDatalogWithNegationBody,
)
from pydbsp.operator import (
    CoreDelay,
    CoreDifferentiate,
    CoreIntegrate,
    Input,
    Lift1,
    Lift2,
    LiftStreamIntroduction,
    Operator,
)
from pydbsp.progress import NodeId, Time
from pydbsp.relational_operators import IncrementalDatalogWithNegationBody, LiftH, LiftJoin
from pydbsp.storage import DictStorage
from pydbsp.zset import ZSet, ZSetAddition


# ---- Value-layer helpers ---------------------------------------------------


def _without_internal_facts(z: ZSet[dlg.Fact]) -> ZSet[dlg.Fact]:
    """Drop zero-weight entries from public Datalog output."""
    return ZSet({fact: w for fact, w in z.inner.items() if w != 0})


def _tainted_positions(program: ZSet[dlg.Rule]) -> set[tuple[str, int]]:
    """The set of ``(predicate, arg-index)`` positions that can carry an
    INVENTED (Skolem-constructed) value. Seeds every head position occupied by
    a Skolem term, then propagates through plain-variable head copies to a
    fixpoint. Used by the generative-recursion check to tell EDB-grounded key
    positions apart from invented value positions."""
    tainted: set[tuple[str, int]] = set()
    rules = [r for r, w in program.inner.items() if w > 0]
    for rule in rules:
        hp = rule[0][0]
        for i, a in enumerate(rule[0][1]):
            if type(a) is dlg.Skolem:
                tainted.add((hp, i))
    changed = True
    while changed:
        changed = False
        for rule in rules:
            hp = rule[0][0]
            for i, a in enumerate(rule[0][1]):
                if not dlg.is_variable(a) or (hp, i) in tainted:
                    continue
                for batom in rule[1:]:
                    bp = batom[0].lstrip("!")
                    if any(bt == a and (bp, j) in tainted for j, bt in enumerate(batom[1])):
                        tainted.add((hp, i))
                        changed = True
                        break
    return tainted


def _check_stratifiable(program: ZSet[dlg.Rule]) -> None:
    """Reject a program if its predicate-dependency graph contains a
    cycle through negation. Tarjan SCC over the combined positive +
    negative dep graph; reject any SCC that contains an internal
    negative edge."""
    if not program.inner:
        return

    nodes: set[str] = set()
    pos_edges: dict[str, set[str]] = {}
    neg_edges: set[tuple[str, str]] = set()
    for rule, w in program.inner.items():
        if w <= 0:
            continue
        head = rule[0][0]
        nodes.add(head)
        for atom in rule[1:]:
            body_pred = atom[0]
            if body_pred.startswith("!"):
                target = body_pred[1:]
                neg_edges.add((head, target))
                nodes.add(target)
            else:
                pos_edges.setdefault(head, set()).add(body_pred)
                nodes.add(body_pred)
    all_edges: dict[str, set[str]] = {n: set() for n in nodes}
    for src, dsts in pos_edges.items():
        all_edges.setdefault(src, set()).update(dsts)
    for src, dst in neg_edges:
        all_edges.setdefault(src, set()).add(dst)

    sccs: list[set[str]] = []
    index_counter = [0]
    stack: list[str] = []
    on_stack: set[str] = set()
    idx_of: dict[str, int] = {}
    lowlink: dict[str, int] = {}

    def strongconnect(v: str) -> None:
        idx_of[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in all_edges.get(v, ()):
            if w not in idx_of:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], idx_of[w])
        if lowlink[v] == idx_of[v]:
            scc: set[str] = set()
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.add(w)
                if w == v:
                    break
            sccs.append(scc)

    for v in list(nodes):
        if v not in idx_of:
            strongconnect(v)

    for scc in sccs:
        for src, dst in neg_edges:
            if src in scc and dst in scc:
                raise ValueError(
                    f"Program is not stratifiable: cycle through negation "
                    f"within SCC {sorted(scc)} via neg edge "
                    f"{src!r} → {dst!r}"
                )

    # Generative (Skolem-bearing) heads INVENT terms, and function symbols make
    # Datalog Turing-complete — so generative recursion can derive unbounded
    # terms. But the structural transpiler rule
    #   py_expr(N, infix(Op,La,Ra)) :- c_binary(N,Op,L,R), py_expr(L,La), py_expr(R,Ra)
    # terminates: the recursion KEY (the node id) strictly DESCENDS an EDB
    # relation (N → L,R via c_binary), so the invented value rides a VALUE
    # position and never drives the loop. Reject a generative cyclic SCC only
    # when it is NOT key-grounded by such a descent — the head has no untainted
    # EDB-bound key, or some recursive (in-SCC) atom fails to descend to a
    # DIFFERENT EDB-bound key. (Sound provided the descent EDB relations are
    # acyclic — always true for an AST; `max_inner` bounds any violation.)
    # A no-op for any program without Skolems.
    generative_rules = [
        rule
        for rule, w in program.inner.items()
        if w > 0 and any(type(a) is dlg.Skolem for a in rule[0][1])
    ]
    if generative_rules:
        tainted = _tainted_positions(program)
        scc_of = {p: frozenset(scc) for scc in sccs for p in scc}
        cyclic: set[str] = set()
        for scc in sccs:
            if len(scc) > 1 or next(iter(scc)) in all_edges.get(next(iter(scc)), ()):
                cyclic |= scc
        for rule in generative_rules:
            hp = rule[0][0]
            if hp not in cyclic:
                continue  # acyclic generative head — always terminating
            s = scc_of[hp]
            # vars bound at an untainted position by an out-of-SCC (EDB / lower
            # stratum) positive body atom — the EDB-grounded keys.
            edb_bound = {
                bt
                for batom in rule[1:]
                if not batom[0].startswith("!") and batom[0] not in s
                for j, bt in enumerate(batom[1])
                if dlg.is_variable(bt) and (batom[0], j) not in tainted
            }
            head_keys = {
                a
                for i, a in enumerate(rule[0][1])
                if dlg.is_variable(a) and (hp, i) not in tainted and a in edb_bound
            }
            if not head_keys:
                raise ValueError(
                    f"Program is not stratifiable: recursive term construction "
                    f"in cyclic SCC {sorted(s)} is not key-grounded — head {hp!r} "
                    f"has no untainted EDB-bound key to descend."
                )
            for batom in rule[1:]:
                if batom[0].startswith("!") or batom[0] not in s:
                    continue  # only recursive (in-SCC) atoms must descend
                descended = any(
                    dlg.is_variable(bt) and (batom[0], j) not in tainted and bt in edb_bound and bt not in head_keys
                    for j, bt in enumerate(batom[1])
                )
                if not descended:
                    raise ValueError(
                        f"Program is not stratifiable: recursive term construction "
                        f"in cyclic SCC {sorted(s)} is not key-grounded — recursive "
                        f"atom {batom[0]!r} does not descend to a new EDB-bound key "
                        f"(the invented value would drive the recursion)."
                    )


@dataclass(frozen=True)
class LiftLiftLiftJoin[A, B, C](LiftJoin[A, B, C]):
    """``↑↑↑J``. Pointwise lift of value-level join to a 3-D stream.
    ``Lift1`` / ``Lift2`` are arity-agnostic so this subclass exists
    only to document the intended lattice shape."""


@dataclass(frozen=True)
class LiftLiftLiftH[V](LiftH[V]):
    """``↑↑↑H``. Pointwise lift of value-level H to a 3-D stream."""


@dataclass(frozen=True)
class DeltaLiftedDeltaLiftedDeltaLiftedJoin[A, B, C](Operator):
    """8-term triply-incremental bilinear join on a 3-axis flat product
    lattice (outer = axis 0, stratum = axis 1, inner = axis 2).

    The 3-D delta of the cumulative join

        c(o, s, i) = J(Iᵒ Iˢ Iⁱ a, Iᵒ Iˢ Iⁱ b)(o, s, i)

    decomposes by iterated 1-D bilinearity (``ΔJ(X, Y) = J(ΔX, Y) +
    J(z⁻¹X, ΔY)`` applied once per axis) into a sum over subsets
    ``Sₐ ⊆ {o, s, i}``:

        Δ³ c  =  Σ_{Sₐ ⊆ {o, s, i}}  J( z⁻¹^Sₐ I^Sₐ a , I^(complement) b )

    where ``I^S = ∏_{k ∈ S} Iᵏ`` and ``z⁻¹^S = ∏_{k ∈ S} z⁻¹ᵏ``. Each
    axis ``k`` either goes to the ``a`` side (with ``z⁻¹ᵏ Iᵏ``
    applied) or to the ``b`` side (with ``Iᵏ`` applied, no delay).
    ``2³ = 8`` subsets, so 8 terms.

    Reduces to the 4-term 2-D ``DeltaLiftedDeltaLiftedJoin`` when the
    third axis is dropped — both decompositions compute the same Δ²
    of the cumulative join, just with a different but algebraically
    equivalent distribution of ``z⁻¹`` factors.

    ``inputs = (diff_a, diff_b)``. ``group_a`` / ``group_b`` are the
    Z-set groups for the two inputs (needed by the Integrate / Delay
    primitives). ``out_group`` is the group for the joined output.

    Connects as 14 (a-side) + 7 (b-side) + 8 (joins) + 7 (sums) = 36
    nodes."""

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
            raise ValueError(f"DeltaLiftedDeltaLiftedDeltaLiftedJoin takes 2 inputs; got {len(inputs)}")
        diff_a, diff_b = inputs

        # For every subset S ⊆ {0, 1, 2} build the nodes for
        # ``z⁻¹^S I^S diff_a`` and ``I^S diff_b``. Operators on
        # different axes commute so the build order doesn't matter.
        a_paths: dict[frozenset[int], NodeId] = {frozenset(): diff_a}
        b_paths: dict[frozenset[int], NodeId] = {frozenset(): diff_b}
        for axis in (0, 1, 2):
            new_a: dict[frozenset[int], NodeId] = {}
            for subset, node in a_paths.items():
                integrated = CoreIntegrate[ZSet[A]](axis=axis, group=self.group_a).connect(circuit, (node,))
                delayed = CoreDelay[ZSet[A]](axis=axis, group=self.group_a).connect(circuit, (integrated,))
                new_a[subset | {axis}] = delayed
            a_paths.update(new_a)
            new_b: dict[frozenset[int], NodeId] = {}
            for subset, node in b_paths.items():
                integrated = CoreIntegrate[ZSet[B]](axis=axis, group=self.group_b).connect(circuit, (node,))
                new_b[subset | {axis}] = integrated
            b_paths.update(new_b)

        # Each of the 8 terms takes one subset of axes on the a-side
        # and the complement on the b-side.
        all_axes = frozenset({0, 1, 2})
        lifted_join = LiftLiftLiftJoin[A, B, C](pred=self.pred, proj=self.proj)
        terms: list[NodeId] = []
        for sa_axes in (
            frozenset(),
            frozenset({0}),
            frozenset({1}),
            frozenset({2}),
            frozenset({0, 1}),
            frozenset({0, 2}),
            frozenset({1, 2}),
            frozenset({0, 1, 2}),
        ):
            sb_axes = all_axes - sa_axes
            terms.append(lifted_join.connect(circuit, (a_paths[sa_axes], b_paths[sb_axes])))

        add = self.out_group.add
        result = terms[0]
        for t in terms[1:]:
            result = Lift2[ZSet[C], ZSet[C], ZSet[C]](op=add).connect(circuit, (result, t))
        return result


# ---- 3-D distinct (triply-incremental H-based distinct) --------------------


@dataclass(frozen=True)
class DeltaLiftedDeltaLiftedDeltaLiftedDistinct[V](Operator):
    """Triply-incremental distinct on a 3-axis lattice:

        Dᵒ ( Dˢ ( H( z⁻¹ⁱ Iᵒ Iˢ Iⁱ s,  Iᵒ Iˢ s ) ) )

    Mirrors the 2-D ``Dᵒ(H(z⁻¹ⁱ Iⁱ Iᵒ s, Iᵒ s))`` with one extra
    stratum-axis integrate before the threshold and one extra outer
    differentiate after, so the result is the full triple-delta of
    distinct over the cumulative.

    H pointwise on a 3-D stream is just ``Lift2`` of the value-level
    H (``LiftLiftLiftH``).

    ``inputs = (diff_stream,)``. ``inner_group`` is the Z-set group
    for the stream's element type.

    Connects as 4 Integrate / Delay primitives + 1 H + 2
    Differentiate composites (3 nodes each) = 11 nodes."""

    inner_group: ZSetAddition[V]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        (diff_stream,) = inputs
        int_o = CoreIntegrate[ZSet[V]](axis=0, group=self.inner_group).connect(circuit, (diff_stream,))
        int_os = CoreIntegrate[ZSet[V]](axis=1, group=self.inner_group).connect(circuit, (int_o,))
        int_osi = CoreIntegrate[ZSet[V]](axis=2, group=self.inner_group).connect(circuit, (int_os,))
        del_int_osi = CoreDelay[ZSet[V]](axis=2, group=self.inner_group).connect(circuit, (int_osi,))
        h = LiftLiftLiftH[V]().connect(circuit, (del_int_osi, int_os))
        diff_s = CoreDifferentiate[ZSet[V]](axis=1, group=self.inner_group).connect(circuit, (h,))
        return CoreDifferentiate[ZSet[V]](axis=0, group=self.inner_group).connect(circuit, (diff_s,))


# ---- 3-D negation Datalog body --------------------------------------------
#
# The 3-D body for stratified, negation-aware Datalog. The body itself
# carries no stratification logic; :class:`IncrementalDatalogStratified`
# sequences rule pushes on the 3-D lattice to enforce stratum
# ordering.


@dataclass(frozen=True)
class IncrementalDatalogBodyWithNegation(Operator):
    """Body of a negation-aware Datalog interpreter on the
    ``(outer, stratum, inner)`` lattice. Naïve mechanical lift of
    :class:`pydbsp.relational_operators.IncrementalDatalogWithNegationBody`:
    every 2-D operator in the 2-D body has a corresponding 3-D
    operator here. The body carries no stratification logic; the
    stratified driver enforces stratum ordering by sequencing rule
    pushes on the 3-D lattice. See the module docstring for the
    operator-mapping table.

    ``inputs = (edb_1d, program_3d, state_facts_3d, state_rewrites_3d,
    seed_1d)``.

    * ``edb_1d`` (``ZSet[Fact]``, 1-D): pushed at outer ticks.
    * ``program_3d`` (``ZSet[Rule]``, 3-D): rule stream on the 3-D
      lattice. For the simple use case (single batch, single
      stratum), wrap a 1-D ``Input`` in two ``LiftStreamIntroduction``
      calls. For the stratified use case, push rule deltas directly
      at ``(outer, stratum, inner_cursor)`` cells.
    * ``state_facts_3d`` / ``state_rewrites_3d``: 3-D ``Input`` nodes.
      The caller pushes the body's own diffs back as state at
      ``(outer, stratum, inner)``.
    * ``seed_1d`` (``ZSet[ProvenanceIndexedRewrite]``, 1-D): rewrite
      seed, pushed once at outer 0.

    Output is a single ``NodeId`` whose value at each
    ``(outer, stratum, inner)`` cell is the pair ``(facts_diff,
    rewrites_diff)``.

    SPLIT NEGATION PORT (optional 6th input ``neg_ref_3d``): when given,
    the negation test (``anti_product``) reads THIS input through a
    genuine ``z⁻¹`` on the MIDDLE axis (``CoreDelay(axis=1)``) instead of
    the moving axis-2-delayed facts state. The reference is therefore (a)
    one alternation tick old and (b) FROZEN along the inner axis (nothing
    is pushed at inner coordinates > 0; the anti-join's internal
    integrals hold it constant). This is the uniformly-lifted
    semipositive circuit of the alternating-fixpoint evaluator
    (:class:`DatalogAlternating`): the middle axis is the alternation
    axis, and the ONLY new operator relative to the mechanical lift is
    this single delay edge. With 5 inputs, behavior is unchanged."""

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
                f"IncrementalDatalogBodyWithNegation takes 5 inputs "
                f"(edb, program, state_facts, state_rewrites, seed) or 6 "
                f"(…, neg_ref); got {len(inputs)}"
            )
        edb_1d, program_3d, state_facts_3d, state_rewrites_3d, seed_1d = inputs[:5]
        neg_ref_3d = inputs[5] if len(inputs) == 6 else None

        edb_2d_step = LiftStreamIntroduction[ZSet[dlg.Fact]](group=self.fact_group).connect(circuit, (edb_1d,))
        edb_3d = LiftStreamIntroduction[ZSet[dlg.Fact]](group=self.fact_group).connect(circuit, (edb_2d_step,))
        seed_2d_step = LiftStreamIntroduction[ZSet[dlg.ProvenanceIndexedRewrite]](group=self.rewrite_group).connect(
            circuit, (seed_1d,)
        )
        seed_3d = LiftStreamIntroduction[ZSet[dlg.ProvenanceIndexedRewrite]](group=self.rewrite_group).connect(
            circuit, (seed_2d_step,)
        )

        delayed_facts = CoreDelay[ZSet[dlg.Fact]](axis=2, group=self.fact_group).connect(circuit, (state_facts_3d,))
        delayed_rewrites = CoreDelay[ZSet[dlg.ProvenanceIndexedRewrite]](axis=2, group=self.rewrite_group).connect(
            circuit, (state_rewrites_3d,)
        )

        # Negation reference: with the split port, a real z⁻¹ on the ALTERNATION
        # axis — the one new operator of the alternating-fixpoint circuit.
        neg_ref_wire = (
            CoreDelay[ZSet[dlg.Fact]](axis=1, group=self.fact_group).connect(circuit, (neg_ref_3d,))
            if neg_ref_3d is not None
            else delayed_facts
        )

        sig_3d = Lift1[ZSet[dlg.Rule], ZSet[dlg.Signal]](f=dlg.sig).connect(circuit, (program_3d,))
        dir_3d = Lift1[ZSet[dlg.Rule], ZSet[dlg.Direction]](f=dlg.dir).connect(circuit, (program_3d,))

        gatekeep = DeltaLiftedDeltaLiftedDeltaLiftedJoin[
            dlg.ProvenanceIndexedRewrite,
            dlg.Direction,
            dlg.AtomWithSourceRewriteAndProvenance,
        ](
            pred=lambda left, right: left[0] == right[0],
            proj=lambda left, right: (right[1], right[2], left[1]),
            group_a=self.rewrite_group,
            group_b=self.dir_group,
            out_group=self.gatekeep_group,
        ).connect(circuit, (delayed_rewrites, dir_3d))

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
        # interpreted (built-in) atoms — a stateless guard on the rewrite: keep the
        # gatekeep entries whose decidable predicate holds under their (ground, by
        # safety) bindings, projected to the unchanged rewrite. No facts-join, no delay.
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

        product = DeltaLiftedDeltaLiftedDeltaLiftedJoin[
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

        anti_product = DeltaLiftedDeltaLiftedDeltaLiftedJoin[
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

        # final_product = product + proj − anti_product
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
        # final_product = product + proj − anti_product + interp_proj
        final_product = Lift2[
            ZSet[dlg.ProvenanceIndexedRewrite],
            ZSet[dlg.ProvenanceIndexedRewrite],
            ZSet[dlg.ProvenanceIndexedRewrite],
        ](op=self.rewrite_group.add).connect(circuit, (final_product_neg, interp_proj))

        ground = DeltaLiftedDeltaLiftedDeltaLiftedJoin[
            dlg.ProvenanceIndexedRewrite,
            dlg.Signal,
            dlg.Fact,
        ](
            pred=lambda left, right: left[0] == right[0],
            proj=lambda left, right: left[1].apply(right[1]),
            group_a=self.rewrite_group,
            group_b=self.signal_group,
            out_group=self.fact_group,
        ).connect(circuit, (final_product, sig_3d))

        ground_plus_edb = Lift2[ZSet[dlg.Fact], ZSet[dlg.Fact], ZSet[dlg.Fact]](op=self.fact_group.add).connect(
            circuit, (ground, edb_3d)
        )
        next_facts = DeltaLiftedDeltaLiftedDeltaLiftedDistinct[dlg.Fact](inner_group=self.fact_group).connect(
            circuit, (ground_plus_edb,)
        )

        final_plus_seed = Lift2[
            ZSet[dlg.ProvenanceIndexedRewrite],
            ZSet[dlg.ProvenanceIndexedRewrite],
            ZSet[dlg.ProvenanceIndexedRewrite],
        ](op=self.rewrite_group.add).connect(circuit, (final_product, seed_3d))
        next_rewrites = DeltaLiftedDeltaLiftedDeltaLiftedDistinct[dlg.ProvenanceIndexedRewrite](
            inner_group=self.rewrite_group
        ).connect(circuit, (final_plus_seed,))

        return Lift2[
            ZSet[dlg.Fact],
            ZSet[dlg.ProvenanceIndexedRewrite],
            tuple[ZSet[dlg.Fact], ZSet[dlg.ProvenanceIndexedRewrite]],
        ](op=lambda a, b: (a, b)).connect(circuit, (next_facts, next_rewrites))


def _frontier_max(frontier: Antichain[Time], axis: int) -> int:
    """Largest coordinate on ``axis`` covered anywhere by ``frontier``'s
    down-set (= the max over its maximal elements)."""
    if frontier.is_universal:
        return -1
    out = -1
    for elt in frontier.elements:
        if len(elt) < 3:
            continue
        v = elt[axis]
        if isinstance(v, int) and v > out:
            out = v
    return out


class IncrementalDatalogStratified:
    """``stratSpec`` of ``V3/StratifiedCircuit.lean``, on the
    ``(outer, stratum, inner)`` lattice::

        stratSpec = integral ; prepInject ; feedback2 stratumBody

    Three things make this the Lean circuit rather than merely an evaluator
    that agrees with it.

    *Keyed introduction.* ``placedRules S k`` is pushed at stratum cell ``k``:
    a rule lives at the cell whose coordinate equals its head's level. No rule
    is rewritten, and no control fact exists -- stratum ordering is the wiring.

    *Stratum delay.* ``eCell = delta0 E [k] + z^-1 cand [k]``. The base facts
    reach ``(t, 0, 0)`` through the body's own introductions, and cell ``k``
    opens on the summed store of cell ``k - 1``.

    *Pointwise stratum lift.* The body is built at ``inner_axis=2``, so every
    integrate and delay runs on axes 0 and 2 and nothing touches axis 1. The
    join is the 4-term form lifted pointwise over the strata -- three lifts,
    two incrementalizations -- not the 8-term join that incrementalizes all
    three axes and lets strata mix.

    ``stage(k)`` exposes the ``k``-th stage, which is what
    ``stratSpec_computes`` is a statement about; ``saturate`` returns the
    running cumulative, and every cell at or above the top level repeats the
    full stratified model (``stratSpec_computes_full``).

    No compaction is used: storage retains every cell across batches.
    """

    _indexed: bool = False

    def __init__(self, *, max_inner: int = 1024) -> None:
        self._max_inner = max_inner
        self._fact_group = ZSetAddition[dlg.Fact]()
        self._rule_group = ZSetAddition[dlg.Rule]()
        self._rewrite_group = ZSetAddition[dlg.ProvenanceIndexedRewrite]()
        self._cumulative_rules: ZSet[dlg.Rule] = ZSet({})
        self._pending_rules: ZSet[dlg.Rule] = ZSet({})
        self._pending_edb: ZSet[dlg.Fact] = ZSet({})
        self._cumulative_facts: ZSet[dlg.Fact] = ZSet({})
        self._stages: list[ZSet[dlg.Fact]] = []
        self._derived: list[ZSet[dlg.Fact]] = []
        self._depth: list[int] = []
        self._placed: list[ZSet[dlg.Rule]] = []
        self._opened: list[ZSet[dlg.Fact]] = []
        self._next_outer = 0
        self._e: Evaluator[Time] | None = None
        self._h: tuple = ()

    # -- inputs ------------------------------------------------------------
    def push_rules(self, rules: ZSet[dlg.Rule]) -> None:
        merged = self._rule_group.add(
            self._cumulative_rules, self._rule_group.add(self._pending_rules, rules)
        )
        _check_stratifiable(merged)
        dlg.check_interpreted_safety(merged)
        self._pending_rules = self._rule_group.add(self._pending_rules, rules)

    def push_facts(self, facts: ZSet[dlg.Fact]) -> None:
        self._pending_edb = self._fact_group.add(self._pending_edb, facts)

    # -- the circuit -------------------------------------------------------
    def _build(self) -> None:
        e = Evaluator(
            circuit=Circuit(), storage=DictStorage(),
            ctx=ComputeCtx(lattice=dbsp_time(3)), group=self._fact_group,
        )

        def inp(axes: int) -> NodeId:
            return Input(frontier=Antichain(dbsp_time(axes))).connect(e.circuit, ())

        edb_1d, seed_1d = inp(1), inp(1)
        program_3d, facts_3d, rewrites_3d = inp(3), inp(3), inp(3)
        fg, rug, rwg = self._fact_group, self._rule_group, self._rewrite_group
        if self._indexed:
            body_op = IndexedIncrementalDatalogWithNegationBody(
                fact_group=fg,
                rule_group=rug,
                rewrite_group=rwg,
                signal_group=ZSetAddition(),
                ext_dir_group=ZSetAddition(),
                jorder_group=ZSetAddition(),
                gatekeep_group=ZSetAddition(),
                indexed_fact_group=ZSetAddition(),
                inner_axis=2,
            )
        else:
            body_op = IncrementalDatalogWithNegationBody(
                fact_group=fg,
                rule_group=rug,
                rewrite_group=rwg,
                signal_group=ZSetAddition(),
                dir_group=ZSetAddition(),
                gatekeep_group=ZSetAddition(),
                inner_axis=2,
            )
        body = body_op.connect(e.circuit, (edb_1d, program_3d, facts_3d, rewrites_3d, seed_1d))
        e.push(seed_1d, ZSet({(0, dlg._rewrite_monoid.identity()): 1}))
        self._e = e
        self._h = (edb_1d, program_3d, facts_3d, rewrites_3d, body)

    # -- the stratum loop --------------------------------------------------
    def saturate(self) -> ZSet[dlg.Fact]:
        if self._e is None:
            self._build()
        e = self._e
        assert e is not None
        edb_1d, program_3d, facts_3d, rewrites_3d, body = self._h
        fg = self._fact_group

        self._cumulative_rules = self._rule_group.add(self._cumulative_rules, self._pending_rules)
        self._pending_rules = ZSet({})
        outer, self._next_outer = self._next_outer, self._next_outer + 1
        if self._pending_edb.inner:
            e.push(edb_1d, self._pending_edb, t=(outer,))
            self._pending_edb = ZSet({})

        rules = {r: w for r, w in self._cumulative_rules.inner.items() if w > 0}
        lv = stratum_levels(ZSet(rules))
        top = max(lv.values(), default=0)

        seed = ZSet({(0, dlg._rewrite_monoid.identity()): 1})
        stages: list[ZSet[dlg.Fact]] = []
        cand = fg.identity()
        for k in range(top + 1):
            first_visit = k >= len(self._depth)
            # keyed introduction: the rules of level k belong at cell k. The
            # body integrates on axis 0, so push the DELTA against whatever
            # this cell already holds -- a rule that changes level is a
            # retraction at the old cell and an insertion at the new one.
            placed = ZSet({r: w for r, w in rules.items() if lv[r[0][0]] == k})
            prior = self._placed[k] if k < len(self._placed) else ZSet({})
            delta_rules = self._rule_group.add(placed, self._rule_group.neg(prior))
            if delta_rules.inner:
                e.push(program_3d, delta_rules, t=(outer, k, 0))
            while len(self._placed) <= k:
                self._placed.append(ZSet({}))
            self._placed[k] = placed

            if k > 0:
                # stratum delay: cell k opens on the summed store of cell
                # k - 1, again as a delta against what it opened on before.
                prior_open = self._opened[k] if k < len(self._opened) else fg.identity()
                delta_open = fg.add(cand, fg.neg(prior_open))
                if delta_open.inner:
                    e.push(facts_3d, delta_open, t=(outer, k, 0))
                while len(self._opened) <= k:
                    self._opened.append(fg.identity())
                self._opened[k] = cand
                if first_visit:
                    e.push(rewrites_3d, seed, t=(outer, k, 0))

            # An empty cell certifies the fixed point only past the depth
            # earlier pushes reached: a doubly incremental operator reads the
            # outer sum at its own depth, so below that an empty cell means
            # "nothing new here", not "converged".
            # Track actual support, not the empty tick that certifies it.
            # The feedback delay can expose that support one tick later.
            deepest = self._depth[k] if not first_visit else -1
            if k > 0 and (delta_open.inner or first_visit):
                deepest = max(deepest, 0)
            floor = deepest + 1
            total = fg.identity()
            j = 0 if k == 0 else 1
            while j < self._max_inner:
                df, dr = e.read(body, (outer, k, j))
                if not df.inner and not dr.inner and j > floor:
                    break
                if df.inner or dr.inner:
                    deepest = max(deepest, j)
                    total = fg.add(total, df)
                    if df.inner:
                        e.push(facts_3d, df, t=(outer, k, j))
                    if dr.inner:
                        e.push(rewrites_3d, dr, t=(outer, k, j))
                j += 1
            else:
                raise RuntimeError(f"stratum {k} did not converge in {self._max_inner} inner ticks")

            # Two accumulations meet here and must not be confused. Along the
            # stratum axis, cell k's store is what it inherited from k - 1 plus
            # what k derives. Along the push axis, what k derives accumulates
            # over outer ticks, because the body emits deltas. So the
            # elimination on the iteration axis feeds the per-cell running
            # total, and the stratum store is the inherited store plus that.
            while len(self._depth) <= k:
                self._depth.append(-1)
            self._depth[k] = deepest
            prev_derived = self._derived[k] if k < len(self._derived) else fg.identity()
            derived = fg.add(prev_derived, total)
            while len(self._derived) <= k:
                self._derived.append(fg.identity())
            self._derived[k] = derived
            cand = fg.add(cand, derived)
            stages.append(cand)

        self._stages = stages
        self._cumulative_facts = cand
        return self._cumulative_facts

    def stage(self, k: int) -> ZSet[dlg.Fact]:
        """The ``k``-th stage. Cells at or above the top level repeat the
        full stratified model, so the list is read saturating."""
        if not self._stages:
            self.saturate()
        return self._stages[min(k, len(self._stages) - 1)]


class IndexedIncrementalDatalogStratified(IncrementalDatalogStratified):
    """Drop-in for :class:`IncrementalDatalogStratified` whose body uses the
    SORT-MERGE-indexed joins
    (:class:`~pydbsp.indexed_relational_operators.IndexedIncrementalDatalogWithNegationBody`
    at ``inner_axis=2``). Identical API, identical results — only the
    per-saturation join cost differs (O(matching keys) vs O(|left|·|right|)),
    so it scales to large dense EDBs."""

    _indexed = True


class IndexedDatalog:
    """Persistent 2-D indexed positive Datalog driver.

    Each ``saturate()`` advances one outer batch, feeds pending rule/fact
    deltas through the same circuit, and returns the full cumulative model.
    Inputs must have set-valued cumulative contents and rules must be safe.
    The inner scan revisits earlier coordinates while retaining all history,
    including the deepest support from previous batches. No compaction is used.
    """

    _body_type: type[IndexedIncrementalDatalogBody] | type[IndexedIncrementalDatalogWithNegationBody] = (
        IndexedIncrementalDatalogBody
    )

    def __init__(self, *, max_inner: int = 8192) -> None:
        if max_inner <= 0:
            raise ValueError("max_inner must be positive")
        self._max_inner = max_inner
        self._rule_group = ZSetAddition[dlg.Rule]()
        self._fact_group = ZSetAddition[dlg.Fact]()
        self._rules: ZSet[dlg.Rule] = ZSet({})
        self._pending_rules: ZSet[dlg.Rule] = ZSet({})
        self._pending_facts: ZSet[dlg.Fact] = ZSet({})
        self._model: ZSet[dlg.Fact] = ZSet({})
        self._next_outer = 0
        self._deepest_inner = -1
        self._failed = False
        self._e: Evaluator[Time] | None = None

    def _validate_rules(self, rules: ZSet[dlg.Rule]) -> None:
        if any(pred.startswith("!") or pred in dlg.INTERPRETED for r in rules.inner for pred, _ in r[1:]):
            raise ValueError("Use IndexedDatalogWithNegation for negation or interpreted predicates")
        _check_stratifiable(rules)

    def push_rules(self, rules: ZSet[dlg.Rule]) -> None:
        merged = self._rule_group.add(self._rules, rules)
        self._validate_rules(merged)
        self._rules = merged
        self._pending_rules = self._rule_group.add(self._pending_rules, rules)

    def push_facts(self, facts: ZSet[dlg.Fact]) -> None:
        self._pending_facts = self._fact_group.add(self._pending_facts, facts)

    def _setup_circuit(self) -> None:
        fg, rug = self._fact_group, self._rule_group
        rg = ZSetAddition[dlg.ProvenanceIndexedRewrite]()
        e = Evaluator[Time](circuit=Circuit[Time](), storage=DictStorage(), ctx=ComputeCtx(lattice=dbsp_time(2)), group=fg)
        edb_1d = Input[ZSet[dlg.Fact]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
        program_1d = Input[ZSet[dlg.Rule]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
        sf = Input[ZSet[dlg.Fact]](frontier=Antichain(dbsp_time(2))).connect(e.circuit, ())
        sr = Input[ZSet[dlg.ProvenanceIndexedRewrite]](frontier=Antichain(dbsp_time(2))).connect(e.circuit, ())
        seed = Input[ZSet[dlg.ProvenanceIndexedRewrite]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
        program_2d = LiftStreamIntroduction[ZSet[dlg.Rule]](group=rug).connect(e.circuit, (program_1d,))
        body = self._body_type(
            fact_group=fg, rule_group=rug, rewrite_group=rg, signal_group=ZSetAddition[dlg.Signal](),
            ext_dir_group=ZSetAddition[dlg.ExtendedDirection](), jorder_group=ZSetAddition[tuple[str, dlg.ColumnReference]](),
            gatekeep_group=ZSetAddition[dlg.IndexedGatekeepEntry](), indexed_fact_group=ZSetAddition[dlg.IndexedFact](),
        ).connect(e.circuit, (edb_1d, program_2d, sf, sr, seed))
        e.push(seed, ZSet({(0, dlg._rewrite_monoid.identity()): 1}), t=(0,))
        self._e = e
        self._edb, self._program = edb_1d, program_1d
        self._sf, self._sr, self._body = sf, sr, body

    def saturate(self) -> ZSet[dlg.Fact]:
        if self._failed:
            raise RuntimeError("The previous batch did not converge; create a new driver")
        if self._e is None:
            self._setup_circuit()
        e = self._e
        assert e is not None
        outer = self._next_outer
        e.push(self._edb, self._pending_facts, t=(outer,))
        e.push(self._program, self._pending_rules, t=(outer,))
        self._pending_facts, self._pending_rules = ZSet({}), ZSet({})
        # Prior fact AND rewrite support can arrive late through outer integrals.
        # Include the entry delay before accepting an empty inner cell.
        floor = self._deepest_inner + 1
        batch = self._fact_group.identity()
        for k in range(self._max_inner):
            df, dr = e.read(self._body, (outer, k))
            if not df.inner and not dr.inner:
                if k > floor:
                    break
            else:
                self._deepest_inner = max(self._deepest_inner, k)
                batch = self._fact_group.add(batch, df)
                if df.inner:
                    e.push(self._sf, df, t=(outer, k))
                if dr.inner:
                    e.push(self._sr, dr, t=(outer, k))
        else:
            self._failed = True
            raise RuntimeError(f"outer {outer} did not converge within {self._max_inner} inner ticks")
        self._model = self._fact_group.add(self._model, batch)
        self._next_outer += 1
        return _without_internal_facts(self._model)


class IndexedDatalogWithNegation(IndexedDatalog):
    """The same persistent 2-D driver with the indexed negation body.

    Correctness requires safe rules and a cumulative program tight below
    negation at each batch. Stratifiability and interpreted-predicate safety
    are checked here; tightness and negation safety remain caller obligations.
    This is not a general stratified-negation evaluator.
    """

    _body_type = IndexedIncrementalDatalogWithNegationBody

    def _validate_rules(self, rules: ZSet[dlg.Rule]) -> None:
        _check_stratifiable(rules)
        dlg.check_interpreted_safety(rules)


class DatalogWithNegation:
    """Single-batch 2-D (outer × inner) NON-INDEXED Datalog WITH NEGATION — the
    non-indexed counterpart of :class:`IndexedDatalogWithNegation`, driving the plain
    :class:`~pydbsp.relational_operators.IncrementalDatalogWithNegationBody` (``pred``-based
    joins, no sort-merge index). Same ``push_rules`` / ``push_facts`` / ``saturate`` API and
    tight-below-negation correctness domain, but rebuilds on each saturation.
    Used to validate the non-indexed body against the indexed one."""

    def __init__(self, *, max_inner: int = 8192) -> None:
        self._max_inner = max_inner
        self._rule_group = ZSetAddition[dlg.Rule]()
        self._fact_group = ZSetAddition[dlg.Fact]()
        self._rules: ZSet[dlg.Rule] = ZSet({})
        self._facts: ZSet[dlg.Fact] = ZSet({})

    def push_rules(self, rules: ZSet[dlg.Rule]) -> None:
        merged = self._rule_group.add(self._rules, rules)
        _check_stratifiable(merged)
        dlg.check_interpreted_safety(merged)
        self._rules = merged

    def push_facts(self, facts: ZSet[dlg.Fact]) -> None:
        self._facts = self._fact_group.add(self._facts, facts)

    def saturate(self) -> ZSet[dlg.Fact]:
        fg, rug = self._fact_group, self._rule_group
        rg = ZSetAddition[dlg.ProvenanceIndexedRewrite]()
        e = Evaluator[Time](circuit=Circuit[Time](), storage=DictStorage(), ctx=ComputeCtx(lattice=dbsp_time(2)), group=fg)
        edb_1d = Input[ZSet[dlg.Fact]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
        program_1d = Input[ZSet[dlg.Rule]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
        sf = Input[ZSet[dlg.Fact]](frontier=Antichain(dbsp_time(2))).connect(e.circuit, ())
        sr = Input[ZSet[dlg.ProvenanceIndexedRewrite]](frontier=Antichain(dbsp_time(2))).connect(e.circuit, ())
        seed = Input[ZSet[dlg.ProvenanceIndexedRewrite]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
        program_2d = LiftStreamIntroduction[ZSet[dlg.Rule]](group=rug).connect(e.circuit, (program_1d,))
        body = IncrementalDatalogWithNegationBody(
            fact_group=fg, rule_group=rug, rewrite_group=rg, signal_group=ZSetAddition[dlg.Signal](),
            dir_group=ZSetAddition[dlg.Direction](),
            gatekeep_group=ZSetAddition[dlg.AtomWithSourceRewriteAndProvenance](),
        ).connect(e.circuit, (edb_1d, program_2d, sf, sr, seed))
        e.push(edb_1d, self._facts)
        e.push(program_1d, self._rules)
        e.push(seed, ZSet({(0, dlg._rewrite_monoid.identity()): 1}))
        acc: dict[dlg.Fact, int] = {}
        for k in range(self._max_inner):
            diff_facts, diff_rewrites = e.read(body, (0, k))
            if not diff_facts.inner and not diff_rewrites.inner:
                break
            for f, m in diff_facts.inner.items():
                acc[f] = acc.get(f, 0) + m
            e.push(sf, diff_facts, t=(0, k))
            e.push(sr, diff_rewrites, t=(0, k))
        return _without_internal_facts(ZSet({f: m for f, m in acc.items() if m != 0}))


def stratum_levels(rules: ZSet[dlg.Rule]) -> dict[str, int]:
    """Least level assignment over the cumulative program:
    ``level(head) >= level(positive derived dep)`` and
    ``level(head) >= level(negated DERIVED dep) + 1``. EDB and interpreted
    predicates are level 0 and impose no constraints. Terminates because the
    program is stratifiable (validated at push time)."""
    derived = {rule[0][0] for rule in rules.inner}
    lv: dict[str, int] = {p: 0 for p in derived}
    changed = True
    while changed:
        changed = False
        for rule in rules.inner:
            head = rule[0][0]
            req = 0
            for pred, _args in rule[1:]:
                if pred.startswith("!"):
                    q = pred.lstrip("!")
                    if q in derived:
                        req = max(req, lv[q] + 1)
                elif pred in derived:
                    req = max(req, lv[pred])
            if req > lv[head]:
                lv[head] = req
                changed = True
    return lv


class DatalogStratifiedComposition:
    """Stratified-negation evaluation by PER-STRATUM COMPOSITION of the 2-D
    semipositive circuit: stratification is realized as rule PLACEMENT on the
    stratum axis, not as in-circuit control flow.

    ``F(s) = B(rules placed at s, E ∪ F(s-1))``, ``F(-1) = ∅``, where ``B``
    is the plain 2-D negation body (:class:`DatalogWithNegation`). Each
    stratum gets a fresh inner axis; all cross-stratum flow goes through the
    "saturated snapshot forwarded as EDB" bridge, so every negated predicate
    is FINAL in the consuming stage's EDB by construction — no in-flight
    retraction can occur at any negation depth, and positive cycles anywhere
    are harmless (ghost-free by construction).

    Replaces the level-subgraph / stratum_ready-guard / shared-cursor control
    plane of :class:`IncrementalDatalogStratified`, whose scheduling misses
    stratum-axis diffs below the shared inner cursor and computes wrong models
    at negation depth >= 2. Differentially validated against Soufflé:
    0/270 divergent fuzzed programs (the guard-based driver: 11-43 per round).

    Same single-batch ``push_rules`` / ``push_facts`` / ``saturate`` API as
    the 2-D drivers."""

    def __init__(self) -> None:
        self._group = ZSetAddition[dlg.Fact]()
        self._rule_group = ZSetAddition[dlg.Rule]()
        self._rules: ZSet[dlg.Rule] = ZSet({})
        self._facts: ZSet[dlg.Fact] = ZSet({})

    def push_rules(self, rules: ZSet[dlg.Rule]) -> None:
        merged = self._rule_group.add(self._rules, rules)
        _check_stratifiable(merged)
        dlg.check_interpreted_safety(merged)
        self._rules = merged

    def push_facts(self, facts: ZSet[dlg.Fact]) -> None:
        self._facts = self._group.add(self._facts, facts)

    def saturate(self) -> ZSet[dlg.Fact]:
        lv = stratum_levels(self._rules)
        top = max(lv.values(), default=0)
        facts = self._facts
        for s in range(top + 1):
            placed = ZSet({r: w for r, w in self._rules.inner.items() if lv[r[0][0]] == s})
            if not placed.inner:
                continue
            stage = DatalogWithNegation()
            stage.push_rules(placed)
            stage.push_facts(facts)  # E ∪ F(s-1): lower strata are this stage's EDB
            facts = stage.saturate()
        return facts


class DatalogAlternating:
    """Stratified-negation evaluation as the ALTERNATING FIXPOINT (Van Gelder)
    of the semipositive circuit, as ONE physical 3-D circuit on the
    ``(outer, alternation, inner)`` lattice (``dbsp_time(3)``):

    * the body is the uniformly-lifted semipositive negation body
      (:class:`IncrementalDatalogBodyWithNegation` — 8-term joins, 3-D
      distinct), with its split negation-reference port;
    * the ONLY new operator is the alternation feedback edge: the model's
      axis-1 deltas, read by the anti-join through a genuine
      ``CoreDelay(axis=1)`` (``z⁻¹`` on the middle axis);
    * program, EDB and seed are pushed ONCE at outer tick 0 — no
      ``stratum_levels``, no rule placement, no guards, no per-stratum
      stages. Stratifiability is a correctness HYPOTHESIS (validated at
      push time), never a mechanism.

    Semantics of the alternation axis: tick ``a`` computes
    ``M_a = Γ(M_{a-1})`` where ``Γ(N)`` is the semipositive fixpoint with
    every negation evaluated against the FROZEN reference ``N``
    (``M_{-1} = ∅``). The freeze is structural, not scheduled: the reference
    enters through ``z⁻¹`` on axis 1 and nothing is pushed at inner
    coordinates > 0, so within a tick the anti-join's integrals hold it
    constant — negation can never read a wire that moves during the inner
    fixpoint (the guard-based driver's fatal flaw). ``Γ`` is antitone, so
    even ticks over-approximate, odd ticks under-approximate, and for
    stratifiable programs both limits meet at the perfect model in at most
    ``max stratum + 2`` ticks; transient over-derivations are retracted by
    NEGATIVE WEIGHTS flowing through the joins' cross-axis delta terms.
    Convergence is detected as physical quiescence: an alternation tick all
    of whose cells read zero (the ``fix_unique`` criterion — zero delta at
    a strict-feedback fixpoint is stable forever).

    Same single-batch ``push_rules`` / ``push_facts`` / ``saturate`` API as
    the other drivers."""

    def __init__(self, *, max_inner: int = 8192, max_alternation: int = 64) -> None:
        self._max_inner = max_inner
        self._max_alternation = max_alternation
        self._rule_group = ZSetAddition[dlg.Rule]()
        self._fact_group = ZSetAddition[dlg.Fact]()
        self._rules: ZSet[dlg.Rule] = ZSet({})
        self._facts: ZSet[dlg.Fact] = ZSet({})

    def push_rules(self, rules: ZSet[dlg.Rule]) -> None:
        merged = self._rule_group.add(self._rules, rules)
        _check_stratifiable(merged)  # hypothesis, not mechanism: levels are never computed
        dlg.check_interpreted_safety(merged)
        self._rules = merged

    def push_facts(self, facts: ZSet[dlg.Fact]) -> None:
        self._facts = self._fact_group.add(self._facts, facts)

    def saturate(self) -> ZSet[dlg.Fact]:
        fg, rug = self._fact_group, self._rule_group
        rg = ZSetAddition[dlg.ProvenanceIndexedRewrite]()
        e = Evaluator[Time](
            circuit=Circuit[Time](), storage=DictStorage(), ctx=ComputeCtx(lattice=dbsp_time(3)), group=fg
        )
        edb_1d = Input[ZSet[dlg.Fact]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
        program_1d = Input[ZSet[dlg.Rule]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
        sf = Input[ZSet[dlg.Fact]](frontier=Antichain(dbsp_time(3))).connect(e.circuit, ())
        sr = Input[ZSet[dlg.ProvenanceIndexedRewrite]](frontier=Antichain(dbsp_time(3))).connect(e.circuit, ())
        neg_ref = Input[ZSet[dlg.Fact]](frontier=Antichain(dbsp_time(3))).connect(e.circuit, ())
        seed = Input[ZSet[dlg.ProvenanceIndexedRewrite]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
        program_2d = LiftStreamIntroduction[ZSet[dlg.Rule]](group=rug).connect(e.circuit, (program_1d,))
        program_3d = LiftStreamIntroduction[ZSet[dlg.Rule]](group=rug).connect(e.circuit, (program_2d,))
        body = IncrementalDatalogBodyWithNegation(
            fact_group=fg, rule_group=rug, rewrite_group=rg, signal_group=ZSetAddition[dlg.Signal](),
            dir_group=ZSetAddition[dlg.Direction](),
            gatekeep_group=ZSetAddition[dlg.AtomWithSourceRewriteAndProvenance](),
        ).connect(e.circuit, (edb_1d, program_3d, sf, sr, seed, neg_ref))

        # Value plane, all in-circuit (elementary axis-generic operators on the
        # body's facts output Δ³C): I² gives D⁰D¹C — at outer 0, the tick's model
        # delta M_a − M_{a-1} (the alternation feedback value AND the convergence
        # signal); I¹ of that gives D⁰C — the model itself.
        facts_out = Lift1[
            tuple[ZSet[dlg.Fact], ZSet[dlg.ProvenanceIndexedRewrite]],
            ZSet[dlg.Fact],
        ](f=lambda pair: pair[0]).connect(e.circuit, (body,))
        tick_delta_node = CoreIntegrate[ZSet[dlg.Fact]](axis=2, group=fg).connect(e.circuit, (facts_out,))
        model_node = CoreIntegrate[ZSet[dlg.Fact]](axis=1, group=fg).connect(e.circuit, (tick_delta_node,))

        e.push(edb_1d, self._facts)
        e.push(program_1d, self._rules)
        e.push(seed, ZSet({(0, dlg._rewrite_monoid.identity()): 1}))

        # Scan bounds from the Δ³-support rule: output deltas live in the
        # ∨-closure of pushed cells, shifted ONCE per explicit entry delay —
        # sf/sr enter the body only through z⁻¹ on axis 2 (+1 inner), neg_ref
        # only through z⁻¹ on axis 1 (+1 tick), δ₀ imports at coordinate 0.
        # The bounds read the LIVE input antichains (pushes advance them), so
        # cascade, cross-tick holes and cross-outer joins need no special cases.
        sf_ac = e.circuit.progress_rules[sf].frontier
        sr_ac = e.circuit.progress_rules[sr].frontier
        nr_ac = e.circuit.progress_rules[neg_ref].frontier

        def inner_bound() -> int:
            return 1 + max(_frontier_max(sf_ac, 2), _frontier_max(sr_ac, 2))

        def alt_bound() -> int:
            return max(
                _frontier_max(sf_ac, 1), _frontier_max(sr_ac, 1), 1 + _frontier_max(nr_ac, 1)
            )

        for a in range(self._max_alternation):
            quiet = True
            i = 0
            while i < self._max_inner:
                diff_facts, diff_rewrites = e.read(body, (0, a, i))
                if not diff_facts.inner and not diff_rewrites.inner:
                    if i > inner_bound():
                        break
                else:
                    quiet = False
                    if diff_facts.inner:
                        e.push(sf, diff_facts, t=(0, a, i))
                    if diff_rewrites.inner:
                        e.push(sr, diff_rewrites, t=(0, a, i))
                i += 1
            else:
                raise RuntimeError(f"alternation tick {a} did not converge in {self._max_inner} inner ticks")
            tick_delta = e.read(tick_delta_node, (0, a, i))
            if tick_delta.inner:
                # the alternation feedback edge: ferry the tick's model delta from
                # the output port to the reference port (the driver-materialized
                # ∫-barrier); consumed at tick a+1 through the in-circuit z⁻¹ on axis 1
                e.push(neg_ref, tick_delta, t=(0, a, 0))
            if quiet and a >= alt_bound():  # all support sites visited and zero: stable forever
                return _without_internal_facts(e.read(model_node, (0, a, inner_bound())))
        raise RuntimeError(
            f"alternating fixpoint did not converge within {self._max_alternation} ticks "
            f"(non-total well-founded model? stratifiability is checked at push time)"
        )


class IncrementalDatalogAlternating:
    """MULTI-BATCH (t1-incremental) driver for the alternating-fixpoint 3-D
    circuit — the same circuit as :class:`DatalogAlternating` (every operator
    is already the Δ³ form, so incrementality on all three axes is a property
    of the circuit; this class only supplies the persistent multi-batch crank).

    Protocol = three nested fixpoints: each ``saturate()`` call advances ONE
    outer tick (pending fact/rule deltas are pushed at ``(t1,)``), walks the
    alternation axis to quiescence, and walks the inner axis to quiescence per
    alternation tick. The restart is positional, not computational: cell
    ``(t1+1, a, i)`` evaluates to the DELTA against ``(t1, a, i)`` through the
    operators' internal integrals, so undisturbed cells read zero and the work
    is proportional to the delta cone. Retractions (negative-weight fact or
    RULE deltas) ride the same path; because no stratification is ever
    computed, rule updates that change the stratum structure need zero
    special-casing — the alternation simply re-converges.

    Scan floors come from the evaluator's Antichain progress plane, on BOTH
    axes: work sites at ``(t1, a, i)`` are seeded by ``z⁻¹`` shifts from the
    prior alternation tick (same outer), the prior outer (same and prior
    tick), and the inner cascade — so the inner scan floor is the max of the
    two frontier reads, and the ALTERNATION axis gets the symmetric treatment
    (a batch delta touching only high strata reads zero at low ticks and fires
    deeper: alternation-axis holes, the inner-hole physics one axis up).
    Alternation quiescence = zero ticks through the prior batch's alternation
    depth; by that floor, every cell past the quiescent tick is provably zero,
    so the cumulative-model read (the ``I⁰`` tap) at the covering coordinates
    is exact.

    ``push_rules`` validates stratifiability of the CUMULATIVE program
    (hypothesis, not mechanism). ``compact_each_tick=True`` runs
    :meth:`Evaluator.compact` after each outer tick's model read (GC)."""

    def __init__(
        self,
        *,
        max_inner: int = 8192,
        max_alternation: int = 64,
        compact_each_tick: bool = False,
    ) -> None:
        self._max_inner = max_inner
        self._max_alternation = max_alternation
        self._compact_each_tick = compact_each_tick
        self._rule_group = ZSetAddition[dlg.Rule]()
        self._fact_group = ZSetAddition[dlg.Fact]()
        self._all_rules: ZSet[dlg.Rule] = ZSet({})
        self._pending_rules: ZSet[dlg.Rule] = ZSet({})
        self._pending_edb: ZSet[dlg.Fact] = ZSet({})
        self._next_outer = 0
        self._e: Evaluator[Time] | None = None

    def push_rules(self, rules: ZSet[dlg.Rule]) -> None:
        merged = self._rule_group.add(self._all_rules, rules)
        _check_stratifiable(merged)  # on the cumulative program; commit only if it holds
        dlg.check_interpreted_safety(merged)
        self._all_rules = merged
        self._pending_rules = self._rule_group.add(self._pending_rules, rules)

    def push_facts(self, facts: ZSet[dlg.Fact]) -> None:
        self._pending_edb = self._fact_group.add(self._pending_edb, facts)

    def _setup_circuits(self) -> None:
        fg, rug = self._fact_group, self._rule_group
        rg = ZSetAddition[dlg.ProvenanceIndexedRewrite]()
        e = Evaluator[Time](
            circuit=Circuit[Time](), storage=DictStorage(), ctx=ComputeCtx(lattice=dbsp_time(3)), group=fg
        )
        self._edb_1d = Input[ZSet[dlg.Fact]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
        self._program_1d = Input[ZSet[dlg.Rule]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
        self._sf = Input[ZSet[dlg.Fact]](frontier=Antichain(dbsp_time(3))).connect(e.circuit, ())
        self._sr = Input[ZSet[dlg.ProvenanceIndexedRewrite]](frontier=Antichain(dbsp_time(3))).connect(e.circuit, ())
        self._neg_ref = Input[ZSet[dlg.Fact]](frontier=Antichain(dbsp_time(3))).connect(e.circuit, ())
        seed = Input[ZSet[dlg.ProvenanceIndexedRewrite]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
        program_2d = LiftStreamIntroduction[ZSet[dlg.Rule]](group=rug).connect(e.circuit, (self._program_1d,))
        program_3d = LiftStreamIntroduction[ZSet[dlg.Rule]](group=rug).connect(e.circuit, (program_2d,))
        self._body = IncrementalDatalogBodyWithNegation(
            fact_group=fg, rule_group=rug, rewrite_group=rg, signal_group=ZSetAddition[dlg.Signal](),
            dir_group=ZSetAddition[dlg.Direction](),
            gatekeep_group=ZSetAddition[dlg.AtomWithSourceRewriteAndProvenance](),
        ).connect(e.circuit, (self._edb_1d, program_3d, self._sf, self._sr, seed, self._neg_ref))
        facts_out = Lift1[
            tuple[ZSet[dlg.Fact], ZSet[dlg.ProvenanceIndexedRewrite]],
            ZSet[dlg.Fact],
        ](f=lambda pair: pair[0]).connect(e.circuit, (self._body,))
        # I² = the outer-delta of the tick delta (the neg_ref feed at every outer);
        # I¹ of that = the outer-delta of the model; I⁰ of that = the model.
        self._tick_delta_node = CoreIntegrate[ZSet[dlg.Fact]](axis=2, group=fg).connect(e.circuit, (facts_out,))
        model_delta_node = CoreIntegrate[ZSet[dlg.Fact]](axis=1, group=fg).connect(e.circuit, (self._tick_delta_node,))
        self._cum_model_node = CoreIntegrate[ZSet[dlg.Fact]](axis=0, group=fg).connect(e.circuit, (model_delta_node,))
        e.push(seed, ZSet({(0, dlg._rewrite_monoid.identity()): 1}), t=(0,))
        self._e = e

    def saturate(self) -> ZSet[dlg.Fact]:
        if self._e is None:
            self._setup_circuits()
        e = self._e
        assert e is not None
        rule_delta, self._pending_rules = self._pending_rules, ZSet({})
        edb_delta, self._pending_edb = self._pending_edb, ZSet({})
        t1 = self._next_outer
        self._next_outer += 1
        if rule_delta.inner:
            e.push(self._program_1d, rule_delta, t=(t1,))
        if edb_delta.inner:
            e.push(self._edb_1d, edb_delta, t=(t1,))

        # Scan bounds from the Δ³-support rule: output deltas live in the
        # ∨-closure of pushed cells, shifted ONCE per explicit entry delay
        # (sf/sr enter via z⁻¹ on axis 2, neg_ref via z⁻¹ on axis 1; δ₀ imports
        # sit at coordinate 0). Lattice joins lift any HISTORICAL cell to the
        # current outer, so the bounds are global frontier maxima — read live
        # from the input antichains, which grow as the scan pushes.
        sf_ac = e.circuit.progress_rules[self._sf].frontier
        sr_ac = e.circuit.progress_rules[self._sr].frontier
        nr_ac = e.circuit.progress_rules[self._neg_ref].frontier

        def inner_bound() -> int:
            return 1 + max(_frontier_max(sf_ac, 2), _frontier_max(sr_ac, 2))

        def alt_bound() -> int:
            return max(
                _frontier_max(sf_ac, 1), _frontier_max(sr_ac, 1), 1 + _frontier_max(nr_ac, 1)
            )

        a = 0
        while a < self._max_alternation:
            quiet = True
            i = 0
            while i < self._max_inner:
                diff_facts, diff_rewrites = e.read(self._body, (t1, a, i))
                if not diff_facts.inner and not diff_rewrites.inner:
                    if i > inner_bound():
                        break
                else:
                    quiet = False
                    if diff_facts.inner:
                        e.push(self._sf, diff_facts, t=(t1, a, i))
                    if diff_rewrites.inner:
                        e.push(self._sr, diff_rewrites, t=(t1, a, i))
                i += 1
            else:
                raise RuntimeError(
                    f"outer {t1} alternation tick {a} did not converge in {self._max_inner} inner ticks"
                )
            tick_delta = e.read(self._tick_delta_node, (t1, a, i))
            if tick_delta.inner:
                e.push(self._neg_ref, tick_delta, t=(t1, a, 0))
            if quiet and a >= alt_bound():  # all support sites visited and zero
                break
            a += 1
        else:
            raise RuntimeError(
                f"outer {t1}: alternating fixpoint did not converge within "
                f"{self._max_alternation} ticks (non-total well-founded model?)"
            )
        model = e.read(self._cum_model_node, (t1, max(a, alt_bound()), inner_bound()))
        if self._compact_each_tick:
            e.compact()
        return _without_internal_facts(model)
