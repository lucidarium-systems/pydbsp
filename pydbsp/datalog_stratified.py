"""Stratified-negation Datalog: helpers, body operator, and driver.

The body operates on a ``(outer, stratum, inner)`` lattice with
primitives that are the 3-D analog of the 2-D bilinear / distinct
operators in :mod:`pydbsp.relational_operators`:

* :class:`DeltaLiftedDeltaLiftedDeltaLiftedJoin` — 8-term bilinear
  join (the 3-D extension of the 2-D 4-term bilinear).
* :class:`DeltaLiftedDeltaLiftedDeltaLiftedDistinct` — 3-D form of
  ``Differentiate(H(z⁻¹ I s, I s))``.

The ``Lift1`` / ``Lift2`` value-layer wrappers are arity-agnostic and
carry over unchanged.

This module exposes :class:`IncrementalDatalogBodyWithNegation`, the
negation-aware Datalog body, plus the stratified driver
:class:`IncrementalDatalogStratified` that wraps user rules with
monotone ``stratum_ready`` guards derived from the running level
subgraph and seeds those readiness facts into the body's feedback
state at the physical inner coordinate where each stratum starts.

The value-layer helpers below feed the in-circuit level subgraph via
``Lift1`` (per-rule extractors) and the guard rewriter via ``Lift2``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydbsp import datalog as dlg
from pydbsp.circuit import Circuit
from pydbsp.compute import ComputeCtx
from pydbsp.core import Antichain, dbsp_time
from pydbsp.evaluate import Evaluator
from pydbsp.indexed_relational_operators import (
    IndexedDeltaLiftedDeltaLiftedDeltaLiftedJoin,
    IndexedIncrementalDatalogWithNegationBody,
    LiftGroupBy,
    LiftIndex,
)
from pydbsp.indexed_zset import IndexedZSetAddition
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


def _extract_head_preds(program: ZSet[dlg.Rule]) -> ZSet[tuple[str]]:
    """Per-rule head-predicate extractor. Each rule contributes a
    single ``(head_pred,)`` 1-tuple at the rule's weight. Designed as
    a value-level Lift1 input — for the in-circuit level Datalog,
    these are the ``head(P)`` facts from rule 1."""
    out: dict[tuple[str], int] = {}
    for rule, w in program.inner.items():
        if w == 0:
            continue
        key = (rule[0][0],)
        out[key] = out.get(key, 0) + w
    return ZSet({k: v for k, v in out.items() if v != 0})


def _extract_pos_dep_pairs(
    program: ZSet[dlg.Rule],
) -> ZSet[tuple[str, str]]:
    """Per-rule positive-dep edges. For each rule with head ``P`` and
    each positive body atom whose predicate is ``Q``, emit ``(P, Q)``.
    Weight is the rule's weight, summed across rules contributing the
    same edge."""
    out: dict[tuple[str, str], int] = {}
    for rule, w in program.inner.items():
        if w == 0:
            continue
        head_pred = rule[0][0]
        for atom in rule[1:]:
            body_pred = atom[0]
            if body_pred.startswith("!") or body_pred in dlg.INTERPRETED:
                continue  # interpreted predicates are guards, not relations → no dep edge
            key = (head_pred, body_pred)
            out[key] = out.get(key, 0) + w
    return ZSet({k: v for k, v in out.items() if v != 0})


def _extract_neg_dep_pairs(
    program: ZSet[dlg.Rule],
) -> ZSet[tuple[str, str]]:
    """Per-rule negative-dep edges. For each rule with head ``P`` and
    each negated body atom ``!Q``, emit ``(P, Q)``."""
    out: dict[tuple[str, str], int] = {}
    for rule, w in program.inner.items():
        if w == 0:
            continue
        head_pred = rule[0][0]
        for atom in rule[1:]:
            body_pred = atom[0]
            if not body_pred.startswith("!"):
                continue
            key = (head_pred, body_pred[1:])
            out[key] = out.get(key, 0) + w
    return ZSet({k: v for k, v in out.items() if v != 0})


_STRATUM_READY_PRED = "stratum_ready"


def _guard_program_with_levels(
    program: ZSet[dlg.Rule],
    levels: ZSet[tuple[tuple[str], int]],
) -> ZSet[dlg.Rule]:
    """Per-cell ``Lift2`` join: prepend a stratum-readiness guard to
    each rule according to its head's current effective level.
    ``levels`` is the running level subgraph's
    ``effective_level_out`` cumulative — entries ``((P,), max_level)``.
    A rule whose head has no level entry yet defaults to level 0
    (e.g., the very first push, before the level subgraph has had a
    chance to propagate).

    Level changes become ordinary signed rule-value changes after this
    function is composed with ``Differentiate``: the old guarded rule is
    retracted and the new guarded rule is inserted."""
    level_map: dict[str, int] = {}
    for (pred_key, lvl), w in levels.inner.items():
        if w <= 0:
            continue
        pred = pred_key[0]
        prior = level_map.get(pred)
        if prior is None or lvl > prior:
            level_map[pred] = lvl
    out: dict[dlg.Rule, int] = {}
    for rule, w in program.inner.items():
        if w == 0:
            continue
        head_pred = rule[0][0]
        level = level_map.get(head_pred, 0)
        key = (
            rule[0],
            (_STRATUM_READY_PRED, (level - 1,)),
            *rule[1:],
        )
        out[key] = out.get(key, 0) + w
    return ZSet({k: v for k, v in out.items() if v != 0})


def _stratum_ready_delta(stratum: int) -> ZSet[dlg.Fact]:
    """Monotone readiness control fact delta for ``stratum``.

    The cumulative fact set at stratum ``s`` contains every readiness
    fact up to ``stratum_ready(s - 1)``. Keeping lower readiness facts
    active preserves lower-stratum derived facts as the body advances
    to higher strata; higher-stratum rules still cannot fire early
    because their own readiness fact has not been inserted yet."""
    current = (_STRATUM_READY_PRED, (stratum - 1,))
    return ZSet({current: 1})


def _without_internal_facts(z: ZSet[dlg.Fact]) -> ZSet[dlg.Fact]:
    """Filter internal control facts out of public Datalog output."""
    return ZSet({fact: w for fact, w in z.inner.items() if fact[0] != _STRATUM_READY_PRED and w != 0})


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


@dataclass(frozen=True)
class IndexedIncrementalDatalogBodyWithNegation(Operator):
    """SORT-MERGE-INDEXED 3-D (outer, stratum, inner) negation body — the indexed
    analog of :class:`IncrementalDatalogBodyWithNegation`. Byte-for-byte the same
    body, with every :class:`DeltaLiftedDeltaLiftedDeltaLiftedJoin` replaced by an
    :class:`~pydbsp.indexed_relational_operators.IndexedDeltaLiftedDeltaLiftedDeltaLiftedJoin`
    (operands pre-indexed via :class:`LiftIndex`), exactly as the 2-D
    :class:`~pydbsp.indexed_relational_operators.IndexedIncrementalDatalogWithNegationBody`
    indexes the 2-D plain body. Produces the SAME ``(facts_diff, rewrites_diff)`` IDB;
    only the join cost changes (O(matching keys) vs O(|left|·|right|)).

    ``inputs = (edb_1d, program_3d, state_facts_3d, state_rewrites_3d, seed_1d)``."""

    fact_group: ZSetAddition[dlg.Fact]
    rule_group: ZSetAddition[dlg.Rule]
    rewrite_group: ZSetAddition[dlg.ProvenanceIndexedRewrite]
    signal_group: ZSetAddition[dlg.Signal]
    ext_dir_group: ZSetAddition[dlg.ExtendedDirection]
    jorder_group: ZSetAddition[tuple[str, dlg.ColumnReference]]
    gatekeep_group: ZSetAddition[dlg.IndexedGatekeepEntry]
    indexed_fact_group: ZSetAddition[dlg.IndexedFact]

    def connect(self, circuit: Circuit[Time], inputs: tuple[NodeId, ...]) -> NodeId:
        if len(inputs) != 5:
            raise ValueError(f"IndexedIncrementalDatalogBodyWithNegation takes 5 inputs; got {len(inputs)}")
        edb_1d, program_3d, state_facts_3d, state_rewrites_3d, seed_1d = inputs

        edb_2d = LiftStreamIntroduction[ZSet[dlg.Fact]](group=self.fact_group).connect(circuit, (edb_1d,))
        edb_3d = LiftStreamIntroduction[ZSet[dlg.Fact]](group=self.fact_group).connect(circuit, (edb_2d,))
        seed_2d = LiftStreamIntroduction[ZSet[dlg.ProvenanceIndexedRewrite]](group=self.rewrite_group).connect(circuit, (seed_1d,))
        seed_3d = LiftStreamIntroduction[ZSet[dlg.ProvenanceIndexedRewrite]](group=self.rewrite_group).connect(circuit, (seed_2d,))

        delayed_facts = CoreDelay[ZSet[dlg.Fact]](axis=2, group=self.fact_group).connect(circuit, (state_facts_3d,))
        delayed_rewrites = CoreDelay[ZSet[dlg.ProvenanceIndexedRewrite]](axis=2, group=self.rewrite_group).connect(circuit, (state_rewrites_3d,))

        sig_3d = Lift1[ZSet[dlg.Rule], ZSet[dlg.Signal]](f=dlg.sig).connect(circuit, (program_3d,))
        ext_dir_3d = Lift1[ZSet[dlg.Rule], ZSet[dlg.ExtendedDirection]](f=dlg.ext_dir).connect(circuit, (program_3d,))
        jorder_3d = Lift1[ZSet[dlg.Rule], ZSet[tuple[str, dlg.ColumnReference]]](f=dlg.jorder).connect(circuit, (program_3d,))

        # gatekeep: delayed_rewrites ⋈ ext_dir on provenance
        indexed_rewrites = LiftIndex[dlg.ProvenanceIndexedRewrite, dlg.Provenance](indexer=lambda rw: rw[0]).connect(circuit, (delayed_rewrites,))
        indexed_ext_dir = LiftIndex[dlg.ExtendedDirection, dlg.Provenance](indexer=lambda ed: ed[0]).connect(circuit, (ext_dir_3d,))
        rewrite_idx_group = IndexedZSetAddition[dlg.Provenance, dlg.ProvenanceIndexedRewrite](self.rewrite_group, lambda rw: rw[0])
        ext_dir_idx_group = IndexedZSetAddition[dlg.Provenance, dlg.ExtendedDirection](self.ext_dir_group, lambda ed: ed[0])
        gatekeep = IndexedDeltaLiftedDeltaLiftedDeltaLiftedJoin[
            dlg.Provenance, dlg.ProvenanceIndexedRewrite, dlg.ExtendedDirection, dlg.IndexedGatekeepEntry
        ](
            proj=lambda _k, l, r: (r[1], r[2], l[1], r[3]),
            group_a=rewrite_idx_group, group_b=ext_dir_idx_group, out_group=self.gatekeep_group,
        ).connect(circuit, (indexed_rewrites, indexed_ext_dir))

        # facts ⋈ jorder on predicate → indexed (column-ref-keyed) facts
        indexed_facts_by_pred = LiftIndex[dlg.Fact, str](indexer=lambda f: f[0]).connect(circuit, (delayed_facts,))
        indexed_jorder_by_pred = LiftIndex[tuple[str, dlg.ColumnReference], str](indexer=lambda j: j[0]).connect(circuit, (jorder_3d,))
        fact_pred_group = IndexedZSetAddition[str, dlg.Fact](self.fact_group, lambda f: f[0])
        jorder_pred_group = IndexedZSetAddition[str, tuple[str, dlg.ColumnReference]](self.jorder_group, lambda j: j[0])
        indexed_facts_raw = IndexedDeltaLiftedDeltaLiftedDeltaLiftedJoin[
            str, dlg.Fact, tuple[str, dlg.ColumnReference], dlg.IndexedFact
        ](
            proj=lambda _k, fact, jo: dlg._index_fact(jo[1], fact),
            group_a=fact_pred_group, group_b=jorder_pred_group, out_group=self.indexed_fact_group,
        ).connect(circuit, (indexed_facts_by_pred, indexed_jorder_by_pred))
        distinct_indexed_facts = DeltaLiftedDeltaLiftedDeltaLiftedDistinct[dlg.IndexedFact](
            inner_group=self.indexed_fact_group
        ).connect(circuit, (indexed_facts_raw,))

        positive_atoms = Lift1[ZSet[dlg.IndexedGatekeepEntry], ZSet[dlg.IndexedGatekeepEntry]](
            f=lambda z: ZSet(
                {gk: w for gk, w in z.inner.items() if gk[1] is None or ("!" not in gk[1][0] and gk[1][0] not in dlg.INTERPRETED)}
            )
        ).connect(circuit, (gatekeep,))
        negative_atoms = Lift1[ZSet[dlg.IndexedGatekeepEntry], ZSet[dlg.IndexedGatekeepEntry]](
            f=lambda z: ZSet({gk: w for gk, w in z.inner.items() if not (gk[1] is None or "!" not in gk[1][0])})
        ).connect(circuit, (gatekeep,))
        # interpreted (built-in) atoms — stateless guard, projected to the unchanged rewrite.
        interp_proj = Lift1[ZSet[dlg.IndexedGatekeepEntry], ZSet[dlg.ProvenanceIndexedRewrite]](
            f=lambda z: ZSet(
                {
                    (gk[0], gk[2]): w
                    for gk, w in z.inner.items()
                    if gk[1] is not None and gk[1][0] in dlg.INTERPRETED and dlg.eval_interpreted(gk[1], gk[2])
                }
            )
        ).connect(circuit, (gatekeep,))

        gatekeep_idx_group = IndexedZSetAddition[dlg.JoinKey, dlg.IndexedGatekeepEntry](self.gatekeep_group, dlg._gatekeep_join_key)
        indexed_fact_idx_group = IndexedZSetAddition[dlg.JoinKey, dlg.IndexedFact](self.indexed_fact_group, lambda ifact: ifact[0])
        keyed_positive = LiftIndex[dlg.IndexedGatekeepEntry, dlg.JoinKey](indexer=dlg._gatekeep_join_key).connect(circuit, (positive_atoms,))
        keyed_facts = LiftIndex[dlg.IndexedFact, dlg.JoinKey](indexer=lambda ifact: ifact[0]).connect(circuit, (distinct_indexed_facts,))

        product = IndexedDeltaLiftedDeltaLiftedDeltaLiftedJoin[
            dlg.JoinKey, dlg.IndexedGatekeepEntry, dlg.IndexedFact, dlg.ProvenanceIndexedRewrite
        ](
            proj=dlg._indexed_product_proj,
            group_a=gatekeep_idx_group, group_b=indexed_fact_idx_group, out_group=self.rewrite_group,
        ).connect(circuit, (keyed_positive, keyed_facts))

        proj = Lift1[ZSet[dlg.IndexedGatekeepEntry], ZSet[dlg.ProvenanceIndexedRewrite]](
            f=lambda z: ZSet({(gk[0], gk[2]): w for gk, w in z.inner.items()})
        ).connect(circuit, (negative_atoms,))

        keyed_negative = LiftIndex[dlg.IndexedGatekeepEntry, dlg.JoinKey](indexer=dlg._gatekeep_join_key).connect(circuit, (negative_atoms,))
        anti_product = IndexedDeltaLiftedDeltaLiftedDeltaLiftedJoin[
            dlg.JoinKey, dlg.IndexedGatekeepEntry, dlg.IndexedFact, dlg.ProvenanceIndexedRewrite
        ](
            proj=lambda _k, gk, _ifact: (gk[0], gk[2]),
            group_a=gatekeep_idx_group, group_b=indexed_fact_idx_group, out_group=self.rewrite_group,
        ).connect(circuit, (keyed_negative, keyed_facts))

        # final_product = product + proj − anti_product
        product_plus_proj = Lift2[ZSet[dlg.ProvenanceIndexedRewrite], ZSet[dlg.ProvenanceIndexedRewrite], ZSet[dlg.ProvenanceIndexedRewrite]](
            op=self.rewrite_group.add
        ).connect(circuit, (product, proj))
        neg_anti = Lift1[ZSet[dlg.ProvenanceIndexedRewrite], ZSet[dlg.ProvenanceIndexedRewrite]](f=self.rewrite_group.neg).connect(circuit, (anti_product,))
        final_product_neg = Lift2[ZSet[dlg.ProvenanceIndexedRewrite], ZSet[dlg.ProvenanceIndexedRewrite], ZSet[dlg.ProvenanceIndexedRewrite]](
            op=self.rewrite_group.add
        ).connect(circuit, (product_plus_proj, neg_anti))
        # final_product = product + proj − anti_product + interp_proj
        final_product = Lift2[ZSet[dlg.ProvenanceIndexedRewrite], ZSet[dlg.ProvenanceIndexedRewrite], ZSet[dlg.ProvenanceIndexedRewrite]](
            op=self.rewrite_group.add
        ).connect(circuit, (final_product_neg, interp_proj))

        # ground: final_product ⋈ sig on provenance → facts
        indexed_final_product = LiftIndex[dlg.ProvenanceIndexedRewrite, dlg.Provenance](indexer=lambda r: r[0]).connect(circuit, (final_product,))
        indexed_sig = LiftIndex[dlg.Signal, dlg.Provenance](indexer=lambda s: s[0]).connect(circuit, (sig_3d,))
        sig_idx_group = IndexedZSetAddition[dlg.Provenance, dlg.Signal](self.signal_group, lambda s: s[0])
        ground = IndexedDeltaLiftedDeltaLiftedDeltaLiftedJoin[dlg.Provenance, dlg.ProvenanceIndexedRewrite, dlg.Signal, dlg.Fact](
            proj=lambda _k, l, r: l[1].apply(r[1]),
            group_a=rewrite_idx_group, group_b=sig_idx_group, out_group=self.fact_group,
        ).connect(circuit, (indexed_final_product, indexed_sig))

        ground_plus_edb = Lift2[ZSet[dlg.Fact], ZSet[dlg.Fact], ZSet[dlg.Fact]](op=self.fact_group.add).connect(circuit, (ground, edb_3d))
        next_facts = DeltaLiftedDeltaLiftedDeltaLiftedDistinct[dlg.Fact](inner_group=self.fact_group).connect(circuit, (ground_plus_edb,))

        final_plus_seed = Lift2[ZSet[dlg.ProvenanceIndexedRewrite], ZSet[dlg.ProvenanceIndexedRewrite], ZSet[dlg.ProvenanceIndexedRewrite]](
            op=self.rewrite_group.add
        ).connect(circuit, (final_product, seed_3d))
        next_rewrites = DeltaLiftedDeltaLiftedDeltaLiftedDistinct[dlg.ProvenanceIndexedRewrite](inner_group=self.rewrite_group).connect(circuit, (final_plus_seed,))

        return Lift2[ZSet[dlg.Fact], ZSet[dlg.ProvenanceIndexedRewrite], tuple[ZSet[dlg.Fact], ZSet[dlg.ProvenanceIndexedRewrite]]](
            op=lambda a, b: (a, b)
        ).connect(circuit, (next_facts, next_rewrites))


LevelAtLeast = tuple[str, int]
"""A ``(predicate, level)`` pair. ``level_at_least(P, k)`` semantics:
P's stratum is *at least* k, derived from the dependency graph."""

EffectiveLevel = tuple[tuple[str], int]
"""``(predicate_key, max_level)`` after grouping by predicate. The key
is wrapped in a 1-tuple because :class:`LiftIndex` keys are arbitrary
tuples — keeps the type uniform."""


def _max_inner_before_outer(
    frontier: Antichain[Time],
    outer: int,
    stratum: int,
) -> int:
    """Largest inner coordinate covered by ``frontier`` at any prior
    outer for the given ``stratum``. An antichain point at a higher
    stratum also covers lower strata in the down-set, so it contributes
    to the bound."""
    if frontier.is_universal:
        return -1
    out = -1
    for elt in frontier.elements:
        if len(elt) < 3 or not isinstance(elt[0], int) or elt[0] >= outer:
            continue
        if elt[1] < stratum:
            continue
        inner = elt[2]
        if isinstance(inner, int) and inner > out:
            out = inner
    return out


def _max_inner_at_outer(frontier: Antichain[Time], outer: int) -> int:
    """Largest inner coordinate covered by ``frontier`` at ``outer``, over ALL
    middle-axis coordinates. The covering depth for reads of axis-2 integrals."""
    if frontier.is_universal:
        return -1
    out = -1
    for elt in frontier.elements:
        if len(elt) < 3 or elt[0] != outer:
            continue
        inner = elt[2]
        if isinstance(inner, int) and inner > out:
            out = inner
    return out


def _connect_running_level_3d(
    e: Evaluator[Time],
    program_1d: NodeId,
    state_level_3d: NodeId,
    level_group: ZSetAddition[LevelAtLeast],
) -> tuple[NodeId, NodeId, NodeId]:
    """Wire the running stratification subgraph on the main 3-D
    ``(outer, stratum, inner)`` lattice.

    ``level_at_least_out`` is the ordinary feedback delta. A value
    ``(P, k)`` means the circuit has established that predicate ``P``
    has effective level at least ``k``. The stratum axis carries the
    negative-depth wave: rule 3 reads level state from the prior
    stratum via ``Delay(axis=1)``.

    ``level_next_stratum_out`` is not fed back. It is a driver-visible
    lookahead saying "the current stratum has enough information to
    produce work in the next stratum." This avoids computing a final
    Python-side level map before deciding to clock stratum ``s + 1``.
    """
    head_group = ZSetAddition[tuple[str]]()
    edge_group = ZSetAddition[tuple[str, str]]()

    head_1d = Lift1[ZSet[dlg.Rule], ZSet[tuple[str]]](
        f=_extract_head_preds,
    ).connect(e.circuit, (program_1d,))
    pos_dep_1d = Lift1[ZSet[dlg.Rule], ZSet[tuple[str, str]]](
        f=_extract_pos_dep_pairs,
    ).connect(e.circuit, (program_1d,))
    neg_dep_1d = Lift1[ZSet[dlg.Rule], ZSet[tuple[str, str]]](
        f=_extract_neg_dep_pairs,
    ).connect(e.circuit, (program_1d,))

    head_2d = LiftStreamIntroduction[ZSet[tuple[str]]](
        group=head_group,
    ).connect(e.circuit, (head_1d,))
    head_3d = LiftStreamIntroduction[ZSet[tuple[str]]](
        group=head_group,
    ).connect(e.circuit, (head_2d,))
    pos_dep_2d = LiftStreamIntroduction[ZSet[tuple[str, str]]](
        group=edge_group,
    ).connect(e.circuit, (pos_dep_1d,))
    pos_dep_3d = LiftStreamIntroduction[ZSet[tuple[str, str]]](
        group=edge_group,
    ).connect(e.circuit, (pos_dep_2d,))
    neg_dep_2d = LiftStreamIntroduction[ZSet[tuple[str, str]]](
        group=edge_group,
    ).connect(e.circuit, (neg_dep_1d,))
    neg_dep_3d = LiftStreamIntroduction[ZSet[tuple[str, str]]](
        group=edge_group,
    ).connect(e.circuit, (neg_dep_2d,))

    within_stratum_level = CoreDelay[ZSet[LevelAtLeast]](
        axis=2,
        group=level_group,
    ).connect(e.circuit, (state_level_3d,))
    prior_stratum_level = CoreDelay[ZSet[LevelAtLeast]](
        axis=1,
        group=level_group,
    ).connect(e.circuit, (state_level_3d,))

    contrib_1 = Lift1[ZSet[tuple[str]], ZSet[LevelAtLeast]](
        f=lambda heads: ZSet({(head[0], 0): w for head, w in heads.inner.items()}),
    ).connect(e.circuit, (head_3d,))

    contrib_2 = DeltaLiftedDeltaLiftedDeltaLiftedJoin[
        tuple[str, str],
        LevelAtLeast,
        LevelAtLeast,
    ](
        pred=lambda pd, lvl: pd[1] == lvl[0],
        proj=lambda pd, lvl: (pd[0], lvl[1]),
        group_a=edge_group,
        group_b=level_group,
        out_group=level_group,
    ).connect(e.circuit, (pos_dep_3d, within_stratum_level))

    contrib_3 = DeltaLiftedDeltaLiftedDeltaLiftedJoin[
        tuple[str, str],
        LevelAtLeast,
        LevelAtLeast,
    ](
        pred=lambda nd, lvl: nd[1] == lvl[0],
        proj=lambda nd, lvl: (nd[0], lvl[1] + 1),
        group_a=edge_group,
        group_b=level_group,
        out_group=level_group,
    ).connect(e.circuit, (neg_dep_3d, prior_stratum_level))

    sum_12 = Lift2[
        ZSet[LevelAtLeast],
        ZSet[LevelAtLeast],
        ZSet[LevelAtLeast],
    ](op=level_group.add).connect(e.circuit, (contrib_1, contrib_2))
    total = Lift2[
        ZSet[LevelAtLeast],
        ZSet[LevelAtLeast],
        ZSet[LevelAtLeast],
    ](op=level_group.add).connect(e.circuit, (sum_12, contrib_3))
    level_at_least_out = DeltaLiftedDeltaLiftedDeltaLiftedDistinct[LevelAtLeast](inner_group=level_group).connect(
        e.circuit, (total,)
    )

    # Feed-forward lookahead: if a level fact produced in this stratum
    # can traverse a negative edge, the predicate's effective level is
    # already higher than the current stratum. This is not fed back as
    # same-stratum level state; it only annotates the running guarded
    # program and tells the driver to clock the next stratum.
    level_next_stratum_out = DeltaLiftedDeltaLiftedDeltaLiftedJoin[
        tuple[str, str],
        LevelAtLeast,
        LevelAtLeast,
    ](
        pred=lambda nd, lvl: nd[1] == lvl[0],
        proj=lambda nd, lvl: (nd[0], lvl[1] + 1),
        group_a=edge_group,
        group_b=level_group,
        out_group=level_group,
    ).connect(e.circuit, (neg_dep_3d, level_at_least_out))

    effective_level_delta = Lift2[
        ZSet[LevelAtLeast],
        ZSet[LevelAtLeast],
        ZSet[LevelAtLeast],
    ](op=level_group.add).connect(
        e.circuit,
        (level_at_least_out, level_next_stratum_out),
    )

    int_o = CoreIntegrate[ZSet[LevelAtLeast]](
        axis=0,
        group=level_group,
    ).connect(e.circuit, (effective_level_delta,))
    int_os = CoreIntegrate[ZSet[LevelAtLeast]](
        axis=1,
        group=level_group,
    ).connect(e.circuit, (int_o,))
    cum_level = CoreIntegrate[ZSet[LevelAtLeast]](
        axis=2,
        group=level_group,
    ).connect(e.circuit, (int_os,))

    indexed_cum = LiftIndex[LevelAtLeast, tuple[str]](
        indexer=lambda pk: (pk[0],),
    ).connect(e.circuit, (cum_level,))
    effective_level_out = LiftGroupBy[LevelAtLeast, tuple[str], int](
        aggregate=lambda items: max(p_k[1] for p_k, _ in items),
    ).connect(e.circuit, (indexed_cum,))

    return (
        level_at_least_out,
        level_next_stratum_out,
        effective_level_out,
    )


def _build_stratified_body_circuit(indexed: bool = False) -> tuple[
    Evaluator[Time],
    NodeId,  # edb_1d
    NodeId,  # program_1d (driver pushes raw rule deltas)
    NodeId,  # state_level_3d
    NodeId,  # state_facts_3d
    NodeId,  # state_rewrites_3d
    NodeId,  # seed_1d
    NodeId,  # level_at_least_out
    NodeId,  # level_next_stratum_out
    NodeId,  # guarded_delta_3d
    NodeId,  # body_out (facts + rewrites pair)
]:
    """Persistent 3-D body circuit using
    :class:`IncrementalDatalogBodyWithNegation`. The body rewrites
    each rule with a stratum-readiness guard derived from the 3-D
    running level subgraph, then feeds guarded-rule deltas to the main
    Datalog body. Monotone ``stratum_ready`` facts are seeded into the
    fact-state feedback at each stratum's physical inner start."""
    fact_group = ZSetAddition[dlg.Fact]()
    rule_group = ZSetAddition[dlg.Rule]()
    rewrite_group = ZSetAddition[dlg.ProvenanceIndexedRewrite]()
    level_group = ZSetAddition[LevelAtLeast]()
    e = Evaluator[Time](
        circuit=Circuit[Time](),
        storage=DictStorage(),
        ctx=ComputeCtx(lattice=dbsp_time(3)),
        group=fact_group,
    )
    edb_1d = Input[ZSet[dlg.Fact]](
        frontier=Antichain(dbsp_time(1)),
    ).connect(e.circuit, ())
    program_1d = Input[ZSet[dlg.Rule]](
        frontier=Antichain(dbsp_time(1)),
    ).connect(e.circuit, ())
    state_level_3d = Input[ZSet[LevelAtLeast]](
        frontier=Antichain(dbsp_time(3)),
    ).connect(e.circuit, ())
    state_facts_3d = Input[ZSet[dlg.Fact]](
        frontier=Antichain(dbsp_time(3)),
    ).connect(e.circuit, ())
    state_rewrites_3d = Input[ZSet[dlg.ProvenanceIndexedRewrite]](
        frontier=Antichain(dbsp_time(3)),
    ).connect(e.circuit, ())
    seed_1d = Input[ZSet[dlg.ProvenanceIndexedRewrite]](
        frontier=Antichain(dbsp_time(1)),
    ).connect(e.circuit, ())

    (
        level_at_least_out,
        level_next_stratum_out,
        effective_level_3d,
    ) = _connect_running_level_3d(
        e,
        program_1d,
        state_level_3d,
        level_group,
    )

    # Lift the raw 1-D rule stream to 3-D, integrate over all three
    # axes to get the cumulative program, rewrite each rule with its
    # readiness guard against the current effective-level snapshot,
    # then differentiate back to a per-cell guarded-rule delta.
    program_2d = LiftStreamIntroduction[ZSet[dlg.Rule]](
        group=rule_group,
    ).connect(e.circuit, (program_1d,))
    program_3d = LiftStreamIntroduction[ZSet[dlg.Rule]](
        group=rule_group,
    ).connect(e.circuit, (program_2d,))
    cum_program_o = CoreIntegrate[ZSet[dlg.Rule]](
        axis=0,
        group=rule_group,
    ).connect(e.circuit, (program_3d,))
    cum_program_os = CoreIntegrate[ZSet[dlg.Rule]](
        axis=1,
        group=rule_group,
    ).connect(e.circuit, (cum_program_o,))
    cum_program_3d = CoreIntegrate[ZSet[dlg.Rule]](
        axis=2,
        group=rule_group,
    ).connect(e.circuit, (cum_program_os,))
    guarded_cum_3d = Lift2[
        ZSet[dlg.Rule],
        ZSet[tuple[tuple[str], int]],
        ZSet[dlg.Rule],
    ](
        op=_guard_program_with_levels,
    ).connect(e.circuit, (cum_program_3d, effective_level_3d))
    guarded_delta_i = CoreDifferentiate[ZSet[dlg.Rule]](
        axis=2,
        group=rule_group,
    ).connect(e.circuit, (guarded_cum_3d,))
    guarded_delta_si = CoreDifferentiate[ZSet[dlg.Rule]](
        axis=1,
        group=rule_group,
    ).connect(e.circuit, (guarded_delta_i,))
    guarded_delta_3d = CoreDifferentiate[ZSet[dlg.Rule]](
        axis=0,
        group=rule_group,
    ).connect(e.circuit, (guarded_delta_si,))

    if indexed:
        # SORT-MERGE-indexed body — identical IDB, asymptotically faster joins.
        body = IndexedIncrementalDatalogBodyWithNegation(
            fact_group=fact_group,
            rule_group=rule_group,
            rewrite_group=rewrite_group,
            signal_group=ZSetAddition[dlg.Signal](),
            ext_dir_group=ZSetAddition[dlg.ExtendedDirection](),
            jorder_group=ZSetAddition[tuple[str, dlg.ColumnReference]](),
            gatekeep_group=ZSetAddition[dlg.IndexedGatekeepEntry](),
            indexed_fact_group=ZSetAddition[dlg.IndexedFact](),
        )
    else:
        body = IncrementalDatalogBodyWithNegation(
            fact_group=fact_group,
            rule_group=rule_group,
            rewrite_group=rewrite_group,
            signal_group=ZSetAddition[dlg.Signal](),
            dir_group=ZSetAddition[dlg.Direction](),
            gatekeep_group=(ZSetAddition[dlg.AtomWithSourceRewriteAndProvenance]()),
        )
    body_out = body.connect(
        e.circuit,
        (
            edb_1d,
            guarded_delta_3d,
            state_facts_3d,
            state_rewrites_3d,
            seed_1d,
        ),
    )
    return (
        e,
        edb_1d,
        program_1d,
        state_level_3d,
        state_facts_3d,
        state_rewrites_3d,
        seed_1d,
        level_at_least_out,
        level_next_stratum_out,
        guarded_delta_3d,
        body_out,
    )


class IncrementalDatalogStratified:
    """Stratified-negation Datalog evaluator. Truly incremental across
    outer ticks w.r.t. both facts AND rules.

    Construct, push rule and fact deltas, call :meth:`saturate` to
    advance an outer tick. Each ``saturate()`` returns the running
    cumulative derived-facts set. The fused 3-D circuit is built on
    the first call and persists; subsequent calls only push the
    per-outer deltas.

    Stratifiability is validated at each :meth:`push_rules` call —
    a program that ever forms a cycle through negation raises
    immediately. The stratum axis maintains running levels
    incrementally, and the body circuit turns level changes into
    signed guarded-rule value changes rather than moving raw rules
    between timestamps."""

    _indexed: bool = False  # subclasses flip this to use the sort-merge body

    def __init__(
        self,
        *,
        max_inner: int = 1024,
    ) -> None:
        self._max_inner = max_inner
        self._rule_group = ZSetAddition[dlg.Rule]()
        self._fact_group = ZSetAddition[dlg.Fact]()

        self._cumulative_rules: ZSet[dlg.Rule] = ZSet({})
        self._pending_rules: ZSet[dlg.Rule] = ZSet({})
        self._pending_edb: ZSet[dlg.Fact] = ZSet({})

        self._next_outer = 0
        self._cumulative_facts: ZSet[dlg.Fact] = ZSet({})
        self._body_e: Evaluator[Time] | None = None
        self._body_handles: tuple = ()

        # ``_scheduled_strata`` is monotone — it does not shrink when
        # the current program's max level decreases, because historical
        # upper-stratum state still needs cleanup retractions.
        self._scheduled_strata: Antichain[Time] = Antichain(dbsp_time(1))
        self._ready_strata: Antichain[Time] = Antichain(dbsp_time(1))

    def push_rules(self, rules: ZSet[dlg.Rule]) -> None:
        """Accumulate a rule delta for the next :meth:`saturate`.
        Validates stratifiability of the program-so-far each time;
        raises if a cycle through negation appears."""
        new_program = self._rule_group.add(
            self._cumulative_rules,
            self._rule_group.add(self._pending_rules, rules),
        )
        _check_stratifiable(new_program)
        dlg.check_interpreted_safety(new_program)
        self._pending_rules = self._rule_group.add(
            self._pending_rules,
            rules,
        )

    def push_facts(self, facts: ZSet[dlg.Fact]) -> None:
        """Accumulate a fact delta for the next :meth:`saturate`."""
        self._pending_edb = self._fact_group.add(self._pending_edb, facts)

    def _setup_circuits(self) -> None:
        """Build the persistent 3-D circuit on the first
        :meth:`saturate`. Pushes the rewrite-identity seed at outer 0."""
        (
            body_e,
            edb_1d,
            program_1d,
            state_level_3d,
            sf_3d,
            sr_3d,
            seed_1d,
            level_out,
            level_next,
            guarded_delta_3d,
            body_out,
        ) = _build_stratified_body_circuit(indexed=self._indexed)
        body_e.push(
            seed_1d,
            ZSet({(0, dlg._rewrite_monoid.identity()): 1}),
        )
        self._body_e = body_e
        self._body_handles = (
            edb_1d,
            program_1d,
            state_level_3d,
            sf_3d,
            sr_3d,
            seed_1d,
            level_out,
            level_next,
            guarded_delta_3d,
            body_out,
        )

    def saturate(self) -> ZSet[dlg.Fact]:
        """Advance one outer tick. Push pending rule + EDB deltas;
        walk the stratum axis while the running level subgraph exposes
        work, drive the shared inner loop per stratum, accumulate the
        outer-delta, and return the running cumulative."""
        if self._body_e is None:
            self._setup_circuits()
        assert self._body_e is not None

        rule_delta = self._pending_rules
        self._pending_rules = ZSet({})
        edb_delta = self._pending_edb
        self._pending_edb = ZSet({})
        self._cumulative_rules = self._rule_group.add(
            self._cumulative_rules,
            rule_delta,
        )

        outer = self._next_outer
        self._next_outer += 1

        (
            edb_1d,
            program_1d,
            state_level_3d,
            sf_3d,
            sr_3d,
            _seed_1d,
            level_out,
            level_next,
            guarded_delta_3d,
            body_out,
        ) = self._body_handles

        if rule_delta.inner:
            self._body_e.push(program_1d, rule_delta, t=(outer,))
        if edb_delta.inner:
            self._body_e.push(edb_1d, edb_delta, t=(outer,))
        frontiers = self._body_e.frontiers()
        level_state_frontier = frontiers[state_level_3d]
        body_state_frontier = frontiers[sf_3d]

        # Walk the stratum axis while the running level subgraph
        # exposes work: clock stratum s if its historical frontier
        # covers it OR the previous stratum produced negative-edge
        # lookahead.
        outer_delta = self._fact_group.identity()
        inner_cursor = 0
        s = 0
        self._scheduled_strata.insert((0,))
        while self._scheduled_strata.covers((s,)):
            seed_ready = not self._ready_strata.covers((s,))
            prior_stratum_depth = -1
            if outer > 0:
                prior_stratum_depth = max(
                    _max_inner_before_outer(
                        level_state_frontier,
                        outer,
                        s,
                    ),
                    _max_inner_before_outer(
                        body_state_frontier,
                        outer,
                        s,
                    ),
                )
            min_inner = max(prior_stratum_depth, inner_cursor - 1)
            k = 0 if s == 0 else (prior_stratum_depth if prior_stratum_depth >= 0 else inner_cursor)
            schedule_next = False
            while k < self._max_inner:
                diff_level = self._body_e.read(level_out, (outer, s, k))
                next_level = self._body_e.read(level_next, (outer, s, k))
                guarded_delta = self._body_e.read(
                    guarded_delta_3d,
                    (outer, s, k),
                )
                diff_facts, diff_rewrites = self._body_e.read(
                    body_out,
                    (outer, s, k),
                )
                state_seed = (
                    _stratum_ready_delta(s) if seed_ready and k == inner_cursor else self._fact_group.identity()
                )
                empty = (
                    not diff_level.inner
                    and not next_level.inner
                    and not guarded_delta.inner
                    and not diff_facts.inner
                    and not diff_rewrites.inner
                    and not state_seed.inner
                )
                if next_level.inner:
                    schedule_next = True
                if empty and k > min_inner:
                    break
                if diff_facts.inner:
                    public_diff = _without_internal_facts(diff_facts)
                    outer_delta = self._fact_group.add(
                        outer_delta,
                        public_diff,
                    )
                self._body_e.push(
                    state_level_3d,
                    diff_level,
                    t=(outer, s, k),
                )
                state_facts = self._fact_group.add(diff_facts, state_seed)
                self._body_e.push(sf_3d, state_facts, t=(outer, s, k))
                self._body_e.push(sr_3d, diff_rewrites, t=(outer, s, k))
                k += 1
            else:
                raise RuntimeError(f"stratum {s} at outer {outer} did not converge in {self._max_inner} inner ticks")
            inner_cursor = k
            if schedule_next:
                self._scheduled_strata.insert((s + 1,))
            self._ready_strata.insert((s,))
            s += 1

        self._cumulative_facts = self._fact_group.add(
            self._cumulative_facts,
            outer_delta,
        )
        return self._cumulative_facts


class IndexedIncrementalDatalogStratified(IncrementalDatalogStratified):
    """Drop-in for :class:`IncrementalDatalogStratified` whose body uses the
    SORT-MERGE-indexed joins (:class:`IndexedIncrementalDatalogBodyWithNegation`).
    Identical API, identical results — only the per-saturation join cost differs
    (O(matching keys) vs O(|left|·|right|)), so it scales to large dense EDBs."""

    _indexed = True


class IndexedDatalogWithNegation:
    """Single-batch 2-D (outer × inner) SORT-MERGE-indexed Datalog WITH NEGATION —
    **no stratum axis**. Same ``push_rules`` / ``push_facts`` / ``saturate`` API as
    :class:`IncrementalDatalogStratified`, but drives only the 2-D indexed negation
    body (:class:`~pydbsp.indexed_relational_operators.IndexedIncrementalDatalogWithNegationBody`)
    over a single outer tick to the inner fixpoint.

    CORRECT for programs whose negation is a SINGLE stratum — every negated predicate
    is computable independently of (never recursive through) the predicate that
    negates it. For such programs the single 2-D fixpoint coincides with the
    stratified semantics, and this is markedly faster than the 3-D stratified driver
    (one fewer lattice axis + far less machinery). NOT a general stratified evaluator:
    a genuine multi-stratum program (``c :- !b`` where ``b :- !a``) needs
    :class:`IncrementalDatalogStratified`."""

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
        body = IndexedIncrementalDatalogWithNegationBody(
            fact_group=fg, rule_group=rug, rewrite_group=rg, signal_group=ZSetAddition[dlg.Signal](),
            ext_dir_group=ZSetAddition[dlg.ExtendedDirection](), jorder_group=ZSetAddition[tuple[str, dlg.ColumnReference]](),
            gatekeep_group=ZSetAddition[dlg.IndexedGatekeepEntry](), indexed_fact_group=ZSetAddition[dlg.IndexedFact](),
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


class DatalogWithNegation:
    """Single-batch 2-D (outer × inner) NON-INDEXED Datalog WITH NEGATION — the
    non-indexed counterpart of :class:`IndexedDatalogWithNegation`, driving the plain
    :class:`~pydbsp.relational_operators.IncrementalDatalogWithNegationBody` (``pred``-based
    joins, no sort-merge index). Same ``push_rules`` / ``push_facts`` / ``saturate`` API and
    the same single-stratum-negation correctness domain; slower (O(n²) joins) but index-free.
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

        for a in range(self._max_alternation):
            # Scan floor from the evaluator's OWN progress plane: cross-tick join
            # terms (z⁻¹ on axis 1) can fire at any inner coordinate the previous
            # tick reached, so scan at least as deep as the sf frontier covers.
            floor = _max_inner_before_outer(e.frontiers()[sf], 1, a - 1) if a > 0 else -1
            i = 0
            while i < self._max_inner:
                diff_facts, diff_rewrites = e.read(body, (0, a, i))
                if not diff_facts.inner and not diff_rewrites.inner and i >= floor:
                    break
                e.push(sf, diff_facts, t=(0, a, i))
                e.push(sr, diff_rewrites, t=(0, a, i))
                i += 1
            else:
                raise RuntimeError(f"alternation tick {a} did not converge in {self._max_inner} inner ticks")
            tick_delta = e.read(tick_delta_node, (0, a, i))
            if not tick_delta.inner:  # physically quiescent: zero delta at the fixpoint, stable forever
                # cover the deepest inner cell ANY tick reached (earlier ticks may
                # have derived, then retracted, deeper than the quiescent tick)
                cov = max(i, _max_inner_at_outer(e.frontiers()[sf], 0))
                return _without_internal_facts(e.read(model_node, (0, a, cov)))
            # the alternation feedback edge: ferry the tick's model delta from the
            # output port to the reference port (the driver-materialized ∫-barrier,
            # same discipline as every pydbsp feedback edge); it is consumed at
            # tick a+1 through the in-circuit z⁻¹ on axis 1
            e.push(neg_ref, tick_delta, t=(0, a, 0))
        raise RuntimeError(
            f"alternating fixpoint did not converge within {self._max_alternation} ticks "
            f"(non-total well-founded model? stratifiability is checked at push time)"
        )
