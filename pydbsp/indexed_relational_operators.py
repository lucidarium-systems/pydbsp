"""Indexed relational operators. Sort-merge variants of the bilinear
joins, building on :mod:`pydbsp.indexed_zset`.

Reuses :class:`pydbsp.indexed_zset.IndexedZSet` (a Z-set sidecar with
a key→value index) and its :class:`IndexedZSetAddition` group as-is.
The algebra-layer integration is value-only:

* :class:`LiftIndex`. Wraps a ``ZSet[V]`` stream into an
  ``IndexedZSet[K, V]`` stream pointwise (a single :class:`Lift1`).
* :class:`IndexedDeltaLiftedDeltaLiftedJoin`. The same 4-term
  bilinear DBSP join as :class:`DeltaLiftedDeltaLiftedJoin`, but each
  inner ``Lift2`` term invokes :func:`sort_merge_join` over the
  shared index, replacing the ``O(|left| × |right|)`` nested-loop
  with an ``O(|matching|)`` key-matched scan.
* :class:`LiftGroupBy`. The general relational ``GROUP BY ...
  AGGREGATE``: takes an ``IndexedZSet[K, V]`` stream and an aggregate
  function over the bucket's ``(V, int)`` records, emits
  ``ZSet[(K, C)]`` per cell. The grouping is already materialised
  by an upstream :class:`LiftIndex`, so the reducer is a single pass
  over the index.

Storage protocol is unchanged: indexed values are stored opaquely
just like plain Z-sets. The performance win is entirely value-layer.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from pydbsp import datalog as dlg
from pydbsp import datalog as dli
from pydbsp.indexed_zset import IndexedZSet, IndexedZSetAddition, _Comparable
from pydbsp.indexed_zset import sort_merge_join
from pydbsp.zset import ZSet, ZSetAddition

from pydbsp.circuit import Circuit
from pydbsp.operator import (
    CoreDelay,
    CoreIntegrate,
    Delay,
    Differentiate,
    Integrate,
    Lift1,
    Lift2,
    LiftDelay,
    LiftDifferentiate,
    LiftIntegrate,
    LiftStreamIntroduction,
    Operator,
)
from pydbsp.progress import NodeId, Time
from pydbsp.relational_operators import (
    DeltaLiftedDeltaLiftedDistinct,
    LiftLiftProject,
    LiftLiftSelect,
    LiftSelect,
)


@dataclass(frozen=True)
class LiftIndex[V, K: _Comparable](Operator):
    """``↑Index``. Wrap a ``ZSet[V]`` stream into an
    ``IndexedZSet[K, V]`` stream, pointwise.

    Pure :class:`Lift1`. The indexing is per-cell. The benefit is
    that downstream operators consuming the indexed view can do
    ``O(|matching|)`` key-lookups instead of scanning the full Z-set.

    The output value type is ``IndexedZSet[K, V]``. An
    :class:`IndexedZSetAddition` group is required by downstream
    Integrate / Delay primitives. Construct it once at circuit-build
    time and pass to the consumers (see
    :class:`IndexedDeltaLiftedDeltaLiftedJoin`).

    ``inputs = (diff_stream,)``."""

    indexer: Callable[[V], K]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        (s,) = inputs
        indexer = self.indexer
        return Lift1[ZSet[V], IndexedZSet[K, V]](f=lambda z: IndexedZSet(z.inner, indexer)).connect(circuit, (s,))


@dataclass(frozen=True)
class LiftLiftIndex[V, K: _Comparable](LiftIndex[V, K]):
    """``↑↑Index``. Pointwise lift to a 2-D stream-of-streams. Same
    implementation as :class:`LiftIndex` (Lift1 is arity-agnostic). The distinct name documents the intended stream shape."""


@dataclass(frozen=True)
class LiftGroupBy[V, K: _Comparable, C](Operator):
    """``↑γ``. The general relational ``GROUP BY ... AGGREGATE``
    on an :class:`IndexedZSet` stream.

    Per cell, for each key ``k`` present in the input's index, apply
    ``aggregate`` to the bucket's ``(record, weight)`` records and
    emit ``(k, aggregate(bucket)) → 1`` into the output Z-set.

    The aggregator receives an ``Iterable[tuple[V, int]]``. The
    bucket's records with their multiplicities. No sub-Z-set is
    materialised per group. The iterator is over the indexed
    bucket directly.

    Composes with :class:`LiftIndex` upstream::

        indexed = LiftIndex[V, K](indexer=key_fn).connect(c, (src,))
        agg = LiftGroupBy[V, K, C](aggregate=lambda items: ...).connect(c, (indexed,))

    Multiple aggregations on the same key share the indexing work::

        total = LiftGroupBy(aggregate=sum_amount).connect(c, (indexed,))
        peak = LiftGroupBy(aggregate=max_amount).connect(c, (indexed,))

    Per-tick / cumulative semantics:

    * Per-tick delta aggregate: ``LiftGroupBy(LiftIndex(diff))``. The
      output Z-set is a per-tick snapshot of each group's aggregate.
    * Cumulative across ticks: ``LiftGroupBy(Integrate(LiftIndex(diff)))``.
      Re-aggregates the full cumulative every tick (``O(|state|)``),
      correct for any aggregate. For sparse updates, prefer
      :class:`DeltaLiftedDeltaLiftedGroupBy`, which emits the cumulative
      view's delta touching only changed keys.

    Output encoding: each group emits ``(K, C) → 1``. This is the
    *value-encoded* form. If you compose with :class:`Integrate`, be
    aware: distinct ``C`` values for the same ``K`` accumulate as
    separate entries, which is correct for "list of all aggregates
    seen" semantics but **not** for "running sum-of-aggregates"
    semantics. For linear aggregates (sum / count), prefer the
    weight-encoded ``Lift1`` trick over this operator.

    ``inputs = (indexed_stream,)``."""

    aggregate: Callable[[Iterable[tuple[V, int]]], C]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        (s,) = inputs
        agg = self.aggregate

        def reduce(z: IndexedZSet[K, V]) -> ZSet[tuple[K, C]]:
            out: dict[tuple[K, C], int] = {}
            for k, bucket in z.index_to_value.items():
                # IndexedZSetAddition's invariant: buckets only hold
                # records with non-zero weight in z.inner. No filter
                # needed here.
                items = ((t, z.inner[t]) for t in bucket)
                out[(k, agg(items))] = 1
            return ZSet(out)

        return Lift1[IndexedZSet[K, V], ZSet[tuple[K, C]]](f=reduce).connect(circuit, (s,))


@dataclass(frozen=True)
class LiftLiftGroupBy[V, K: _Comparable, C](LiftGroupBy[V, K, C]):
    """``↑↑γ``. Pointwise lift to a 2-D stream-of-streams. Same
    implementation as :class:`LiftGroupBy`."""


@dataclass(frozen=True)
class DeltaLiftedDeltaLiftedGroupBy[V, K: _Comparable, C](Operator):
    """``Dᵒ(G_agg(z⁻¹ⁱ Iⁱ Iᵒ s, Iᵒ s))``. Doubly-incremental
    ``GROUP BY ... AGGREGATE`` — the incremental counterpart of
    :class:`LiftGroupBy`, generalizing
    :class:`pydbsp.relational_operators.DeltaLiftedDeltaLiftedDistinct`
    (distinct is this operator with ``K`` = element identity, ``aggregate``
    = positive-weight indicator, ``H`` as its per-element kernel).

    Per cell, the kernel touches only the **keys changed this step**:
    for each, it re-reads the bucket from the doubly-integrated state,
    aggregates old vs. old+delta, and emits ``-(k, agg(old)) + (k,
    agg(new))``. So ``O(Σ_{k changed} |bucket_k|)`` per cell, vs.
    ``LiftGroupBy(Integrate(...))``'s ``O(|state|)``. Correct for
    **any** aggregate (min / max / median included): changed buckets
    are re-scanned in full, so no invertibility is assumed. The output
    is a proper Z-set diff, so ``Integrate`` of it is the current
    ``{(k, agg(group_k)) → 1}`` snapshot — without the value-encoding
    footgun of integrating :class:`LiftGroupBy`'s output.

    ``inputs = (indexed_diff_stream,)`` (``IndexedZSet[K, V]``, usually
    :class:`LiftIndex` over a 2-D source). ``group`` is the indexed
    group for Integrate / Delay; ``out_group`` the ``ZSet[(K, C)]``
    group for the final Differentiate. ``aggregate`` takes the bucket's
    ``Iterable[tuple[V, int]]`` ``(record, weight)`` pairs, as in
    :class:`LiftGroupBy`."""

    aggregate: Callable[[Iterable[tuple[V, int]]], C]
    group: IndexedZSetAddition[K, V]
    out_group: ZSetAddition[tuple[K, C]]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        (s,) = inputs
        agg = self.aggregate

        def kernel(i: IndexedZSet[K, V], d: IndexedZSet[K, V]) -> ZSet[tuple[K, C]]:
            # i = z⁻¹ⁱ Iⁱ Iᵒ s : previous cumulative grouped state.
            # d = Iᵒ s         : this inner step's changed keys (small).
            # Iterate only the changed keys; look up old buckets in i.
            out: dict[tuple[K, C], int] = {}
            for k, delta_bucket in d.index_to_value.items():
                old_bucket = i.index_to_value.get(k)
                if old_bucket:
                    old_kv = (k, agg([(t, i.inner[t]) for t in old_bucket]))
                    out[old_kv] = out.get(old_kv, 0) - 1
                    new_weights = {t: i.inner[t] for t in old_bucket}
                else:
                    new_weights = {}
                for t in delta_bucket:
                    new_weights[t] = new_weights.get(t, 0) + d.inner[t]
                new_items = [(t, w) for t, w in new_weights.items() if w != 0]
                if new_items:
                    new_kv = (k, agg(new_items))
                    out[new_kv] = out.get(new_kv, 0) + 1
            return ZSet({kv: w for kv, w in out.items() if w != 0})

        integrated = Integrate[IndexedZSet[K, V]](group=self.group).connect(circuit, (s,))
        int_int = LiftIntegrate[IndexedZSet[K, V]](group=self.group).connect(circuit, (integrated,))
        del_int_int = LiftDelay[IndexedZSet[K, V]](group=self.group).connect(circuit, (int_int,))
        g = Lift2[IndexedZSet[K, V], IndexedZSet[K, V], ZSet[tuple[K, C]]](op=kernel).connect(
            circuit, (del_int_int, integrated)
        )
        return Differentiate[ZSet[tuple[K, C]]](group=self.out_group).connect(circuit, (g,))


@dataclass(frozen=True)
class IndexedDeltaLiftedDeltaLiftedJoin[K: _Comparable, A, B, C](Operator):
    """Indexed 4-term doubly-incremental sort-merge join on a 2-axis
    lattice. Same algebraic shape as
    :class:`pydbsp.relational_operators.DeltaLiftedDeltaLiftedJoin`:

        J(z⁻¹ᵒ Iᵒ a,   z⁻¹ⁱ Iⁱ b)
      + J(Iᵒ Iⁱ a,      b)
      + J(Iⁱ a,          z⁻¹ᵒ Iᵒ b)
      + J(a,             z⁻¹ⁱ Iᵒ Iⁱ b)

    Both operands must already be ``IndexedZSet[K, *]`` streams
    (typically the output of :class:`LiftIndex` on the same key
    function). The inner ``Lift2`` joins delegate to
    :func:`pydbsp.indexed_zset.sort_merge_join`
    over the shared key index. Output is a plain ``ZSet[C]``.

    ``inputs = (diff_indexed_a, diff_indexed_b)``.
    ``proj(k, a, b) -> C | None``. Return ``None`` to drop a pair.
    """

    proj: Callable[[K, A, B], C | None]
    group_a: IndexedZSetAddition[K, A]
    group_b: IndexedZSetAddition[K, B]
    out_group: ZSetAddition[C]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        if len(inputs) != 2:
            raise ValueError(f"IndexedDeltaLiftedDeltaLiftedJoin takes 2 inputs; got {len(inputs)}")
        diff_a, diff_b = inputs

        int_a_o = Integrate[IndexedZSet[K, A]](group=self.group_a).connect(circuit, (diff_a,))
        del_int_a_o = Delay[IndexedZSet[K, A]](group=self.group_a).connect(circuit, (int_a_o,))

        int_b_o = Integrate[IndexedZSet[K, B]](group=self.group_b).connect(circuit, (diff_b,))
        del_int_b_o = Delay[IndexedZSet[K, B]](group=self.group_b).connect(circuit, (int_b_o,))

        int_a_i = LiftIntegrate[IndexedZSet[K, A]](group=self.group_a).connect(circuit, (diff_a,))
        int_b_i = LiftIntegrate[IndexedZSet[K, B]](group=self.group_b).connect(circuit, (diff_b,))

        int_a_oi = Integrate[IndexedZSet[K, A]](group=self.group_a).connect(circuit, (int_a_i,))
        int_b_oi = Integrate[IndexedZSet[K, B]](group=self.group_b).connect(circuit, (int_b_i,))

        del_int_b_i = LiftDelay[IndexedZSet[K, B]](group=self.group_b).connect(circuit, (int_b_i,))
        del_int_b_oi = LiftDelay[IndexedZSet[K, B]](group=self.group_b).connect(circuit, (int_b_oi,))

        # Inner Lift2 is sort-merge over the shared index — O(|matching
        # keys|) per cell instead of the plain bilinear's O(|left| · |right|).
        proj = self.proj
        term_op: Callable[[IndexedZSet[K, A], IndexedZSet[K, B]], ZSet[C]] = lambda la, lb: sort_merge_join(
            la, lb, proj
        )

        def _join(left: NodeId, right: NodeId) -> NodeId:
            return Lift2[IndexedZSet[K, A], IndexedZSet[K, B], ZSet[C]](op=term_op).connect(circuit, (left, right))

        j1 = _join(del_int_a_o, del_int_b_i)
        j2 = _join(int_a_oi, diff_b)
        j3 = _join(int_a_i, del_int_b_o)
        j4 = _join(diff_a, del_int_b_oi)

        add = self.out_group.add
        sum_12 = Lift2[ZSet[C], ZSet[C], ZSet[C]](op=add).connect(circuit, (j1, j2))
        sum_34 = Lift2[ZSet[C], ZSet[C], ZSet[C]](op=add).connect(circuit, (j3, j4))
        return Lift2[ZSet[C], ZSet[C], ZSet[C]](op=add).connect(circuit, (sum_12, sum_34))


@dataclass(frozen=True)
class IndexedDeltaLiftedDeltaLiftedDeltaLiftedJoin[K: _Comparable, A, B, C](Operator):
    """Indexed 8-term triply-incremental sort-merge join on a 3-axis lattice
    (outer = axis 0, stratum = axis 1, inner = axis 2). The indexed analog of
    :class:`pydbsp.datalog_stratified.DeltaLiftedDeltaLiftedDeltaLiftedJoin`:
    identical subset-of-axes decomposition (``Δ³c = Σ_{Sₐ⊆{o,s,i}} J(z⁻¹^Sₐ I^Sₐ a,
    I^complement b)``, 2³ = 8 terms), with the per-term value-level join replaced by
    a :func:`sort_merge_join` over the shared key index — O(matching keys) per cell
    instead of O(|left|·|right|).

    Both operands are ``IndexedZSet[K, *]`` streams (typically ``LiftIndex`` output).
    ``inputs = (diff_indexed_a, diff_indexed_b)``; ``proj(k, a, b) -> C | None``.
    """

    proj: Callable[[K, A, B], C | None]
    group_a: IndexedZSetAddition[K, A]
    group_b: IndexedZSetAddition[K, B]
    out_group: ZSetAddition[C]

    def connect(self, circuit: Circuit[Time], inputs: tuple[NodeId, ...]) -> NodeId:
        if len(inputs) != 2:
            raise ValueError(f"IndexedDeltaLiftedDeltaLiftedDeltaLiftedJoin takes 2 inputs; got {len(inputs)}")
        diff_a, diff_b = inputs

        # For every subset S ⊆ {0,1,2}: ``z⁻¹^S I^S diff_a`` (delayed) and ``I^S diff_b``.
        a_paths: dict[frozenset[int], NodeId] = {frozenset(): diff_a}
        b_paths: dict[frozenset[int], NodeId] = {frozenset(): diff_b}
        for axis in (0, 1, 2):
            new_a: dict[frozenset[int], NodeId] = {}
            for subset, node in a_paths.items():
                integrated = CoreIntegrate[IndexedZSet[K, A]](axis=axis, group=self.group_a).connect(circuit, (node,))
                delayed = CoreDelay[IndexedZSet[K, A]](axis=axis, group=self.group_a).connect(circuit, (integrated,))
                new_a[subset | {axis}] = delayed
            a_paths.update(new_a)
            new_b: dict[frozenset[int], NodeId] = {}
            for subset, node in b_paths.items():
                integrated = CoreIntegrate[IndexedZSet[K, B]](axis=axis, group=self.group_b).connect(circuit, (node,))
                new_b[subset | {axis}] = integrated
            b_paths.update(new_b)

        # Inner term: sort-merge over the shared index (a single Lift2 — the lattice
        # arity, not the lift depth, supplies the 3-D-ness; cf. LiftJoin == one Lift2).
        proj = self.proj
        term_op: Callable[[IndexedZSet[K, A], IndexedZSet[K, B]], ZSet[C]] = lambda la, lb: sort_merge_join(la, lb, proj)

        def _join(left: NodeId, right: NodeId) -> NodeId:
            return Lift2[IndexedZSet[K, A], IndexedZSet[K, B], ZSet[C]](op=term_op).connect(circuit, (left, right))

        all_axes = frozenset({0, 1, 2})
        terms = [
            _join(a_paths[sa], b_paths[all_axes - sa])
            for sa in (
                frozenset(), frozenset({0}), frozenset({1}), frozenset({2}),
                frozenset({0, 1}), frozenset({0, 2}), frozenset({1, 2}), frozenset({0, 1, 2}),
            )
        ]
        add = self.out_group.add
        result = terms[0]
        for t in terms[1:]:
            result = Lift2[ZSet[C], ZSet[C], ZSet[C]](op=add).connect(circuit, (result, t))
        return result


# ============================================================================
# Indexed body operators — same wiring as the plain versions in
# ``pydbsp.relational_operators`` but each ``DLDJoin`` is replaced
# with an :class:`IndexedDeltaLiftedDeltaLiftedJoin`, with the
# operands pre-indexed via :class:`LiftIndex`. Reuses pydbsp's
# value-layer helpers (``ext_dir``, ``jorder``, ``_index_fact``,
# ``_gatekeep_join_key``, ``_indexed_product_proj``) directly.
# ============================================================================


@dataclass(frozen=True)
class IndexedIncrementalReachabilityBody[N: _Comparable](Operator):
    """Indexed variant of
    :class:`pydbsp.relational_operators.IncrementalReachabilityBody`.

    Replaces the bilinear hop join with a sort-merge variant: edges
    are pre-indexed by source vertex, delayed state by destination
    vertex, and the join scans matching keys.

    ``inputs = (edges_1d, state_2d)``. Same as the plain version."""

    edge_group: ZSetAddition[tuple[N, N]]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        if len(inputs) != 2:
            raise ValueError(f"IndexedIncrementalReachabilityBody takes 2 inputs; got {len(inputs)}")
        edges_1d, state_2d = inputs

        edges_2d = LiftStreamIntroduction[ZSet[tuple[N, N]]](group=self.edge_group).connect(circuit, (edges_1d,))
        delayed_state = LiftDelay[ZSet[tuple[N, N]]](group=self.edge_group).connect(circuit, (state_2d,))

        indexed_state = LiftIndex[tuple[N, N], N](indexer=lambda e: e[1]).connect(circuit, (delayed_state,))
        indexed_edges = LiftIndex[tuple[N, N], N](indexer=lambda e: e[0]).connect(circuit, (edges_2d,))

        state_group = IndexedZSetAddition[N, tuple[N, N]](self.edge_group, lambda e: e[1])
        edge_group = IndexedZSetAddition[N, tuple[N, N]](self.edge_group, lambda e: e[0])

        hop = IndexedDeltaLiftedDeltaLiftedJoin[N, tuple[N, N], tuple[N, N], tuple[N, N]](
            proj=lambda _k, l, r: (l[0], r[1]),
            group_a=state_group,
            group_b=edge_group,
            out_group=self.edge_group,
        ).connect(circuit, (indexed_state, indexed_edges))

        fresh = Lift2[ZSet[tuple[N, N]], ZSet[tuple[N, N]], ZSet[tuple[N, N]]](op=self.edge_group.add).connect(
            circuit, (hop, edges_2d)
        )

        return DeltaLiftedDeltaLiftedDistinct[tuple[N, N]](inner_group=self.edge_group).connect(circuit, (fresh,))


@dataclass(frozen=True)
class IndexedIncrementalDatalogBody(Operator):
    """Indexed variant of
    :class:`pydbsp.relational_operators.IncrementalDatalogBody`.

    Mirrors :func:`pydbsp.algorithms.datalog_indexed.IncrementalDatalogWithIndexing`:

    * uses :func:`pydbsp.algorithms.datalog_indexed.ext_dir` /
      :func:`jorder` instead of :func:`datalog.dir`.
    * pre-indexes facts by ``(predicate, col_ref)`` via
      :func:`_index_fact`.
    * each of the three sequential joins becomes an indexed sort-merge.

    ``inputs = (edb_1d, program_1d, state_facts_2d, state_rewrites_2d,
    seed_1d)``. Same as the plain body."""

    fact_group: ZSetAddition[dlg.Fact]
    rule_group: ZSetAddition[dlg.Rule]
    rewrite_group: ZSetAddition[dlg.ProvenanceIndexedRewrite]
    signal_group: ZSetAddition[dlg.Signal]
    ext_dir_group: ZSetAddition[dli.ExtendedDirection]
    jorder_group: ZSetAddition[tuple[str, dli.ColumnReference]]
    gatekeep_group: ZSetAddition[dli.IndexedGatekeepEntry]
    indexed_fact_group: ZSetAddition[dli.IndexedFact]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        if len(inputs) != 5:
            raise ValueError(f"IndexedIncrementalDatalogBody takes 5 inputs; got {len(inputs)}")
        edb_1d, program_2d, state_facts_2d, state_rewrites_2d, seed_1d = inputs

        edb_2d = LiftStreamIntroduction[ZSet[dlg.Fact]](group=self.fact_group).connect(circuit, (edb_1d,))
        seed_2d = LiftStreamIntroduction[ZSet[dlg.ProvenanceIndexedRewrite]](group=self.rewrite_group).connect(
            circuit, (seed_1d,)
        )

        delayed_facts = LiftDelay[ZSet[dlg.Fact]](group=self.fact_group).connect(circuit, (state_facts_2d,))
        delayed_rewrites = LiftDelay[ZSet[dlg.ProvenanceIndexedRewrite]](group=self.rewrite_group).connect(
            circuit, (state_rewrites_2d,)
        )

        sig_2d = Lift1[ZSet[dlg.Rule], ZSet[dlg.Signal]](f=dlg.sig).connect(circuit, (program_2d,))
        ext_dir_2d = Lift1[ZSet[dlg.Rule], ZSet[dli.ExtendedDirection]](f=dli.ext_dir).connect(circuit, (program_2d,))
        jorder_2d = Lift1[ZSet[dlg.Rule], ZSet[tuple[str, dli.ColumnReference]]](f=dli.jorder).connect(
            circuit, (program_2d,)
        )

        indexed_rewrites = LiftIndex[
            dlg.ProvenanceIndexedRewrite,
            dlg.Provenance,
        ](indexer=lambda rw: rw[0]).connect(circuit, (delayed_rewrites,))
        indexed_ext_dir = LiftIndex[
            dli.ExtendedDirection,
            dlg.Provenance,
        ](indexer=lambda ed: ed[0]).connect(circuit, (ext_dir_2d,))

        rewrite_idx_group = IndexedZSetAddition[
            dlg.Provenance,
            dlg.ProvenanceIndexedRewrite,
        ](self.rewrite_group, lambda rw: rw[0])
        ext_dir_idx_group = IndexedZSetAddition[
            dlg.Provenance,
            dli.ExtendedDirection,
        ](self.ext_dir_group, lambda ed: ed[0])

        gatekeep = IndexedDeltaLiftedDeltaLiftedJoin[
            dlg.Provenance,
            dlg.ProvenanceIndexedRewrite,
            dli.ExtendedDirection,
            dli.IndexedGatekeepEntry,
        ](
            proj=lambda _k, l, r: (r[1], r[2], l[1], r[3]),
            group_a=rewrite_idx_group,
            group_b=ext_dir_idx_group,
            out_group=self.gatekeep_group,
        ).connect(circuit, (indexed_rewrites, indexed_ext_dir))

        indexed_facts_by_pred = LiftIndex[dlg.Fact, str](indexer=lambda f: f[0]).connect(circuit, (delayed_facts,))
        indexed_jorder_by_pred = LiftIndex[
            tuple[str, dli.ColumnReference],
            str,
        ](indexer=lambda j: j[0]).connect(circuit, (jorder_2d,))

        fact_pred_group = IndexedZSetAddition[str, dlg.Fact](self.fact_group, lambda f: f[0])
        jorder_pred_group = IndexedZSetAddition[
            str,
            tuple[str, dli.ColumnReference],
        ](self.jorder_group, lambda j: j[0])

        indexed_facts_raw = IndexedDeltaLiftedDeltaLiftedJoin[
            str,
            dlg.Fact,
            tuple[str, dli.ColumnReference],
            dli.IndexedFact,
        ](
            proj=lambda _k, fact, jo: dli._index_fact(jo[1], fact),
            group_a=fact_pred_group,
            group_b=jorder_pred_group,
            out_group=self.indexed_fact_group,
        ).connect(circuit, (indexed_facts_by_pred, indexed_jorder_by_pred))

        distinct_indexed_facts = DeltaLiftedDeltaLiftedDistinct[dli.IndexedFact](
            inner_group=self.indexed_fact_group
        ).connect(circuit, (indexed_facts_raw,))

        gatekeep_idx_group = IndexedZSetAddition[
            dli.JoinKey,
            dli.IndexedGatekeepEntry,
        ](self.gatekeep_group, dli._gatekeep_join_key)
        indexed_fact_idx_group = IndexedZSetAddition[
            dli.JoinKey,
            dli.IndexedFact,
        ](self.indexed_fact_group, lambda ifact: ifact[0])

        keyed_gatekeep = LiftIndex[
            dli.IndexedGatekeepEntry,
            dli.JoinKey,
        ](indexer=dli._gatekeep_join_key).connect(circuit, (gatekeep,))
        keyed_facts = LiftIndex[
            dli.IndexedFact,
            dli.JoinKey,
        ](indexer=lambda ifact: ifact[0]).connect(circuit, (distinct_indexed_facts,))

        product = IndexedDeltaLiftedDeltaLiftedJoin[
            dli.JoinKey,
            dli.IndexedGatekeepEntry,
            dli.IndexedFact,
            dlg.ProvenanceIndexedRewrite,
        ](
            proj=dli._indexed_product_proj,
            group_a=gatekeep_idx_group,
            group_b=indexed_fact_idx_group,
            out_group=self.rewrite_group,
        ).connect(circuit, (keyed_gatekeep, keyed_facts))

        indexed_product = LiftIndex[
            dlg.ProvenanceIndexedRewrite,
            dlg.Provenance,
        ](indexer=lambda r: r[0]).connect(circuit, (product,))
        indexed_sig = LiftIndex[dlg.Signal, dlg.Provenance](indexer=lambda s: s[0]).connect(circuit, (sig_2d,))

        sig_idx_group = IndexedZSetAddition[dlg.Provenance, dlg.Signal](self.signal_group, lambda s: s[0])

        ground = IndexedDeltaLiftedDeltaLiftedJoin[
            dlg.Provenance,
            dlg.ProvenanceIndexedRewrite,
            dlg.Signal,
            dlg.Fact,
        ](
            proj=lambda _k, l, r: l[1].apply(r[1]),
            group_a=rewrite_idx_group,
            group_b=sig_idx_group,
            out_group=self.fact_group,
        ).connect(circuit, (indexed_product, indexed_sig))

        ground_plus_edb = Lift2[ZSet[dlg.Fact], ZSet[dlg.Fact], ZSet[dlg.Fact]](op=self.fact_group.add).connect(
            circuit, (ground, edb_2d)
        )
        next_facts = DeltaLiftedDeltaLiftedDistinct[dlg.Fact](inner_group=self.fact_group).connect(
            circuit, (ground_plus_edb,)
        )

        product_plus_seed = Lift2[
            ZSet[dlg.ProvenanceIndexedRewrite],
            ZSet[dlg.ProvenanceIndexedRewrite],
            ZSet[dlg.ProvenanceIndexedRewrite],
        ](op=self.rewrite_group.add).connect(circuit, (product, seed_2d))
        next_rewrites = DeltaLiftedDeltaLiftedDistinct[dlg.ProvenanceIndexedRewrite](
            inner_group=self.rewrite_group
        ).connect(circuit, (product_plus_seed,))

        return Lift2[
            ZSet[dlg.Fact],
            ZSet[dlg.ProvenanceIndexedRewrite],
            tuple[ZSet[dlg.Fact], ZSet[dlg.ProvenanceIndexedRewrite]],
        ](op=lambda a, b: (a, b)).connect(circuit, (next_facts, next_rewrites))


@dataclass(frozen=True)
class IndexedIncrementalDatalogWithNegationBody(Operator):
    """Indexed variant of
    :class:`pydbsp.relational_operators.IncrementalDatalogWithNegationBody`.

    Same shape as :class:`IndexedIncrementalDatalogBody` plus the
    split-gatekeep + anti-product chain from the plain negation body.
    Each of the four joins (gatekeep, indexed_facts_raw, product,
    anti_product, ground) is sort-merge-indexed.

    ``inputs = (edb_1d, program_2d, state_facts_2d, state_rewrites_2d,
    seed_1d)``. Same as the plain negation body. ``program_2d`` is the
    rule stream already lifted to 2-D; wrap a 1-D ``Input`` in
    :class:`LiftStreamIntroduction` if you do not have a 2-D source."""

    fact_group: ZSetAddition[dlg.Fact]
    rule_group: ZSetAddition[dlg.Rule]
    rewrite_group: ZSetAddition[dlg.ProvenanceIndexedRewrite]
    signal_group: ZSetAddition[dlg.Signal]
    ext_dir_group: ZSetAddition[dli.ExtendedDirection]
    jorder_group: ZSetAddition[tuple[str, dli.ColumnReference]]
    gatekeep_group: ZSetAddition[dli.IndexedGatekeepEntry]
    indexed_fact_group: ZSetAddition[dli.IndexedFact]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        if len(inputs) != 5:
            raise ValueError(f"IndexedIncrementalDatalogWithNegationBody takes 5 inputs; got {len(inputs)}")
        edb_1d, program_2d, state_facts_2d, state_rewrites_2d, seed_1d = inputs

        edb_2d = LiftStreamIntroduction[ZSet[dlg.Fact]](group=self.fact_group).connect(circuit, (edb_1d,))
        seed_2d = LiftStreamIntroduction[ZSet[dlg.ProvenanceIndexedRewrite]](group=self.rewrite_group).connect(
            circuit, (seed_1d,)
        )

        delayed_facts = LiftDelay[ZSet[dlg.Fact]](group=self.fact_group).connect(circuit, (state_facts_2d,))
        delayed_rewrites = LiftDelay[ZSet[dlg.ProvenanceIndexedRewrite]](group=self.rewrite_group).connect(
            circuit, (state_rewrites_2d,)
        )

        sig_2d = Lift1[ZSet[dlg.Rule], ZSet[dlg.Signal]](f=dlg.sig).connect(circuit, (program_2d,))
        ext_dir_2d = Lift1[ZSet[dlg.Rule], ZSet[dli.ExtendedDirection]](f=dli.ext_dir).connect(circuit, (program_2d,))
        jorder_2d = Lift1[ZSet[dlg.Rule], ZSet[tuple[str, dli.ColumnReference]]](f=dli.jorder).connect(
            circuit, (program_2d,)
        )

        indexed_rewrites = LiftIndex[
            dlg.ProvenanceIndexedRewrite,
            dlg.Provenance,
        ](indexer=lambda rw: rw[0]).connect(circuit, (delayed_rewrites,))
        indexed_ext_dir = LiftIndex[
            dli.ExtendedDirection,
            dlg.Provenance,
        ](indexer=lambda ed: ed[0]).connect(circuit, (ext_dir_2d,))

        rewrite_idx_group = IndexedZSetAddition[
            dlg.Provenance,
            dlg.ProvenanceIndexedRewrite,
        ](self.rewrite_group, lambda rw: rw[0])
        ext_dir_idx_group = IndexedZSetAddition[
            dlg.Provenance,
            dli.ExtendedDirection,
        ](self.ext_dir_group, lambda ed: ed[0])

        gatekeep = IndexedDeltaLiftedDeltaLiftedJoin[
            dlg.Provenance,
            dlg.ProvenanceIndexedRewrite,
            dli.ExtendedDirection,
            dli.IndexedGatekeepEntry,
        ](
            proj=lambda _k, l, r: (r[1], r[2], l[1], r[3]),
            group_a=rewrite_idx_group,
            group_b=ext_dir_idx_group,
            out_group=self.gatekeep_group,
        ).connect(circuit, (indexed_rewrites, indexed_ext_dir))

        indexed_facts_by_pred = LiftIndex[dlg.Fact, str](indexer=lambda f: f[0]).connect(circuit, (delayed_facts,))
        indexed_jorder_by_pred = LiftIndex[
            tuple[str, dli.ColumnReference],
            str,
        ](indexer=lambda j: j[0]).connect(circuit, (jorder_2d,))

        fact_pred_group = IndexedZSetAddition[str, dlg.Fact](self.fact_group, lambda f: f[0])
        jorder_pred_group = IndexedZSetAddition[
            str,
            tuple[str, dli.ColumnReference],
        ](self.jorder_group, lambda j: j[0])

        indexed_facts_raw = IndexedDeltaLiftedDeltaLiftedJoin[
            str,
            dlg.Fact,
            tuple[str, dli.ColumnReference],
            dli.IndexedFact,
        ](
            proj=lambda _k, fact, jo: dli._index_fact(jo[1], fact),
            group_a=fact_pred_group,
            group_b=jorder_pred_group,
            out_group=self.indexed_fact_group,
        ).connect(circuit, (indexed_facts_by_pred, indexed_jorder_by_pred))
        distinct_indexed_facts = DeltaLiftedDeltaLiftedDistinct[dli.IndexedFact](
            inner_group=self.indexed_fact_group
        ).connect(circuit, (indexed_facts_raw,))

        positive_atoms = LiftLiftSelect[dli.IndexedGatekeepEntry](
            pred=lambda gk: gk[1] is None or ("!" not in gk[1][0])
        ).connect(circuit, (gatekeep,))
        negative_atoms = LiftLiftSelect[dli.IndexedGatekeepEntry](
            pred=lambda gk: not (gk[1] is None or ("!" not in gk[1][0]))
        ).connect(circuit, (gatekeep,))

        gatekeep_idx_group = IndexedZSetAddition[
            dli.JoinKey,
            dli.IndexedGatekeepEntry,
        ](self.gatekeep_group, dli._gatekeep_join_key)
        indexed_fact_idx_group = IndexedZSetAddition[
            dli.JoinKey,
            dli.IndexedFact,
        ](self.indexed_fact_group, lambda ifact: ifact[0])

        keyed_positive = LiftIndex[
            dli.IndexedGatekeepEntry,
            dli.JoinKey,
        ](indexer=dli._gatekeep_join_key).connect(circuit, (positive_atoms,))
        keyed_facts = LiftIndex[
            dli.IndexedFact,
            dli.JoinKey,
        ](indexer=lambda ifact: ifact[0]).connect(circuit, (distinct_indexed_facts,))

        product = IndexedDeltaLiftedDeltaLiftedJoin[
            dli.JoinKey,
            dli.IndexedGatekeepEntry,
            dli.IndexedFact,
            dlg.ProvenanceIndexedRewrite,
        ](
            proj=dli._indexed_product_proj,
            group_a=gatekeep_idx_group,
            group_b=indexed_fact_idx_group,
            out_group=self.rewrite_group,
        ).connect(circuit, (keyed_positive, keyed_facts))

        proj = LiftLiftProject[
            dli.IndexedGatekeepEntry,
            dlg.ProvenanceIndexedRewrite,
        ](f=lambda gk: (gk[0], gk[2])).connect(circuit, (negative_atoms,))

        keyed_negative = LiftIndex[
            dli.IndexedGatekeepEntry,
            dli.JoinKey,
        ](indexer=dli._gatekeep_join_key).connect(circuit, (negative_atoms,))

        anti_product = IndexedDeltaLiftedDeltaLiftedJoin[
            dli.JoinKey,
            dli.IndexedGatekeepEntry,
            dli.IndexedFact,
            dlg.ProvenanceIndexedRewrite,
        ](
            proj=lambda _k, gk, _ifact: (gk[0], gk[2]),
            group_a=gatekeep_idx_group,
            group_b=indexed_fact_idx_group,
            out_group=self.rewrite_group,
        ).connect(circuit, (keyed_negative, keyed_facts))

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
        final_product = Lift2[
            ZSet[dlg.ProvenanceIndexedRewrite],
            ZSet[dlg.ProvenanceIndexedRewrite],
            ZSet[dlg.ProvenanceIndexedRewrite],
        ](op=self.rewrite_group.add).connect(circuit, (product_plus_proj, neg_anti))

        indexed_final_product = LiftIndex[
            dlg.ProvenanceIndexedRewrite,
            dlg.Provenance,
        ](indexer=lambda r: r[0]).connect(circuit, (final_product,))
        indexed_sig = LiftIndex[dlg.Signal, dlg.Provenance](indexer=lambda s: s[0]).connect(circuit, (sig_2d,))

        sig_idx_group = IndexedZSetAddition[dlg.Provenance, dlg.Signal](self.signal_group, lambda s: s[0])

        ground = IndexedDeltaLiftedDeltaLiftedJoin[
            dlg.Provenance,
            dlg.ProvenanceIndexedRewrite,
            dlg.Signal,
            dlg.Fact,
        ](
            proj=lambda _k, l, r: l[1].apply(r[1]),
            group_a=rewrite_idx_group,
            group_b=sig_idx_group,
            out_group=self.fact_group,
        ).connect(circuit, (indexed_final_product, indexed_sig))

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


@dataclass(frozen=True)
class StratificationRewriter(Operator):
    """Takes a user rule stream plus the level circuit's ``level(P, S)``
    fact stream, and emits a rule stream where every rule has been
    prepended with a ``("stratum_ready", (s - 1,))`` guard atom. ``s``
    is the max level derived for the rule's head predicate.

    With the guard in place, no rewrite in
    :class:`IncrementalDatalogBody` (or its negation variant) can walk
    past the rule's first body atom until the caller pushes
    ``stratum_ready(s - 1)`` into the body's EDB. That mechanism lets a
    stratified-Datalog driver collapse to a "saturate, push next
    ready, repeat" loop with no explicit per-stratum bookkeeping.

    ``inputs = (program_2d, level_facts_2d)``.

    * ``program_2d`` (``ZSet[Rule]``, 2-D): user rules, already lifted.
      For a 1-D ``Input``, wrap in :class:`LiftStreamIntroduction`.
    * ``level_facts_2d`` (``ZSet[Fact]``, 2-D): the fact stream from
      the level circuit's body, where ``level(P, S)`` rows live
      alongside the dependency EDB. The rewriter filters to
      ``level/2`` internally.

    Output is a 2-D ``ZSet[Rule]`` of guarded rules, plug-compatible
    with the body's ``program_2d`` input slot.

    Implementation: the cumulative-max-per-predicate step uses the
    standard ``Integrate → LiftIndex → LiftGroupBy → Differentiate``
    aggregate pattern. The join itself is a 4-term indexed bilinear,
    keyed on the head predicate name. Re-stratification on rule deltas
    happens for free via Z-set retraction propagating through the
    bilinear formula."""

    rule_group: ZSetAddition[dlg.Rule]
    fact_group: ZSetAddition[dlg.Fact]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        if len(inputs) != 2:
            raise ValueError(f"StratificationRewriter takes 2 inputs (program, level_facts); got {len(inputs)}")
        program_2d, level_facts_2d = inputs

        level_only = LiftSelect[dlg.Fact](
            pred=lambda fact: fact[0] == "level",
        ).connect(circuit, (level_facts_2d,))

        # Cumulative-max per predicate: integrate on both axes (level
        # facts come out across multiple inner ticks while the level
        # circuit saturates), group-by predicate taking max, then
        # differentiate back to delta form for the bilinear join below.
        outer_cum = Integrate[ZSet[dlg.Fact]](
            group=self.fact_group,
        ).connect(circuit, (level_only,))
        full_cum = LiftIntegrate[ZSet[dlg.Fact]](
            group=self.fact_group,
        ).connect(circuit, (outer_cum,))

        indexed_cum = LiftIndex[dlg.Fact, str](
            indexer=lambda fact: fact[1][0],
        ).connect(circuit, (full_cum,))
        max_levels_cum = LiftGroupBy[dlg.Fact, str, int](
            aggregate=lambda items: max(fact[1][1] for fact, w in items if w > 0),
        ).connect(circuit, (indexed_cum,))

        kv_group: ZSetAddition[tuple[str, int]] = ZSetAddition()
        max_levels_inner_delta = LiftDifferentiate[ZSet[tuple[str, int]]](
            group=kv_group,
        ).connect(circuit, (max_levels_cum,))
        max_levels_delta = Differentiate[ZSet[tuple[str, int]]](
            group=kv_group,
        ).connect(circuit, (max_levels_inner_delta,))

        indexed_rules = LiftIndex[dlg.Rule, str](
            indexer=lambda rule: rule[0][0],
        ).connect(circuit, (program_2d,))
        indexed_levels = LiftIndex[tuple[str, int], str](
            indexer=lambda kv: kv[0],
        ).connect(circuit, (max_levels_delta,))

        rule_idx_group = IndexedZSetAddition[str, dlg.Rule](self.rule_group, lambda r: r[0][0])
        level_idx_group = IndexedZSetAddition[str, tuple[str, int]](kv_group, lambda kv: kv[0])

        # ``proj`` prepends a ``stratum_ready(level - 1)`` guard atom so
        # the rule fires only once its readiness fact is in the cube.
        return IndexedDeltaLiftedDeltaLiftedJoin[
            str,
            dlg.Rule,
            tuple[str, int],
            dlg.Rule,
        ](
            proj=lambda _k, rule, kv: (
                rule[0],
                ("stratum_ready", (kv[1] - 1,)),
                *rule[1:],
            ),
            group_a=rule_idx_group,
            group_b=level_idx_group,
            out_group=self.rule_group,
        ).connect(circuit, (indexed_rules, indexed_levels))
