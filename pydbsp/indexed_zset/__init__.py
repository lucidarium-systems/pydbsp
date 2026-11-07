"""Indexed Z-set: a Z-set carrying a side-car B-tree index over some
derived key ``I = indexer(T)``. The abelian group semantics are
identical to ``ZSet``. The index is an auxiliary structure that
accelerates sort-merge joins.

``IndexedZSetAddition`` is parameterised by the ``indexer`` function. The indexer is part of the group instance, not the carrier.
"""

from bisect import bisect_right, insort
from collections.abc import Callable, Generator, Iterator
from typing import Any, Protocol, TypeVar

from pydbsp.core import AbelianGroupOperation
from pydbsp.zset import ZSet, ZSetAddition


class _Comparable(Protocol):
    """Minimal ordering bound: anything that supports ``<``.
    ``bisect``, ``insort``, and ``sort_merge_keys`` all require it.

    ``other`` is typed ``Any`` rather than ``object`` because the
    stdlib types (``str``, ``int``, ``tuple``) declare ``__lt__`` against
    their own type, not ``object`` — the wider ``Any`` parameter is
    necessary for them to satisfy this Protocol."""

    def __lt__(self, other: Any, /) -> bool: ...


class AppendOnlySpine[T: _Comparable]:
    """Flat B-tree for sorted iteration. Ported from v0.6.0. A
    leaf-level chunked list keyed by per-chunk max.
    """

    _len: int
    _load: int
    _lists: list[list[T]]
    _maxes: list[T]

    def __init__(self) -> None:
        self._len = 0
        self._load = 1024
        self._lists = []
        self._maxes = []

    def __len__(self) -> int:
        return self._len

    def add(self, value: T) -> None:
        if self._maxes:
            pos = bisect_right(self._maxes, value)
            if pos == len(self._maxes):
                pos -= 1
                self._lists[pos].append(value)
                self._maxes[pos] = value
            else:
                insort(self._lists[pos], value)
            self._expand(pos)
        else:
            self._lists.append([value])
            self._maxes.append(value)
        self._len += 1

    def __iter__(self) -> Generator[T, None, None]:
        for sublist in self._lists:
            yield from sublist

    def _expand(self, pos: int) -> None:
        if len(self._lists[pos]) > (self._load << 1):
            chunk = self._lists[pos]
            half = chunk[self._load :]
            del chunk[self._load :]
            self._maxes[pos] = chunk[-1]
            self._lists.insert(pos + 1, half)
            self._maxes.insert(pos + 1, half[-1])


def sort_merge_keys[K: _Comparable](a: AppendOnlySpine[K], b: AppendOnlySpine[K]) -> Iterator[K]:
    """Yield keys present in both spines (in sorted order). Keys are
    consumed one at a time. Callers handle multiplicity."""
    it_a: Iterator[K] = iter(a)
    it_b: Iterator[K] = iter(b)
    try:
        x = next(it_a)
        y = next(it_b)
        while True:
            if x < y:
                x = next(it_a)
            elif y < x:
                y = next(it_b)
            else:
                yield x
                x = next(it_a)
                y = next(it_b)
    except StopIteration:
        return


I = TypeVar("I")


class IndexedZSet[I: _Comparable, T]:
    """Z-set with an index on ``indexer(value)``. The ``inner`` dict
    is the Z-set carrier. ``index_to_value`` and ``index`` are the
    side-car lookup structures maintained on insertion.
    """

    inner: dict[T, int]
    index_to_value: dict[I, set[T]]
    indexer: Callable[[T], I]
    index: AppendOnlySpine[I]

    def __init__(self, values: dict[T, int], indexer: Callable[[T], I]) -> None:
        self.inner = {}
        self.index_to_value = {}
        self.index = AppendOnlySpine()
        self.indexer = indexer
        for k, v in values.items():
            self[k] = v

    @classmethod
    def _from_parts(
        cls,
        inner: dict[T, int],
        index_to_value: dict[I, set[T]],
        index: AppendOnlySpine[I],
        indexer: Callable[[T], I],
    ) -> "IndexedZSet[I, T]":
        out: IndexedZSet[I, T] = cls.__new__(cls)
        out.inner = inner
        out.index_to_value = index_to_value
        out.index = index
        out.indexer = indexer
        return out

    def __repr__(self) -> str:
        return self.inner.__repr__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IndexedZSet):
            return False
        return self.inner == other.inner

    def __contains__(self, item: T) -> bool:
        return item in self.inner

    def __getitem__(self, item: T) -> int:
        return self.inner.get(item, 0)

    def __setitem__(self, key: T, weight: int) -> None:
        already = key in self.inner
        self.inner[key] = weight
        if already:
            return
        idx = self.indexer(key)
        bucket = self.index_to_value.get(idx)
        if bucket is None:
            self.index_to_value[idx] = {key}
            self.index.add(idx)
        else:
            bucket.add(key)


class IndexedZSetAddition[I: _Comparable, T](AbelianGroupOperation[IndexedZSet[I, T]]):
    """Abelian group over ``IndexedZSet[I, T]``. Semantics match
    ``ZSetAddition``. Weight-wise add, zero-weight elimination.
    The indexer is captured in the group instance. ``add`` returns
    a fresh indexed zset with a rebuilt index.
    """

    inner_group: ZSetAddition[T]
    indexer: Callable[[T], I]

    def __init__(self, inner_group: ZSetAddition[T], indexer: Callable[[T], I]) -> None:
        self.inner_group = inner_group
        self.indexer = indexer

    def add(self, a: IndexedZSet[I, T], b: IndexedZSet[I, T]) -> IndexedZSet[I, T]:
        # Light COW:
        #   ``inner`` always forks (small, changes every add).
        #   ``index_to_value`` — shallow-copy top-level dict on first
        #   bucket mutation; per-bucket forks only when touched.
        #   ``index`` spine — deep-clone only when a new index key
        #   needs appending (``b``'s keys whose ``indexer(k)`` is
        #   absent from ``a``'s bucket map).
        # Relies on the convention that results are read-only after
        # construction (no post-hoc ``__setitem__``).
        # The scheme assumes ``b`` is the small side; add is
        # commutative, so orient the operands to guarantee it (the
        # ``Foldl``/``Sum`` recurrence passes the cumulative as ``b``,
        # which would otherwise fork every bucket and clone the spine
        # on every tick).
        if not b.inner:
            return a
        if not a.inner:
            return b
        if len(a.inner) < len(b.inner):
            a, b = b, a
        inner: dict[T, int] = dict(a.inner)
        i2v: dict[I, set[T]] | None = None
        spine: AppendOnlySpine[I] | None = None
        forked_buckets: set[I] = set()
        for k, v in b.inner.items():
            prev = inner.get(k, 0)
            w = prev + v
            if w == 0:
                inner.pop(k, None)
            else:
                inner[k] = w
            # Index transitions: prev==0,w!=0 → add; prev!=0,w==0 → remove;
            # prev!=0,w!=0 → no change (key already indexed).
            if prev == 0 and w == 0:
                continue
            if prev != 0 and w != 0:
                continue
            idx = self.indexer(k)
            if i2v is None:
                i2v = dict(a.index_to_value)
            if prev == 0:
                # New key: add to bucket/spine.
                bucket = i2v.get(idx)
                if bucket is None:
                    if spine is None:
                        spine = AppendOnlySpine()
                        spine._len = a.index._len
                        spine._load = a.index._load
                        spine._lists = [list(chunk) for chunk in a.index._lists]
                        spine._maxes = list(a.index._maxes)
                    spine.add(idx)
                    i2v[idx] = {k}
                    forked_buckets.add(idx)
                else:
                    if idx not in forked_buckets:
                        bucket = set(bucket)
                        i2v[idx] = bucket
                        forked_buckets.add(idx)
                    bucket.add(k)
            else:
                # Cancellation (w==0): remove k from the bucket; if the
                # bucket becomes empty, drop it from ``i2v``. ``spine`` is
                # append-only so orphan idx entries can remain — downstream
                # ``sort_merge_keys`` tolerates them because buckets are
                # looked up via ``index_to_value`` which we keep clean.
                bucket = i2v.get(idx)
                if bucket is not None:
                    if idx not in forked_buckets:
                        bucket = set(bucket)
                        i2v[idx] = bucket
                        forked_buckets.add(idx)
                    bucket.discard(k)
                    if not bucket:
                        del i2v[idx]
        return IndexedZSet._from_parts(
            inner,
            a.index_to_value if i2v is None else i2v,
            a.index if spine is None else spine,
            self.indexer,
        )

    def neg(self, a: IndexedZSet[I, T]) -> IndexedZSet[I, T]:
        # Keys are unchanged — share ``index_to_value`` and ``index``.
        return IndexedZSet._from_parts(
            {k: -v for k, v in a.inner.items()},
            a.index_to_value,
            a.index,
            self.indexer,
        )

    def identity(self) -> IndexedZSet[I, T]:
        return IndexedZSet({}, self.indexer)


# ---- Sort-merge join (value-layer helper) ---------------------------------


def sort_merge_join[I: _Comparable, A, B, C](
    left: IndexedZSet[I, A],
    right: IndexedZSet[I, B],
    proj: Callable[[I, A, B], "C | None"],
) -> ZSet[C]:
    """Equi-join via sort-merge over the shared index. ``proj`` may
    return ``None`` to drop a pair.

    **Invariant:** ``index_to_value[k]`` only contains values whose
    weight in ``inner`` is nonzero. Producers (IndexedZSetAddition)
    must maintain this. We no longer guard per-pair.
    """
    from itertools import product as cartesian_product

    out: dict[C, int] = {}
    l_inner = left.inner
    r_inner = right.inner
    for key in sort_merge_keys(left.index, right.index):
        # Spine is append-only; a cancellation in
        # ``IndexedZSetAddition.add`` removes a key from
        # ``index_to_value`` but leaves it in ``index``. Use ``.get``
        # and skip orphan keys.
        l_bucket = left.index_to_value.get(key)
        r_bucket = right.index_to_value.get(key)
        if not l_bucket or not r_bucket:
            continue
        for l_val, r_val in cartesian_product(l_bucket, r_bucket):
            c = proj(key, l_val, r_val)
            if c is None:
                continue
            out[c] = out.get(c, 0) + l_inner[l_val] * r_inner[r_val]
    return ZSet({k: v for k, v in out.items() if v != 0})
