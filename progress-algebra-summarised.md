# Antichain Algebra

## The time lattice

Every stream in pydbsp lives over **`DBSPTime[N]`**, the product of `N` copies of `ℕ ∪ {ω}` ordered componentwise. A *timestamp* is an `N`-tuple of non-negative integers (with `ω` allowed). One tuple is ≤ another iff every component is.

For `N = 1`, the order is plain `≤` on natural numbers. For `N ≥ 2`, two timestamps can be **incomparable**. For example, `(1, 3)` and `(3, 1)` neither dominates the other. That incomparability is the whole reason we need the machinery below. Subsets of a partial order cannot be described by a single max or min.

## Down-sets

A **down-set** of the time lattice is a subset closed downward. If `t ∈ S` and `t' ≤ t`, then `t' ∈ S`. Examples:

- The single timestamp `(5, 3)` has the down-set `{ (i, j) : i ≤ 5 ∧ j ≤ 3 }`.
- The set of timestamps an observer is contractually allowed to read on a given stream is a down-set. Observation grows monotonically, so once `t` becomes observable, every `t' ≤ t` was observable already.

A down-set is fully described by its **maximal elements**, the elements with no element of the set strictly above them. For a single timestamp those maxima are just `{t}`. For the union of two down-sets they may be several incomparable points.

## Antichains

An **antichain** is exactly that, a set of pairwise-incomparable elements. Every antichain `A` represents a down-set

```
down-set(A) = { t : ∃ a ∈ A. t ≤ a }
```

and every finitely-generated down-set has a unique antichain of maximal elements. Antichains are the **compact representation of progress**. Instead of tracking a possibly-infinite set of settled timestamps, we keep a small set of frontier points.

`pydbsp.core.Antichain` enforces the invariant in one method:

- `Antichain.insert(t)`. If `t` is already in the down-set, no-op. Otherwise drop any existing element that `t` dominates, then add `t`. The down-set strictly grows.

### Running example

A 2-D input growing under successive pushes. Each row shows the antichain `A` after the push and the down-set it represents.

| step | act | antichain `A` | `down-set(A)` |
|---|---|---|---|
| 0 | init | `⊥` (empty) | `∅` |
| 1 | push `(1, 0)` | `{(1, 0)}` | `{(0,0), (1,0)}` |
| 2 | push `(0, 1)` | `{(1, 0), (0, 1)}`. Incomparable, both kept | `{(0,0), (1,0), (0,1)}` |
| 3 | push `(2, 0)` | `{(0, 1), (2, 0)}`. `(1,0)` is dominated and dropped | `{(0,0), (1,0), (2,0), (0,1)}` |
| 4 | push `(1, 1)` | `{(2, 0), (1, 1)}`. `(0,1)` is dominated and dropped | `{(0,0), (1,0), (2,0), (0,1), (1,1)}` |

Step 4 visualised. `■` is covered, `□` is not, antichain elements are circled.

```
 j
 2  □    □    □
 1  ■   (■)   □       ← (1,1) ∈ A
 0  ■    ■   (■)      ← (2,0) ∈ A
    0    1    2    i
```

Notice. At step 3 and step 4 an existing antichain element is dropped by `insert`, yet the down-set strictly grows. The antichain is the *compact* representation. The truth is the down-set.

## Antichain lattice

Antichains over the time lattice form their **own** bounded distributive lattice, ordered by down-set inclusion.

| | meaning | computed as |
|---|---|---|
| `⊥` | empty antichain, down-set = ∅ | `Antichain(L)` with no elements |
| `⊤` | universal, down-set = entire lattice | `Antichain.universal(L)` |
| `A ⊑ B` | `down-set(A) ⊆ down-set(B)` | every element of `A` is covered by some element of `B` |
| `A ⊔ B` | union of down-sets | insert every element of `A ∪ B` into a fresh antichain |
| `A ⊓ B` | intersection of down-sets | insert every pairwise time-lattice meet `a ⊓_L b` into a fresh antichain |
| `A.covers(t)` | `t ∈ down-set(A)` | `∃ a ∈ A. t ≤_L a` |

This is the toolkit on which the rest of the document depends.

## Per-axis primitives on `DBSPTime` antichains

`pydbsp.progress` adds five primitives that exploit the product-of-`NaturalChain` structure of `DBSPTime`. A generic `BoundedBelowLattice` antichain has no axes, so these primitives type-check against the lattice carried by the antichain and raise on a mismatch.

| primitive | meaning |
|---|---|
| `shift(A, axis)` | per-element successor on `axis`. Empty stays empty. Universal is a fixed point |
| `retreat(A, axis)` | per-element predecessor on `axis`. Elements at the axis bottom drop |
| `retreat_omega_fill(A, axis)` | predecessor on `axis`, with every non-`axis` coordinate replaced by `ω`. HISTORICAL: the pre-v2.1.2 backward contribution, refuted in `V3/ProgressAlgebra.lean` (`rofDead_not_sound_atOrAfter`, `rofDead_not_tight_strictAdvance`); no shape uses it since v2.1.2 |
| `dead_below(A)` | the largest antichain-representable dead region STRICTLY below the up-set of `A`: per element, one `(c[j] − 1 at axis j, ω elsewhere)` piece for every axis with `0 < c[j] < ω`, met across elements. The `maxDead` complement rule, machine-checked (`maxDead_sound` / `maxDead_maximal` / `maxDead_eq_iUnion`) |
| `drop_axis(A, axis)` | project onto a lattice with one fewer axis by removing `axis` |
| `insert_axis(A, axis, v)` | lift onto a lattice with one more axis by inserting `v` at `axis`. Use `ω` for "settled everywhere on this axis" |

## Settled frontier

Every node has a **settled frontier**, an antichain whose down-set is exactly the set of timestamps at which the node is **observable**. The contract for callers is that a read at `t` on a node is legal iff the node's frontier covers `t`. The value rule of the node is total. It returns `group.identity()` for cells that nobody has pushed yet. The frontier carves out the subset the observer is allowed to look at.

Progress is the frontier's down-set monotonically growing. A node "advances" when an upstream input antichain gains an element and the new frontier (recomputed by `propagate_forward`) covers more cells than before.

## Forward propagation

Progress flows from inputs to roots. `pydbsp.progress.propagate_forward(nodes)` walks the arena in declaration order and returns a per-node antichain. At every node, the rule reads its inputs' frontiers (which the walk has already computed) and produces its own.

Six **progress shapes** cover the operators currently in the library. Each shape is a frozen dataclass that pairs its `inputs` tuple with a `forward` and a `backward` method.

| shape | operator | `forward` rule |
|---|---|---|
| `Input` | source nodes | the antichain stored on the shape. Mutated by `Evaluator.push` |
| `Identity` | `Lift1` (and the input edge of `Integrate`) | passthrough of the single input |
| `Meet` | `Lift2` and any N-ary pointwise op | `⊓` of all inputs' frontiers |
| `AxisShift` | `Delay(axis)`, `LiftDelay` | `shift(input, axis)`, plus a `⊥` seed when the input frontier is empty |
| `AxisIntroduction` | `StreamIntroduction`, `LiftStreamIntroduction` | `insert_axis(input, axis, ω)`. Arity changes |
| `Feedback` | `Integrate(axis)`, `LiftIntegrate` | identity on the input. The strict-feedback theorem closes the loop |

Composite operators connect as multiple nodes. `Differentiate(axis)` lays down three nodes (one `AxisShift`, one `Identity`, one `Meet`) rather than carrying its own progress shape.

## Backward propagation

Compaction reverses direction. `Evaluator.compact(cursors)` seeds `propagate_backward(nodes, cursors)` with caller-supplied cursors (each a promise that future reads at that node happen at-or-after some cursor element), walks the arena in reverse order, and returns a per-node **dead antichain**. Storage entries whose timestamps fall in a dead down-set are evicted.

**Seeding (v2.1.2).** The seed dead region is `dead_below` of the cursor — strictly below its up-set, so cursor cells themselves stay readable. Seeding AT the cursor was the pre-v2.1.2 rule; it evicted the at-cursor memo and is refuted as `atCursor_not_sound` in `V3/ProgressAlgebra.lean`. From the seeds on, the accumulators hold DEAD REGIONS, and each shape's `backward` transports a dead region to its inputs — the argument a shape receives is a dead region, not a reader cursor.

The walk is one-shot rather than fixpoint-iterating. At every node, consumer contributions are met into the accumulator before the node's own `backward` runs. For `Feedback` shapes the self-edge feeds back into the accumulator only after the call returns, so the contribution to other targets uses the pre-meet region. Iterating to fixpoint would converge to `⊥` and free nothing.

| shape | `forward` | `backward` (dead-region transport, v2.1.2+) |
|---|---|---|
| `Input` | stored antichain | empty (no upstream) |
| `Identity` | passthrough | passthrough on the input |
| `Meet` | `⊓` at fan-in | every input receives the dead region |
| `AxisShift` | `shift(input, axis)` | `retreat(dead, axis)` on the input (`needed_eq_Ici`) |
| `AxisIntroduction` | `insert_axis(input, axis, ω)` | ω-filtered `drop_axis`: only elements settled at `ω` on the introduced axis project down (`introBackward_sound` in `V3/EliminationAlgebra.lean` — an input cell is safe only when its ENTIRE output fiber is dead) |
| `Feedback` | passthrough | input gets the dead region; self-edge gets `retreat(dead, axis)`. Sound under the boundary-memo discipline: recomputes of live cells stop at the retained boundary snapshot (`snapshotStore_sound` / `per_tick_lower_bound` — snapshot-plus-deltas is the minimum retained set) |

**History.** Until v2.1.2 the `AxisShift` and `Feedback` self-edge rows used `retreat_omega_fill` on the CURSOR, justified by a tightness argument ("Future `Delay` reads at `(2, k)` will read input at `(1, k)`, never at outer = 0"). `V3/ProgressAlgebra.lean` refuted that rule twice: it is unsound under the documented at-or-after reader contract (`rofDead_not_sound_atOrAfter`, at this document's own example cursor `(1, 29)`), and not tight even under its own strict-advance reading (`rofDead_not_tight_strictAdvance`). The tight rule is the complement of the needed up-set — the per-axis union `dead_below` now computes at the seed (`maxDead_maximal`). Sixteen pre-v2.1.2 backward tests were re-derived by hand against the corrected rules.

**Caller contract caveat.** The no-arg `Evaluator.compact()` uses each node's settled frontier as its cursor — the monotone-forward contract. Drivers whose read pattern RESTARTS an axis each outer tick (a stratum or alternation walk revisits low coordinates at the next tick) violate that contract once ω-closures coarsen the frontier; they must pass explicit next-tick-start cursors instead, and the `Feedback` boundary memos then carry the history (snapshot-plus-deltas).
