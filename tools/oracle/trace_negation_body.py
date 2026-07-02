"""Instrumented trace of the NON-INDEXED 2-D negation body.

Rebuilds the exact wiring of ``IncrementalDatalogWithNegationBody`` inline
(so every intermediate node id is held), runs the ``DatalogWithNegation``
driver loop, and prints per inner step k:

  - the DIFF each node emits at (0, k)
  - the CUMULATIVE facts F_k and rewrites R_k (the Lean ``s_k``)
  - orbit-invariant checks:
      (i)   E subset of F_k          (facts re-inject E)
      (ii)  negated-pred facts in E  (semipositive: never violated)
      (iii) any NEGATIVE weight anywhere (semipositive: never appears)

Run:  uv run python trace_negation_body.py  (from the pydbsp repo root)
"""

from __future__ import annotations

import pydbsp.datalog_stratified as ds
import pydbsp.relational_operators as ro
from pydbsp import datalog as dlg
from pydbsp.zset import ZSet, ZSetAddition

MAXW = 100


def fmt_item(x) -> str:
    s = repr(x)
    return s if len(s) <= MAXW else s[: MAXW - 1] + "…"


def zfmt(z: ZSet) -> str:
    if not z.inner:
        return "∅"
    parts = [f"{fmt_item(k)}:{w:+d}" for k, w in z.inner.items()]
    return "{ " + " ; ".join(parts) + " }"


def build(rules: ZSet, facts: ZSet):
    fg = ZSetAddition[dlg.Fact]()
    rug = ZSetAddition[dlg.Rule]()
    rg = ZSetAddition[dlg.ProvenanceIndexedRewrite]()
    dg = ZSetAddition[dlg.Direction]()
    sg = ZSetAddition[dlg.Signal]()
    gg = ZSetAddition[dlg.AtomWithSourceRewriteAndProvenance]()

    e = ds.Evaluator(
        circuit=ds.Circuit(),
        storage=ds.DictStorage(),
        ctx=ds.ComputeCtx(lattice=ds.dbsp_time(2)),
        group=fg,
    )
    c = e.circuit
    edb_1d = ds.Input(frontier=ds.Antichain(ds.dbsp_time(1))).connect(c, ())
    program_1d = ds.Input(frontier=ds.Antichain(ds.dbsp_time(1))).connect(c, ())
    sf = ds.Input(frontier=ds.Antichain(ds.dbsp_time(2))).connect(c, ())
    sr = ds.Input(frontier=ds.Antichain(ds.dbsp_time(2))).connect(c, ())
    seed = ds.Input(frontier=ds.Antichain(ds.dbsp_time(1))).connect(c, ())
    program_2d = ds.LiftStreamIntroduction(group=rug).connect(c, (program_1d,))

    n: dict[str, object] = {}
    n["edb_2d"] = ro.LiftStreamIntroduction(group=fg).connect(c, (edb_1d,))
    n["seed_2d"] = ro.LiftStreamIntroduction(group=rg).connect(c, (seed,))
    n["delayed_facts"] = ro.LiftDelay(group=fg).connect(c, (sf,))
    n["delayed_rewrites"] = ro.LiftDelay(group=rg).connect(c, (sr,))
    n["sig"] = ro.Lift1(f=dlg.sig).connect(c, (program_2d,))
    n["dir"] = ro.Lift1(f=dlg.dir).connect(c, (program_2d,))

    n["gatekeep"] = ro.DeltaLiftedDeltaLiftedJoin(
        pred=lambda left, right: left[0] == right[0],
        proj=lambda left, right: (right[1], right[2], left[1]),
        group_a=rg,
        group_b=dg,
        out_group=gg,
    ).connect(c, (n["delayed_rewrites"], n["dir"]))

    n["positive_atoms"] = ro.Lift1(
        f=lambda z: ZSet(
            {
                gk: w
                for gk, w in z.inner.items()
                if gk[1] is None or ("!" not in gk[1][0] and gk[1][0] not in dlg.INTERPRETED)
            }
        ),
    ).connect(c, (n["gatekeep"],))
    n["negative_atoms"] = ro.Lift1(
        f=lambda z: ZSet({gk: w for gk, w in z.inner.items() if not (gk[1] is None or "!" not in gk[1][0])}),
    ).connect(c, (n["gatekeep"],))
    n["interp_proj"] = ro.Lift1(
        f=lambda z: ZSet(
            {
                (gk[0], gk[2]): w
                for gk, w in z.inner.items()
                if gk[1] is not None and gk[1][0] in dlg.INTERPRETED and dlg.eval_interpreted(gk[1], gk[2])
            }
        ),
    ).connect(c, (n["gatekeep"],))

    n["product"] = ro.DeltaLiftedDeltaLiftedJoin(
        pred=lambda left, right: (
            left[1] is None or (left[1][0] == right[0] and dlg.unify(left[2].apply(left[1]), right) is not None)
        ),
        proj=dlg.rewrite_product_projection,
        group_a=gg,
        group_b=fg,
        out_group=rg,
    ).connect(c, (n["positive_atoms"], n["delayed_facts"]))

    n["proj"] = ro.Lift1(
        f=lambda z: ZSet({(gk[0], gk[2]): w for gk, w in z.inner.items()}),
    ).connect(c, (n["negative_atoms"],))

    n["anti_product"] = ro.DeltaLiftedDeltaLiftedJoin(
        pred=lambda left, right: (
            left[1] is None
            or (left[1][0].strip("!") == right[0] and dlg.unify(left[2].apply(left[1]), right) is not None)
        ),
        proj=lambda left, _right: (left[0], left[2]),
        group_a=gg,
        group_b=fg,
        out_group=rg,
    ).connect(c, (n["negative_atoms"], n["delayed_facts"]))

    n["product_plus_proj"] = ro.Lift2(op=rg.add).connect(c, (n["product"], n["proj"]))
    n["neg_anti"] = ro.Lift1(f=rg.neg).connect(c, (n["anti_product"],))
    n["final_product_neg"] = ro.Lift2(op=rg.add).connect(c, (n["product_plus_proj"], n["neg_anti"]))
    n["final_product"] = ro.Lift2(op=rg.add).connect(c, (n["final_product_neg"], n["interp_proj"]))

    n["ground"] = ro.DeltaLiftedDeltaLiftedJoin(
        pred=lambda left, right: left[0] == right[0],
        proj=lambda left, right: left[1].apply(right[1]),
        group_a=rg,
        group_b=sg,
        out_group=fg,
    ).connect(c, (n["final_product"], n["sig"]))

    n["ground_plus_edb"] = ro.Lift2(op=fg.add).connect(c, (n["ground"], n["edb_2d"]))
    n["next_facts"] = ro.DeltaLiftedDeltaLiftedDistinct(inner_group=fg).connect(c, (n["ground_plus_edb"],))
    n["final_plus_seed"] = ro.Lift2(op=rg.add).connect(c, (n["final_product"], n["seed_2d"]))
    n["next_rewrites"] = ro.DeltaLiftedDeltaLiftedDistinct(inner_group=rg).connect(c, (n["final_plus_seed"],))

    e.push(edb_1d, facts)
    e.push(program_1d, rules)
    e.push(seed, ZSet({(0, dlg._rewrite_monoid.identity()): 1}))
    return e, n, sf, sr


TRACE_NODES = [
    "gatekeep",
    "product",
    "proj",
    "anti_product",
    "final_product",
    "ground",
    "next_facts",
    "next_rewrites",
]


def run(name: str, rules: ZSet, facts: ZSet, max_inner: int = 30) -> None:
    print(f"\n{'=' * 100}\nCASE: {name}\n{'=' * 100}")
    negpreds = {
        atom[0].strip("!")
        for rule in rules.inner
        for atom in rule[1:]
        if "!" in atom[0]
    }
    E = dict(facts.inner)
    print(f"negated predicates: {negpreds or '∅'};  EDB: {sorted(E)}")

    e, n, sf, sr = build(rules, facts)
    F: dict = {}  # cumulative facts  = Lean facts_k
    R: dict = {}  # cumulative rewrites
    saw_negative = False

    for k in range(max_inner):
        diffs = {name_: e.read(n[name_], (0, k)) for name_ in TRACE_NODES}
        df, dr = diffs["next_facts"], diffs["next_rewrites"]
        if not df.inner and not dr.inner:
            print(f"\n-- k={k}: quiescent (both diffs empty) → SATURATED after {k} inner steps")
            break

        print(f"\n-- inner step k={k} " + "-" * 78)
        for name_ in TRACE_NODES:
            z = diffs[name_]
            if z.inner:
                neg = any(w < 0 for w in z.inner.values())
                mark = "   ⚠ NEGATIVE WEIGHTS" if neg and name_ not in ("anti_product",) else ""
                saw_negative |= neg and name_ in ("next_facts", "next_rewrites", "final_product", "ground")
                print(f"   {name_:>14}: {zfmt(z)}{mark}")

        for f_, m in df.inner.items():
            F[f_] = F.get(f_, 0) + m
        for r_, m in dr.inner.items():
            R[r_] = R.get(r_, 0) + m
        F = {a: w for a, w in F.items() if w != 0}
        R = {a: w for a, w in R.items() if w != 0}

        Fset = set(F)
        inv_i = set(E) <= Fset
        bad_ii = {a for a in Fset if a[0] in negpreds and a not in E}
        print(f"   F_{k + 1} (cumulative facts, {len(F)}): {sorted(Fset)}")
        print(
            f"   invariant (i)  E ⊆ F: {'OK' if inv_i else 'VIOLATED'}   "
            f"(ii) negated-pred facts ⊆ E: {'OK' if not bad_ii else f'VIOLATED by {bad_ii}'}"
        )

        e.push(sf, df, t=(0, k))
        e.push(sr, dr, t=(0, k))

    print(f"\n   signed retraction observed in facts/rewrites diffs: {'YES' if saw_negative else 'no'}")
    final = {a for a, w in F.items() if w > 0 and not a[0].startswith("__")}
    print(f"   FINAL MODEL: {sorted(final)}")


# ---------------------------------------------------------------------------
# CASE A — SEMIPOSITIVE: negation on the EDB predicate `dead` only.
# Positive recursion (reach) forces several inner steps, so the orbit is long
# enough to watch the invariant hold at every k.
# ---------------------------------------------------------------------------
RULES_A = ZSet(
    {
        (("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))): 1,
        (("reach", ("?X", "?Z")), ("edge", ("?X", "?Y")), ("reach", ("?Y", "?Z"))): 1,
        (("alive", ("?P",)), ("person", ("?P",)), ("!dead", ("?P",))): 1,
    }
)
EDB_A = ZSet(
    {
        ("edge", (0, 1)): 1,
        ("edge", (1, 2)): 1,
        ("edge", (2, 3)): 1,
        ("person", ("alice",)): 1,
        ("person", ("bob",)): 1,
        ("dead", ("bob",)): 1,
    }
)

# ---------------------------------------------------------------------------
# CASE B — NOT semipositive: negation on the DERIVED predicate `reach`.
# (PARITY_CASES[2].) Expect optimistic not-rewrites emitted early, then
# RETRACTED by signed anti_product contributions as reach facts arrive —
# the non-monotonicity the semipositive restriction excludes.
# ---------------------------------------------------------------------------
RULES_B = ZSet(
    {
        (("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))): 1,
        (("reach", ("?X", "?Z")), ("edge", ("?X", "?Y")), ("reach", ("?Y", "?Z"))): 1,
        (("lonely", ("?X",)), ("node", ("?X",)), ("!reach", ("?X", "?Y"))): 1,
    }
)
EDB_B = ZSet(
    {
        ("edge", (0, 1)): 1,
        ("edge", (1, 2)): 1,
        ("node", (0,)): 1,
        ("node", (1,)): 1,
        ("node", (2,)): 1,
    }
)

# ---------------------------------------------------------------------------
# CASE C — derived negation where the BLOCKER ARRIVES LATE. `far(X)` is
# derived optimistically at k=2 (before reach(1,3)/reach(0,3) exist), then
# must be RETRACTED at k=3/k=4 as the recursive chain completes. This is the
# signed, non-monotone orbit the semipositive restriction excludes.
# ---------------------------------------------------------------------------
RULES_C = ZSet(
    {
        (("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))): 1,
        (("reach", ("?X", "?Z")), ("edge", ("?X", "?Y")), ("reach", ("?Y", "?Z"))): 1,
        (("far", ("?X",)), ("node", ("?X",)), ("!reach", ("?X", 3))): 1,
    }
)
EDB_C = ZSet(
    {
        ("edge", (0, 1)): 1,
        ("edge", (1, 2)): 1,
        ("edge", (2, 3)): 1,
        ("node", (0,)): 1,
        ("node", (1,)): 1,
        ("node", (2,)): 1,
        ("node", (3,)): 1,
    }
)

if __name__ == "__main__":
    run("A: SEMIPOSITIVE (negation on EDB `dead`)", RULES_A, EDB_A)
    run("B: DERIVED NEGATION (negation on derived `reach`) — NOT semipositive", RULES_B, EDB_B)
    run("C: DERIVED NEGATION, LATE BLOCKER — the signed retraction orbit", RULES_C, EDB_C)
