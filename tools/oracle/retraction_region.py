"""THE FRAGMENT PREDICATE: tight-below-negation, plus its partition fuzzer.

The 2-D negation drivers are conjectured correct on exactly the stratifiable,
well-formed programs that are TIGHT BELOW NEGATION: no positive cycle is
positively reachable from a retraction origin (a head whose rule negates a
DERIVED predicate). Until now this region existed only by generator
construction (attack_acyclic_claim.py); here it is a checkable predicate,
and the fuzzer partitions random programs by it:

  class 1  semipositive                      -> expect 0 divergence (proven, Lean task N)
  class 2  tight & not semipositive          -> expect 0 divergence (THE CONJECTURE)
  class 3  stratifiable & not tight          -> divergence allowed; every observed
                                                divergence must land here (ghosts)

Two boundary candidates are computed and ASSERTED EQUAL on every program:
refined (retraction propagates through positive edges only) and coarse
(any-edge downstream). They coincide because a negative edge with a derived
source makes its target an origin, and EDB-sourced negative edges have no
in-edges to reach them; the assert is the machine check of that argument
(Lean lemma `coarse_iff_refined`).

Run from the pydbsp repo root:
    uv run python tools/oracle/retraction_region.py --per-class 60
"""

from __future__ import annotations

import argparse
import random
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from souffle_oracle import run_driver, run_souffle  # noqa: E402

from pydbsp.datalog_stratified import (  # noqa: E402
    DatalogWithNegation,
    IndexedDatalogWithNegation,
)
from pydbsp.zset import ZSet  # noqa: E402

D2 = (DatalogWithNegation, IndexedDatalogWithNegation)
SHORT = {"DatalogWithNegation": "2D", "IndexedDatalogWithNegation": "2Di"}


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def dep_graph(rules: dict):
    """Predicate-level dependency edges, flow direction src -> dst = src feeds dst."""
    pos, neg, idb = set(), set(), set()
    for rule in rules:
        head, *body = rule
        idb.add(head[0])
        for pred, _args in body:
            if pred.startswith("!"):
                neg.add((pred[1:], head[0]))
            else:
                pos.add((pred, head[0]))
    return pos, neg, idb


def _closure(edges: set, nodes: set) -> dict:
    """reach[n] = set of nodes reachable from n in >= 1 step."""
    succ = {n: set() for n in nodes}
    for s, d in edges:
        succ.setdefault(s, set()).add(d)
    reach = {n: set(succ.get(n, ())) for n in nodes}
    changed = True
    while changed:
        changed = False
        for n in nodes:
            new = reach[n] | {x for m in list(reach[n]) for x in reach.get(m, ())}
            if len(new) != len(reach[n]):
                reach[n] = new
                changed = True
    return reach


def classify(rules: dict) -> dict:
    """All region predicates for one program (rules in pydbsp tuple format)."""
    pos, neg, idb = dep_graph(rules)
    nodes = {p for e in pos | neg for p in e} | idb
    all_reach = _closure(pos | neg, nodes)
    pos_reach = _closure(pos, nodes)

    stratifiable = all(not (d == s or s in all_reach.get(d, ())) for s, d in neg)
    semipositive = all(s not in idb for s, _d in neg)

    origins = {r[0][0] for r in rules
               if any(a[0].startswith("!") and a[0][1:] in idb for a in r[1:])}
    cyclic = {n for n in nodes if n in pos_reach.get(n, ())}

    def downstream(reach):
        return origins | {p for o in origins for p in reach.get(o, ())}

    tight_refined = not (downstream(pos_reach) & cyclic)
    tight_coarse = not (downstream(all_reach) & cyclic)

    return {
        "stratifiable": stratifiable,
        "semipositive": semipositive,
        "tight": tight_refined,
        "tight_coarse": tight_coarse,
        "origins": origins,
        "cyclic": cyclic,
    }


def region_class(rules: dict) -> str | None:
    """Partition label, or None if outside the experiment (not stratifiable)."""
    c = classify(rules)
    assert c["tight"] == c["tight_coarse"], (
        "coarse/refined boundary split — the origin-closure argument is wrong: "
        f"{sorted(rules)}")
    if not c["stratifiable"]:
        return None
    if c["semipositive"]:
        return "1-semipositive"
    if c["tight"]:
        return "2-tight-idb-negation"
    return "3-ghost-prone"


# ---------------------------------------------------------------------------
# Generator: stratifiable + well-formed, on BOTH sides of the boundary
# ---------------------------------------------------------------------------

CONSTS = [0, 1, 2]

# level-0 lateness machinery: `slow` arrives via a 4-chain of unary rules.
CHAIN = {
    (("s1", ("?X",)), ("go", ("?X",))): 1,
    (("s2", ("?X",)), ("s1", ("?X",))): 1,
    (("s3", ("?X",)), ("s2", ("?X",))): 1,
    (("slow", ("?X",)), ("s3", ("?X",))): 1,
}


def gen(rng: random.Random) -> tuple[dict, dict]:
    rules: dict = dict(CHAIN)
    # class-1 bias: ~1/4 of programs negate EDB predicates ONLY (semipositive
    # by construction) — random IDB-heavy pools starve the semipositive quota.
    edb_only_neg = rng.random() < 0.25
    # level-0 recursion: binary TC (a positive cycle NOT downstream of negation)
    if rng.random() < 0.6:
        rules[(("reach", ("?X", "?Y")), ("edge", ("?X", "?Y")))] = 1
        rules[(("reach", ("?X", "?Z")), ("edge", ("?X", "?Y")), ("reach", ("?Y", "?Z")))] = 1

    base = ["e0", "e1", "slow", "s2"]
    derived: list[str] = []
    n_levels = rng.randint(2, 4)
    for lvl in range(n_levels):
        width = rng.randint(1, 3)
        level_preds = [f"d{lvl}_{i}" for i in range(width)]
        lower_idb = [p for p in derived] + ["slow", "s2", "s3"]
        for p in level_preds:
            # 1-2 rules per head: positives from lower/base, negations from
            # strictly lower (stratifiable by construction), negation LAST
            # (well-formed order), unary ?X everywhere (NSafe by construction).
            for _ in range(rng.randint(1, 2)):
                body = [(rng.choice(base + derived), ("?X",))
                        for _ in range(rng.randint(1, 2))]
                nnegs = rng.choices([0, 1, 2], weights=[3, 6, 1])[0]
                if edb_only_neg:
                    pool = ["e0", "e1", "go"]
                else:
                    pool = lower_idb + (["e0", "e1"] if rng.random() < 0.4 else [])
                for np_ in rng.sample(pool, k=min(nnegs, len(pool))):
                    body.append(("!" + np_, ("?X",)))
                rules[((p, ("?X",)), *body)] = 1
        # positive 2-cycles WITHIN the new level (stratifiability unharmed:
        # same-level positive edges are fine). Whether the cycle lands in the
        # retraction region depends on the random wiring above -> both classes.
        if width >= 2 and rng.random() < 0.5:
            a, b = rng.sample(level_preds, k=2)
            rules[((a, ("?X",)), (b, ("?X",)))] = 1
            rules[((b, ("?X",)), (a, ("?X",)))] = 1
        derived += level_preds

    # explicit ghost motif sometimes, so class 3 is never starved: a head
    # guarded by a LATE derived blocker feeds a fresh positive 2-cycle.
    # (skipped in EDB-only mode: it negates `slow`, an IDB pred)
    if not edb_only_neg and rng.random() < 0.35:
        rules[(("gb", ("?X",)), ("e0", ("?X",)), ("!slow", ("?X",)))] = 1
        rules[(("gp", ("?X",)), ("gb", ("?X",)))] = 1
        rules[(("gq", ("?X",)), ("gp", ("?X",)))] = 1
        rules[(("gp", ("?X",)), ("gq", ("?X",)))] = 1

    facts: dict = {}
    for c in CONSTS:
        for cc in CONSTS:
            if rng.random() < 0.4:
                facts[("edge", (c, cc))] = 1
        for ep in ("e0", "e1", "go"):
            if rng.random() < 0.6:
                facts[(ep, (c,))] = 1
    if not facts:
        facts[("e0", (0,))] = 1
    return rules, facts


class _Timeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise _Timeout()


def adjudicate(rules: dict, facts: dict) -> list[str]:
    """Return list of divergence messages (empty = all drivers match Soufflé)."""
    oracle = run_souffle(ZSet(rules), ZSet(facts))
    if isinstance(oracle, str):
        return [oracle]
    msgs = []
    for cls in D2:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(30)
        try:
            got = run_driver(cls, ZSet(rules), ZSet(facts))
            if got != oracle:
                msgs.append(f"{SHORT[cls.__name__]}: extra={sorted(got - oracle) or '∅'} "
                            f"missing={sorted(oracle - got) or '∅'}")
        except _Timeout:
            msgs.append(f"{SHORT[cls.__name__]}: HANG(30s)")
        except Exception as ex:
            msgs.append(f"{SHORT[cls.__name__]}: EXC {type(ex).__name__}: {ex}")
        finally:
            signal.alarm(0)
    return msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260705)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    quota = {"1-semipositive": args.per_class,
             "2-tight-idb-negation": args.per_class,
             "3-ghost-prone": args.per_class}
    seen = {k: 0 for k in quota}
    diverged = {k: [] for k in quota}
    souffle_err = 0
    attempts = 0

    while any(seen[k] < quota[k] for k in quota) and attempts < 6000:
        attempts += 1
        rules, facts = gen(rng)
        label = region_class(rules)
        if label is None or seen[label] >= quota[label]:
            continue
        msgs = adjudicate(rules, facts)
        if msgs and msgs[0].startswith("SOUFFLE ERROR"):
            souffle_err += 1
            continue
        seen[label] += 1
        if msgs:
            diverged[label].append((rules, facts, msgs))

    print(f"attempts={attempts} souffle_err={souffle_err}")
    ok = True
    for k in sorted(quota):
        n_div = len(diverged[k])
        verdict = "OK" if (n_div == 0 or k == "3-ghost-prone") else "*** CONJECTURE BROKEN ***"
        if k != "3-ghost-prone" and n_div:
            ok = False
        print(f"  {k:>24}: {seen[k]:3d} programs, {n_div:3d} divergent   {verdict}")
    for k in sorted(quota):
        for rules, facts, msgs in diverged[k][:3]:
            print(f"\n  [{k}] {' | '.join(msgs)}")
            for rule in sorted(rules):
                head, *body = rule
                print("      " + f"{head[0]}{head[1]} <- "
                      + ", ".join(f"{a[0]}{a[1]}" for a in body))
            print(f"      EDB: {sorted(facts)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
