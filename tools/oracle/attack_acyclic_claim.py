"""ADVERSARIAL fuzz: try to BREAK "acyclic retraction region => 2-D correct".

Part 1 — handcrafted attack cases (worst-case arithmetic/timing, all
well-formed + stratifiable + acyclic retraction region).
Part 2 — nasty generator: binary preds + unification, diamonds (multi-path
heads), up to TWO negations per rule, staggered delay chains, level-0
recursion (binary TC). Retraction region kept acyclic (higher-level rules
reference strictly lower preds).
"""

from __future__ import annotations

import random
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


def check(name, rules, facts, verbose=True):
    oracle = run_souffle(ZSet(rules), ZSet(facts))
    if isinstance(oracle, str):
        print(f"  {name:>34}: {oracle}")
        return True
    ok = True
    msgs = []
    for cls in D2:
        try:
            got = run_driver(cls, ZSet(rules), ZSet(facts))
        except Exception as ex:
            got = None
            msgs.append(f"{SHORT[cls.__name__]}: EXC {ex}")
            ok = False
            continue
        if got != oracle:
            ok = False
            msgs.append(f"{SHORT[cls.__name__]}: extra={sorted(got - oracle) or '∅'} missing={sorted(oracle - got) or '∅'}")
    if verbose or not ok:
        print(f"  {name:>34}: {'OK' if ok else 'BROKEN  ' + ' | '.join(msgs)}")
    return ok


# ---------- Part 1: handcrafted attacks ------------------------------------
# `slow(X)` arrives late via a 4-chain; `slow2(X)` via a 6-chain.
CHAIN = {
    (("s1", ("?X",)), ("go", ("?X",))): 1,
    (("s2", ("?X",)), ("s1", ("?X",))): 1,
    (("s3", ("?X",)), ("s2", ("?X",))): 1,
    (("slow", ("?X",)), ("s3", ("?X",))): 1,
    (("t1", ("?X",)), ("slow", ("?X",))): 1,
    (("t2", ("?X",)), ("t1", ("?X",))): 1,
    (("slow2", ("?X",)), ("t2", ("?X",))): 1,
}
E0 = {("go", (0,)): 1, ("base", (0,)): 1}

CASES = []

# A1: TWO negations in one rule; first blocker arrives late.
CASES.append(("A1 double-negation one rule", {
    **CHAIN,
    (("h", ("?X",)), ("base", ("?X",)), ("!slow", ("?X",)), ("!nope", ("?X",))): 1,
    (("dh", ("?X",)), ("h", ("?X",))): 1,   # consequence to drag through retraction
}, E0))

# A2: simultaneous retraction of BOTH join inputs (Δu·Δv cross-term).
CASES.append(("A2 simultaneous double retraction", {
    **CHAIN,
    (("u", ("?X",)), ("base", ("?X",)), ("!slow", ("?X",))): 1,
    (("v", ("?X",)), ("base", ("?X",)), ("!slow", ("?X",))): 1,
    (("h", ("?X",)), ("u", ("?X",)), ("v", ("?X",))): 1,
    (("dh", ("?X",)), ("h", ("?X",))): 1,
}, E0))

# A3: retracted derived support, but the fact is ALSO in the EDB — must survive.
CASES.append(("A3 EDB+derived mixed support", {
    **CHAIN,
    (("u", ("?X",)), ("base", ("?X",)), ("!slow", ("?X",))): 1,
    (("h", ("?X",)), ("u", ("?X",))): 1,
}, {**E0, ("h", (0,)): 1}))

# A4: resurrection with a consequence chain (late INSERTION wave after retraction).
CASES.append(("A4 late insertion after retraction", {
    **CHAIN,
    (("b", ("?X",)), ("base", ("?X",)), ("!slow", ("?X",))): 1,   # optimistic, dies
    (("a", ("?X",)), ("base", ("?X",)), ("!b", ("?X",))): 1,      # fires LATE
    (("x1", ("?X",)), ("a", ("?X",))): 1,
    (("x2", ("?X",)), ("x1", ("?X",))): 1,
}, E0))

# A5: diamond — two doomed paths + one safe path into the same head.
CASES.append(("A5 diamond, partial survival", {
    **CHAIN,
    (("m", ("?X",)), ("base", ("?X",)), ("!slow", ("?X",))): 1,
    (("l1", ("?X",)), ("m", ("?X",))): 1,
    (("l2", ("?X",)), ("m", ("?X",))): 1,
    (("h", ("?X",)), ("l1", ("?X",))): 1,
    (("h", ("?X",)), ("l2", ("?X",))): 1,
    (("h", ("?X",)), ("base", ("?X",))): 1,   # safe third path
}, E0))

# A6: binary derived negation with real unification; asymmetric late blockers.
CASES.append(("A6 binary negation on reach", {
    (("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))): 1,
    (("reach", ("?X", "?Z")), ("edge", ("?X", "?Y")), ("reach", ("?Y", "?Z"))): 1,
    (("oneway", ("?X", "?Y")), ("edge", ("?X", "?Y")), ("!reach", ("?Y", "?X"))): 1,
    (("c1", ("?X",)), ("oneway", ("?X", "?Y"))): 1,
}, {("edge", (0, 1)): 1, ("edge", (1, 2)): 1, ("edge", (2, 0)): 1, ("edge", (2, 3)): 1}))

# A7: negation on a diamond-supported fact retracted in TWO staggered waves.
CASES.append(("A7 staggered diamond then negation", {
    **CHAIN,
    (("u1", ("?X",)), ("base", ("?X",)), ("!slow", ("?X",))): 1,   # dies at wave 1
    (("u2", ("?X",)), ("base", ("?X",)), ("!slow2", ("?X",))): 1,  # dies at wave 2
    (("m", ("?X",)), ("u1", ("?X",))): 1,
    (("m", ("?X",)), ("u2", ("?X",))): 1,
    (("a", ("?X",)), ("base", ("?X",)), ("!m", ("?X",))): 1,       # fires only after BOTH
    (("y", ("?X",)), ("a", ("?X",))): 1,
}, E0))

# A8: negated atom in the MIDDLE of the body (binder before, positive after).
CASES.append(("A8 negation mid-body", {
    **CHAIN,
    (("h", ("?X",)), ("base", ("?X",)), ("!slow", ("?X",)), ("go", ("?X",))): 1,
}, E0))

# ---------- Part 2: nasty generator -----------------------------------------
CONSTS = [0, 1, 2]


def gen(rng: random.Random):
    rules: dict = {}
    # level-0: binary TC (genuine recursion, never retracted)
    rules[(("reach", ("?X", "?Y")), ("edge", ("?X", "?Y")))] = 1
    rules[(("reach", ("?X", "?Z")), ("edge", ("?X", "?Y")), ("reach", ("?Y", "?Z")))] = 1
    chain = [f"c{i}" for i in range(rng.randint(1, 5))]
    prev = "e0"
    for cp in chain:
        rules[((cp, ("?X",)), (prev, ("?X",)))] = 1
        prev = cp
    base = ["e0", "e1"] + chain
    derived = [f"d{i}" for i in range(rng.randint(3, 7))]
    for i, p in enumerate(derived):
        lower_all = base + derived[:i]
        lower_idb = chain + derived[:i]  # unary pool only; binary !reach handled in the binary branch
        for _ in range(rng.randint(1, 2)):  # multiple rules per head => diamonds
            shape = rng.random()
            if shape < 0.35:
                # binary binder + optional negation on reach or unary idb
                body = [("edge", ("?X", "?Y"))]
                if rng.random() < 0.5:
                    body.append(("reach", ("?Y", "?X")) if rng.random() < 0.5 else (rng.choice(lower_all), ("?Y",)))
                if rng.random() < 0.8:
                    body.append(("!reach", ("?Y", "?X")) if rng.random() < 0.5
                                else ("!" + rng.choice(lower_idb), ("?Y",)))
                rules[((p, ("?X",)), *body)] = 1
            else:
                body = [(rng.choice(lower_all), ("?X",)) for _ in range(rng.randint(1, 2))]
                nnegs = rng.choices([0, 1, 2], weights=[2, 6, 2])[0]
                negs = rng.sample(lower_idb, k=min(nnegs, len(lower_idb)))
                for np_ in negs:
                    body.append(("!" + np_, ("?X",)))
                rules[((p, ("?X",)), *body)] = 1
    facts: dict = {}
    for c in CONSTS:
        for cc in CONSTS:
            if rng.random() < 0.4:
                facts[("edge", (c, cc))] = 1
        for ep in ("e0", "e1"):
            if rng.random() < 0.6:
                facts[(ep, (c,))] = 1
    if not facts:
        facts[("e0", (0,))] = 1
    return rules, facts


if __name__ == "__main__":
    print("=== Part 1: handcrafted attacks ===")
    broken = 0
    for name, rules, facts in CASES:
        broken += 0 if check(name, rules, facts) else 1

    print("\n=== Part 2: nasty generator (3 seeds x 50) ===")
    for seed in (31337, 424242, 271828):
        rng = random.Random(seed)
        results = {"match": 0, "err": 0}
        fails = []
        for i in range(50):
            rules, facts = gen(rng)
            oracle = run_souffle(ZSet(rules), ZSet(facts))
            if isinstance(oracle, str):
                results["err"] += 1
                continue
            wrong = []
            for cls in D2:
                try:
                    if run_driver(cls, ZSet(rules), ZSet(facts)) != oracle:
                        wrong.append(SHORT[cls.__name__])
                except Exception as ex:
                    wrong.append(f"{SHORT[cls.__name__]}(EXC:{type(ex).__name__})")
            if wrong:
                fails.append((i, tuple(wrong), rules, facts))
            else:
                results["match"] += 1
        print(f"  seed {seed}: {results['match']} match, {len(fails)} BROKEN, {results['err']} souffle-err")
        for i, wrong, rules, facts in fails[:2]:
            print(f"    program #{i} broke {wrong}:")
            for rule in sorted(rules):
                head, *body = rule
                print("      " + f"{head[0]}{head[1]} <- " + ", ".join(f"{a[0]}{a[1]}" for a in body))
            print(f"      EDB: {sorted(facts)}")
