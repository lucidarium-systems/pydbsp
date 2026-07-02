"""Multi-outer-tick behavior of DatalogStratifiedComposition.

Drives ONE driver instance across several push/saturate rounds (t1 ticks),
including fact AND rule retractions, and checks every tick's output against
Soufflé evaluated on the NET (cumulative) program+facts at that tick.

Directed scenarios stress the tricky flips:
  S1  semipositive: retracting an EDB blocker resurrects a not-fact
  S2  depth-2 cascade: one edge retraction flips TWO negation levels
      (mark disappears -> b appears -> a disappears)
  S3  restratification: adding/retracting a rule changes the level structure
      (mark not-derived -> derived -> not-derived)
  S4  cycle toggle: retract/re-add the edge gating a negation whose output
      feeds a positive cycle (legit activation, then full retraction)

Fuzz: A->B->A episodes on the acyclic-region generator: push rules + half
the EDB, add the rest, retract a random fact subset, retract a random rule
subset — Soufflé-checked at every tick.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from attack_acyclic_claim import gen as gen_nasty  # noqa: E402
from souffle_oracle import run_souffle, _san, _const  # noqa: E402

from pydbsp.datalog_stratified import DatalogStratifiedComposition  # noqa: E402
from pydbsp.zset import ZSet, ZSetAddition  # noqa: E402

G = ZSetAddition()


def model_of(driver) -> set:
    return {(_san(a[0]), tuple(_const(x) for x in a[1])) for a, w in driver.saturate().inner.items() if w > 0}


def tick(driver, net_rules, net_facts, drules=None, dfacts=None):
    if drules and drules.inner:
        driver.push_rules(drules)
        net_rules = G.add(net_rules, drules)
    if dfacts and dfacts.inner:
        driver.push_facts(dfacts)
        net_facts = G.add(net_facts, dfacts)
    oracle = run_souffle(net_rules, net_facts)
    got = model_of(driver)
    ok = got == oracle
    return ok, net_rules, net_facts, (None if ok else (sorted(got - oracle), sorted(oracle - got)))


def scenario(name, steps):
    d = DatalogStratifiedComposition()
    nr, nf = ZSet({}), ZSet({})
    for i, (drules, dfacts) in enumerate(steps):
        ok, nr, nf, diff = tick(d, nr, nf, drules, dfacts)
        tag = "OK" if ok else f"WRONG extra={diff[0]} missing={diff[1]}"
        print(f"  {name} tick{i}: {tag}")
        if not ok:
            return False
    return True


ALIVE = ZSet({(("alive", ("?P",)), ("person", ("?P",)), ("!dead", ("?P",))): 1})
REACH = {
    (("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))): 1,
    (("reach", ("?X", "?Z")), ("edge", ("?X", "?Y")), ("reach", ("?Y", "?Z"))): 1,
}
D2 = ZSet({
    **REACH,
    (("mark", ("?X",)), ("probe", ("?X",)), ("reach", ("?X", 3))): 1,
    (("b", ("?X",)), ("probe", ("?X",)), ("!mark", ("?X",))): 1,
    (("a", ("?X",)), ("probe", ("?X",)), ("!b", ("?X",))): 1,
})
GHOSTP = ZSet({
    **REACH,
    (("mark", ("?X",)), ("probe", ("?X",)), ("reach", ("?X", 3))): 1,
    (("b", ("?X",)), ("probe", ("?X",)), ("!mark", ("?X",))): 1,
    (("p", ("?X",)), ("b", ("?X",))): 1,
    (("q", ("?X",)), ("p", ("?X",))): 1,
    (("p", ("?X",)), ("q", ("?X",))): 1,
})
CHAINE = ZSet({("edge", (0, 1)): 1, ("edge", (1, 2)): 1, ("edge", (2, 3)): 1, ("probe", (0,)): 1})
MARKRULE = ZSet({(("mark", ("?X",)), ("special", ("?X",))): 1})

print("— directed multi-outer scenarios —")
ok = True
ok &= scenario("S1 semipositive blocker retract", [
    (ALIVE, ZSet({("person", ("alice",)): 1, ("person", ("bob",)): 1, ("dead", ("bob",)): 1})),
    (None, ZSet({("dead", ("bob",)): -1})),          # alive(bob) must appear
    (None, ZSet({("dead", ("alice",)): 1})),         # alive(alice) must vanish
])
ok &= scenario("S2 depth-2 cascade edge retract ", [
    (D2, CHAINE),                                     # mark(0), a(0)
    (None, ZSet({("edge", (2, 3)): -1})),             # mark gone -> b(0) -> no a(0)
    (None, ZSet({("edge", (2, 3)): 1})),              # and back
])
ok &= scenario("S3 restratification by rule delta", [
    (ZSet({(("b", ("?X",)), ("probe", ("?X",)), ("!mark", ("?X",))): 1,
           (("a", ("?X",)), ("probe", ("?X",)), ("!b", ("?X",))): 1}),
     ZSet({("probe", (0,)): 1})),                     # mark not derived: b(0), no a
    (MARKRULE, ZSet({("special", (0,)): 1})),         # mark derived: levels shift; a(0)
    (G.neg(MARKRULE), None),                          # retract rule: levels collapse back
])
ok &= scenario("S4 cycle-feeding negation toggle ", [
    (GHOSTP, CHAINE),                                 # mark blocks b: no p/q
    (None, ZSet({("edge", (2, 3)): -1})),             # b,p,q legitimately appear
    (None, ZSet({("edge", (2, 3)): 1})),              # all must retract again
])

print("— A->B->A episode fuzz (acyclic-region generator) —")
rng = random.Random(20260703)
bad = 0
for ep in range(15):
    rules_d, facts_d = gen_nasty(rng)
    rules, facts = ZSet(rules_d), ZSet(facts_d)
    fl = list(facts.inner)
    half = ZSet({f: 1 for f in fl[: len(fl) // 2]})
    rest = ZSet({f: 1 for f in fl[len(fl) // 2:]})
    retract_f = ZSet({f: -1 for f in fl if rng.random() < 0.4})
    rl = list(rules.inner)
    retract_r = ZSet({r: -1 for r in rl if rng.random() < 0.3})
    d = DatalogStratifiedComposition()
    nr, nf = ZSet({}), ZSet({})
    for j, (dr, df) in enumerate([(rules, half), (None, rest), (None, retract_f), (retract_r, None)]):
        okt, nr, nf, diff = tick(d, nr, nf, dr, df)
        if not okt:
            bad += 1
            print(f"  episode {ep} tick {j}: WRONG extra={diff[0]} missing={diff[1]}")
            break
print(f"  episodes: {15 - bad}/15 fully correct across all 4 ticks")
print(f"\nDIRECTED: {'ALL OK' if ok else 'FAILURES ABOVE'}")
