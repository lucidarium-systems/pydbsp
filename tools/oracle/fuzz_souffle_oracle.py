"""Differential fuzz: random stratified programs, pydbsp drivers vs Soufflé.

Generator (bounded negation only — footgun 1 is known; we hunt NEW classes):
  - unary predicates; constants {0,1,2};
  - EDB preds e0..e2 with random facts;
  - derived preds d0..d5, each assigned a stratum level 0..2;
  - each derived pred gets 1-2 rules: 1-2 positive atoms of preds at level
    <= own (same-level => positive cycles arise naturally) +, at level > 0,
    optionally one negated atom of a strictly-lower-level pred (var bound);
  - stratifiable by construction; safety by construction (single var ?X,
    >= 1 positive atom per rule).

Buckets divergences: which subset of drivers disagrees with Soufflé.
Deterministic seed for reproducibility.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from souffle_oracle import run_driver, run_souffle  # noqa: E402

from pydbsp.datalog_stratified import (  # noqa: E402
    DatalogAlternating,
    DatalogWithNegation,
    IncrementalDatalogStratified,
    IndexedDatalogWithNegation,
)
from pydbsp.zset import ZSet  # noqa: E402

DRIVERS = (DatalogWithNegation, IndexedDatalogWithNegation, IncrementalDatalogStratified, DatalogAlternating)
CONSTS = [0, 1, 2]
N_PROGRAMS = 60


def gen_program(rng: random.Random) -> tuple[ZSet, ZSet]:
    """Biased toward the dangerous shapes: delay chains (late blockers),
    negation preferentially on DERIVED preds one level down, and same-level
    positive references (cycle fodder)."""
    edb_preds = ["e0", "e1"]
    # delay chain: c0 <- e0, c_{i+1} <- c_i  (stretches inner time)
    chain_len = rng.randint(0, 4)
    chain = [f"c{i}" for i in range(chain_len)]
    derived = [f"d{i}" for i in range(rng.randint(3, 6))]
    level = {p: 0 for p in chain}
    for p in derived:
        level[p] = rng.choice([0, 1, 1, 2, 2])  # bias upward

    rules: dict = {}
    prev = "e0"
    for cp in chain:
        rules[((cp, ("?X",)), (prev, ("?X",)))] = 1
        prev = cp

    for p in derived:
        for _ in range(rng.randint(1, 2)):
            lp = level[p]
            # positive body: EDB/chain/derived at level <= lp (same level => cycles)
            pos_pool = edb_preds + chain + [q for q in derived if level[q] <= lp]
            body = [(rng.choice(pos_pool), ("?X",)) for _ in range(rng.randint(1, 2))]
            if lp > 0 and rng.random() < 0.8:
                lower_derived = [q for q in derived + chain if level[q] < lp]
                lower_derived = [q for q in lower_derived if q not in edb_preds]
                if lower_derived and rng.random() < 0.8:
                    body.append(("!" + rng.choice(lower_derived), ("?X",)))
                else:
                    body.append(("!" + rng.choice(edb_preds), ("?X",)))
            rules[((p, ("?X",)), *body)] = 1

    facts: dict = {}
    for ep in edb_preds:
        for c in CONSTS:
            if rng.random() < 0.6:
                facts[(ep, (c,))] = 1
    if not facts:
        facts[("e0", (0,))] = 1
    return ZSet(rules), ZSet(facts)


def fmt_rules(rules: ZSet) -> str:
    out = []
    for rule in rules.inner:
        head, *body = rule
        out.append(f"    {head[0]}(X) <- " + ", ".join(f"{a[0]}(X)" for a in body))
    return "\n".join(sorted(out))


if __name__ == "__main__":
    rng = random.Random(20260702)
    buckets: dict[tuple, list[int]] = {}
    samples: dict[tuple, str] = {}
    errors = 0

    for i in range(N_PROGRAMS):
        rules, facts = gen_program(rng)
        oracle = run_souffle(rules, facts)
        if isinstance(oracle, str):
            errors += 1
            continue
        wrong = tuple(
            cls.__name__ for cls in DRIVERS if run_driver(cls, rules, facts) != oracle
        )
        buckets.setdefault(wrong, []).append(i)
        if wrong and wrong not in samples:
            diffs = []
            for cls in DRIVERS:
                got = run_driver(cls, rules, facts)
                if got != oracle:
                    diffs.append(
                        f"      {cls.__name__}: extra={sorted(got - oracle) or '∅'} "
                        f"missing={sorted(oracle - got) or '∅'}"
                    )
            samples[wrong] = (
                f"  program #{i}:\n{fmt_rules(rules)}\n"
                f"    EDB: {sorted(facts.inner)}\n" + "\n".join(diffs)
            )

    print(f"fuzzed {N_PROGRAMS} programs (souffle errors: {errors})\n")
    for wrong, idxs in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        label = "ALL MATCH" if not wrong else " + ".join(wrong) + " WRONG"
        print(f"  [{len(idxs):3d}] {label}   (programs: {idxs[:8]}{'…' if len(idxs) > 8 else ''})")
    for wrong, sample in samples.items():
        print(f"\n--- sample divergence [{' + '.join(wrong)}] ---\n{sample}")
