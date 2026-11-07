"""Episode fuzz for IncrementalDatalogAlternating: random stratifiable
programs driven through random DELTA EPISODES (fact adds/retracts, rule
adds/retracts) over multiple outer ticks.

Per tick, TRIPLE-GATED:
  - vs a fresh single-batch DatalogAlternating on the cumulative input
    (the primary gate: the multi-batch driver must be tick-for-tick
    equivalent to recompute);
  - vs DatalogStratifiedComposition on the cumulative input;
  - vs Soufflé on the cumulative input (ground truth).

Episode shapes per program (all deltas end in a stratifiable cumulative
program by construction — subsets of a stratifiable program are
stratifiable, so staged rule ARRIVAL is always safe, and rule REMOVAL
only deletes dependency edges):
  t0: first half of the rules + ~60% of the facts
  t1: remaining rules + more facts
  t2: retract a random t0 fact
  t3: retract a random rule (weight -1)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fuzz_souffle_oracle import gen_program  # noqa: E402
from souffle_oracle import _const, _san, run_souffle  # noqa: E402

from pydbsp.datalog_stratified import (  # noqa: E402
    DatalogAlternating,
    DatalogStratifiedComposition,
    IncrementalDatalogAlternating,
)
from pydbsp.zset import ZSet  # noqa: E402

N_PER_SEED = 25
SEEDS = [20260702, 1, 42, 777, 987654, 31337, 8, 999]


def _support(z: ZSet) -> set:
    return {(_san(a[0]), tuple(_const(x) for x in a[1])) for a, w in z.inner.items() if w > 0}


def _merge(cum: dict, delta: dict) -> None:
    for k, w in delta.items():
        cum[k] = cum.get(k, 0) + w
        if cum[k] == 0:
            del cum[k]


def run_episode(rng: random.Random, rules: ZSet, facts: ZSet) -> str | None:
    """Returns None on success, else a failure description."""
    rule_items = sorted(rules.inner, key=repr)
    fact_items = sorted(facts.inner, key=repr)
    half = len(rule_items) // 2
    t0_facts = [f for f in fact_items if rng.random() < 0.6] or fact_items[:1]
    later_facts = [f for f in fact_items if f not in t0_facts]

    episodes: list[tuple[dict, dict]] = [
        ({r: 1 for r in rule_items[:half]}, {f: 1 for f in t0_facts}),
        ({r: 1 for r in rule_items[half:]}, {f: 1 for f in later_facts}),
        ({}, {rng.choice(t0_facts): -1}),
    ]
    if rule_items:
        episodes.append(({rng.choice(rule_items): -1}, {}))

    inc = IncrementalDatalogAlternating()
    cum_rules: dict = {}
    cum_facts: dict = {}
    for t, (rd, fd) in enumerate(episodes):
        if rd:
            inc.push_rules(ZSet(rd))
        if fd:
            inc.push_facts(ZSet(fd))
        _merge(cum_rules, rd)
        _merge(cum_facts, fd)
        got = _support(inc.saturate())
        for cls in (DatalogAlternating, DatalogStratifiedComposition):
            ref = cls()
            ref.push_rules(ZSet(cum_rules))
            ref.push_facts(ZSet(cum_facts))
            exp = _support(ref.saturate())
            if got != exp:
                return (
                    f"tick {t} vs {cls.__name__}: extra={sorted(got - exp)} "
                    f"missing={sorted(exp - got)}"
                )
        oracle = run_souffle(ZSet(cum_rules), ZSet(cum_facts))
        if isinstance(oracle, str):
            continue  # souffle error; recompute gates still checked
        if got != oracle:
            return f"tick {t} vs SOUFFLE: extra={sorted(got - oracle)} missing={sorted(oracle - got)}"
    return None


if __name__ == "__main__":
    total = fails = 0
    for seed in SEEDS:
        rng = random.Random(seed)
        bad = 0
        for _ in range(N_PER_SEED):
            rules, facts = gen_program(rng)
            total += 1
            failure = run_episode(rng, rules, facts)
            if failure:
                bad += 1
                fails += 1
                print(f"seed {seed}: FAIL — {failure}")
                for r in sorted(rules.inner, key=repr):
                    head, *body = r
                    print(f"    {head[0]}(X) <- " + ", ".join(a[0] for a in body))
                print(f"    EDB: {sorted(facts.inner)}")
        print(f"seed {seed}: {N_PER_SEED} episodes, {bad} failures")
    print(f"\nTOTAL: {total} episodes ({total * 4} gated ticks), {fails} failures")
