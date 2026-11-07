"""Soufflé as ground-truth oracle for pydbsp's stratified negation.

Translates pydbsp-style programs (tuple rules, "?X" vars, "!pred" negation)
to Soufflé .dl + .facts, runs Soufflé (via `nix run nixpkgs#souffle`), and
diffs the model against each pydbsp driver.

Translation notes:
  - every constant becomes a `symbol` (ints stringified);
  - predicate names sanitized (non-alphanumerics -> "_");
  - "?X" -> "X"; a var of a negated atom UNBOUND by the positive body -> "_"
    (Soufflé's don't-care = the existential reading pydbsp implements);
  - every predicate is .decl'd; EDB predicates get .input, all get .output.

Run from the pydbsp repo root:
    uv run python souffle_oracle.py
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from pydbsp.datalog_stratified import (
    DatalogAlternating,
    DatalogWithNegation,
    IncrementalDatalogStratified,
    IndexedDatalogWithNegation,
)
from pydbsp.zset import ZSet

DRIVERS = (DatalogWithNegation, IndexedDatalogWithNegation, IncrementalDatalogStratified, DatalogAlternating)
SOUFFLE = ["nix", "run", "nixpkgs#souffle", "--"]


def _san(pred: str) -> str:
    return "p_" + re.sub(r"[^A-Za-z0-9_]", "_", pred.lstrip("!"))


def _is_var(t) -> bool:
    return isinstance(t, str) and t.startswith("?")


def _const(t) -> str:
    return str(t)


def _term(t, bound: set, negated: bool) -> str:
    if _is_var(t):
        name = "V" + t[1:]
        if negated and t not in bound:
            return "_"
        return name
    return '"' + _const(t).replace('"', '\\"') + '"'


def to_souffle(rules: ZSet, facts: ZSet) -> tuple[str, dict[str, list[tuple]]]:
    arity: dict[str, int] = {}
    edb_preds: set[str] = set()
    for (pred, args), _w in facts.inner.items():
        arity[pred] = len(args)
        edb_preds.add(pred)
    for rule in rules.inner:
        for atom in rule:
            p = atom[0].lstrip("!")
            arity.setdefault(p, len(atom[1]))

    lines = []
    for p, n in sorted(arity.items()):
        cols = ", ".join(f"a{i}: symbol" for i in range(n))
        lines.append(f".decl {_san(p)}({cols})")
        if p in edb_preds:
            lines.append(f".input {_san(p)}")
        lines.append(f".output {_san(p)}")

    for rule in rules.inner:
        head, *body = rule
        bound = {t for atom in body if "!" not in atom[0] for t in atom[1] if _is_var(t)}
        h = f"{_san(head[0])}({', '.join(_term(t, bound, False) for t in head[1])})"
        parts = []
        for pred, args in body:
            neg = "!" in pred
            a = f"{_san(pred)}({', '.join(_term(t, bound, neg) for t in args)})"
            parts.append(("!" if neg else "") + a)
        lines.append(f"{h} :- {', '.join(parts)}.")

    fact_rows: dict[str, list[tuple]] = {}
    for (pred, args), w in facts.inner.items():
        if w > 0:
            fact_rows.setdefault(_san(pred), []).append(tuple(_const(a) for a in args))
    return "\n".join(lines) + "\n", fact_rows


def run_souffle(rules: ZSet, facts: ZSet) -> set[tuple] | str:
    dl, fact_rows = to_souffle(rules, facts)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "prog.dl").write_text(dl)
        fdir, odir = d / "facts", d / "out"
        fdir.mkdir(), odir.mkdir()
        for p, rows in fact_rows.items():
            (fdir / f"{p}.facts").write_text("\n".join("\t".join(r) for r in rows) + ("\n" if rows else ""))
        # touch empty .facts for declared-but-absent EDB preds
        r = subprocess.run(
            SOUFFLE + ["-F", str(fdir), "-D", str(odir), str(d / "prog.dl")],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            return f"SOUFFLE ERROR: {r.stderr.strip().splitlines()[:3]}"
        model: set[tuple] = set()
        for csv in odir.glob("*.csv"):
            pred = csv.stem
            for line in csv.read_text().splitlines():
                if line:
                    model.add((pred, tuple(line.split("\t"))))
        return model


def run_driver(cls, rules: ZSet, facts: ZSet) -> set[tuple] | str:
    s = cls()
    s.push_rules(rules)
    s.push_facts(facts)
    out = s.saturate()
    return {
        (_san(a[0]), tuple(_const(x) for x in a[1]))
        for a, w in out.inner.items()
        if w > 0
    }


def compare(name: str, rules: ZSet, facts: ZSet) -> None:
    print(f"\n=== {name} ===")
    oracle = run_souffle(rules, facts)
    if isinstance(oracle, str):
        print(f"  {oracle}")
        return
    for cls in DRIVERS:
        got = run_driver(cls, rules, facts)
        if got == oracle:
            print(f"  {cls.__name__:>28}: MATCHES souffle ({len(oracle)} facts)")
        else:
            extra, missing = sorted(got - oracle), sorted(oracle - got)
            print(f"  {cls.__name__:>28}: *** DIVERGES ***  extra(ghosts)={extra or '∅'}  missing={missing or '∅'}")


# ---------------------------------------------------------------------------
CASES: list[tuple[str, ZSet, ZSet]] = []

# 1-5: the parity suite (single-level negation) — sanity: everything should match.
from tests.test_nonindexed_2d_negation import PARITY_CASES  # noqa: E402

for nm, r, f in PARITY_CASES:
    CASES.append((f"parity: {nm}", r, f))

# 6: footgun 1 — over-retraction (unbound negated var + independent derivation).
CASES.append((
    "footgun1: over-retraction (2 blockers + special)",
    ZSet({
        (("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))): 1,
        (("reach", ("?X", "?Z")), ("edge", ("?X", "?Y")), ("reach", ("?Y", "?Z"))): 1,
        (("lonely", ("?X",)), ("node", ("?X",)), ("!reach", ("?X", "?Y"))): 1,
        (("lonely", ("?X",)), ("special", ("?X",))): 1,
    }),
    ZSet({("edge", (0, 1)): 1, ("edge", (1, 2)): 1, ("node", (0,)): 1, ("special", (0,)): 1}),
))

# 7: ghost cycle (retraction + positive cycle downstream).
CASES.append((
    "ghost cycle: late mark, b -> p/q cycle",
    ZSet({
        (("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))): 1,
        (("reach", ("?X", "?Z")), ("edge", ("?X", "?Y")), ("reach", ("?Y", "?Z"))): 1,
        (("mark", ("?X",)), ("probe", ("?X",)), ("reach", ("?X", 3))): 1,
        (("b", ("?X",)), ("probe", ("?X",)), ("!mark", ("?X",))): 1,
        (("p", ("?X",)), ("b", ("?X",))): 1,
        (("q", ("?X",)), ("p", ("?X",))): 1,
        (("p", ("?X",)), ("q", ("?X",))): 1,
    }),
    ZSet({("edge", (0, 1)): 1, ("edge", (1, 2)): 1, ("edge", (2, 3)): 1, ("probe", (0,)): 1}),
))

# 8-10: depth-2 / depth-3 chains (the stratified driver's failure zone).
CASES.append((
    "depth-2 V1 (mark fires)",
    ZSet({
        (("mark", ("?X",)), ("probe", ("?X",))): 1,
        (("b", ("?X",)), ("probe", ("?X",)), ("!mark", ("?X",))): 1,
        (("a", ("?X",)), ("probe", ("?X",)), ("!b", ("?X",))): 1,
    }),
    ZSet({("probe", (0,)): 1}),
))
CASES.append((
    "depth-3 V3",
    ZSet({
        (("mark", ("?X",)), ("probe", ("?X",))): 1,
        (("b", ("?X",)), ("probe", ("?X",)), ("!mark", ("?X",))): 1,
        (("a", ("?X",)), ("probe", ("?X",)), ("!b", ("?X",))): 1,
        (("c", ("?X",)), ("probe", ("?X",)), ("!a", ("?X",))): 1,
    }),
    ZSet({("probe", (0,)): 1}),
))
CASES.append((
    "depth-2 mixed per-constant fire/block",
    ZSet({
        (("mark", ("?X",)), ("special", ("?X",))): 1,
        (("b", ("?X",)), ("probe", ("?X",)), ("!mark", ("?X",))): 1,
        (("a", ("?X",)), ("probe", ("?X",)), ("!b", ("?X",))): 1,
    }),
    ZSet({("probe", (0,)): 1, ("probe", (1,)): 1, ("special", (0,)): 1}),
))

# 11: two-negation-level chain over recursion (2-D got this right).
CASES.append((
    "depth-2 over recursion (late mark)",
    ZSet({
        (("reach", ("?X", "?Y")), ("edge", ("?X", "?Y"))): 1,
        (("reach", ("?X", "?Z")), ("edge", ("?X", "?Y")), ("reach", ("?Y", "?Z"))): 1,
        (("mark", ("?X",)), ("probe", ("?X",)), ("reach", ("?X", 3))): 1,
        (("b", ("?X",)), ("probe", ("?X",)), ("!mark", ("?X",))): 1,
        (("a", ("?X",)), ("probe", ("?X",)), ("!b", ("?X",))): 1,
    }),
    ZSet({("edge", (0, 1)): 1, ("edge", (1, 2)): 1, ("edge", (2, 3)): 1, ("probe", (0,)): 1}),
))

if __name__ == "__main__":
    for nm, r, f in CASES:
        compare(nm, r, f)
