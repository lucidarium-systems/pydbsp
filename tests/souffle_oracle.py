"""Soufflé as ground truth for stratified negation.

The specification lives in Lean (V3/StratifiedCircuit.lean: ``stratSpec``,
``stratSpec_computes``), but it is ``noncomputable`` there, so it cannot be
run and diffed against. Transcribing it into Python would make a second
source of truth that can drift from the first, so the semantic oracle is an
INDEPENDENT implementation of the same standard semantics instead. Soufflé
is that. The structural properties of the circuit -- pointwise stratum lift,
stage per cell, no control facts -- are checked separately, and those need no
oracle because they are statements about shape, not about meaning.

Translation, pydbsp program -> Soufflé:
  constants become ``symbol`` (ints stringified, quoted in rules, raw in
  facts); predicates are sanitised to ``p_<name>``; ``?X`` becomes ``VX``; a
  negated atom's variable left unbound by the positive body becomes ``_``,
  which is Soufflé's don't-care and matches pydbsp's existential negation.
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def souffle_argv():
    """Soufflé is usually not installed; nix can supply it."""
    if shutil.which("souffle"):
        return ["souffle"]
    if shutil.which("nix"):
        return ["nix", "run", "nixpkgs#souffle", "--"]
    return None


available = souffle_argv() is not None


def _pred(name):
    return "p_" + re.sub(r"[^A-Za-z0-9_]", "_", name.lstrip("!"))


def _var(t):
    return "V" + t[1:] if isinstance(t, str) and t.startswith("?") else None


def _arity(rules, facts):
    ar = {}
    for f in facts:
        ar[_pred(f[0])] = len(f[1])
    for r in rules:
        for atom in r:
            ar[_pred(atom[0])] = len(atom[1])
    return ar


def model(rules, facts):
    """The stratified model of ``rules`` over ``facts``, as Soufflé computes
    it. Returns a set of pydbsp-shaped ``(pred, terms)`` tuples."""
    argv = souffle_argv()
    if argv is None:
        raise RuntimeError("souffle unavailable")
    ar = _arity(rules, facts)
    edb = {_pred(f[0]) for f in facts}

    lines = []
    for p, n in sorted(ar.items()):
        cols = ", ".join(f"c{i}: symbol" for i in range(n))
        lines.append(f".decl {p}({cols})")
        if p in edb:
            lines.append(f".input {p}")
        lines.append(f".output {p}")
    for r in rules:
        head, body = r[0], r[1:]
        bound = {v for atom in body if not atom[0].startswith("!")
                 for v in map(_var, atom[1]) if v}

        def render(atom, allow_wild):
            terms = []
            for t in atom[1]:
                v = _var(t)
                if v is None:
                    terms.append(f'"{t}"')
                elif allow_wild and v not in bound:
                    terms.append("_")
                else:
                    terms.append(v)
            return f"{_pred(atom[0])}({', '.join(terms)})"

        parts = [("!" if a[0].startswith("!") else "")
                 + render(a, a[0].startswith("!")) for a in body]
        lines.append(f"{render(head, False)} :- {', '.join(parts)}.")

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "facts").mkdir()
        (d / "out").mkdir()
        (d / "prog.dl").write_text("\n".join(lines) + "\n")
        by_pred = {}
        for f in facts:
            by_pred.setdefault(_pred(f[0]), []).append(f[1])
        for p, rows in by_pred.items():
            (d / "facts" / f"{p}.facts").write_text(
                "".join("\t".join(str(c) for c in row) + "\n" for row in rows))
        res = subprocess.run(
            argv + ["-F", str(d / "facts"), "-D", str(d / "out"), str(d / "prog.dl")],
            capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"souffle failed: {res.stderr[-800:]}")
        unsan = {_pred(p): p for p in
                 {a[0].lstrip("!") for r in rules for a in r} | {f[0] for f in facts}}
        out = set()
        for csv in (d / "out").glob("*.csv"):
            pred = unsan.get(csv.stem, csv.stem)
            for line in csv.read_text().splitlines():
                if line:
                    out.add((pred, tuple(line.split("\t"))))
        return out


def as_souffle_shape(zset_facts):
    """pydbsp facts in Soufflé's shape: every constant stringified."""
    return {(p, tuple(str(c) for c in terms)) for p, terms in zset_facts}
