"""Datalog value-layer terms: facts, rules, rewrites, unification, and
the indexed-join helpers.

This module is pure data and pure functions. No Circuit / Evaluator /
Stream dependencies. It is the substrate over which the v2 circuit
operators in :mod:`pydbsp.relational_operators` and
:mod:`pydbsp.indexed_relational_operators` (and the algorithm modules
that follow them) construct their bodies.

Two layers of types stack here:

* **Plain (positive) Datalog** . ``Constant``, ``Variable``, ``Atom``,
  ``Fact``, ``Rule``, ``Program``, ``EDB``, plus the provenance /
  signal / direction types that drive the gatekeep machinery.
* **Indexed extensions** . ``ColumnReference``, ``IndexedFact``,
  ``IndexedGatekeepEntry``, ``JoinKey``, ``ExtendedDirection``, plus
  the ``ext_dir`` / ``jorder`` / ``_index_fact`` / ``_gatekeep_join_key``
  helpers that the sort-merge-join variant uses.

The :class:`Rewrite` value type and its monoid (``_rewrite_monoid``)
are shared by both layers. So is ``unify``.

Originally split across ``pydbsp.algorithms.datalog`` and
``pydbsp.algorithms.datalog_indexed`` in v1. Consolidated here for the
v2 layout, with the v1 evaluator/stream imports stripped.
"""

from functools import total_ordering
from typing import Any, Callable, NewType, TypeAlias, cast

from pydbsp.zset import ZSet


# ---- Datalog types ----------------------------------------------------------

Constant = Any
_Variable: TypeAlias = str
Variable = NewType("Variable", _Variable)
Term = Constant | Variable


def is_variable(term: object) -> bool:
    """Return ``True`` iff ``term`` is a Variable. Variables are strings
    beginning with ``?``. Any other value (ints, tuples, lists, strings
    without the ``?`` prefix) is a Constant. The prefix is the only
    discriminator the unifier and :meth:`Rewrite.apply` consult, so
    callers must construct Variables as ``Variable("?X")``, never as
    a bare ``"X"``."""
    return isinstance(term, str) and term.startswith("?")


@total_ordering
class Skolem:
    """A term constructor for IDB value invention — a Skolem function symbol.

    ``Skolem(functor, args)`` placed in a rule HEAD is grounded by
    :meth:`Rewrite.apply`, which recursively substitutes the rule's body
    bindings into ``args`` (each a Variable, Constant, or nested ``Skolem``).
    The result is a deterministic compound *value*: a fresh identity that is a
    pure function of its bound arguments. No gensym counter is needed, and
    incremental maintenance stays exact — a change to the source bindings
    retracts precisely the Skolem values keyed on them. This is what lets a
    Datalog program *construct* trees (one invented node per source binding)
    rather than only project existing facts.

    A *ground* Skolem (no Variables left in ``args``) is an ordinary hashable
    Constant: :func:`is_variable` returns False for it, so the unifier and the
    indexed-join keys treat it as an opaque value and never destructure it.
    Term construction therefore lives only in heads; a body consumes the
    invented value through a Variable that binds to it.

    Function symbols make Datalog Turing-complete, so a generative head inside
    a recursive cycle can derive infinite terms. The stratification check
    (``_check_stratifiable`` in :mod:`pydbsp.datalog_stratified`) rejects any
    cyclic SCC that passes through a Skolem-bearing head, which guarantees
    termination for the structural (acyclic-over-source) programs this targets.
    """

    __slots__ = ("functor", "args", "_hash")

    def __init__(self, functor: Constant, args: tuple[Term, ...]) -> None:
        self.functor = functor
        self.args = tuple(args)
        self._hash = None

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Skolem)
            and self.functor == other.functor
            and self.args == other.args
        )

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((Skolem, self.functor, self.args))
        return self._hash

    def __lt__(self, other: object) -> bool:
        # A total, type-stable order so ground Skolems can live in the indexed
        # sort-merge spines alongside ordinary constants (the ``_Comparable``
        # contract). ``_order_key`` tags by type so heterogeneous terms never
        # raise; Skolems sort after all plain constants.
        return _order_key(self) < _order_key(other)

    def __repr__(self) -> str:
        return f"Skolem({self.functor!r}, {self.args!r})"


def _order_key(term: object) -> tuple:
    """Total ordering key for any Datalog term. Plain constants key as
    ``(0, typename, value)`` — the typename string decides cross-type order
    before any raw value comparison, so mixed-type terms never raise. Skolems
    key as ``(1, functor_key, arg_keys)``, sorting after all plain constants
    and recursing structurally."""
    if type(term) is Skolem:
        return (1, _order_key(term.functor), tuple(_order_key(a) for a in term.args))
    return (0, type(term).__name__, term)


Predicate: TypeAlias = str
Atom: TypeAlias = tuple[Predicate, tuple[Term, ...]]
Fact: TypeAlias = tuple[Predicate, tuple[Constant, ...]]
Rule: TypeAlias = tuple[Atom, ...]

Program: TypeAlias = ZSet[Rule]
EDB: TypeAlias = ZSet[Fact]

Provenance: TypeAlias = int
Direction: TypeAlias = tuple[Provenance, Provenance, Atom | None]
ProvenanceChain: TypeAlias = ZSet[Direction]
Signal: TypeAlias = tuple[Provenance, Atom]
GroundingSignals: TypeAlias = ZSet[Signal]

ProvenanceIndexedRewrite: TypeAlias = tuple[Provenance, "Rewrite"]
AtomWithSourceRewriteAndProvenance: TypeAlias = tuple[Provenance, Atom | None, "Rewrite"]


# ---- Rewrites ---------------------------------------------------------------


class Rewrite:
    inner: dict[Variable, Constant]
    _hash: int | None

    def __init__(self, substitutions: dict[Variable, Constant]) -> None:
        self.inner = substitutions
        self._hash = None

    def __repr__(self) -> str:
        return self.inner.__repr__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rewrite):
            return False
        return self.inner == other.inner

    def __contains__(self, variable: Variable) -> bool:
        return variable in self.inner

    def __getitem__(self, variable: Variable) -> Constant | None:
        return self.inner.get(variable)

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(frozenset(self.inner.items()))
        return self._hash

    def _ground_skolem(self, term: "Skolem") -> "Skolem":
        """Substitute this rewrite into a Skolem constructor's args, recursing
        into nested Skolems. The common var/constant cases stay inline so the
        non-Skolem fast path in :meth:`apply` is untouched."""
        inner = self.inner
        return Skolem(
            term.functor,
            tuple(
                inner.get(a, a)
                if is_variable(a)
                else self._ground_skolem(a)
                if type(a) is Skolem
                else a
                for a in term.args
            ),
        )

    def apply(self, atom: Atom) -> Atom:
        inner = self.inner
        if not inner:
            return atom
        return (
            atom[0],
            tuple(
                inner.get(t, t)
                if is_variable(t)
                else self._ground_skolem(t)
                if type(t) is Skolem
                else t
                for t in atom[1]
            ),
        )


class RewriteMonoid:
    def add(self, a: Rewrite, b: Rewrite) -> Rewrite:
        # ``a`` takes precedence on key collision (matches the old
        # "first seen wins" loop). ``{**b, **a}`` is one C-implemented
        # dict merge instead of two Python loops with a redundant
        # ``Variable(var)`` no-op wrapper per iteration.
        if not a.inner:
            return b
        if not b.inner:
            return a
        return Rewrite({**b.inner, **a.inner})

    def identity(self) -> Rewrite:
        return Rewrite({})


_rewrite_monoid = RewriteMonoid()


_UNBOUND = object()  # sentinel so ``None`` stays a valid constant value


def unify(atom: Atom, fact: Fact) -> Rewrite | None:
    # Build the binding dict in-place; skip the per-term
    # ``Rewrite({term: constant})`` + ``RewriteMonoid.add`` intermediates
    # the original loop allocated on every iteration.
    atom_args = atom[1]
    fact_args = fact[1]
    if len(atom_args) != len(fact_args):
        return _rewrite_monoid.identity()
    bindings: dict[Variable, Constant] = {}
    for term, constant in zip(atom_args, fact_args):
        if is_variable(term):
            existing = bindings.get(term, _UNBOUND)
            if existing is _UNBOUND:
                bindings[term] = constant
            elif existing != constant:
                return None
        elif term != constant:
            return None
    return Rewrite(bindings)


# ---- Interpreted (built-in) predicates: lazily-evaluated infinite relations -
#
# A predicate like ``lt`` denotes the INFINITE relation ``{(a, b) : a < b}`` — it
# has infinite support, so it can never be a (finite) ZSet of facts. Instead it is
# an *interpreted* predicate: a decidable guard evaluated on bindings already
# produced by finite positive atoms. Range-restriction (every argument bound by a
# preceding positive body atom — see :func:`check_interpreted_safety`) keeps the
# join finite and the guard TOTAL over ground values. The body operators route
# these atoms to a stateless select (no facts-join, no delay), exactly parallel to
# negation's pass-through branch — so the infinite relation never becomes an operand.

INTERPRETED: dict[str, Callable[..., bool]] = {
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "neq": lambda a, b: a != b,
}


def register_interpreted(name: str, fn: Callable[..., bool]) -> None:
    """Register an interpreted predicate ``name`` → a decidable guard ``fn``."""
    INTERPRETED[name] = fn


def eval_interpreted(atom: Atom, rewrite: "Rewrite") -> bool:
    """Evaluate an interpreted atom under ``rewrite``. Safety guarantees every arg is
    ground, so this is a total Boolean function of the bound values."""
    return INTERPRETED[atom[0]](*rewrite.apply(atom)[1])


def _term_vars(terms: tuple[Term, ...]) -> set[Variable]:
    """Variables occurring in ``terms`` (recursing into Skolem args)."""
    out: set[Variable] = set()
    for t in terms:
        if is_variable(t):
            out.add(cast(Variable, t))
        elif type(t) is Skolem:
            out |= _term_vars(t.args)
    return out


def check_interpreted_safety(program: "ZSet[Rule]") -> None:
    """Reject any rule where an interpreted atom's arguments are not ALL bound by a
    preceding positive (non-interpreted, non-negated) body atom. Safety is what makes
    the interpreted guard total over ground values — without it the guard would meet
    an unbound ``?var`` and compare a string."""
    for rule, w in program.inner.items():
        if w <= 0:
            continue
        bound: set[Variable] = set()
        for atom in rule[1:]:
            pred = atom[0]
            if pred in INTERPRETED:
                unbound = _term_vars(atom[1]) - bound
                if unbound:
                    raise ValueError(
                        f"unsafe interpreted atom {pred}{atom[1]} in rule with head "
                        f"{rule[0]}: {sorted(unbound)} not bound by a preceding positive atom"
                    )
            elif not pred.startswith("!"):
                bound |= _term_vars(atom[1])
    return None


# ---- Program-derived functions ---------------------------------------------


def sig(program: Program) -> GroundingSignals:
    # ACCUMULATE shared keys: sig must be LINEAR in the program Z-set — it is
    # applied to DELTA cells by Lift1, so sig(P + ΔP) = sig(P) + sig(ΔP) is
    # load-bearing (the Lean model proves this as `sigFunc.linear_ok`). Dict
    # ASSIGNMENT here silently under-counted shared keys (rules with permuted
    # bodies and the same head share the running total); single-batch runs
    # never notice (distinct clamps multiplicity), but rule DELETION of one
    # sharer then annihilates the survivors' signal.
    signals: dict[Signal, int] = {}
    for rule, weight in program.inner.items():
        if weight == 0:
            continue
        running = 0
        if len(rule) > 1:
            for dep in rule[1:]:
                running += hash(dep)
        key = (running, rule[0])
        signals[key] = signals.get(key, 0) + weight
    return ZSet({k: w for k, w in signals.items() if w != 0})


def dir(program: Program) -> ProvenanceChain:
    # ACCUMULATE shared keys: the provenance chain deliberately shares rewrite
    # prefixes across rules (running atom-hash, no rule id), so distinct rules
    # ROUTINELY contribute the same direction. Linearity (`dirFunc.linear_ok`
    # in the Lean model) requires the weights to ADD; see sig() for the
    # rule-deletion failure mode of the old dict-assignment version.
    directions: dict[Direction, int] = {}
    for rule, weight in program.inner.items():
        if weight == 0:
            continue
        running = 0
        if len(rule) > 1:
            for dep in rule[1:]:
                prev = running
                running += hash(dep)
                key = (prev, running, dep)
                directions[key] = directions.get(key, 0) + weight
    return ZSet({k: w for k, w in directions.items() if w != 0})


def rewrite_product_projection(
    gatekeep: AtomWithSourceRewriteAndProvenance,
    fact: Fact,
) -> ProvenanceIndexedRewrite:
    provenance = gatekeep[0]
    possible_atom = gatekeep[1]
    rewrite = gatekeep[2]

    fresh = _rewrite_monoid.identity()
    if possible_atom is not None:
        fresh = cast(Rewrite, unify(rewrite.apply(possible_atom), fact))
    return provenance, _rewrite_monoid.add(rewrite, fresh)


# ---- Indexed extensions ----------------------------------------------------
#
# The sort-merge-join variant of incremental Datalog needs three
# additional pieces of bookkeeping on top of the plain Rule / Atom
# vocabulary above:
#
# * A ``ColumnReference`` per body atom — the indices of arguments that
#   were already bound by an earlier body atom in the rule. These
#   columns define the join key.
# * An ``ExtendedDirection`` chain — the standard provenance chain with
#   each body atom's ``col_ref`` attached.
# * An ``IndexedFact`` — a fact paired with its ``(pred, values_at_col_ref)``
#   join key, so both sides of the join are pre-indexed.

ColumnReference = tuple[int, ...]
ExtendedDirection = tuple[Provenance, Provenance, Atom | None, ColumnReference]
ExtendedProvenanceChain = ZSet[ExtendedDirection]

IndexedGatekeepEntry = tuple[Provenance, Atom | None, Rewrite, ColumnReference]
JoinKey = tuple[str, tuple]
IndexedFact = tuple[JoinKey, Fact]


def _rule_col_refs(rule: Rule) -> list[tuple[Atom, ColumnReference]]:
    variables: set[Variable] = set()
    out: list[tuple[Atom, ColumnReference]] = []
    for body_atom in rule[1:]:
        cols: list[int] = []
        fresh: set[Variable] = set()
        for idx, term in enumerate(body_atom[1]):
            if isinstance(term, _Variable) and term not in variables:
                fresh.add(cast(Variable, term))
            else:
                cols.append(idx)
        out.append((body_atom, tuple(cols)))
        variables |= fresh
    return out


def ext_dir(program: Program) -> ExtendedProvenanceChain:
    """Direction chain with per-body-atom ``col_ref`` attached. ACCUMULATES
    shared keys — same linearity requirement as :func:`dir`."""
    entries: dict[ExtendedDirection, int] = {}
    for rule, weight in program.inner.items():
        if weight == 0:
            continue
        running = 0
        for body_atom, col_ref in _rule_col_refs(rule):
            prev = running
            running += hash(body_atom)
            key = (prev, running, body_atom, col_ref)
            entries[key] = entries.get(key, 0) + weight
    return ZSet({k: w for k, w in entries.items() if w != 0})


def jorder(program: Program) -> ZSet[tuple[str, ColumnReference]]:
    """``(pred, col_ref)`` pairs across all rule body atoms, ACCUMULATING
    shared pairs -- same linearity requirement as :func:`dir` (C5 Def 4.6
    defines the join order Z-set as weighted). The distinct after the
    index join renormalizes the multiplicities, and linearity is what
    retracts an index configuration when its last rule is removed."""
    entries: dict[tuple[str, ColumnReference], int] = {}
    for rule, weight in program.inner.items():
        if weight == 0:
            continue
        for body_atom, col_ref in _rule_col_refs(rule):
            key = (body_atom[0].lstrip("!"), col_ref)
            entries[key] = entries.get(key, 0) + weight
    return ZSet({k: w for k, w in entries.items() if w != 0})


def _index_fact(col_ref: ColumnReference, fact: Fact) -> IndexedFact:
    values = tuple(fact[1][i] for i in col_ref)
    return ((fact[0], values), fact)


def _gatekeep_join_key(gk: IndexedGatekeepEntry) -> JoinKey:
    _prov, atom, rewrite, col_ref = gk
    if atom is None:
        return ("", ())
    applied = rewrite.apply(atom)
    return (applied[0].lstrip("!"), tuple(applied[1][i] for i in col_ref))


def _indexed_product_proj(
    _key: JoinKey,
    gk: IndexedGatekeepEntry,
    ifact: IndexedFact,
) -> ProvenanceIndexedRewrite | None:
    prov, atom, rewrite, _col_ref = gk
    fact = ifact[1]
    if atom is None:
        return prov, rewrite
    fresh = unify(rewrite.apply(atom), fact)
    if fresh is None:
        return None
    return prov, _rewrite_monoid.add(rewrite, fresh)
