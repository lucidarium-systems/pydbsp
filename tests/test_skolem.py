"""Skolemized Datalog — term construction (value invention) in rule heads.

A ``Skolem(functor, args)`` term in a head is grounded by ``Rewrite.apply``,
which substitutes the body's bindings into ``args`` (recursing into nested
Skolems). This turns Datalog from a project-only language into one that
*constructs* trees: one invented node per source binding, identity a pure
function of the binding (so incremental maintenance stays exact).

Covered here:
  * value-layer: ``Rewrite.apply`` grounds (nested) Skolem head terms;
  * a ground Skolem is an opaque hashable Constant (``is_variable`` False);
  * end-to-end over the public ``IncrementalDatalogStratified`` driver —
    inventing nodes, then building a parent node *over* an invented child,
    then consuming an invented value as a downstream join key;
  * the same on the INDEXED body (``IndexedIncrementalDatalogBody``),
    including a Skolem value used as a sort-merge join key;
  * the termination guard: a recursive generative head is rejected at
    ``push_rules``, a non-recursive one is accepted.
"""

from __future__ import annotations

import pytest

from pydbsp import datalog as dlg
from pydbsp.circuit import Circuit
from pydbsp.compute import ComputeCtx
from pydbsp.core import Antichain, dbsp_time
from pydbsp.datalog import Rewrite, Skolem
from pydbsp.datalog_stratified import IncrementalDatalogStratified
from pydbsp.evaluate import Evaluator
from pydbsp.indexed_relational_operators import IndexedIncrementalDatalogBody
from pydbsp.operator import Input, LiftStreamIntroduction
from pydbsp.progress import Time
from pydbsp.storage import DictStorage
from pydbsp.zset import ZSet, ZSetAddition


# ---- value layer ------------------------------------------------------------


def test_apply_grounds_skolem_head() -> None:
    """A head Skolem term grounds by substituting the rewrite into its args."""
    rw = Rewrite({"?N": 42})
    head = ("py_node", (Skolem("wrap_u32", ("?N",)), "call"))
    assert rw.apply(head) == ("py_node", (Skolem("wrap_u32", (42,)), "call"))


def test_apply_grounds_nested_skolem() -> None:
    """Substitution recurses through nested Skolem constructors."""
    rw = Rewrite({"?C": 7})
    head = ("t", (Skolem("parent", (Skolem("child", ("?C",)), "k")),))
    assert rw.apply(head) == (
        "t",
        (Skolem("parent", (Skolem("child", (7,)), "k")),),
    )


def test_apply_without_skolem_is_unchanged() -> None:
    """The non-Skolem fast path is untouched: plain projection still works,
    and a plain tuple constant is NOT treated as a constructor."""
    rw = Rewrite({"?X": 1, "?Y": 2})
    assert rw.apply(("e", ("?X", "?Y"))) == ("e", (1, 2))
    # a bare tuple constant carrying a "?"-string is left alone (only Skolem
    # instances are descended into).
    assert rw.apply(("e", (("?X", "lit"),))) == ("e", (("?X", "lit"),))


def test_ground_skolem_is_opaque_constant() -> None:
    """A ground Skolem is a Constant: not a Variable, hashable, value-equal."""
    s = Skolem("wrap", (1,))
    assert dlg.is_variable(s) is False
    assert s == Skolem("wrap", (1,))
    assert hash(s) == hash(Skolem("wrap", (1,)))
    assert s != Skolem("wrap", (2,))
    # usable as a dict / ZSet key
    assert ZSet({("n", (s,)): 1}).inner[("n", (s,))] == 1


# ---- end-to-end over the public stratified driver ---------------------------


def _rules(*rs: dlg.Rule) -> ZSet[dlg.Rule]:
    return ZSet({r: 1 for r in rs})


def _facts(*fs: dlg.Fact) -> ZSet[dlg.Fact]:
    return ZSet({f: 1 for f in fs})


def test_skolem_invents_one_node_per_source() -> None:
    """``py_node(wrap(?N)) :- c_node(?N)`` mints a distinct invented node for
    each source fact — the core tree-construction primitive."""
    s = IncrementalDatalogStratified()
    s.push_rules(_rules((("py_node", (Skolem("wrap", ("?N",)),)), ("c_node", ("?N",)))))
    s.push_facts(_facts(("c_node", (1,)), ("c_node", (2,))))
    out = s.saturate()
    assert out.inner.get(("py_node", (Skolem("wrap", (1,)),))) == 1
    assert out.inner.get(("py_node", (Skolem("wrap", (2,)),))) == 1


def test_skolem_builds_parent_over_invented_child() -> None:
    """Two generative rules compose: rule 2 builds a node *over* the value
    rule 1 invented, yielding a nested term — i.e. an output subtree."""
    s = IncrementalDatalogStratified()
    s.push_rules(
        _rules(
            (("leaf", (Skolem("wrap", ("?N",)),)), ("c_node", ("?N",))),
            (("parent", (Skolem("neg", ("?P",)),)), ("leaf", ("?P",))),
        )
    )
    s.push_facts(_facts(("c_node", (5,))))
    out = s.saturate()
    child = Skolem("wrap", (5,))
    assert out.inner.get(("leaf", (child,))) == 1
    assert out.inner.get(("parent", (Skolem("neg", (child,)),))) == 1


def test_skolem_value_consumed_as_join_key() -> None:
    """An invented value flows through a Variable and joins against an EDB
    fact that carries the same ground Skolem constant."""
    s = IncrementalDatalogStratified()
    s.push_rules(
        _rules(
            (("node", (Skolem("wrap", ("?N",)),)), ("src", ("?N",))),
            (("keep", ("?P",)), ("node", ("?P",)), ("want", ("?P",))),
        )
    )
    s.push_facts(_facts(("src", (1,)), ("src", (2,)), ("want", (Skolem("wrap", (1,)),))))
    out = s.saturate()
    assert out.inner.get(("keep", (Skolem("wrap", (1,)),))) == 1
    assert ("keep", (Skolem("wrap", (2,)),)) not in out.inner  # 2 not wanted


# ---- indexed body -----------------------------------------------------------


def _drive_indexed_2d(edb: ZSet[dlg.Fact], program: ZSet[dlg.Rule], max_k: int = 64) -> ZSet[dlg.Fact]:
    """Wire the INDEXED positive Datalog body at 2-D, push EDB+program at the
    outer tick, run the inner fixpoint, and return the cumulative facts
    (EDB + every derived diff). Mirrors tests/test_3d_datalog.py::_drive_2d."""
    fact_group = ZSetAddition[dlg.Fact]()
    rewrite_group = ZSetAddition[dlg.ProvenanceIndexedRewrite]()
    rule_group = ZSetAddition[dlg.Rule]()
    e = Evaluator[Time](
        circuit=Circuit[Time](),
        storage=DictStorage(),
        ctx=ComputeCtx(lattice=dbsp_time(2)),
        group=fact_group,
    )
    edb_1d = Input[ZSet[dlg.Fact]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    program_1d = Input[ZSet[dlg.Rule]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    state_facts = Input[ZSet[dlg.Fact]](frontier=Antichain(dbsp_time(2))).connect(e.circuit, ())
    state_rewrites = Input[ZSet[dlg.ProvenanceIndexedRewrite]](frontier=Antichain(dbsp_time(2))).connect(
        e.circuit, ()
    )
    seed_1d = Input[ZSet[dlg.ProvenanceIndexedRewrite]](frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
    program_2d = LiftStreamIntroduction[ZSet[dlg.Rule]](group=rule_group).connect(e.circuit, (program_1d,))
    body = IndexedIncrementalDatalogBody(
        fact_group=fact_group,
        rule_group=rule_group,
        rewrite_group=rewrite_group,
        signal_group=ZSetAddition[dlg.Signal](),
        ext_dir_group=ZSetAddition[dlg.ExtendedDirection](),
        jorder_group=ZSetAddition[tuple[str, dlg.ColumnReference]](),
        gatekeep_group=ZSetAddition[dlg.IndexedGatekeepEntry](),
        indexed_fact_group=ZSetAddition[dlg.IndexedFact](),
    ).connect(e.circuit, (edb_1d, program_2d, state_facts, state_rewrites, seed_1d))

    e.push(edb_1d, edb)
    e.push(program_1d, program)
    e.push(seed_1d, ZSet({(0, dlg._rewrite_monoid.identity()): 1}))
    acc = dict(edb.inner)
    for k in range(max_k):
        diff_facts, diff_rewrites = e.read(body, (0, k))
        if not diff_facts.inner and not diff_rewrites.inner:
            break
        for f, w in diff_facts.inner.items():
            acc[f] = acc.get(f, 0) + w
        e.push(state_facts, diff_facts, t=(0, k))
        e.push(state_rewrites, diff_rewrites, t=(0, k))
    return ZSet({f: w for f, w in acc.items() if w != 0})


def test_skolem_on_indexed_body_with_skolem_join_key() -> None:
    """The INDEXED body grounds Skolem heads AND keys a sort-merge join on an
    invented value — proving the fast path handles value invention end to end."""
    program = _rules(
        (("node", (Skolem("wrap", ("?N",)),)), ("src", ("?N",))),
        (("keep", ("?P",)), ("node", ("?P",)), ("want", ("?P",))),
    )
    edb = _facts(("src", (1,)), ("src", (2,)), ("want", (Skolem("wrap", (2,)),)))
    out = _drive_indexed_2d(edb, program)
    assert out.inner.get(("node", (Skolem("wrap", (1,)),))) == 1
    assert out.inner.get(("node", (Skolem("wrap", (2,)),))) == 1
    assert out.inner.get(("keep", (Skolem("wrap", (2,)),))) == 1
    assert ("keep", (Skolem("wrap", (1,)),)) not in out.inner


# ---- termination guard (generative-head stratification) ---------------------


def test_recursive_generative_head_rejected() -> None:
    """``g(wrap(?X)) :- g(?X)`` invents a strictly larger term every round —
    rejected at push time so it can never run to non-termination."""
    s = IncrementalDatalogStratified()
    with pytest.raises(ValueError, match="recursive term construction"):
        s.push_rules(_rules((("g", (Skolem("wrap", ("?X",)),)), ("g", ("?X",)))))


def test_indirect_generative_cycle_rejected() -> None:
    """A generative head in a 2-predicate cycle (g → h → g) is also rejected:
    the invented value feeds back through ``h``."""
    s = IncrementalDatalogStratified()
    with pytest.raises(ValueError, match="recursive term construction"):
        s.push_rules(
            _rules(
                (("g", (Skolem("w", ("?X",)),)), ("h", ("?X",))),
                (("h", ("?Y",)), ("g", ("?Y",))),
            )
        )


def test_nonrecursive_generative_head_accepted() -> None:
    """A generative head NOT in a cycle (the structural rewrite case) passes
    stratifiability and runs."""
    s = IncrementalDatalogStratified()
    s.push_rules(_rules((("py", (Skolem("w", ("?X",)),)), ("c", ("?X",)))))  # must not raise
    s.push_facts(_facts(("c", (9,))))
    out = s.saturate()
    assert out.inner.get(("py", (Skolem("w", (9,)),))) == 1


# ---- key-grounded generative recursion (the transpiler's structural case) ---
#
# The blanket "reject any cyclic generative SCC" guard is SOUND but too
# conservative: the transpiler's core rule recurses on AST sub-nodes, which
# terminates because the recursion KEY (the node id) descends an EDB relation.
# The refinement accepts generative recursion that strictly DESCENDS an
# EDB-grounded key (head key ≠ recursive-atom key, both bound by an out-of-SCC
# atom), and still rejects recursion where the invented value drives the loop.


def _ast_edb() -> ZSet[dlg.Fact]:
    """`(a * c) + b` as c_binary nodes; leaves seeded via `operand` facts."""
    return _facts(
        ("c_binary", ("n1", "+", "n2", "lb")),
        ("c_binary", ("n2", "*", "la", "lc")),
        ("operand", ("la", "a")),
        ("operand", ("lb", "b")),
        ("operand", ("lc", "c")),
    )


_AST_RULES = _rules(
    # base: a leaf renders to its operand string (a plain projection, not generative)
    (("py_expr", ("?L", "?S")), ("operand", ("?L", "?S"))),
    # recursive generative rule: build an infix node over the operand sub-nodes.
    # py_expr is generative (Skolem in VALUE pos 1) AND self-recursive, but the
    # recursion descends the KEY (pos 0: ?N → ?L/?R via the EDB `c_binary`).
    (
        ("py_expr", ("?N", Skolem("infix", ("?Op", "?La", "?Ra")))),
        ("c_binary", ("?N", "?Op", "?L", "?R")),
        ("py_expr", ("?L", "?La")),
        ("py_expr", ("?R", "?Ra")),
    ),
)


def test_structural_recursion_accepted_and_runs() -> None:
    """The transpiler's core shape: a generative rule recursing on operand
    sub-nodes. Accepted (key descends an EDB relation) and computes the nested
    Python-AST term bottom-up."""
    s = IncrementalDatalogStratified()
    s.push_rules(_AST_RULES)  # must NOT raise
    s.push_facts(_ast_edb())
    out = s.saturate()
    mul = Skolem("infix", ("*", "a", "c"))
    add = Skolem("infix", ("+", mul, "b"))
    assert out.inner.get(("py_expr", ("n2", mul))) == 1
    assert out.inner.get(("py_expr", ("n1", add))) == 1


def test_tree_copy_recursion_accepted() -> None:
    """A clean node→value tree fold (key descends `child`/`tnode` EDB) is
    accepted and folds the whole tree."""
    rules_ = _rules(
        (("fold", ("?N", Skolem("leaf", ("?K",)))), ("tnode", ("?N", "?K")), ("isleaf", ("?N",))),
        (
            ("fold", ("?N", Skolem("br", ("?Lc", "?Rc")))),
            ("kids", ("?N", "?L", "?R")),
            ("fold", ("?L", "?Lc")),
            ("fold", ("?R", "?Rc")),
        ),
    )
    s = IncrementalDatalogStratified()
    s.push_rules(rules_)  # must NOT raise
    s.push_facts(
        _facts(
            ("kids", ("r", "x", "y")),
            ("tnode", ("x", "X")),
            ("isleaf", ("x",)),
            ("tnode", ("y", "Y")),
            ("isleaf", ("y",)),
        )
    )
    out = s.saturate()
    assert out.inner.get(("fold", ("r", Skolem("br", (Skolem("leaf", ("X",)), Skolem("leaf", ("Y",))))))) == 1


def test_skolem_into_recursion_key_rejected() -> None:
    """The invented value flows into the position the recursion ITERATES, with
    only a same-key EDB guard (no descent) — unbounded, must still be rejected.
    `walk(succ(?P), ?G) :- walk(?P, ?G), limit(?G)`: ?G is EDB-bound but it is
    the head's key too (no descent); the growth is on the tainted pos 0."""
    s = IncrementalDatalogStratified()
    with pytest.raises(ValueError, match="term construction"):
        s.push_rules(
            _rules(
                (
                    ("walk", (Skolem("succ", ("?P",)), "?G")),
                    ("walk", ("?P", "?G")),
                    ("limit", ("?G",)),
                )
            )
        )
