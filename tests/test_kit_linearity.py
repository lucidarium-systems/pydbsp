"""Regression: the kit functions are linear in the program Z-set.

``dir`` / ``sig`` / ``ext_dir`` are applied to DELTA cells by ``Lift1``,
so ``f(P + dP) = f(P) + f(dP)`` is load-bearing (``dirFunc.linear_ok`` /
``sigFunc.linear_ok`` in the Lean model). The dict-assignment versions
under-counted keys shared between rules (the seed and transitive rules
of Pi_path share their first-atom direction), and rule DELETION of one
sharer then annihilated the survivors' entry. Fixed in 0754ed2; these
tests pin it."""

from pydbsp.datalog import dir, ext_dir, jorder, sig
from pydbsp.zset import ZSet, ZSetAddition

SEED = (("path", ("?X", "?Y")), ("edge", ("?X", "?Y")))
TRANS = (
    ("path", ("?X", "?Z")),
    ("edge", ("?X", "?Y")),
    ("path", ("?Y", "?Z")),
)


def _assert_linear(f) -> None:
    g = ZSetAddition()
    combined = f(ZSet({SEED: 1, TRANS: 1}))
    additive = g.add(f(ZSet({SEED: 1})), f(ZSet({TRANS: 1})))
    assert combined.inner == additive.inner


def test_dir_is_linear_on_shared_prefixes() -> None:
    """Seed and trans share their first-atom direction — weights add."""
    _assert_linear(dir)


def test_sig_is_linear() -> None:
    _assert_linear(sig)


def test_ext_dir_is_linear_on_shared_prefixes() -> None:
    _assert_linear(ext_dir)


def test_shared_direction_survives_single_rule_deletion() -> None:
    """Deleting one sharer must leave the other's direction standing."""
    g = ZSetAddition()
    state = dir(ZSet({SEED: 1, TRANS: 1}))
    after = g.add(state, dir(ZSet({TRANS: -1})))
    shared = [w for (prev, _run, _dep), w in after.inner.items() if prev == 0]
    assert shared == [1]


def test_jorder_is_linear() -> None:
    """jorder(a + b) == jorder(a) + jorder(b): shared (pred, col_ref)
    pairs accumulate weight, and removing one sharer keeps the survivor
    (C5 Def 4.6: the join order Z-set is weighted)."""
    seed = (("path", ("?X", "?Y")), ("edge", ("?X", "?Y")))
    trans = (("path", ("?X", "?Z")), ("edge", ("?X", "?Y")), ("path", ("?Y", "?Z")))
    both = ZSet({seed: 1, trans: 1})
    jo = jorder(both)
    # seed and trans share the unindexed edge configuration.
    assert jo.inner[("edge", ())] == 2
    assert jo.inner[("path", (0,))] == 1
    # Linearity: the sum of the parts equals the whole.
    g: ZSetAddition[tuple[str, tuple[int, ...]]] = ZSetAddition()
    assert g.add(jorder(ZSet({seed: 1})), jorder(ZSet({trans: 1}))).inner == jo.inner
    # Removing trans annihilates its share and only its share.
    after = g.add(jo, jorder(ZSet({trans: -1})))
    assert after.inner == {("edge", ()): 1}
