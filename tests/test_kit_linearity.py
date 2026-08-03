"""Regression: the kit functions are linear in the program Z-set.

``dir`` / ``sig`` / ``ext_dir`` are applied to DELTA cells by ``Lift1``,
so ``f(P + dP) = f(P) + f(dP)`` is load-bearing (``dirFunc.linear_ok`` /
``sigFunc.linear_ok`` in the Lean model). The dict-assignment versions
under-counted keys shared between rules (the seed and transitive rules
of Pi_path share their first-atom direction), and rule DELETION of one
sharer then annihilated the survivors' entry. Fixed in 0754ed2; these
tests pin it."""

from pydbsp.datalog import dir, ext_dir, sig
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
