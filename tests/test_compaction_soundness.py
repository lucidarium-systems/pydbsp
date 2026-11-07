"""Regression: compaction is sound for at-or-after readers.

v2.1.1 seeded the backward walk AT the cursors and transported dead
regions with ``retreat_omega_fill``, which broke the documented
at-or-after reader contract twice: the memo at the cursor itself was
evicted (``atCursor_not_sound`` in `V3/ProgressAlgebra.lean`), and a
delay's input lost cells that reads at the same shifted coordinate but
larger other coordinates still recompute through
(``rofDead_not_sound_atOrAfter``). v2.1.2 seeds with ``dead_below`` and
transports with ``retreat``. These tests pin both repros and the
Feedback snapshot discipline.
"""

from __future__ import annotations

from pydbsp.circuit import Circuit
from pydbsp.compute import ComputeCtx
from pydbsp.core import Antichain, dbsp_time
from pydbsp.evaluate import Evaluator
from pydbsp.operator import CoreDelay, Input, Integrate
from pydbsp.storage import DictStorage
from pydbsp.zset import ZSet, ZSetAddition


def _delay_chain():
    eg = ZSetAddition[tuple[str, tuple[int, ...]]]()
    ev = Evaluator(
        circuit=Circuit(),
        storage=DictStorage(),
        ctx=ComputeCtx(lattice=dbsp_time(2)),
        group=eg,
    )
    inp = Input(frontier=Antichain(dbsp_time(2))).connect(ev.circuit, ())
    dly = CoreDelay(axis=0, group=eg).connect(ev.circuit, (inp,))
    for k in range(4):
        ev.push(inp, ZSet({("v", (k,)): 1}), t=(0, k))
    return ev, inp, dly


def test_delay_reads_at_cursor_survive_explicit_compact() -> None:
    """Explicit cursor at the delay: reads AT and at-or-after the
    cursor return the pre-compaction values."""
    ev, _, dly = _delay_chain()
    truth = {t: ev.read(dly, t) for t in [(1, 2), (1, 3), (2, 1)]}
    ev.compact({dly: Antichain(dbsp_time(2), [(1, 2)])})
    for t, want in truth.items():
        assert ev.read(dly, t) == want, t


def test_delay_reads_at_frontier_survive_default_compact() -> None:
    """No-arg compact (cursors = settled frontiers): reading the delay
    at its own frontier still returns the true value."""
    ev, _, dly = _delay_chain()
    truth = ev.read(dly, (1, 3))
    ev.compact()
    assert ev.read(dly, (1, 3)) == truth


def test_integrate_snapshot_discipline_survives_compact() -> None:
    """The Feedback boundary snapshot: after a compact at the cursor,
    the cumulative value at the cursor and the next tick both read
    correctly (recomputes stop at the retained boundary memo)."""
    eg = ZSetAddition[tuple[str, tuple[int, ...]]]()
    ev = Evaluator(
        circuit=Circuit(),
        storage=DictStorage(),
        ctx=ComputeCtx(lattice=dbsp_time(1)),
        group=eg,
    )
    inp = Input(frontier=Antichain(dbsp_time(1))).connect(ev.circuit, ())
    acc = Integrate(group=eg).connect(ev.circuit, (inp,))
    for _ in range(3):
        ev.push(inp, ZSet({("e", (1,)): 1}))
    truth = ev.read(acc, (2,))
    assert truth == ZSet({("e", (1,)): 3})
    evicted = ev.compact({acc: Antichain(dbsp_time(1), [(2,)])})
    assert evicted > 0
    assert ev.read(acc, (2,)) == truth
    ev.push(inp, ZSet({("e", (1,)): 1}))
    assert ev.read(acc, (3,)) == ZSet({("e", (1,)): 4})
