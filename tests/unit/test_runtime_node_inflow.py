"""P2.3 — NodeLateralInflow runtime action factory.

Construction tier (space/validation) needs no engine; integration tier
applies an inflow to the real engine and reads it back (no mocks).

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.unit._base import BaseEngineTest

from openswmm_gymnasium.spaces import NodeLateralInflow


class TestConstruction(unittest.TestCase):
    def test_space_bounds_and_shape(self):
        f = NodeLateralInflow(["J1", "J2"], max_inflow=5.0)
        self.assertEqual(f.name, "node_lateral_inflow")
        self.assertEqual(f.space.shape, (2,))
        self.assertEqual(float(f.space.high[0]), 5.0)
        self.assertEqual(float(f.space.low[0]), 0.0)

    def test_rejects_empty(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            NodeLateralInflow([], max_inflow=1.0)

    def test_rejects_nonpositive_max(self):
        with self.assertRaisesRegex(ValueError, "max_inflow"):
            NodeLateralInflow(["J1"], max_inflow=0.0)


class _FakeNodes:
    """Minimal stand-in with the real ``_NodesCompat`` attribute layout."""

    def __init__(self) -> None:
        self.applied: dict[int, float] = {}

    def get_index(self, node_id: str) -> int:
        return {"J1": 0, "J2": 1}[node_id]

    def set_lateral_inflow(self, idx: int, value: float) -> None:
        self.applied[idx] = value


class _FakeAdapter:
    """Adapter shape without the scalar setters -- they live on the collections.

    Deliberately does **not** define ``set_lateral_inflow``: the real
    ``SolverAdapter`` doesn't either, and there is no ``__getattr__``
    delegation. ``apply`` reaching for it directly raised ``AttributeError``
    until the call was routed through ``adapter.nodes``.
    """

    def __init__(self) -> None:
        self.nodes = _FakeNodes()


class TestApplyDelegation(unittest.TestCase):
    """Engine-free guard on the collection the setter is reached through."""

    def test_apply_goes_through_the_nodes_collection(self):
        adapter = _FakeAdapter()
        f = NodeLateralInflow(["J1", "J2"], max_inflow=10.0)
        f.bind(adapter)
        f.apply(adapter, np.array([2.5, 7.5], dtype=np.float32))
        self.assertEqual(adapter.nodes.applied, {0: 2.5, 1: 7.5})

    def test_values_are_clipped_to_the_action_range(self):
        adapter = _FakeAdapter()
        f = NodeLateralInflow(["J1", "J2"], max_inflow=5.0)
        f.bind(adapter)
        f.apply(adapter, np.array([-3.0, 99.0], dtype=np.float32))
        self.assertEqual(adapter.nodes.applied, {0: 0.0, 1: 5.0})

    def test_apply_before_bind_raises(self):
        f = NodeLateralInflow(["J1"], max_inflow=1.0)
        with self.assertRaisesRegex(RuntimeError, "bind"):
            f.apply(_FakeAdapter(), np.array([0.5], dtype=np.float32))


class TestApply(BaseEngineTest):
    def test_apply_sets_inflow(self):
        adapter = self.make_adapter(open=True)
        f = NodeLateralInflow(["J1"], max_inflow=10.0)
        f.bind(adapter)
        f.apply(adapter, np.array([2.5], dtype=np.float32))
        idx = adapter.nodes.get_index("J1")
        # swmm_node_set_lateral_inflow writes the override to user_lat_flow;
        # get_lateral_inflow reads lat_flow, which routing folds the override
        # into. The read-back is therefore only meaningful after a step.
        adapter.step()
        self.assertAlmostEqual(adapter.nodes.get_lateral_inflow(idx), 2.5, places=4)
