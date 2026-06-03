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


class TestApply(BaseEngineTest):
    def test_apply_sets_inflow(self):
        adapter = self.make_adapter(open=True)
        f = NodeLateralInflow(["J1"], max_inflow=10.0)
        f.bind(adapter)
        f.apply(adapter, np.array([2.5], dtype=np.float32))
        idx = adapter.nodes.get_index("J1")
        # The lateral inflow we injected should be read back.
        self.assertAlmostEqual(adapter.nodes.get_lateral_inflow(idx), 2.5, places=4)
