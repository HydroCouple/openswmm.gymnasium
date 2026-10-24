# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 Caleb Buahin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""P2.3 — NodeLateralInflow runtime action factory.

Construction tier (space/validation) needs no engine; integration tier
applies an inflow to the real engine and reads it back (no mocks).

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest

import numpy as np

from openswmm_gymnasium.spaces import NodeLateralInflow
from tests.unit._base import BaseEngineTest


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


class _FakeAdapter:
    """The adapter surface the factory uses: catalog C{index} and C{write}.

    Deliberately has no C{set_lateral_inflow} and no node collection: the
    real C{SolverAdapter} has neither, and there is no C{__getattr__}
    delegation, so C{apply} must write the catalog field
    C{node.lateral_inflow}.
    """

    def __init__(self) -> None:
        self.written: dict[str, dict[int, float]] = {}

    def index(self, kind: str, element_id: str) -> int:
        assert kind == "node", kind
        return {"J1": 0, "J2": 1}[element_id]

    def write(self, path: str, idx: int, value: float) -> None:
        self.written.setdefault(path, {})[idx] = value


class TestApplyDelegation(unittest.TestCase):
    """Engine-free guard on the catalog field the setter writes."""

    def test_apply_writes_the_lateral_inflow_field(self):
        adapter = _FakeAdapter()
        f = NodeLateralInflow(["J1", "J2"], max_inflow=10.0)
        f.bind(adapter)
        f.apply(adapter, np.array([2.5, 7.5], dtype=np.float32))
        self.assertEqual(adapter.written, {"node.lateral_inflow": {0: 2.5, 1: 7.5}})

    def test_values_are_clipped_to_the_action_range(self):
        adapter = _FakeAdapter()
        f = NodeLateralInflow(["J1", "J2"], max_inflow=5.0)
        f.bind(adapter)
        f.apply(adapter, np.array([-3.0, 99.0], dtype=np.float32))
        self.assertEqual(adapter.written, {"node.lateral_inflow": {0: 0.0, 1: 5.0}})

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
        idx = adapter.index("node", "J1")
        # swmm_node_set_lateral_inflow writes the override to user_lat_flow;
        # the node.lateral_inflow read returns lat_flow, which routing folds
        # the override into. The read-back is therefore only meaningful after
        # a step.
        adapter.step()
        self.assertAlmostEqual(adapter.read("node.lateral_inflow", idx), 2.5, places=4)
