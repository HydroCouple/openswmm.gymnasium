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

"""Unit tests for the P2 reward terms.

L{FloodingVolume} P1 coverage stays in C{test_rewards_terms.py}; here
we focus on the four new terms.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest

import numpy as np

from openswmm_gymnasium.rewards import (
    CSOVolume,
    PeakOutflow,
    ReliabilityMargin,
    RewardTerm,
    SetpointSmoothness,
)
from tests.unit._base import BaseEngineTest

# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------

_DIRECTION_CASES = [
    (CSOVolume, {"node_ids": ["J1"]}, "minimize"),
    (PeakOutflow, {"link_ids": ["C1"]}, "minimize"),
    (ReliabilityMargin, {}, "maximize"),
    (SetpointSmoothness, {"link_ids": ["C1"]}, "minimize"),
]

_DEFAULT_NAME_CASES = [
    (CSOVolume, "cso_volume", {"node_ids": ["J1"]}),
    (PeakOutflow, "peak_outflow", {"link_ids": ["C1"]}),
    (ReliabilityMargin, "reliability_margin", {}),
    (SetpointSmoothness, "setpoint_smoothness", {"link_ids": ["C1"]}),
]

_EMPTY_IDS_CASES = [
    (CSOVolume, {"node_ids": []}),
    (PeakOutflow, {"link_ids": []}),
    (SetpointSmoothness, {"link_ids": []}),
]


class TestConstruction(unittest.TestCase):
    def test_direction(self):
        for cls, kwargs, expected_dir in _DIRECTION_CASES:
            with self.subTest(cls=cls, kwargs=kwargs, expected_dir=expected_dir):
                self.assertEqual(cls(**kwargs).direction, expected_dir)

    def test_default_name(self):
        for cls, name, kwargs in _DEFAULT_NAME_CASES:
            with self.subTest(cls=cls, name=name, kwargs=kwargs):
                self.assertEqual(cls(**kwargs).name, name)

    def test_empty_ids_raise(self):
        for cls, kwargs in _EMPTY_IDS_CASES:
            with self.subTest(cls=cls, kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "at least one"):
                    cls(**kwargs)


# -----------------------------------------------------------------------------
# Protocol conformance
# -----------------------------------------------------------------------------

_PROTOCOL_TERMS = [
    CSOVolume(node_ids=["J1"]),
    PeakOutflow(link_ids=["C1"]),
    ReliabilityMargin(),
    SetpointSmoothness(link_ids=["C1"]),
]


class TestProtocolConformance(unittest.TestCase):
    def test_satisfies_reward_term(self):
        for term in _PROTOCOL_TERMS:
            with self.subTest(term=term):
                self.assertIsInstance(term, RewardTerm)


# -----------------------------------------------------------------------------
# Stateful behaviour: PeakOutflow and SetpointSmoothness need engine state
# we don't have in-sandbox. Verify the *logic* against scripted stubs.
# -----------------------------------------------------------------------------


class _StubLinks:
    def __init__(self, flow_series=None, setting_series=None):
        self._flow_series = flow_series or []
        self._setting_series = setting_series or []
        self._flow_step = 0
        self._setting_step = 0

    def get_index(self, lid):
        return 0  # only one link in stubs

    def get_flow(self, idx):
        v = self._flow_series[self._flow_step]
        # advance only when all reads for this step are done; tests
        # call with one idx per step, so always advance.
        self._flow_step += 1
        return v

    def get_control_setting(self, idx):
        v = self._setting_series[self._setting_step]
        self._setting_step += 1
        return v


class _StubStatistics:
    """Scripted cumulative C{link_max_flow} readings, one per step."""

    def __init__(self, max_flow_series=None):
        self._series = max_flow_series or []
        self._step = 0

    def link_max_flow(self, idx):
        v = self._series[self._step]
        self._step += 1
        return v


class _StubAdapter:
    def __init__(self, links, statistics=None):
        self.links = links
        self.nodes = None
        self.statistics = statistics


class TestPeakOutflowLogic(unittest.TestCase):
    def test_running_max_increment(self):
        # Flows over 4 steps: 1.0 -> 3.0 -> 2.5 -> 4.0, so the engine's
        # cumulative max reads 1.0 -> 3.0 -> 3.0 -> 4.0.
        # Increments    :    +1.0 +2.0  +0.0  +1.0  (total = 4.0)
        links = _StubLinks(flow_series=[1.0, 3.0, 2.5, 4.0])
        adapter = _StubAdapter(links, _StubStatistics([1.0, 3.0, 3.0, 4.0]))
        term = PeakOutflow(link_ids=["C1"])
        term.bind(adapter)
        term.reset()
        increments = [term.step(adapter, dt_seconds=15.0) for _ in range(4)]
        np.testing.assert_allclose(increments, [1.0, 2.0, 0.0, 1.0], rtol=1e-6)
        self.assertAlmostEqual(sum(increments), 4.0, places=6)  # equals peak observed


class TestSetpointSmoothnessLogic(unittest.TestCase):
    def test_first_step_returns_zero(self):
        links = _StubLinks(setting_series=[0.5])
        adapter = _StubAdapter(links)
        term = SetpointSmoothness(link_ids=["C1"])
        term.bind(adapter)
        term.reset()
        self.assertEqual(term.step(adapter, dt_seconds=15.0), 0.0)

    def test_delta_squared(self):
        # Settings: 0.5 -> 0.8 -> 0.3
        # Δ²      :       0.09   0.25 (total = 0.34)
        links = _StubLinks(setting_series=[0.5, 0.8, 0.3])
        adapter = _StubAdapter(links)
        term = SetpointSmoothness(link_ids=["C1"])
        term.bind(adapter)
        term.reset()
        s = [term.step(adapter, dt_seconds=15.0) for _ in range(3)]
        np.testing.assert_allclose(s, [0.0, 0.09, 0.25], rtol=1e-6)


# -----------------------------------------------------------------------------
# Integration tier — real engine required
# -----------------------------------------------------------------------------


class TestIntegration(BaseEngineTest):
    def test_cso_volume_zero_on_dry_episode(self):
        adapter = self.make_adapter(open=True)
        term = CSOVolume(node_ids=["J1"])
        term.bind(adapter)
        term.reset()
        total = 0.0
        while adapter.is_running:
            adapter.step()
            total += term.step(adapter, dt_seconds=15.0)
        self.assertAlmostEqual(total, 0.0, delta=1e-9)
        adapter.close()

    def test_reliability_margin_within_max_depth(self):
        adapter = self.make_adapter(open=True)
        term = ReliabilityMargin(node_ids=["J1"])
        term.bind(adapter)
        term.reset()
        # Read max depth from the fixture; freeboard must be in [0, max_depth].
        max_d = float(adapter.nodes.get_max_depth(adapter.nodes.get_index("J1")))
        observed = []
        while adapter.is_running:
            adapter.step()
            observed.append(term.step(adapter, dt_seconds=15.0))
        self.assertTrue(all(0.0 <= v <= max_d + 1e-6 for v in observed))
        adapter.close()


if __name__ == "__main__":
    unittest.main()
