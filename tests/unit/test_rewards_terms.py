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

"""Unit tests for L{openswmm_gymnasium.rewards.FloodingVolume}.

Per plan §8.0 the integration tier (RUNNING engine) drives the term
against the minimal fixture. The construction tier (this file's
TestConstruction class) needs only a stub.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest

from tests.unit._base import BaseEngineTest

from openswmm_gymnasium.rewards import FloodingVolume, RewardTerm


class TestConstruction(unittest.TestCase):
    def test_default_name_and_direction(self):
        term = FloodingVolume()
        self.assertEqual(term.name, "flooding_volume")
        self.assertEqual(term.direction, "minimize")

    def test_custom_name(self):
        term = FloodingVolume(name="overflow")
        self.assertEqual(term.name, "overflow")

    def test_node_ids_passed_through(self):
        term = FloodingVolume(node_ids=["J1", "J2"])
        # Internal but worth a sanity check — we expose no public getter.
        self.assertEqual(term._node_ids, ["J1", "J2"])

    def test_step_before_bind_raises(self):
        term = FloodingVolume()

        class _Stub:
            pass

        with self.assertRaisesRegex(AssertionError, "bind"):
            term.step(_Stub(), dt_seconds=15.0)


class TestProtocolConformance(unittest.TestCase):
    def test_flooding_volume_is_reward_term(self):
        """L{FloodingVolume} must satisfy the L{RewardTerm} protocol."""
        self.assertIsInstance(FloodingVolume(), RewardTerm)


# -----------------------------------------------------------------------------
# Integration tier — needs the real engine running
# -----------------------------------------------------------------------------


class TestIntegration(BaseEngineTest):
    def test_zero_flooding_on_minimal_dry_episode(self):
        """On the minimal fixture the inflow drains via C1; no flooding."""
        adapter = self.make_adapter(open=True)
        term = FloodingVolume()
        term.bind(adapter)
        term.reset()
        # Drive a few steps; FloodingVolume should be zero.
        total = 0.0
        while adapter.is_running:
            adapter.step()
            total += term.step(adapter, dt_seconds=15.0)
        self.assertAlmostEqual(total, 0.0, delta=1e-9)
        adapter.close()


if __name__ == "__main__":
    unittest.main()
