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

"""Unit tests for the operational reward terms (plan Phase 0.4).

Construction tier (names, directions, guards) needs no engine. The integration
tier drives the real engine against C{tests/data/minimal.inp} (junction J1,
outfall O1, conduit C1) per the suite's no-mocks policy.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest

from tests.unit._base import BaseEngineTest

from openswmm_gymnasium.rewards import (
    PumpEnergy,
    RewardTerm,
    StorageUnderUtilization,
    UncontrolledDischarge,
)

_DT = 15.0  # matches the fixture ROUTING_STEP


class TestConstruction(unittest.TestCase):
    def test_names_and_directions(self):
        self.assertEqual(UncontrolledDischarge(["C1"]).name, "uncontrolled_discharge")
        self.assertEqual(StorageUnderUtilization(["J1"]).name, "storage_underutilization")
        self.assertEqual(PumpEnergy(["C1"]).name, "pump_energy")
        for t in (UncontrolledDischarge(["C1"]), StorageUnderUtilization(["J1"]), PumpEnergy(["C1"])):
            self.assertEqual(t.direction, "minimize")

    def test_empty_ids_rejected(self):
        for ctor in (UncontrolledDischarge, StorageUnderUtilization, PumpEnergy):
            with self.assertRaises(ValueError):
                ctor([])

    def test_protocol_conformance(self):
        for t in (UncontrolledDischarge(["C1"]), StorageUnderUtilization(["J1"]), PumpEnergy(["C1"])):
            self.assertIsInstance(t, RewardTerm)

    def test_step_before_bind_raises(self):
        class _Stub:
            pass

        with self.assertRaisesRegex(AssertionError, "bind"):
            UncontrolledDischarge(["C1"]).step(_Stub(), dt_seconds=_DT)


class TestIntegration(BaseEngineTest):
    def _drive(self, term) -> float:
        adapter = self.make_adapter(open=True)
        term.bind(adapter)
        term.reset()
        total = 0.0
        while adapter.is_running:
            adapter.step()
            total += term.step(adapter, dt_seconds=_DT)
        adapter.close()
        return total

    def test_uncontrolled_discharge_accumulates_at_outfall(self):
        # The inflow ramp into J1 drains through conduit C1 to the untreated
        # outfall O1, so C1's positive flow integrates to a positive discharge
        # volume. (Node inflow at a FREE outfall reads 0 — the link carries it.)
        total = self._drive(UncontrolledDischarge(["C1"]))
        self.assertGreater(total, 0.0)

    def test_storage_underutilization_bounded(self):
        # J1 stays well below its 5 ft rim, so unused headroom is positive but
        # cannot exceed (1 per step) * dt summed over the run.
        adapter = self.make_adapter(open=True)
        term = StorageUnderUtilization(["J1"])
        term.bind(adapter)
        term.reset()
        total = 0.0
        elapsed = 0.0
        while adapter.is_running:
            adapter.step()
            total += term.step(adapter, dt_seconds=_DT)
            elapsed += _DT
        adapter.close()
        self.assertGreater(total, 0.0)
        self.assertLessEqual(total, elapsed + 1e-9)

    def test_pump_energy_scales_with_rated_power(self):
        # Energy is linear in rated_power: 5x the power -> 5x the term, exactly,
        # since the engine trajectory is deterministic.
        total1 = self._drive(PumpEnergy(["C1"]))
        total5 = self._drive(PumpEnergy(["C1"], rated_power={"C1": 5.0}))
        self.assertGreaterEqual(total1, 0.0)
        if total1 > 0.0:
            self.assertAlmostEqual(total5 / total1, 5.0, places=9)
        else:  # pragma: no cover - conduit setting expected > 0
            self.assertEqual(total5, 0.0)


if __name__ == "__main__":
    unittest.main()
