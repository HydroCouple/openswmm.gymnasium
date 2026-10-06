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

"""Unit tests for L{openswmm_gymnasium.rewards.RewardRegistry}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest

from openswmm_gymnasium.rewards import (
    CSOVolume,
    FloodingVolume,
    PeakOutflow,
    ReliabilityMargin,
    RewardRegistry,
    SetpointSmoothness,
)


class TestBuiltInRegistrations(unittest.TestCase):
    def test_all_five_builtins_registered(self):
        names = RewardRegistry.names()
        self.assertTrue(
            {
                "flooding_volume",
                "cso_volume",
                "peak_outflow",
                "reliability_margin",
                "setpoint_smoothness",
            }.issubset(set(names))
        )

    def test_get_returns_correct_class(self):
        self.assertIs(RewardRegistry.get("flooding_volume"), FloodingVolume)
        self.assertIs(RewardRegistry.get("cso_volume"), CSOVolume)
        self.assertIs(RewardRegistry.get("peak_outflow"), PeakOutflow)
        self.assertIs(RewardRegistry.get("reliability_margin"), ReliabilityMargin)
        self.assertIs(RewardRegistry.get("setpoint_smoothness"), SetpointSmoothness)


class TestRegistration(unittest.TestCase):
    def test_unknown_name_raises(self):
        with self.assertRaisesRegex(KeyError, "not registered"):
            RewardRegistry.get("does_not_exist")

    def test_idempotent_re_register_same_class(self):
        # Built-ins already registered; re-registering the same class
        # is a no-op.
        RewardRegistry.register("flooding_volume", FloodingVolume)

    def test_conflicting_re_register_raises(self):
        class Imposter:
            pass

        with self.assertRaisesRegex(ValueError, "already registered"):
            RewardRegistry.register("flooding_volume", Imposter)


if __name__ == "__main__":
    unittest.main()
