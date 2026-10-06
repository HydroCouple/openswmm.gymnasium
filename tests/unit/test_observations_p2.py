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

"""Unit tests for the P2 collectors added to L{ObservationBuilder}.

P1 collectors (node depths, link flows) are covered in
C{test_observations_builder.py}. Here we focus on the new ones plus the
clock collector's pure-math feature computation.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import math
import unittest

import numpy as np
from gymnasium import spaces

from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.observations.builder import _ClockCollector


_SIZE_CASES = [
    ("add_node_heads", ["J1", "J2"], 2),
    ("add_node_inflows", ["J1"], 1),
    ("add_node_overflows", ["J1", "J2", "J3"], 3),
    ("add_link_depths", ["C1"], 1),
    ("add_link_settings", ["C1", "C2"], 2),
    ("add_subcatch_runoff", ["S1"], 1),
    ("add_rainfall", ["RainGage"], 1),
]

_EMPTY_IDS_CASES = [
    "add_node_heads",
    "add_node_inflows",
    "add_node_overflows",
    "add_link_depths",
    "add_link_settings",
    "add_subcatch_runoff",
    "add_rainfall",
]


class TestNewCollectorsSize(unittest.TestCase):
    """Each new collector should contribute the right number of features."""

    def test_size(self):
        for method, ids, expected_size in _SIZE_CASES:
            with self.subTest(method=method, ids=ids, expected_size=expected_size):
                b = ObservationBuilder()
                getattr(b, method)(ids)
                self.assertEqual(b.space().shape, (expected_size,))
                self.assertEqual(b.space().dtype, np.float32)

    def test_empty_ids_raises(self):
        for method in _EMPTY_IDS_CASES:
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "at least one"):
                    getattr(ObservationBuilder(), method)([])


class TestClock(unittest.TestCase):
    def test_default_features(self):
        b = ObservationBuilder().add_clock()
        self.assertEqual(b.space().shape, (3,))

    def test_feature_subset(self):
        b = ObservationBuilder().add_clock(features=["elapsed_frac"])
        self.assertEqual(b.space().shape, (1,))

    def test_unknown_feature_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown clock features"):
            ObservationBuilder().add_clock(features=["not_a_real_feature"])

    def test_collect_against_stub_adapter(self):
        """Clock math against a stub — no engine needed.

        The collector reads C{adapter.current_time / start_time /
        end_time}, which are pure floats. We can satisfy that without
        the engine.
        """
        c = _ClockCollector()

        class _StubAdapter:
            # 6:00 AM on day 100 → fractional part 0.25 → angle = π/2
            # → sin=1, cos≈0
            current_time = 100.25
            start_time = 100.0
            end_time = 101.0

        c.bind(_StubAdapter())
        out = c.collect(_StubAdapter())
        self.assertEqual(out.shape, (3,))
        self.assertAlmostEqual(out[0], math.sin(math.pi / 2), delta=1e-6)  # hour_sin
        self.assertAlmostEqual(out[1], math.cos(math.pi / 2), delta=1e-6)  # hour_cos
        self.assertAlmostEqual(out[2], 0.25, delta=1e-6)  # elapsed_frac

    def test_collect_clamps_elapsed_frac_at_one(self):
        c = _ClockCollector(features=["elapsed_frac"])

        class _StubAdapter:
            current_time = 102.0  # past end
            start_time = 100.0
            end_time = 101.0

        c.bind(_StubAdapter())
        out = c.collect(_StubAdapter())
        self.assertAlmostEqual(out[0], 1.0, places=6)


class TestTenFeatureChain(unittest.TestCase):
    """The whole P2 collector set composes into a 10+ -feature observation."""

    def test_ten_features(self):
        b = (
            ObservationBuilder()
            .add_node_depths(["J1"])
            .add_node_heads(["J1"])
            .add_node_inflows(["J1"])
            .add_node_overflows(["J1"])
            .add_link_flows(["C1"])
            .add_link_depths(["C1"])
            .add_link_settings(["C1"])
            .add_subcatch_runoff(["S1"])
            .add_rainfall(["RainGage"])
            .add_clock(features=["elapsed_frac"])
        )
        s = b.space()
        self.assertIsInstance(s, spaces.Box)
        self.assertEqual(s.shape, (10,))


if __name__ == "__main__":
    unittest.main()
