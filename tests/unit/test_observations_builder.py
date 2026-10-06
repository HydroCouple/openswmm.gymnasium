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

"""Unit tests for L{openswmm_gymnasium.observations.ObservationBuilder}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest

import numpy as np
from gymnasium import spaces

from openswmm_gymnasium.observations import ObservationBuilder


class TestConstruction(unittest.TestCase):
    def test_empty_builder_space_raises(self):
        b = ObservationBuilder()
        with self.assertRaisesRegex(ValueError, "empty"):
            b.space()

    def test_chainable(self):
        b = ObservationBuilder()
        ret = b.add_node_depths(["J1"]).add_link_flows(["C1"])
        self.assertIs(ret, b)

    def test_node_depths_empty_raises(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            ObservationBuilder().add_node_depths([])

    def test_link_flows_empty_raises(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            ObservationBuilder().add_link_flows([])


class TestSpace(unittest.TestCase):
    def test_single_node_box_shape(self):
        b = ObservationBuilder().add_node_depths(["J1"])
        s = b.space()
        self.assertIsInstance(s, spaces.Box)
        self.assertEqual(s.shape, (1,))
        self.assertEqual(s.dtype, np.float32)

    def test_concatenation_size(self):
        b = ObservationBuilder().add_node_depths(["J1", "J2", "J3"]).add_link_flows(["C1", "C2"])
        s = b.space()
        self.assertEqual(s.shape, (5,))

    def test_bounds_infinite(self):
        b = ObservationBuilder().add_node_depths(["J1"])
        s = b.space()
        self.assertTrue(np.isneginf(s.low).all())
        self.assertTrue(np.isposinf(s.high).all())


if __name__ == "__main__":
    unittest.main()
