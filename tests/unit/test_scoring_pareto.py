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

"""Unit tests for L{openswmm_gymnasium.scoring.pareto}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest

import numpy as np

from openswmm_gymnasium.scoring import is_dominated, pareto_front


class TestIsDominated(unittest.TestCase):
    def test_dominated_by_strictly_better(self):
        self.assertTrue(is_dominated([2, 2], np.array([[1, 1]])))

    def test_not_dominated_by_equal(self):
        self.assertFalse(is_dominated([1, 1], np.array([[1, 1]])))

    def test_not_dominated_when_incomparable(self):
        self.assertFalse(is_dominated([1, 3], np.array([[3, 1]])))

    def test_dominated_by_one_of_many(self):
        others = np.array([[5, 5], [0.5, 0.5], [3, 3]])
        self.assertTrue(is_dominated([1, 1], others))


class TestParetoFront(unittest.TestCase):
    def test_empty_input(self):
        out = pareto_front(np.empty((0, 2)))
        self.assertEqual(out.shape, (0, 2))

    def test_single_point(self):
        out = pareto_front(np.array([[1.0, 2.0]]))
        np.testing.assert_array_equal(out, [[1.0, 2.0]])

    def test_2d_simple(self):
        """Three non-dominated points + one dominated."""
        pts = np.array(
            [
                [0.0, 4.0],  # ND
                [1.0, 3.0],  # ND
                [2.0, 1.0],  # ND
                [3.0, 5.0],  # dominated by (1, 3) and (2, 1)
            ]
        )
        out = pareto_front(pts)
        # Order preserved relative to input.
        expected = pts[:3]
        np.testing.assert_array_equal(out, expected)

    def test_2d_with_duplicates(self):
        """Duplicate non-dominated points are both retained (mutual ties)."""
        pts = np.array([[1.0, 1.0], [1.0, 1.0]])
        out = pareto_front(pts)
        self.assertEqual(out.shape, (2, 2))

    def test_3d_basic(self):
        pts = np.array(
            [
                [0, 0, 1],  # ND
                [0, 1, 0],  # ND
                [1, 0, 0],  # ND
                [1, 1, 1],  # dominated by all three above
            ],
            dtype=float,
        )
        out = pareto_front(pts)
        self.assertEqual(out.shape, (3, 3))
        # Original ordering preserved.
        np.testing.assert_array_equal(out, pts[:3])

    def test_invalid_shape_raises(self):
        with self.assertRaisesRegex(ValueError, "2-D"):
            pareto_front(np.array([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
