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

"""Unit tests for IGD, IGD+, epsilon, spread, R2 scoring metrics.

Closed-form references used:

  - IGD against itself = 0.
  - IGD+ against itself = 0.
  - IGD+ is zero when approximation dominates every reference point.
  - epsilon between identical fronts = 0.
  - epsilon is the max coordinate gap in a shifted-front case.
  - spread of one or zero points = 0.
  - R2 against the ideal itself = 0 for any weight vector.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest

import numpy as np

from openswmm_gymnasium.scoring import (
    epsilon_indicator,
    igd,
    igd_plus,
    r2_indicator,
    spread,
)

# ---------------------------------------------------------------------------
# IGD / IGD+
# ---------------------------------------------------------------------------


class TestIGD(unittest.TestCase):
    def test_self_equals_zero(self):
        front = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        self.assertAlmostEqual(igd(front, front), 0.0, places=6)

    def test_empty_returns_inf(self):
        self.assertEqual(igd(np.empty((0, 2)), np.array([[0.0, 0.0]])), float("inf"))
        self.assertEqual(igd(np.array([[0.0, 0.0]]), np.empty((0, 2))), float("inf"))

    def test_single_point_shift(self):
        approx = np.array([[1.0, 1.0]])
        ref = np.array([[0.0, 0.0]])
        # Distance = sqrt(2)
        self.assertAlmostEqual(igd(approx, ref), np.sqrt(2.0), places=6)


class TestIGDPlus(unittest.TestCase):
    def test_self_equals_zero(self):
        front = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        self.assertAlmostEqual(igd_plus(front, front), 0.0, places=6)

    def test_dominating_approx_zero(self):
        """If every approx point weakly dominates every reference point,
        IGD+ is zero by construction."""
        approx = np.array([[0.0, 0.0]])
        ref = np.array([[1.0, 1.0], [2.0, 2.0]])
        self.assertAlmostEqual(igd_plus(approx, ref), 0.0, places=6)

    def test_only_dominated_dims_count(self):
        """IGD+ ignores the dimension where approx is better than ref."""
        approx = np.array([[2.0, 0.0]])  # worse in d0 by 2, better in d1 by 1
        ref = np.array([[0.0, 1.0]])
        # Clamped diff = (max(0, 2), max(0, -1)) = (2, 0); dist = 2
        self.assertAlmostEqual(igd_plus(approx, ref), 2.0, places=6)


# ---------------------------------------------------------------------------
# Epsilon
# ---------------------------------------------------------------------------


class TestEpsilon(unittest.TestCase):
    def test_self_equals_zero(self):
        front = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.assertAlmostEqual(epsilon_indicator(front, front), 0.0, places=6)

    def test_uniform_shift(self):
        """Shifting A worse by δ in every dim yields ε = δ."""
        b = np.array([[0.0, 1.0], [1.0, 0.0]])
        a = b + 0.25
        self.assertAlmostEqual(epsilon_indicator(a, b), 0.25, places=6)

    def test_empty_returns_inf(self):
        self.assertEqual(
            epsilon_indicator(np.empty((0, 2)), np.array([[0.0]])), float("inf")
        )


# ---------------------------------------------------------------------------
# Spread
# ---------------------------------------------------------------------------


class TestSpread(unittest.TestCase):
    def test_single_point(self):
        self.assertEqual(spread(np.array([[1.0, 1.0]])), 0.0)

    def test_uniform_spacing_zero_stddev(self):
        # Three collinear evenly-spaced points → uniform spacing →
        # stddev of nearest-neighbour distances is zero.
        front = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        self.assertAlmostEqual(spread(front), 0.0, places=6)

    def test_uneven_spacing_positive(self):
        front = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
        self.assertGreater(spread(front), 0.0)


# ---------------------------------------------------------------------------
# R2
# ---------------------------------------------------------------------------


class TestR2(unittest.TestCase):
    def test_at_reference_zero(self):
        """If the front contains the reference point, R2 = 0 regardless
        of weights (since the min weighted-Tchebycheff utility is 0)."""
        front = np.array([[0.0, 0.0]])
        ref = np.array([0.0, 0.0])
        weights = np.array([[0.25, 0.75], [0.5, 0.5], [0.75, 0.25]])
        self.assertAlmostEqual(r2_indicator(front, weights, ref), 0.0, places=6)

    def test_single_weight_single_point(self):
        # Front {(1, 0)}, weights {(1, 1)}, ref (0, 0):
        # utility = max(1*|1|, 1*|0|) = 1; mean over single weight = 1.
        self.assertAlmostEqual(
            r2_indicator(
                np.array([[1.0, 0.0]]),
                np.array([[1.0, 1.0]]),
                np.array([0.0, 0.0]),
            ),
            1.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
