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

"""Unit tests for L{openswmm_gymnasium.scoring.hypervolume}.

Closed-form references used:

  - B{2-D triangle}: front C{[(0, 1), (1, 0)]} with C{ref=(1, 1)} has
    HV = 0.5 (the lower-left triangle of the unit square).
  - B{2-D L-shape}: front C{[(0, 1), (0.5, 0.5), (1, 0)]} with
    C{ref=(1, 1)} has HV = 0.5 + 0.25 = 0.75 (worked example).
  - B{ZDT1-like front}: f2 = 1 - sqrt(f1), f1 ∈ [0, 1], ref = (1, 1).
    Exact HV = ∫₀¹ sqrt(f1) df1 = 2/3 ≈ 0.667. We approximate the
    continuous front with 200 discrete samples and assert close-to-2/3.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest

import numpy as np

from openswmm_gymnasium.scoring import hypervolume, normalized_hypervolume


class TestEmptyAndDegenerate(unittest.TestCase):
    def test_empty_input_returns_zero(self):
        self.assertEqual(hypervolume(np.empty((0, 2)), [1.0, 1.0]), 0.0)

    def test_all_points_outside_ref_returns_zero(self):
        pts = np.array([[2.0, 2.0]])
        self.assertEqual(hypervolume(pts, [1.0, 1.0]), 0.0)

    def test_1d_simple(self):
        pts = np.array([[0.3], [0.7]])
        # HV = ref - min = 1 - 0.3 = 0.7
        self.assertAlmostEqual(hypervolume(pts, [1.0]), 0.7, places=6)


class TestExact2D(unittest.TestCase):
    def test_unit_triangle(self):
        front = np.array([[0.0, 1.0], [1.0, 0.0]])
        # The front degenerates to two boundary points; only points
        # *strictly* dominating ref count, and neither does (both have
        # one coord == 1.0). So HV = 0.0 by design.
        self.assertEqual(hypervolume(front, [1.0, 1.0]), 0.0)

    def test_interior_two_points(self):
        front = np.array([[0.0, 0.5], [0.5, 0.0]])
        # Two rectangles: [0, 0.5] × [0.5, 1] = 0.25, [0.5, 1] × [0, 1]... wait.
        # Standard sweep: sort by f1, contributions are:
        #   (0.0, 0.5): width = 0.5 - 0.0 = 0.5, height = 1 - 0.5 = 0.5 → 0.25
        #   (0.5, 0.0): width = 1.0 - 0.5 = 0.5, height = 1 - 0.0 = 1.0 → 0.5
        # Total = 0.75
        self.assertAlmostEqual(hypervolume(front, [1.0, 1.0]), 0.75, places=6)

    def test_l_shape(self):
        front = np.array([[0.0, 0.5], [0.25, 0.25], [0.5, 0.0]])
        # Sorted by f1:
        #   (0.0, 0.5): width = 0.25, height = 0.5 → 0.125
        #   (0.25, 0.25): width = 0.25, height = 0.75 → 0.1875
        #   (0.5, 0.0): width = 0.5, height = 1.0 → 0.5
        # Total = 0.8125
        self.assertAlmostEqual(hypervolume(front, [1.0, 1.0]), 0.8125, places=6)

    def test_dominated_point_ignored(self):
        front_no_dom = np.array([[0.0, 0.5], [0.5, 0.0]])
        front_with_dom = np.vstack([front_no_dom, [[0.9, 0.9]]])
        self.assertAlmostEqual(
            hypervolume(front_no_dom, [1.0, 1.0]),
            hypervolume(front_with_dom, [1.0, 1.0]),
            places=6,
        )

    def test_zdt1_approximation_close_to_two_thirds(self):
        """Discrete ZDT1 front; HV should approach 2/3 as N grows."""
        n = 200
        f1 = np.linspace(0.0, 1.0, n)
        f2 = 1.0 - np.sqrt(f1)
        # Drop boundary points that have a coord == 1 (don't dominate ref).
        mask = (f1 < 1.0) & (f2 < 1.0)
        front = np.column_stack([f1[mask], f2[mask]])
        hv = hypervolume(front, [1.0, 1.0])
        # As N → ∞, HV → 2/3. With N=200 (after boundary drop) we get
        # within ~0.01.
        self.assertAlmostEqual(hv, 2.0 / 3.0, delta=0.02)


class TestMonteCarlo(unittest.TestCase):
    def test_2d_mc_matches_exact(self):
        """MC HV converges to the exact 2-D answer within a tolerance."""
        front = np.array([[0.1, 0.6], [0.4, 0.3], [0.7, 0.1]])
        exact = hypervolume(front, [1.0, 1.0])
        rng = np.random.default_rng(seed=42)
        mc = hypervolume(front, [1.0, 1.0], method="mc", mc_samples=200_000, rng=rng)
        self.assertAlmostEqual(mc, exact, delta=abs(exact) * 0.05)

    def test_3d_mc_against_box(self):
        """Single point in 3D — HV should be exact box volume."""
        front = np.array([[0.5, 0.5, 0.5]])
        ref = np.array([1.0, 1.0, 1.0])
        # Single point yields HV = box [0.5, 1] × [0.5, 1] × [0.5, 1] = 0.125
        rng = np.random.default_rng(seed=42)
        hv = hypervolume(front, ref, method="mc", mc_samples=200_000, rng=rng)
        self.assertAlmostEqual(hv, 0.125, delta=0.125 * 0.05)

    def test_exact_rejected_for_3d(self):
        with self.assertRaisesRegex(ValueError, "d <= 2"):
            hypervolume(np.array([[0.5, 0.5, 0.5]]), [1, 1, 1], method="exact")


class TestNormalizedHypervolume(unittest.TestCase):
    def test_in_zero_one(self):
        """Result is in [0, 1] for arbitrary objective scales."""
        front = np.array([[2.0, 3.0], [4.0, 1.0]])
        ideal = np.array([0.0, 0.0])
        ref = np.array([8.0, 4.0])
        nhv = normalized_hypervolume(front, ideal, ref)
        self.assertGreaterEqual(nhv, 0.0)
        self.assertLessEqual(nhv, 1.0)

    def test_at_ideal_full_volume(self):
        """A single point at the ideal yields normalised HV = 1."""
        front = np.array([[0.0, 0.0]])
        ideal = np.array([0.0, 0.0])
        ref = np.array([1.0, 1.0])
        nhv = normalized_hypervolume(front, ideal, ref)
        self.assertAlmostEqual(nhv, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
