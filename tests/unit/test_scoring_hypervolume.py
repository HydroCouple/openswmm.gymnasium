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
@license: MIT
"""

from __future__ import annotations

import numpy as np
import pytest

from openswmm_gymnasium.scoring import hypervolume, normalized_hypervolume


class TestEmptyAndDegenerate:
    def test_empty_input_returns_zero(self):
        assert hypervolume(np.empty((0, 2)), [1.0, 1.0]) == 0.0

    def test_all_points_outside_ref_returns_zero(self):
        pts = np.array([[2.0, 2.0]])
        assert hypervolume(pts, [1.0, 1.0]) == 0.0

    def test_1d_simple(self):
        pts = np.array([[0.3], [0.7]])
        # HV = ref - min = 1 - 0.3 = 0.7
        assert hypervolume(pts, [1.0]) == pytest.approx(0.7)


class TestExact2D:
    def test_unit_triangle(self):
        front = np.array([[0.0, 1.0], [1.0, 0.0]])
        # The front degenerates to two boundary points; only points
        # *strictly* dominating ref count, and neither does (both have
        # one coord == 1.0). So HV = 0.0 by design.
        assert hypervolume(front, [1.0, 1.0]) == 0.0

    def test_interior_two_points(self):
        front = np.array([[0.0, 0.5], [0.5, 0.0]])
        # Two rectangles: [0, 0.5] × [0.5, 1] = 0.25, [0.5, 1] × [0, 1]... wait.
        # Standard sweep: sort by f1, contributions are:
        #   (0.0, 0.5): width = 0.5 - 0.0 = 0.5, height = 1 - 0.5 = 0.5 → 0.25
        #   (0.5, 0.0): width = 1.0 - 0.5 = 0.5, height = 1 - 0.0 = 1.0 → 0.5
        # Total = 0.75
        assert hypervolume(front, [1.0, 1.0]) == pytest.approx(0.75)

    def test_l_shape(self):
        front = np.array([[0.0, 0.5], [0.25, 0.25], [0.5, 0.0]])
        # Sorted by f1:
        #   (0.0, 0.5): width = 0.25, height = 0.5 → 0.125
        #   (0.25, 0.25): width = 0.25, height = 0.75 → 0.1875
        #   (0.5, 0.0): width = 0.5, height = 1.0 → 0.5
        # Total = 0.8125
        assert hypervolume(front, [1.0, 1.0]) == pytest.approx(0.8125)

    def test_dominated_point_ignored(self):
        front_no_dom = np.array([[0.0, 0.5], [0.5, 0.0]])
        front_with_dom = np.vstack([front_no_dom, [[0.9, 0.9]]])
        assert hypervolume(front_no_dom, [1.0, 1.0]) == pytest.approx(
            hypervolume(front_with_dom, [1.0, 1.0])
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
        assert hv == pytest.approx(2.0 / 3.0, abs=0.02)


class TestMonteCarlo:
    def test_2d_mc_matches_exact(self):
        """MC HV converges to the exact 2-D answer within a tolerance."""
        front = np.array([[0.1, 0.6], [0.4, 0.3], [0.7, 0.1]])
        exact = hypervolume(front, [1.0, 1.0])
        rng = np.random.default_rng(seed=42)
        mc = hypervolume(front, [1.0, 1.0], method="mc", mc_samples=200_000, rng=rng)
        assert mc == pytest.approx(exact, rel=0.05)

    def test_3d_mc_against_box(self):
        """Single point in 3D — HV should be exact box volume."""
        front = np.array([[0.5, 0.5, 0.5]])
        ref = np.array([1.0, 1.0, 1.0])
        # Single point yields HV = box [0.5, 1] × [0.5, 1] × [0.5, 1] = 0.125
        rng = np.random.default_rng(seed=42)
        hv = hypervolume(front, ref, method="mc", mc_samples=200_000, rng=rng)
        assert hv == pytest.approx(0.125, rel=0.05)

    def test_exact_rejected_for_3d(self):
        with pytest.raises(ValueError, match="d <= 2"):
            hypervolume(np.array([[0.5, 0.5, 0.5]]), [1, 1, 1], method="exact")


class TestNormalizedHypervolume:
    def test_in_zero_one(self):
        """Result is in [0, 1] for arbitrary objective scales."""
        front = np.array([[2.0, 3.0], [4.0, 1.0]])
        ideal = np.array([0.0, 0.0])
        ref = np.array([8.0, 4.0])
        nhv = normalized_hypervolume(front, ideal, ref)
        assert 0.0 <= nhv <= 1.0

    def test_at_ideal_full_volume(self):
        """A single point at the ideal yields normalised HV = 1."""
        front = np.array([[0.0, 0.0]])
        ideal = np.array([0.0, 0.0])
        ref = np.array([1.0, 1.0])
        nhv = normalized_hypervolume(front, ideal, ref)
        assert nhv == pytest.approx(1.0)
