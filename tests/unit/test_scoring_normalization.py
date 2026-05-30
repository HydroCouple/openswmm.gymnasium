"""Unit tests for L{openswmm_gymnasium.scoring.normalization}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import numpy as np
import pytest

from openswmm_gymnasium.scoring import normalize


class TestNormalize:
    def test_at_ideal_is_zero(self):
        out = normalize([0.0, 0.0], [0.0, 0.0], [1.0, 2.0])
        np.testing.assert_array_equal(out, [0.0, 0.0])

    def test_at_reference_is_one(self):
        out = normalize([1.0, 2.0], [0.0, 0.0], [1.0, 2.0])
        np.testing.assert_array_equal(out, [1.0, 1.0])

    def test_midway(self):
        out = normalize([0.5, 1.0], [0.0, 0.0], [1.0, 2.0])
        np.testing.assert_allclose(out, [0.5, 0.5])

    def test_2d_array_input(self):
        out = normalize(
            np.array([[0.0, 0.0], [1.0, 2.0], [0.5, 1.0]]),
            [0.0, 0.0],
            [1.0, 2.0],
        )
        np.testing.assert_allclose(out, [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])

    def test_zero_span_raises(self):
        with pytest.raises(ValueError, match="strictly greater"):
            normalize([0.0], [0.0], [0.0])

    def test_inverted_span_raises(self):
        with pytest.raises(ValueError, match="strictly greater"):
            normalize([0.0], [1.0], [0.0])

    def test_not_clipped(self):
        """Values beyond [ideal, ref] are returned as-is (no clipping)."""
        out = normalize([-1.0, 3.0], [0.0, 0.0], [1.0, 2.0])
        np.testing.assert_allclose(out, [-1.0, 1.5])
