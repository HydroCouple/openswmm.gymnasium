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

"""Unit tests for L{openswmm_gymnasium.scoring.normalization}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest

import numpy as np

from openswmm_gymnasium.scoring import normalize


class TestNormalize(unittest.TestCase):
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
        with self.assertRaisesRegex(ValueError, "strictly greater"):
            normalize([0.0], [0.0], [0.0])

    def test_inverted_span_raises(self):
        with self.assertRaisesRegex(ValueError, "strictly greater"):
            normalize([0.0], [1.0], [0.0])

    def test_not_clipped(self):
        """Values beyond [ideal, ref] are returned as-is (no clipping)."""
        out = normalize([-1.0, 3.0], [0.0, 0.0], [1.0, 2.0])
        np.testing.assert_allclose(out, [-1.0, 1.5])


if __name__ == "__main__":
    unittest.main()
