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

"""Unit tests for L{openswmm_gymnasium.spaces.runtime.OrificeSetting}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest

import numpy as np
from gymnasium import spaces

from tests.unit._base import BaseEngineTest
from openswmm_gymnasium.spaces.runtime import OrificeSetting


class TestConstruction(unittest.TestCase):
    def test_name_default(self):
        f = OrificeSetting(["C1"])
        self.assertEqual(f.name, "orifice_setting")

    def test_name_custom(self):
        f = OrificeSetting(["C1"], name="gate")
        self.assertEqual(f.name, "gate")

    def test_empty_link_ids_raises(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            OrificeSetting([])


class TestSpace(unittest.TestCase):
    def test_box_shape_matches_links(self):
        f = OrificeSetting(["C1", "C2", "C3"])
        s = f.space
        self.assertIsInstance(s, spaces.Box)
        self.assertEqual(s.shape, (3,))
        self.assertEqual(s.dtype, np.float32)

    def test_bounds_are_unit_interval(self):
        f = OrificeSetting(["C1"])
        s = f.space
        self.assertTrue(np.array_equal(s.low, np.zeros(1, dtype=np.float32)))
        self.assertTrue(np.array_equal(s.high, np.ones(1, dtype=np.float32)))

    def test_sample_in_bounds(self):
        f = OrificeSetting(["C1", "C2"])
        for _ in range(10):
            sample = f.space.sample()
            self.assertEqual(sample.shape, (2,))
            self.assertTrue((sample >= 0.0).all())
            self.assertTrue((sample <= 1.0).all())


class TestApplyRequiresBind(BaseEngineTest):
    def test_apply_before_bind_raises(self):
        # No adapter touched - pure error-path test.
        f = OrificeSetting(["C1"])

        class _Stub:
            """Bare object; apply() should never reach it."""

        with self.assertRaisesRegex(RuntimeError, "bind"):
            f.apply(_Stub(), np.array([0.5], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
