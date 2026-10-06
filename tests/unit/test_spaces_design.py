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

"""Unit tests for L{openswmm_gymnasium.spaces.design} factories.

Engine-touching behaviour (the actual setter calls) is exercised by
the CIP env integration tests; here we focus on space construction,
validation, and bind-before-apply enforcement.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest

import numpy as np
from gymnasium import spaces

from openswmm_gymnasium.spaces.design import (
    DesignActionFactory,
    LinkDiameter,
    LinkLength,
    LinkRoughness,
    NodeMaxDepth,
)

# -----------------------------------------------------------------------------
# Parametrized construction tests
# -----------------------------------------------------------------------------


_BOX_FACTORIES = [
    (LinkRoughness, "link_roughness", "link_ids", ["C1"]),
    (LinkLength, "link_length", "link_ids", ["C1"]),
    (LinkDiameter, "link_diameter", "link_ids", ["C1"]),
    (NodeMaxDepth, "node_max_depth", "node_ids", ["J1"]),
]


class TestConstruction(unittest.TestCase):
    def test_default_name(self):
        for cls, name, kwarg, ids in _BOX_FACTORIES:
            with self.subTest(cls=cls.__name__):
                f = cls(ids, low=0.01, high=0.05)
                self.assertEqual(f.name, name)

    def test_empty_ids_raises(self):
        for cls, name, kwarg, ids in _BOX_FACTORIES:
            with self.subTest(cls=cls.__name__):
                with self.assertRaisesRegex(ValueError, "at least one"):
                    cls([], low=0.0, high=1.0)

    def test_invalid_bounds_raise(self):
        for cls, name, kwarg, ids in _BOX_FACTORIES:
            with self.subTest(cls=cls.__name__):
                with self.assertRaisesRegex(ValueError, "strictly greater"):
                    cls(ids, low=1.0, high=1.0)
                with self.assertRaisesRegex(ValueError, "strictly greater"):
                    cls(ids, low=2.0, high=1.0)


class TestSpace(unittest.TestCase):
    def test_box_shape(self):
        for cls, name, kwarg, ids in _BOX_FACTORIES:
            with self.subTest(cls=cls.__name__):
                f = cls(ids * 3, low=0.0, high=1.0)
                s = f.space
                self.assertIsInstance(s, spaces.Box)
                self.assertEqual(s.shape, (3,))
                self.assertEqual(s.dtype, np.float32)

    def test_bounds_propagate(self):
        for cls, name, kwarg, ids in _BOX_FACTORIES:
            with self.subTest(cls=cls.__name__):
                f = cls(ids, low=0.25, high=0.75)
                s = f.space
                self.assertTrue(np.allclose(s.low, [0.25]))
                self.assertTrue(np.allclose(s.high, [0.75]))

    def test_sample_in_bounds(self):
        for cls, name, kwarg, ids in _BOX_FACTORIES:
            with self.subTest(cls=cls.__name__):
                f = cls(ids, low=0.1, high=0.9)
                for _ in range(20):
                    v = f.space.sample()
                    self.assertEqual(v.shape, (1,))
                    self.assertTrue((v >= 0.1).all() and (v <= 0.9).all())


class TestApplyRequiresBind(unittest.TestCase):
    def test_apply_before_bind_raises(self):
        for cls, name, kwarg, ids in _BOX_FACTORIES:
            with self.subTest(cls=cls.__name__):
                f = cls(ids, low=0.0, high=1.0)

                class _Stub:
                    pass

                with self.assertRaisesRegex(RuntimeError, "bind"):
                    f.apply(_Stub(), np.array([0.5], dtype=np.float32))


class TestProtocolConformance(unittest.TestCase):
    def test_satisfies_factory_protocol(self):
        for cls, name, kwarg, ids in _BOX_FACTORIES:
            with self.subTest(cls=cls.__name__):
                f = cls(ids, low=0.0, high=1.0)
                self.assertIsInstance(f, DesignActionFactory)


if __name__ == "__main__":
    unittest.main()
