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

"""P2.2 — additional bulk-backed observation collectors.

Covers the new node volume / lateral-inflow and link velocity / capacity
collectors: builder size/space contract (construction tier) and bulk-vs-
scalar parity against the real engine (integration tier, C{minimal.inp},
no mocks).

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.unit._base import BaseEngineTest

from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.observations import builder as _b


_SIZE_CASES = [
    ("add_node_volumes", ["J1"], 1),
    ("add_node_lateral_inflows", ["J1"], 1),
    ("add_link_velocities", ["C1"], 1),
    ("add_link_capacities", ["C1"], 1),
    ("add_link_volumes", ["C1"], 1),
]


class TestNewCollectorsSize(unittest.TestCase):
    def test_size(self):
        for method, ids, expected in _SIZE_CASES:
            with self.subTest(method=method):
                b = ObservationBuilder()
                getattr(b, method)(ids)
                self.assertEqual(b.space().shape, (expected,))
                self.assertEqual(b.space().dtype, np.float32)

    def test_empty_ids_raises(self):
        for method, _ids, _ in _SIZE_CASES:
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "at least one"):
                    getattr(ObservationBuilder(), method)([])


class TestNewCollectorsParity(BaseEngineTest):
    CASES = [
        ("_NodeVolumeCollector", ["J1"]),
        ("_NodeLateralInflowCollector", ["J1"]),
        ("_LinkVelocityCollector", ["C1"]),
        ("_LinkCapacityCollector", ["C1"]),
        ("_LinkVolumeCollector", ["C1"]),
    ]

    def test_bulk_equals_scalar(self):
        adapter = self.make_adapter(open=True)
        adapter.step()
        for cls_name, ids in self.CASES:
            with self.subTest(collector=cls_name):
                collector = getattr(_b, cls_name)(ids)
                collector.bind(adapter)
                bulk = collector.collect(adapter)
                scalar = np.fromiter(
                    (collector._read_scalar(adapter, i) for i in collector._idxs),
                    dtype=np.float32,
                    count=len(collector._idxs),
                )
                np.testing.assert_array_equal(bulk, scalar)
