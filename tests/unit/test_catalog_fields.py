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

"""Catalog field access: any engine field as an observation or an action.

The adapter reads and writes element fields by their path in
L{openswmm.engine.catalog}; L{ObservationBuilder.add_field} and
L{FieldSetpoint} build on that. Integration tier: real engine over
C{minimal.inp} (1 junction, 1 outfall, 1 conduit).
"""

from __future__ import annotations

import numpy as np

from openswmm_gymnasium._engine import field_entry
from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.spaces import FieldSetpoint
from tests.unit._base import BaseEngineTest, BaseTestCase


class TestFieldEntry(BaseTestCase):
    def test_known_numeric_field(self):
        self.assertEqual(field_entry("link.stats.max_flow")["units"], "flow")

    def test_unknown_path_is_named(self):
        with self.assertRaisesRegex(ValueError, "node.not_a_field"):
            field_entry("node.not_a_field")

    def test_non_numeric_and_non_element_paths_are_refused(self):
        with self.assertRaisesRegex(ValueError, "not a numeric"):
            field_entry("node.id")
        with self.assertRaisesRegex(ValueError, "not a numeric"):
            field_entry("link.loss_coeff")  # a tuple of floats is not one value
        with self.assertRaisesRegex(ValueError, "element kind"):
            field_entry("mass_balance.max_courant")


class TestAdapterFieldAccess(BaseEngineTest):
    def test_read_matches_compat_getter_and_bulk(self):
        adapter = self.make_adapter()
        for _ in range(20):
            adapter.step()
        j1 = adapter.index("node", "J1")
        self.assertEqual(adapter.read("node.depth", j1), adapter.nodes.get_depth(j1))
        self.assertEqual(adapter.read_all("node.depth")[j1], adapter.read("node.depth", j1))
        self.assertIsNone(adapter.read_all("link.stats.max_flow"))

    def test_write_round_trip_and_read_only_refusal(self):
        adapter = self.make_adapter()
        c1 = adapter.index("link", "C1")
        adapter.write("link.target_setting", c1, 0.5)
        self.assertAlmostEqual(adapter.read("link.target_setting", c1), 0.5)
        with self.assertRaisesRegex(ValueError, "read-only"):
            adapter.write("link.stats.max_flow", c1, 1.0)


class TestAddField(BaseEngineTest):
    def test_any_field_observes_with_named_equivalent(self):
        adapter = self.make_adapter()
        builder = (
            ObservationBuilder()
            .add_field("node.depth", ["J1"])
            .add_node_depths(["J1"])
            .add_field("link.stats.max_flow", ["C1"])
        )
        builder.bind(adapter)
        for _ in range(20):
            adapter.step()
        obs = builder.collect(adapter)
        self.assertEqual(builder.space().shape, (3,))
        self.assertEqual(obs[0], obs[1])
        self.assertGreaterEqual(obs[2], 0.0)

    def test_unknown_field_fails_at_construction(self):
        with self.assertRaisesRegex(ValueError, "node.depthh"):
            ObservationBuilder().add_field("node.depthh", ["J1"])


class TestFieldSetpoint(BaseEngineTest):
    def test_applies_clipped_values(self):
        adapter = self.make_adapter()
        action = FieldSetpoint("link.target_setting", ["C1"], 0.0, 0.6)
        self.assertEqual(action.name, "link.target_setting")
        self.assertEqual(action.space.shape, (1,))
        action.bind(adapter)
        action.apply(adapter, np.array([5.0], dtype=np.float32))
        c1 = adapter.index("link", "C1")
        self.assertAlmostEqual(adapter.read("link.target_setting", c1), 0.6, places=6)

    def test_rejects_read_only_fields_and_bad_bounds(self):
        with self.assertRaisesRegex(ValueError, "read-only"):
            FieldSetpoint("link.stats.max_flow", ["C1"], 0.0, 1.0)
        with self.assertRaisesRegex(ValueError, "low < high"):
            FieldSetpoint("link.target_setting", ["C1"], 1.0, 1.0)


class TestObservationUnits(BaseEngineTest):
    def test_units_follow_the_catalog_and_the_model(self):
        adapter = self.make_adapter()
        builder = (
            ObservationBuilder()
            .add_node_depths(["J1"])
            .add_field("link.flow", ["C1"])
            .add_field("link.stats.max_filling", ["C1"])
            .add_clock(["elapsed_frac"])
        )
        units = builder.units(adapter.unit_system, adapter.flow_units)
        self.assertEqual(units, ["ft", "CFS", "fraction", "dimensionless"])
        self.assertEqual(len(units), builder.space().shape[0])
