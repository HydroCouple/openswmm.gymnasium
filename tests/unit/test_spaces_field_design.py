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

"""L{openswmm_gymnasium.spaces.design.FieldDesign}: any writable scalar field as a design.

Engine-touching tests run the real engine on copies of C{tests/data} models;
every file they write lands in C{tests/_output/spaces_field_design/}.
"""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path

import numpy as np

from openswmm_gymnasium._engine import SolverAdapter
from openswmm_gymnasium.spaces.design import FieldDesign

_DATA = Path(__file__).resolve().parents[1] / "data"
_OUTPUT = Path(__file__).resolve().parents[1] / "_output" / "spaces_field_design"


def _folder(name: str) -> Path:
    path = _OUTPUT / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


class TestConstruction(unittest.TestCase):
    def test_the_path_is_the_default_name_and_ids_span_the_box(self):
        factory = FieldDesign("subcatchment.width", ["S1", "S2"], low=10.0, high=500.0)
        self.assertEqual(factory.name, "subcatchment.width")
        self.assertEqual(factory.space.shape, (2,))
        self.assertEqual(float(factory.space.low[0]), 10.0)
        self.assertEqual(FieldDesign("link.roughness", ["C1"], 0.01, 0.02, name="n").name, "n")

    def test_a_field_that_cannot_be_a_design_is_refused(self):
        with self.assertRaisesRegex(ValueError, "read-only"):
            FieldDesign("link.stats.max_flow", ["C1"], 0.0, 1.0)
        with self.assertRaisesRegex(ValueError, "not a numeric property"):
            FieldDesign("node.tag", ["J1"], 0.0, 1.0)
        with self.assertRaisesRegex(ValueError, "not an engine field"):
            FieldDesign("node.no_such_field", ["J1"], 0.0, 1.0)
        with self.assertRaisesRegex(ValueError, "at least one id"):
            FieldDesign("node.invert_elev", [], 0.0, 1.0)
        with self.assertRaisesRegex(ValueError, "strictly greater"):
            FieldDesign("node.invert_elev", ["J1"], 1.0, 1.0)

    def test_apply_needs_bind(self):
        with self.assertRaisesRegex(RuntimeError, "bind"):
            FieldDesign("node.invert_elev", ["J1"], 0.0, 1.0).apply(None, np.array([0.5]))


class TestEngine(unittest.TestCase):
    def _adapter(self, name: str) -> SolverAdapter:
        out = _folder(name)
        inp = out / "minimal.inp"
        shutil.copy(_DATA / "minimal.inp", inp)
        adapter = SolverAdapter(inp, out / "minimal.rpt", out / "minimal.out")
        adapter.open()
        self.addCleanup(adapter.close)
        return adapter

    def test_float_int_and_bool_fields_take_their_own_type(self):
        adapter = self._adapter("types")
        cases = [
            (FieldDesign("node.invert_elev", ["J1"], -5.0, 5.0), 2.5, 2.5),
            (FieldDesign("link.barrels", ["C1"], 1.0, 4.0), 2.6, 3),
            (FieldDesign("link.flap_gate", ["C1"], 0.0, 1.0), 0.7, True),
        ]
        for factory, value, expected in cases:
            with self.subTest(path=factory.requires[0]):
                factory.bind(adapter)
                factory.apply(adapter, np.array([value]))
                got = adapter.read(
                    factory.requires[0], adapter.index(factory._kind, factory._ids[0])
                )
                self.assertEqual(got, expected)
                self.assertIs(type(got), type(expected))

    def test_values_are_clipped_to_the_design_range(self):
        adapter = self._adapter("clip")
        factory = FieldDesign("node.invert_elev", ["J1"], -1.0, 1.0)
        factory.bind(adapter)
        factory.apply(adapter, np.array([7.0]))
        self.assertEqual(adapter.read("node.invert_elev", adapter.index("node", "J1")), 1.0)
        adapter.initialize()  # a pre-initialize write the engine accepts


class TestStudyDimensions(unittest.TestCase):
    def test_a_field_design_is_one_dimension_labelled_by_element(self):
        from openswmm_gymnasium.optimize import design_dimensions
        from openswmm_gymnasium.spec.config import EnvConfig, ObservationSpec

        config = EnvConfig(
            env_type="cip",
            inp_path=str(_DATA / "site_drainage_example.inp"),
            observations=ObservationSpec(node_depths=["J1"]),
            design_factories=[
                {
                    "kind": "field_design",
                    "params": {
                        "path": "subcatchment.width",
                        "ids": ["S1", "S2"],
                        "low": 100.0,
                        "high": 900.0,
                    },
                }
            ],
            reward_terms=[{"kind": "flooding_volume"}],
        )
        (dim,) = design_dimensions(config)
        self.assertEqual(dim.key, "subcatchment.width")
        self.assertEqual(dim.labels, ("field_design:S1", "field_design:S2"))
        self.assertEqual(dim.low, (100.0, 100.0))


if __name__ == "__main__":
    unittest.main()
