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

"""Discrete dimensions: a field takes one of listed values, and a study records the value.

The engine tests run C{tests/data/site_drainage_example.inp}; files land in
C{tests/_output/optimize_choices/}.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pydantic")

from openswmm_gymnasium._engine import SolverAdapter  # noqa: E402
from openswmm_gymnasium.optimize import (  # noqa: E402
    OptimizationConfig,
    decision_values,
    design_dimensions,
    evaluate_decisions,
    join_decisions,
    run_study,
)
from openswmm_gymnasium.optimize.manifest import study_definition  # noqa: E402
from openswmm_gymnasium.spaces.design import FieldChoice  # noqa: E402
from openswmm_gymnasium.spec.config import EnvConfig, ObservationSpec  # noqa: E402
from openswmm_gymnasium.spec.errors import SpecError  # noqa: E402

_DATA = Path(__file__).resolve().parents[1] / "data"
_OUTPUT = Path(__file__).resolve().parents[1] / "_output" / "optimize_choices"


def _folder(name: str) -> Path:
    folder = _OUTPUT / name
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    shutil.copy(_DATA / "site_drainage_example.inp", folder / "model.inp")
    return folder


def _config(folder: Path) -> EnvConfig:
    return EnvConfig(
        env_type="cip",
        inp_path=str(folder / "model.inp"),
        observations=ObservationSpec(node_depths=["J1"]),
        design_factories=[
            {
                "kind": "field_choice",
                "params": {"path": "link.barrels", "ids": ["C11"], "choices": [1, 2, 3]},
            },
            {
                "kind": "field_design",
                "params": {
                    "path": "subcatchment.width",
                    "ids": ["S1"],
                    "low": 100.0,
                    "high": 900.0,
                },
            },
        ],
        reward_terms=[{"kind": "peak_outflow", "params": {"link_ids": ["C11"]}}],
    )


def test_an_action_picks_a_choice_by_its_integer_part():
    factory = FieldChoice("link.roughness", ["C1", "C2"], [0.011, 0.013, 0.024])
    assert (float(factory.space.low[0]), float(factory.space.high[0])) == (0.0, 3.0)
    np.testing.assert_array_equal(factory.chosen(np.array([0.0, 0.99])), [0.011, 0.011])
    np.testing.assert_array_equal(factory.chosen(np.array([2.5, 3.0])), [0.024, 0.024])
    for choices, message in (
        ([0.011], "at least two distinct"),
        ([0.011, 0.011], "at least two distinct"),
    ):
        with pytest.raises(ValueError, match=message):
            FieldChoice("link.roughness", ["C1"], choices)
    with pytest.raises(ValueError, match="choices must be whole"):
        FieldChoice("link.barrels", ["C1"], [1, 1.5])
    with pytest.raises(ValueError, match="choices are 0 and 1"):
        FieldChoice("link.flap_gate", ["C1"], [0, 2])


def test_int_and_bool_fields_take_their_choices_typed():
    folder = _folder("typed")
    adapter = SolverAdapter(folder / "model.inp", folder / "model.rpt", folder / "model.out")
    adapter.open()
    try:
        for factory, action, expected in (
            (FieldChoice("link.barrels", ["C11"], [1, 2, 3]), 2.2, 3),
            (FieldChoice("link.flap_gate", ["C11"], [0, 1]), 1.0, True),
        ):
            factory.bind(adapter)
            factory.apply(adapter, np.array([action]))
            got = adapter.read(factory.requires[0], adapter.index("link", "C11"))
            assert got == expected and type(got) is type(expected)
    finally:
        adapter.close()


def test_a_study_records_the_values_chosen_serially_and_on_workers():
    folder = _folder("study")
    config = _config(folder)
    logs = {}
    for workers in (0, 2):
        opt = OptimizationConfig(algorithm="random_search", budget=8, seed=9, workers=workers)
        run_study(config, opt, folder / f"workers_{workers}", manifest=False)
        log = folder / f"workers_{workers}" / "evaluations.jsonl"
        logs[workers] = [json.loads(line) for line in log.read_text().splitlines()]
    chosen = [r["decisions"]["link.barrels"][0] for r in logs[0]]
    assert set(chosen) <= {1.0, 2.0, 3.0} and len(set(chosen)) > 1
    assert [r["decisions"] for r in logs[0]] == [r["decisions"] for r in logs[2]]
    assert [r["objectives"] for r in logs[0]] == [r["objectives"] for r in logs[2]]


def test_a_chosen_design_round_trips_through_its_vector_and_a_study():
    folder = _folder("round_trip")
    config = _config(folder)
    dims = design_dimensions(config)
    designs = [
        {"link.barrels": [3.0], "subcatchment.width": [250.0]},
        {"link.barrels": [1.0], "subcatchment.width": [250.0]},
    ]
    for design in designs:
        back = decision_values(dims, join_decisions(dims, design))
        assert {k: v.tolist() for k, v in back.items()} == design
    opt = OptimizationConfig(algorithm="alternatives", candidates=designs, seed=1, workers=0)
    run_study(config, opt, folder / "alternatives", manifest=False)
    log = (folder / "alternatives" / "evaluations.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in log]
    assert [r["decisions"] for r in records] == designs
    # The number of barrels reaches the engine: the two designs run differently.
    assert records[0]["objectives"] != records[1]["objectives"]
    scored = evaluate_decisions(config, designs[0])
    assert scored["decisions"] == designs[0]
    assert scored["objectives"] == records[0]["objectives"]
    with pytest.raises(SpecError, match="is not one of them"):
        join_decisions(dims, {"link.barrels": [2.5], "subcatchment.width": [250.0]})


def test_the_study_definition_lists_a_discrete_dimension_s_choices():
    folder = _folder("definition")
    config = _config(folder)
    opt = OptimizationConfig(algorithm="random_search", budget=1, seed=1)
    definition = study_definition(config, opt, design_dimensions(config), {}, "model/model.inp")
    barrels, width = definition["dimensions"]
    assert barrels == {
        "key": "link.barrels",
        "label": "field_choice:C11",
        "low": 0.0,
        "high": 3.0,
        "choices": [1.0, 2.0, 3.0],
    }
    assert "choices" not in width
