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

"""Every reward term is a metric of each evaluation, objective or not.

The study tests run the real engine on C{tests/data/site_drainage_example.inp}; files land
in C{tests/_output/optimize_metrics/}.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from pydantic import ValidationError  # noqa: E402

from openswmm_gymnasium.optimize import OptimizationConfig, run_study  # noqa: E402
from openswmm_gymnasium.optimize.space import objective_directions  # noqa: E402
from openswmm_gymnasium.spec.config import EnvConfig, ObservationSpec  # noqa: E402
from openswmm_gymnasium.spec.errors import SpecError  # noqa: E402

_DATA = Path(__file__).resolve().parents[1] / "data"
_OUTPUT = Path(__file__).resolve().parents[1] / "_output" / "optimize_metrics"


def _config(folder: Path, objectives: list[str] | None) -> EnvConfig:
    return EnvConfig(
        env_type="cip",
        inp_path=str(folder / "model.inp"),
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
        reward_terms=[
            {"kind": "flooding_volume"},
            {"kind": "peak_outflow", "params": {"link_ids": ["C11"]}},
        ],
        objectives=objectives,
    )


def _model(name: str) -> Path:
    folder = _OUTPUT / name
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    shutil.copy(_DATA / "site_drainage_example.inp", folder / "model.inp")
    return folder


def test_terms_that_are_not_objectives_are_recorded_as_metrics():
    folder = _model("study")
    config = _config(folder, ["peak_outflow"])
    logs, results = {}, {}
    for workers in (0, 2):
        opt = OptimizationConfig(algorithm="random_search", budget=4, seed=5, workers=workers)
        results[workers] = run_study(config, opt, folder / f"workers_{workers}", manifest=False)
        log = folder / f"workers_{workers}" / "evaluations.jsonl"
        logs[workers] = [json.loads(line) for line in log.read_text().splitlines()]
    assert results[0]["objective_names"] == ["peak_outflow"]
    for record in logs[0]:
        assert list(record["objectives"]) == ["peak_outflow"]
        assert list(record["metrics"]) == ["flooding_volume", "peak_outflow"]
        # A minimized term's cost is its value.
        assert record["objectives"]["peak_outflow"] == record["metrics"]["peak_outflow"]
    assert len({r["metrics"]["peak_outflow"] for r in logs[0]}) > 1
    assert [r["metrics"] for r in logs[0]] == [r["metrics"] for r in logs[2]]


def test_every_term_is_an_objective_unless_objectives_are_named():
    folder = _model("names")
    assert list(objective_directions(_config(folder, None))) == [
        "flooding_volume",
        "peak_outflow",
    ]
    assert list(objective_directions(_config(folder, ["peak_outflow", "flooding_volume"]))) == [
        "peak_outflow",
        "flooding_volume",
    ]
    with pytest.raises(SpecError, match="name no reward term"):
        objective_directions(_config(folder, ["peak"]))
    for bad in ([], ["peak_outflow", "peak_outflow"]):
        with pytest.raises(ValidationError, match="each once"):
            _config(folder, bad)
