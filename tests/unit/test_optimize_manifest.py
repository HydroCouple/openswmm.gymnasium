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

"""The study manifest a local study writes in hydrocouple.platform's contract.

Needs hydrocouple.platform's base install. Files land in
C{tests/_output/optimize_manifest/}.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("hydrocouple_platform.contracts.studies")

from hydrocouple_platform.contracts.studies import StudyManifest  # noqa: E402

from openswmm_gymnasium.optimize import OptimizationConfig, run_study  # noqa: E402
from openswmm_gymnasium.optimize.manifest import MANIFEST_FILE  # noqa: E402
from openswmm_gymnasium.spec.config import EnvConfig, ObservationSpec  # noqa: E402

_DATA = Path(__file__).resolve().parents[1] / "data"
_OUTPUT = Path(__file__).resolve().parents[1] / "_output" / "optimize_manifest"
WIDTH = "subcatchment.width"


def _setup(name: str) -> tuple[Path, EnvConfig, OptimizationConfig]:
    folder = _OUTPUT / name
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    shutil.copy(_DATA / "site_drainage_example.inp", folder / "model.inp")
    config = EnvConfig(
        env_type="cip",
        inp_path=str(folder / "model.inp"),
        observations=ObservationSpec(node_depths=["J1"]),
        design_factories=[
            {
                "kind": "field_design",
                "params": {"path": WIDTH, "ids": ["S1"], "low": 100.0, "high": 3000.0},
            }
        ],
        reward_terms=[{"kind": "flooding_volume"}],
    )
    opt = OptimizationConfig(
        algorithm="alternatives",
        workers=0,
        seed=1,
        candidates=[{WIDTH: [1587.0]}, {WIDTH: [300.0]}],
    )
    return folder, config, opt


def test_a_local_study_writes_its_manifest_in_the_platform_contract():
    folder, config, opt = _setup("written")
    result = run_study(config, opt, folder / "run")
    path = folder / "run" / MANIFEST_FILE
    manifest = StudyManifest.model_validate_json(path.read_text())
    records = [json.loads(line) for line in (folder / "run" / "evaluations.jsonl").open()]
    assert [e.model_dump(exclude_defaults=True) for e in manifest.evaluations] == records
    definition = manifest.definition
    assert definition.workflow == "gym.evaluate"
    assert list(definition.model_inputs) == ["model/model.inp"]  # composition.json left out
    assert definition.env_config["inp_path"] == "model/model.inp"
    assert definition.env_config["rpt_path"] is None
    assert definition.optimization["candidates"] == opt.candidates
    assert "workers" not in definition.optimization
    assert [(d.key, d.label) for d in definition.dimensions] == [(WIDTH, "field_design:S1")]
    assert manifest.result["best"] == result["best"] and "execution" not in manifest.result
    assert manifest.runtime["executor"] == "local"
    assert manifest.runtime["execution"]["workers"] == 0
    assert (folder / "run" / "package" / "revision-manifest.json").is_file()


def test_the_same_study_twice_compares_equal_and_off_writes_none():
    folder, config, opt = _setup("twice")
    run_study(config, opt, folder / "first")
    run_study(config, opt, folder / "second")
    first, second = (
        StudyManifest.model_validate_json((folder / run / MANIFEST_FILE).read_text())
        for run in ("first", "second")
    )
    assert first.comparable() == second.comparable()
    run_study(config, opt, folder / "off", manifest=False)
    assert not (folder / "off" / MANIFEST_FILE).exists()
