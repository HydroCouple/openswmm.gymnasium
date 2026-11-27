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

"""Latin hypercube and Sobol designs: one seed, one design, however it is evaluated.

The study tests run the real engine on C{tests/data/site_drainage_example.inp}; files land
in C{tests/_output/optimize_sampling/}.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy.stats")
pytest.importorskip("pydantic")

from openswmm_gymnasium.optimize import (  # noqa: E402
    LatinHypercube,
    OptimizationConfig,
    Sobol,
    make_optimizer,
    run_study,
)
from openswmm_gymnasium.spec.config import EnvConfig, ObservationSpec  # noqa: E402

_DATA = Path(__file__).resolve().parents[1] / "data"
_OUTPUT = Path(__file__).resolve().parents[1] / "_output" / "optimize_sampling"
LOW, HIGH = np.array([100.0, 0.01]), np.array([900.0, 0.03])


def _drain(optimizer) -> np.ndarray:
    points = []
    while batch := optimizer.ask():
        points.extend(batch)
    return np.array(points)


def test_a_latin_hypercube_samples_every_stratum_of_every_component_once():
    points = _drain(LatinHypercube(LOW, HIGH, budget=8, seed=3, batch=3))
    assert points.shape == (8, 2)
    strata = np.floor((points - LOW) / (HIGH - LOW) * 8).astype(int)
    for component in strata.T:
        assert sorted(component) == list(range(8))


def test_one_seed_gives_one_design_and_another_seed_another():
    for cls in (LatinHypercube, Sobol):
        a = _drain(cls(LOW, HIGH, budget=6, seed=11, batch=1))
        b = _drain(cls(LOW, HIGH, budget=6, seed=11, batch=4))
        c = _drain(cls(LOW, HIGH, budget=6, seed=12, batch=1))
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)
        assert np.all((a >= LOW) & (a <= HIGH))
    assert _drain(make_optimizer("sobol", LOW, HIGH, 1, budget=5, seed=1)).shape == (5, 2)


@pytest.mark.parametrize("algorithm", ["lhs", "sobol"])
def test_a_study_draws_the_same_design_serially_and_on_workers(algorithm):
    folder = _OUTPUT / algorithm
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
    logs = {}
    for workers in (0, 2):
        opt = OptimizationConfig(algorithm=algorithm, budget=6, seed=7, workers=workers)
        run_study(config, opt, folder / f"workers_{workers}", manifest=False)
        log = folder / f"workers_{workers}" / "evaluations.jsonl"
        logs[workers] = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["decisions"] for r in logs[0]] == [r["decisions"] for r in logs[2]]
    assert [r["objectives"] for r in logs[0]] == [r["objectives"] for r in logs[2]]
    assert len(logs[0]) == 6
