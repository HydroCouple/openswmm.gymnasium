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

"""Behavioural sets, prediction bands, and studies run again from their manifests.

The engine tests run C{tests/data/site_drainage_example.inp}; files land in
C{tests/_output/optimize_uncertainty/}.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pydantic")

from pydantic import ValidationError  # noqa: E402

from openswmm_gymnasium.optimize.uncertainty import (  # noqa: E402
    AcceptanceRule,
    behavioural_set,
    prediction_bands,
)
from openswmm_gymnasium.spec.config import EnvConfig, ObservationSpec  # noqa: E402
from openswmm_gymnasium.spec.errors import SpecError  # noqa: E402

_DATA = Path(__file__).resolve().parents[1] / "data"
_OUTPUT = Path(__file__).resolve().parents[1] / "_output" / "optimize_uncertainty"


def _manifest(acceptance=None) -> dict:
    def record(n, kge, rmse, failed=False):
        return {
            "evaluation": n,
            "decisions": {"w": [100.0 * n]},
            "objectives": {"kge_C11": 1e12 if kge is None else -kge, "rmse_C11": rmse},
            "metrics": {"kge_C11": kge, "rmse_C11": rmse},
            "failed": failed,
        }

    optimization = {"algorithm": "lhs", "budget": 6, "seed": 1}
    if acceptance is not None:
        optimization["acceptance"] = acceptance
    return {
        "definition": {
            "objectives": [
                {"name": "kge_C11", "direction": "maximize"},
                {"name": "rmse_C11", "direction": "minimize"},
            ],
            "optimization": optimization,
        },
        "evaluations": [
            record(1, 0.9, 0.2),
            record(2, 0.4, 0.5),
            record(3, 0.7, 0.3),
            record(4, None, 0.1),
            record(5, 0.95, 0.9, failed=True),
            record(6, 0.6, 0.8),
        ],
    }


def test_a_behavioural_set_keeps_what_its_rule_accepts():
    rule = {"thresholds": [{"metric": "kge_C11", "at_least": 0.5}]}
    accepted = behavioural_set(_manifest(rule))
    assert [m["evaluation"] for m in accepted["members"]] == [1, 3, 6]
    assert accepted["evaluated"] == 5  # the failed evaluation is not a candidate
    both = {
        "thresholds": [
            {"metric": "kge_C11", "at_least": 0.5},
            {"metric": "rmse_C11", "at_most": 0.5},
        ]
    }
    assert [m["evaluation"] for m in behavioural_set(_manifest(), both)["members"]] == [1, 3]
    best = behavioural_set(_manifest(), {"best_fraction": 0.4})  # by the first objective
    assert [m["evaluation"] for m in best["members"]] == [1, 3]  # ceil(0.4 * 5) = 2
    by_rmse = behavioural_set(_manifest(), {"best_fraction": 0.2, "by": "rmse_C11"})
    assert [m["evaluation"] for m in by_rmse["members"]] == [4]


def test_a_rule_must_say_something_the_study_records():
    with pytest.raises(SpecError, match="declares no acceptance rule"):
        behavioural_set(_manifest())
    with pytest.raises(SpecError, match="records no metric \\['nse_C11'\\]"):
        behavioural_set(_manifest(), {"thresholds": [{"metric": "nse_C11", "at_least": 0.5}]})
    for bad in ({}, {"thresholds": [{"metric": "kge_C11"}]}, {"by": "kge_C11"}):
        with pytest.raises(ValidationError):
            AcceptanceRule.model_validate(bad)


def _config(folder: Path) -> EnvConfig:
    return EnvConfig(
        env_type="cip",
        inp_path=str(folder / "model.inp"),
        observations=ObservationSpec(node_depths=["J1"]),
        design_factories=[
            {
                "kind": "field_design",
                "params": {
                    "path": "subcatchment.width",
                    "ids": ["S1"],
                    "low": 100.0,
                    "high": 3000.0,
                },
            }
        ],
        reward_terms=[{"kind": "peak_outflow", "params": {"link_ids": ["C11"]}}],
    )


def test_one_member_s_band_is_its_reported_series_and_quantiles_are_ordered():
    from openswmm.engine import OutLinkVar, OutputReader, Solver

    from openswmm_gymnasium._engine import SolverAdapter
    from openswmm_gymnasium.optimize.apply import apply_decisions

    folder = _OUTPUT / "bands"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    shutil.copy(_DATA / "site_drainage_example.inp", folder / "model.inp")
    config = _config(folder)
    member = {"subcatchment.width": [600.0]}

    one = prediction_bands(config, [member], [("link.flow", "C11")], folder / "one")
    # The same design, run by the engine with its own report.
    adapter = SolverAdapter(folder / "model.inp", None, None)
    adapter.open()
    apply_decisions(config, member, adapter)
    adapter.solver.write(str(folder / "member.inp"))
    adapter.close()
    solver = Solver(
        str(folder / "member.inp"), str(folder / "member.rpt"), str(folder / "member.out")
    )
    solver.run()
    solver.destroy()
    reader = OutputReader(str(folder / "member.out"))
    reported = dict(
        zip(
            np.datetime_as_string(np.asarray(reader.period_times), unit="s").tolist(),
            np.asarray(reader.link_series("C11", int(OutLinkVar.FLOW))).tolist(),
            strict=True,
        )
    )
    reader.close()
    median = dict(
        zip(one["times"], one["series"]["link.flow:C11"]["quantiles"]["0.5"], strict=True)
    )
    shared = sorted(set(median) & set(reported))
    assert len(shared) >= len(reported) - 1
    np.testing.assert_allclose(
        [median[t] for t in shared], [reported[t] for t in shared], rtol=1e-5, atol=1e-5
    )

    members = [{"subcatchment.width": [w]} for w in (200.0, 600.0, 1500.0, 2800.0)]
    many = prediction_bands(config, members, [("link.flow", "C11")], folder / "many")
    q = {k: np.asarray(v) for k, v in many["series"]["link.flow:C11"]["quantiles"].items()}
    assert np.all(q["0.05"] <= q["0.5"]) and np.all(q["0.5"] <= q["0.95"])
    assert np.any(q["0.05"] < q["0.95"])
    assert (
        Path(many["series"]["link.flow:C11"]["file"])
        .read_text()
        .startswith("time,q0.05,q0.5,q0.95,member_1,member_2,member_3,member_4\n")
    )


@pytest.mark.parametrize("algorithm", ["lhs", "nsga2"])
def test_a_study_runs_again_from_its_manifest_to_the_same_log(algorithm):
    pytest.importorskip("hydrocouple_platform")
    pytest.importorskip("scipy.stats" if algorithm == "lhs" else "platypus")
    from hydrocouple_platform.contracts.studies import StudyManifest

    from openswmm_gymnasium.optimize import OptimizationConfig, run_study
    from openswmm_gymnasium.optimize.manifest import rerun

    folder = _OUTPUT / f"rerun_{algorithm}"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    shutil.copy(_DATA / "site_drainage_example.inp", folder / "model.inp")
    opt = OptimizationConfig(
        algorithm=algorithm,
        budget=12,
        population_size=6,
        seed=3,
        workers=0,
        acceptance={"best_fraction": 0.25},
    )
    run_study(_config(folder), opt, folder / "first")
    rerun(
        folder / "first" / "study-manifest.json", folder / "model.inp", folder / "again", workers=2
    )
    first, again = (folder / side / "evaluations.jsonl" for side in ("first", "again"))
    assert first.read_bytes() == again.read_bytes()
    manifests = [
        StudyManifest.model_validate_json((folder / side / "study-manifest.json").read_text())
        for side in ("first", "again")
    ]
    assert manifests[0].comparable() == manifests[1].comparable()
    behavioural = behavioural_set(folder / "first" / "study-manifest.json")  # its own rule
    assert len(behavioural["members"]) == -(-behavioural["evaluated"] // 4)

    edited = folder / "edited"
    edited.mkdir()
    text = (folder / "model.inp").read_text().replace("THREADS              1", "THREADS 2")
    (edited / "model.inp").write_text(text)
    with pytest.raises(SpecError, match=r"these files differ: \['model/model.inp'\]"):
        rerun(folder / "first" / "study-manifest.json", edited / "model.inp", folder / "refused")
