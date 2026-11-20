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

"""Single-vector evaluation, the model's own baseline, and alternatives studies.

Runs the real engine on C{tests/data/site_drainage_example.inp}; every file lands in
C{tests/_output/optimize_alternatives/}.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from openswmm_gymnasium._engine import SolverAdapter  # noqa: E402
from openswmm_gymnasium.optimize import (  # noqa: E402
    OptimizationConfig,
    apply_decisions,
    baseline_decisions,
    evaluate_decisions,
    make_optimizer,
    run_study,
)
from openswmm_gymnasium.rewards import FloodingVolume  # noqa: E402
from openswmm_gymnasium.spec.config import EnvConfig, ObservationSpec  # noqa: E402
from openswmm_gymnasium.spec.errors import SpecError  # noqa: E402

_DATA = Path(__file__).resolve().parents[1] / "data"
_OUTPUT = Path(__file__).resolve().parents[1] / "_output" / "optimize_alternatives"
WIDTH = "subcatchment.width"


def _out(name: str) -> Path:
    path = _OUTPUT / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    shutil.copy(_DATA / "site_drainage_example.inp", path / "model.inp")
    return path


def _config(folder: Path, factories=None) -> EnvConfig:
    return EnvConfig(
        env_type="cip",
        inp_path=str(folder / "model.inp"),
        rpt_path=str(folder / "model.rpt"),
        out_path=str(folder / "model.out"),
        observations=ObservationSpec(node_depths=["J1"]),
        design_factories=factories
        or [
            {
                "kind": "field_design",
                "params": {"path": WIDTH, "ids": ["S1", "S2"], "low": 100.0, "high": 3000.0},
            }
        ],
        reward_terms=[{"kind": "flooding_volume"}],
    )


def _plain_flooding(folder: Path) -> float:
    """The unedited model run through the env's reward term, outside any env."""
    adapter = SolverAdapter(folder / "model.inp", folder / "plain.rpt", folder / "plain.out")
    adapter.open()
    adapter.initialize()
    adapter.start()
    term = FloodingVolume()
    term.bind(adapter)
    term.reset()
    total, previous = 0.0, adapter.elapsed
    try:
        while adapter.is_running:
            adapter.step()
            dt = max((adapter.elapsed - previous) * 86400.0, 0.0)
            previous = adapter.elapsed
            total += float(term.step(adapter, dt))
    finally:
        adapter.close()
    return total


def test_the_baseline_vector_scores_the_same_as_a_plain_episode():
    folder = _out("baseline")
    config = _config(folder)
    baseline = baseline_decisions(config)
    assert baseline == {WIDTH: [1587.0, 1653.0]}  # the widths in the model
    scored = evaluate_decisions(config, baseline)
    plain = _plain_flooding(folder)
    assert plain > 0
    assert scored["objectives"] == {"flooding_volume": plain}
    assert scored["decisions"] == baseline and scored["steps"] == 1
    narrower = evaluate_decisions(config, {WIDTH: [300.0, 300.0]})
    assert narrower["objectives"]["flooding_volume"] != plain


def test_an_alternatives_study_evaluates_exactly_the_given_candidates():
    folder = _out("study")
    config = _config(folder)
    candidates = [baseline_decisions(config), {WIDTH: [300.0, 300.0]}, {WIDTH: [2500.0, 900.0]}]
    opt = OptimizationConfig(algorithm="alternatives", candidates=candidates, workers=0)
    result = run_study(config, opt, folder / "run")
    records = [json.loads(line) for line in (folder / "run" / "evaluations.jsonl").open()]
    assert [r["decisions"] for r in records] == candidates
    assert [r["evaluation"] for r in records] == [1, 2, 3]
    assert records[0]["objectives"] == evaluate_decisions(config, candidates[0])["objectives"]
    assert result["optimization"] == {
        "algorithm": "alternatives",
        "budget": 50,
        "seed": None,
        "candidates": 3,
    }
    assert result["evaluations_count"] == 3


def test_candidates_must_fit_the_design():
    folder = _out("refusals")
    config = _config(folder)
    with pytest.raises(ValueError, match="only with, algorithm 'alternatives'"):
        OptimizationConfig(candidates=[{WIDTH: [1.0, 1.0]}])
    with pytest.raises(ValueError, match="exceed the budget"):
        OptimizationConfig(
            algorithm="alternatives", budget=1, candidates=[{WIDTH: [200.0, 200.0]}] * 2
        )
    with pytest.raises(SpecError, match="outside the design bounds at component"):
        run_study(
            config,
            OptimizationConfig(
                algorithm="alternatives", workers=0, candidates=[{WIDTH: [50.0, 200.0]}]
            ),
            folder / "run",
        )
    with pytest.raises(SpecError, match="missing"):
        evaluate_decisions(config, {"subcatchment.area": [1.0, 1.0]})
    with pytest.raises(SpecError, match="takes 2 value"):
        evaluate_decisions(config, {WIDTH: [1.0]})
    with pytest.raises(SpecError, match="alternatives needs a candidate"):
        make_optimizer("alternatives", [0.0], [1.0], 1, budget=1)


def test_a_design_kind_with_no_single_field_has_no_baseline():
    folder = _out("no_baseline")
    config = _config(
        folder, [{"kind": "link_diameter", "params": {"link_ids": ["C1"], "low": 1.0, "high": 3.0}}]
    )
    with pytest.raises(SpecError, match="link_diameter.*no single field"):
        baseline_decisions(config)


def test_a_written_design_scores_what_its_evaluation_scored():
    """A model written after apply_decisions is the design: its own values are the decisions,
    and as a baseline it scores what the evaluation of the decisions scored."""
    folder = _out("written")
    config = _config(folder)
    decisions = {WIDTH: [300.0, 2500.0]}
    adapter = SolverAdapter(folder / "model.inp", folder / "w.rpt", folder / "w.out")
    adapter.open()
    try:
        applied = apply_decisions(config, decisions, adapter)
        adapter.solver.write(str(folder / "derived.inp"))
    finally:
        adapter.close()
    assert [(a["element_id"], a["value"]) for a in applied] == [("S1", 300.0), ("S2", 2500.0)]
    derived = config.model_copy(update={"inp_path": str(folder / "derived.inp")})
    assert baseline_decisions(derived) == decisions
    assert (
        evaluate_decisions(derived, decisions)["objectives"]
        == (evaluate_decisions(config, decisions)["objectives"])
    )
    policy = config.model_copy(update={"env_type": "market"})
    with pytest.raises(SpecError, match="tunes a policy"):
        apply_decisions(policy, decisions, None)
