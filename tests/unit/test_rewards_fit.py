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

"""Fit metrics and the fit_to_observed calibration objective.

The engine tests run C{tests/data/site_drainage_example.inp}; observations are written from
the run's own reported flows, so the fit is known. Files land in
C{tests/_output/rewards_fit/}.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pydantic")

from openswmm.engine import OutLinkVar, OutputReader  # noqa: E402

from openswmm_gymnasium.optimize import evaluate_decisions  # noqa: E402
from openswmm_gymnasium.rewards.fit import (  # noqa: E402
    FitToObserved,
    fit_metrics,
    kge,
    read_observed_csv,
)
from openswmm_gymnasium.spec.config import EnvConfig, ObservationSpec  # noqa: E402

_DATA = Path(__file__).resolve().parents[1] / "data"
_OUTPUT = Path(__file__).resolve().parents[1] / "_output" / "rewards_fit"
T = np.array(
    ["2026-01-01T00:00", "2026-01-01T01:00", "2026-01-01T02:00", "2026-01-01T03:00"],
    dtype="datetime64[s]",
)


def test_metrics_of_a_perfect_and_a_shifted_series():
    obs = np.array([1.0, 3.0, 2.0, 5.0])
    perfect = fit_metrics(T, obs, T, obs)
    assert (perfect["nse"], perfect["kge"], perfect["rmse"], perfect["bias"]) == (
        1.0,
        1.0,
        0.0,
        0.0,
    )
    shifted = fit_metrics(T, obs, T, obs + 1.0)
    assert shifted["bias"] == 1.0 and shifted["rmse"] == 1.0
    sim, mo = obs + 1.0, obs.mean()
    by_hand = 1 - np.sqrt(0 + 0 + (sim.mean() / mo - 1) ** 2)  # r = 1, alpha = 1
    assert kge(sim, obs) == pytest.approx(by_hand)
    assert kge(np.ones(4), obs) is None  # a constant simulation has no correlation


def test_missing_observations_are_left_out(tmp_path=None):
    folder = _OUTPUT / "csv"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    path = folder / "obs.csv"
    path.write_text("time,A,B\n2026-01-01T00:00,1,\n2026-01-01T01:00,3,7\n2026-01-01T02:00,2,8\n")
    times, values = read_observed_csv(path)["B"]
    assert np.isnan(values[0]) and len(times) == 3
    m = fit_metrics(times, values, times, np.array([0.0, 7.0, 8.0]))
    assert m["rmse"] == 0.0  # the empty first cell is not compared


def _observations(folder: Path, links: list[str], scale: float = 1.0) -> Path:
    """Observed link flows: a reference run's own reported flows, times scale."""
    out = folder / "reference.out"
    from openswmm.engine import Solver

    solver = Solver(str(folder / "model.inp"), str(folder / "reference.rpt"), str(out))
    solver.run()
    solver.destroy()
    reader = OutputReader(str(out))
    times = [np.datetime_as_string(t, unit="s") for t in np.asarray(reader.period_times)]
    series = {lk: np.asarray(reader.link_series(lk, int(OutLinkVar.FLOW))) for lk in links}
    reader.close()
    path = folder / "data" / "observed.csv"
    path.parent.mkdir()
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time", *links])
        for i, t in enumerate(times):
            writer.writerow([t, *(repr(float(series[lk][i]) * scale) for lk in links)])
    return path


def _config(folder: Path, metric: str) -> EnvConfig:
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
        reward_terms=[
            {
                "kind": "fit_to_observed",
                "params": {
                    "path": "link.flow",
                    "element_id": "C11",
                    "observed": "data/observed.csv",
                    "metric": metric,
                },
            }
        ],
    )


def test_a_run_reproducing_its_observations_scores_a_perfect_fit():
    folder = _OUTPUT / "twin"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    shutil.copy(_DATA / "site_drainage_example.inp", folder / "model.inp")
    _observations(folder, ["C11"])
    same = {"subcatchment.width": [1587.0]}  # the model's own width
    nse = evaluate_decisions(_config(folder, "nse"), same)["objectives"]["nse_C11"]
    assert nse == pytest.approx(-1.0, abs=1e-6)  # maximized: recorded as a negated cost
    narrow = evaluate_decisions(_config(folder, "nse"), {"subcatchment.width": [150.0]})
    assert -narrow["objectives"]["nse_C11"] < 0.999
    rmse = evaluate_decisions(_config(folder, "rmse"), same)["objectives"]["rmse_C11"]
    # The observations come from the .out file, which stores flows as float32.
    assert rmse == pytest.approx(0.0, abs=1e-5)


def test_the_term_refuses_an_unknown_metric_or_field():
    with pytest.raises(ValueError, match="metric must be one of"):
        FitToObserved("link.flow", "C11", "obs.csv", metric="r2")
    with pytest.raises(ValueError, match="not an engine field"):
        FitToObserved("link.nothing", "C11", "obs.csv")
    assert FitToObserved("link.flow", "C11", "obs.csv", metric="kge").direction == "maximize"
