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

"""
Goodness of fit: one implementation for calibration objectives and comparisons.

L{fit_metrics} scores one series against another (peak, volume, RMSE, NSE, bias and
the Kling-Gupta efficiency), with the second interpolated onto the first's times. It is
what openswmm.mcp's C{compare} reports. L{FitToObserved} is a reward term that scores a
run against observations as it runs, so a study can calibrate a model.

Observations are a wide CSV: a header row, ISO datetimes in the first column and one
numeric column per element id (L{read_observed_csv}).

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from openswmm_gymnasium._engine import SolverAdapter, element_kind, field_entry

#: The metric families L{fit_metrics} reports.
FIT_METRICS = ("peak", "volume", "rmse", "nse", "bias", "kge")

#: L{FitToObserved} metrics and whether a study maximizes or minimizes them.
OBJECTIVE_DIRECTIONS = {
    "nse": "maximize",
    "kge": "maximize",
    "rmse": "minimize",
    "abs_bias": "minimize",
    "peak_error": "minimize",
    "volume_error": "minimize",
}

_OADATE_EPOCH = np.datetime64("1899-12-30T00:00:00")


def read_observed_csv(path: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Element id to (times as C{datetime64}, values) from a wide observations CSV.

    Empty cells are C{NaN} and are left out of every metric.

    @raise OSError, IndexError, ValueError: The file cannot be read or parsed.
    """
    with Path(path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], [r for r in rows[1:] if r]
    times = np.array([np.datetime64(datetime.fromisoformat(r[0].strip())) for r in body])
    return {
        name.strip(): (
            times,
            np.array([float(r[i]) if i < len(r) and r[i].strip() else np.nan for r in body]),
        )
        for i, name in enumerate(header[1:], start=1)
    }


def kge(simulated: np.ndarray, observed: np.ndarray) -> float | None:
    """Kling-Gupta efficiency (Gupta et al., 2009): 1 for a perfect fit.

    C{1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)}, with C{r} the correlation,
    C{alpha} the ratio of standard deviations and C{beta} the ratio of means. C{None} when
    the observations have no spread or a zero mean, or a series is constant.
    """
    if len(observed) < 2:
        return None
    so, ss, mo = float(np.std(observed)), float(np.std(simulated)), float(np.mean(observed))
    if so == 0.0 or ss == 0.0 or mo == 0.0:
        return None
    r = float(np.corrcoef(simulated, observed)[0, 1])
    alpha, beta = ss / so, float(np.mean(simulated)) / mo
    return 1.0 - float(np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def fit_metrics(ta, a, tb, b) -> dict[str, float | None]:
    """Metrics of b against a (a = baseline or observed), with b interpolated onto a's times."""
    xa, xb = ta.astype("datetime64[s]").astype(float), tb.astype("datetime64[s]").astype(float)
    ok = ~np.isnan(a)
    bi = np.interp(xa[ok], xb, b)
    av = a[ok]
    err = bi - av
    denom = float(np.sum((av - av.mean()) ** 2)) if len(av) else 0.0
    dt = np.diff(xa[ok]) if len(av) > 1 else np.array([0.0])

    def vol(v: np.ndarray) -> float:
        return float(np.sum((v[1:] + v[:-1]) / 2 * dt)) if len(v) > 1 else 0.0

    return {
        "peak_a": float(av.max()) if len(av) else None,
        "peak_b": float(bi.max()) if len(bi) else None,
        "peak_diff": float(bi.max() - av.max()) if len(av) else None,
        "volume_a": vol(av),
        "volume_b": vol(bi),
        "rmse": float(np.sqrt(np.mean(err**2))) if len(err) else None,
        "nse": 1.0 - float(np.sum(err**2)) / denom if denom > 0 else None,
        "bias": float(np.mean(err)) if len(err) else None,
        "kge": kge(bi, av),
    }


def objective_value(metrics: dict[str, float | None], metric: str) -> float | None:
    """The value of one L{OBJECTIVE_DIRECTIONS} metric from L{fit_metrics}' output."""
    if metric in ("nse", "kge", "rmse"):
        return metrics[metric]
    if metric == "abs_bias":
        return None if metrics["bias"] is None else abs(metrics["bias"])
    if metric == "peak_error":
        return None if metrics["peak_diff"] is None else abs(metrics["peak_diff"])
    return abs(metrics["volume_b"] - metrics["volume_a"])  # volume_error


class FitToObserved:
    """How well a run reproduces an observed series at one element, as a reward term.

    C{path} is the catalog field simulated (C{"link.flow"}, C{"node.depth"}), C{element_id}
    the element and C{observed} a wide observations CSV; its column C{column} (default the
    element id) holds the observed values. A relative path is resolved against the model's
    folder, so a revision's sidecar C{data/observed.csv} is named as such. As the run passes
    each observation time, the field is interpolated between the two routing steps around
    it; each step reports the change in the metric over the observations passed so far, so
    the episode's total is the metric over every observation the simulation spanned.
    C{nse} and C{kge} are maximized; C{rmse}, C{abs_bias}, C{peak_error} and
    C{volume_error} minimized.
    """

    def __init__(
        self,
        path: str,
        element_id: str,
        observed: str,
        metric: str = "nse",
        column: str | None = None,
        name: str | None = None,
    ) -> None:
        if metric not in OBJECTIVE_DIRECTIONS:
            raise ValueError(
                f"FitToObserved: metric must be one of {sorted(OBJECTIVE_DIRECTIONS)}, not {metric!r}"
            )
        field_entry(path)  # a numeric element field, or ValueError
        self._path, self._kind = path, element_kind(path)
        self._element, self._observed = element_id, observed
        self._column, self._metric = column or element_id, metric
        self.requires: tuple[str, ...] = (path,)
        self.direction = OBJECTIVE_DIRECTIONS[metric]
        self.name = name or f"{metric}_{element_id}"
        self._idx: int | None = None
        self._obs_days: np.ndarray | None = None
        self._obs_times: np.ndarray | None = None
        self._obs: np.ndarray | None = None

    def bind(self, adapter: SolverAdapter) -> None:
        source = Path(self._observed)
        if not source.is_absolute():
            source = Path(adapter.inp_path).parent / source
        series = read_observed_csv(source)
        if self._column not in series:
            raise ValueError(f"FitToObserved: {source} has no column {self._column!r}")
        times, values = series[self._column]
        keep = ~np.isnan(values)
        self._obs_times, self._obs = times[keep], values[keep]
        self._obs_days = (self._obs_times - _OADATE_EPOCH) / np.timedelta64(1, "D")
        self._idx = adapter.index(self._kind, self._element)

    def reset(self) -> None:
        self._sim: list[float] = []
        self._previous: tuple[float, float] | None = None
        self._score = 0.0
        self._sim_scored = 0

    def _value(self) -> float:
        n = len(self._sim)
        if n < 2:
            return 0.0
        metrics = fit_metrics(
            self._obs_times[:n], self._obs[:n], self._obs_times[:n], np.asarray(self._sim)
        )
        value = objective_value(metrics, self._metric)
        return 0.0 if value is None else float(value)

    def step(self, adapter: SolverAdapter, dt_seconds: float) -> float:
        assert self._idx is not None and self._obs_days is not None, "bind() before step()"
        now = (adapter.current_time, float(adapter.read(self._path, self._idx)))
        before = self._previous or now
        self._previous = now
        passed = len(self._sim)
        while passed < len(self._obs_days) and self._obs_days[passed] <= now[0]:
            t = self._obs_days[passed]
            span = now[0] - before[0]
            w = 1.0 if span <= 0 else min(max((t - before[0]) / span, 0.0), 1.0)
            self._sim.append(before[1] + w * (now[1] - before[1]))
            passed += 1
        if passed == len(self._sim) and passed > 0 and self._sim_scored < passed:
            self._sim_scored = passed  # the metric changes only when an observation is passed
            score = self._value()
            increment, self._score = score - self._score, score
            return increment
        return 0.0
