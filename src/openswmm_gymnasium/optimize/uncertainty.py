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
How uncertain a calibration is: its behavioural set and the prediction bands of the set.

A sampling study (Latin hypercube, Sobol, random) scores parameter sets against
observations. L{behavioural_set} keeps the sets its L{AcceptanceRule} accepts, the
generalized likelihood uncertainty estimation of Beven and Binley (1992); the rule is
declared in the study's optimization settings, so it is part of the study manifest.
L{prediction_bands} runs every member and takes unweighted quantiles of simulated series
at each report time.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from openswmm_gymnasium.spec.errors import ErrorCode, SpecError

_OADATE_EPOCH = np.datetime64("1899-12-30T00:00:00", "ms")


class Threshold(BaseModel):
    """A bound on one metric of an evaluation (C{metrics} in its record).

    @ivar metric: Metric name, e.g. C{"kge_C11"}; values are the term's own (not negated).
    @ivar at_least: Lowest accepted value.
    @ivar at_most: Highest accepted value.
    """

    model_config = ConfigDict(extra="forbid")

    metric: str = Field(min_length=1)
    at_least: float | None = None
    at_most: float | None = None

    @model_validator(mode="after")
    def _a_bound(self) -> Threshold:
        if self.at_least is None and self.at_most is None:
            raise ValueError(f"threshold on {self.metric!r} needs at_least or at_most")
        return self


class AcceptanceRule(BaseModel):
    """Which evaluations are behavioural: every threshold met, then the best fraction.

    @ivar thresholds: Bounds every member meets.
    @ivar best_fraction: Keep only the best this fraction of the study's successful
        evaluations (at least one), ranked by objective C{by}'s cost.
    @ivar by: Objective ranking C{best_fraction}; the study's first objective when unset.
    """

    model_config = ConfigDict(extra="forbid")

    thresholds: list[Threshold] = []
    best_fraction: float | None = Field(default=None, gt=0, le=1)
    by: str | None = None

    @model_validator(mode="after")
    def _a_rule(self) -> AcceptanceRule:
        if not self.thresholds and self.best_fraction is None:
            raise ValueError("an acceptance rule needs thresholds or best_fraction")
        if self.by is not None and self.best_fraction is None:
            raise ValueError("by ranks best_fraction; give best_fraction with it")
        return self


def _document(manifest: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(manifest, Mapping):
        return dict(manifest)
    return json.loads(Path(manifest).read_text(encoding="utf-8"))


def behavioural_set(
    manifest: Mapping[str, Any] | str | Path,
    rule: AcceptanceRule | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The evaluations of a study its acceptance rule accepts.

    @param manifest: A study manifest, or the path of its C{study-manifest.json}.
    @param rule: The rule; the study's own (C{optimization.acceptance}) when C{None}.
    @return: C{{"rule", "evaluated", "members"}}: the rule applied, the number of
        successful evaluations, and the accepted ones (C{evaluation}, C{decisions},
        C{objectives}, C{metrics}) in evaluation order.
    @raise SpecError: C{VALIDATION_ERROR} with no rule, or a threshold on a metric the
        study does not record.
    @rtype: dict
    """
    doc = _document(manifest)
    definition = doc["definition"]
    if rule is None:
        rule = definition["optimization"].get("acceptance")
        if rule is None:
            raise SpecError(
                f"[{ErrorCode.VALIDATION_ERROR}] The study declares no acceptance rule "
                "(optimization.acceptance); pass one."
            )
    rule = AcceptanceRule.model_validate(rule)
    ok = [e for e in doc["evaluations"] if not e.get("failed")]
    recorded = {name for e in ok for name in e.get("metrics", {})}
    unknown = sorted({t.metric for t in rule.thresholds} - recorded)
    if unknown:
        raise SpecError(
            f"[{ErrorCode.VALIDATION_ERROR}] The study records no metric {unknown}; it "
            f"records {sorted(recorded)}."
        )

    def accepted(e: dict[str, Any]) -> bool:
        for t in rule.thresholds:
            value = e.get("metrics", {}).get(t.metric)
            if value is None or (t.at_least is not None and value < t.at_least):
                return False
            if t.at_most is not None and value > t.at_most:
                return False
        return True

    members = [e for e in ok if accepted(e)]
    if rule.best_fraction is not None:
        by = rule.by or definition["objectives"][0]["name"]
        keep = max(1, math.ceil(rule.best_fraction * len(ok)))
        ranked = sorted(members, key=lambda e: (e["objectives"][by], e["evaluation"]))[:keep]
        members = sorted(ranked, key=lambda e: e["evaluation"])
    return {
        "rule": rule.model_dump(mode="json", exclude_none=True),
        "evaluated": len(ok),
        "members": [
            {k: e.get(k) for k in ("evaluation", "decisions", "objectives", "metrics")}
            for e in members
        ],
    }


def _iso(days: np.ndarray) -> list[str]:
    stamps = _OADATE_EPOCH + np.round(np.asarray(days) * 86_400_000).astype("timedelta64[ms]")
    return [str(s) for s in np.datetime_as_string(stamps, unit="s")]


def _run_member(
    env_config: Any,
    decisions: Mapping[str, Any],
    series: list[tuple[str, str]],
    event: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One member's (report times, routing-step times, values per series) as OADates."""
    from openswmm_gymnasium._engine import SolverAdapter, element_kind
    from openswmm_gymnasium.optimize.apply import apply_decisions

    adapter = SolverAdapter(env_config.inp_path, os.devnull, os.devnull)
    adapter.open()
    try:
        apply_decisions(env_config, dict(decisions), adapter)
        if event is not None:
            event.apply(adapter)
        adapter.initialize()
        adapter.start()
        idx = [adapter.index(element_kind(path), eid) for path, eid in series]
        start = adapter.read("options.report_start_datetime", None)
        step = float(adapter.get_option("REPORT_STEP")) / 86_400.0
        first = (np.datetime64(start, "ms") - _OADATE_EPOCH) / np.timedelta64(1, "D")
        grid = np.arange(first, adapter.end_time + step / 2, step)
        times = [adapter.current_time]
        values = [[float(adapter.read(p, i)) for (p, _), i in zip(series, idx, strict=True)]]
        while adapter.is_running:
            adapter.step()
            now = adapter.current_time
            if now > times[-1]:  # the engine's final step does not advance the clock
                times.append(now)
                values.append(
                    [float(adapter.read(p, i)) for (p, _), i in zip(series, idx, strict=True)]
                )
        adapter.end()
    finally:
        adapter.close()
    return grid, np.asarray(times), np.asarray(values)


def prediction_bands(
    env_config: Any,
    members: Sequence[Mapping[str, Any]],
    series: Sequence[tuple[str, str]],
    out_dir: str | Path,
    *,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    event: str | None = None,
) -> dict[str, Any]:
    """Unweighted quantiles, at each report time, of series simulated by every member.

    Each member's decisions are written onto the model as a study evaluated them
    (L{apply_decisions<openswmm_gymnasium.optimize.apply.apply_decisions>}), the run is
    stepped to its end, and each series is interpolated between routing steps onto the
    report times (report start to end, every C{REPORT_STEP}).

    @param env_config: The study's L{EnvConfig<openswmm_gymnasium.spec.config.EnvConfig>},
        with C{inp_path} the local model.
    @param members: Decisions of each member (a behavioural set's C{members}, or their
        C{decisions}).
    @param series: C{(catalog path, element id)} pairs, e.g. C{("link.flow", "C11")}.
    @param out_dir: Folder for one CSV per series: time, the quantiles, then each member.
    @param event: Run one of C{env_config.events} instead of the model's own rain.
    @return: C{{"times", "series"}}; each series (C{"<path>:<id>"}) has its
        C{"quantiles"} by value and its C{"file"}.
    @raise SpecError: C{VALIDATION_ERROR} with no members or an unknown event.
    @rtype: dict
    """
    if not members:
        raise SpecError(f"[{ErrorCode.VALIDATION_ERROR}] Prediction bands need members.")
    rain = None
    if event is not None:
        by_name = {e.name: e for e in env_config.events}
        if event not in by_name:
            raise SpecError(
                f"[{ErrorCode.VALIDATION_ERROR}] No event {event!r}; the config's events "
                f"are {list(by_name)}."
            )
        rain = by_name[event].build()
    pairs = [(str(p), str(e)) for p, e in series]
    grid, runs = None, []
    for member in members:
        decisions = member.get("decisions", member)  # a set's member, or its decisions
        member_grid, times, values = _run_member(env_config, decisions, pairs, rain)
        grid = member_grid if grid is None else grid
        runs.append(np.stack([np.interp(grid, times, values[:, k]) for k in range(len(pairs))]))
    stacked = np.stack(runs)  # member, series, time
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamps = _iso(grid)
    result: dict[str, Any] = {"times": stamps, "series": {}}
    for k, (path, eid) in enumerate(pairs):
        bands = {q: np.quantile(stacked[:, k, :], q, axis=0) for q in quantiles}
        file = out / f"bands_{path}_{eid}.csv"
        with file.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["time", *(f"q{q:g}" for q in quantiles)]
                + [f"member_{m + 1}" for m in range(len(runs))]
            )
            for t, stamp in enumerate(stamps):
                writer.writerow(
                    [stamp, *(repr(float(bands[q][t])) for q in quantiles)]
                    + [repr(float(stacked[m, k, t])) for m in range(len(runs))]
                )
        result["series"][f"{path}:{eid}"] = {
            "quantiles": {f"{q:g}": bands[q].tolist() for q in quantiles},
            "file": str(file),
        }
    return result
