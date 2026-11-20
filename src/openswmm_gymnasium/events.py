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
Storm events: one model run several ways, each with its own rain and window.

A L{RainEvent} is applied to an opened model before C{initialize}: the simulation window
becomes the event's, a new time series holds the event's rain, and the event's gage reads
it. The engine then runs its own gage logic, so the run is the run of a model edited to
hold that storm. L{aggregate} combines a term's per-event values into one.

The rain is a CSV in the model's folder (a revision sidecar): a header row, ISO datetimes
in the first column and one numeric column per series, as
L{read_observed_csv<openswmm_gymnasium.rewards.fit.read_observed_csv>} reads.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

#: How the per-event values of a term become one value.
AGGREGATIONS = ("sum", "mean", "max", "weighted_mean")


@dataclass(frozen=True)
class RainEvent:
    """One storm: the gage it falls on, its rain series, and the simulation window.

    @ivar name: Event name; its series is added to the model as C{event:<name>}.
    @ivar series: The rain CSV, relative to the model's folder or absolute.
    @ivar gage: The rain gage that reads the event's series.
    @ivar start: Simulation (and report) start.
    @ivar end: Simulation end.
    @ivar rain_format: C{INTENSITY}, C{VOLUME} or C{CUMULATIVE}, as for a SWMM gage.
    @ivar column: The CSV column holding the rain; C{None} for the file's only column.
    """

    name: str
    series: str
    gage: str
    start: datetime
    end: datetime
    rain_format: str = "INTENSITY"
    column: str | None = None

    def rain(self, model_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """The event's rain records: times as C{datetime64} and values.

        @raise ValueError: The file has no such column, or several and none is named.
        """
        from openswmm_gymnasium.rewards.fit import read_observed_csv

        source = Path(self.series)
        if not source.is_absolute():
            source = Path(model_dir) / source
        columns = read_observed_csv(source)
        if self.column is None and len(columns) != 1:
            raise ValueError(f"event {self.name!r}: {source} has columns {list(columns)}; name one")
        key = self.column if self.column is not None else next(iter(columns))
        if key not in columns:
            raise ValueError(f"event {self.name!r}: {source} has no column {key!r}")
        times, values = columns[key]
        keep = ~np.isnan(values)
        return times[keep], values[keep]

    def apply(self, adapter) -> None:
        """Set the window and point the gage at the event's rain (after open, before
        initialize).

        The gage's rain interval becomes the shortest spacing between records.
        """
        from openswmm.engine import GageRainType, catalog

        times, values = self.rain(Path(adapter.inp_path).parent)
        adapter.write("options.start_datetime", None, self.start)
        adapter.write("options.report_start_datetime", None, self.start)
        adapter.write("options.end_datetime", None, self.end)
        series_id = f"event:{self.name}"
        table = catalog.resolve(adapter.solver, "tables", None).add_timeseries(series_id)
        for when, value in zip(times.astype("datetime64[s]").tolist(), values, strict=True):
            table.add(when, float(value))
        gage = catalog.resolve(adapter.solver, "gage", self.gage)
        gage.set_timeseries(series_id)
        gage.rain_type = GageRainType[self.rain_format]
        if len(times) > 1:
            spacing = np.diff(times.astype("datetime64[s]")).astype(int)
            gage.set_rain_interval(float(spacing[spacing > 0].min()))


def aggregate(values: Sequence[float], weights: Sequence[float], rule: str) -> float:
    """One value from a term's per-event values, by L{AGGREGATIONS} rule.

    C{weighted_mean} is C{sum(w * v) / sum(w)}; the other rules ignore the weights.
    """
    v = np.asarray(values, dtype=float)
    if rule == "sum":
        return float(v.sum())
    if rule == "mean":
        return float(v.mean())
    if rule == "max":
        return float(v.max())
    w = np.asarray(weights, dtype=float)
    return float((w * v).sum() / w.sum())
