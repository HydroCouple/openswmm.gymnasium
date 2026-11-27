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

"""Event sets: one candidate scored on several storms, each as its own run.

The tests run C{tests/data/site_drainage_example.inp} with three storms written as CSV
sidecars; each event's run is compared with a run of the model edited to hold that storm.
Files land in C{tests/_output/events/}.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from pydantic import ValidationError  # noqa: E402

from openswmm_gymnasium.events import aggregate  # noqa: E402
from openswmm_gymnasium.optimize import evaluate_decisions  # noqa: E402
from openswmm_gymnasium.optimize.manifest import sidecar_files  # noqa: E402
from openswmm_gymnasium.spec.config import EnvConfig, ObservationSpec, build_env  # noqa: E402
from openswmm_gymnasium.spec.errors import SpecError  # noqa: E402

_DATA = Path(__file__).resolve().parents[1] / "data"
_OUTPUT = Path(__file__).resolve().parents[1] / "_output" / "events"
_STEP = timedelta(minutes=5)

#: name -> (window start, window end, rain start, intensities in in/h every five minutes)
STORMS = {
    "small": (
        datetime(1998, 1, 1, 1),
        datetime(1998, 1, 1, 4),
        datetime(1998, 1, 1, 1, 10),
        [0.2, 0.6, 1.0, 0.8, 0.4, 0.1],
    ),
    "medium": (
        datetime(1998, 1, 2),
        datetime(1998, 1, 2, 3),
        datetime(1998, 1, 2, 0, 30),
        [0.4, 1.2, 2.8, 3.6, 2.2, 1.1, 0.6, 0.3],
    ),
    "large": (
        datetime(1998, 1, 3, 6),
        datetime(1998, 1, 3, 10),
        datetime(1998, 1, 3, 6, 20),
        [1.0, 2.5, 4.0, 6.5, 5.0, 3.0, 2.0, 1.0, 0.5, 0.2],
    ),
}
DECISIONS = {"subcatchment.width": [900.0, 400.0]}


def _rain(name: str) -> list[tuple[datetime, float]]:
    _start, _end, first, values = STORMS[name]
    return [(first + i * _STEP, v) for i, v in enumerate(values)]


def _folder(name: str) -> Path:
    folder = _OUTPUT / name
    shutil.rmtree(folder, ignore_errors=True)
    (folder / "rain").mkdir(parents=True)
    shutil.copy(_DATA / "site_drainage_example.inp", folder / "model.inp")
    for storm in STORMS:
        rows = "".join(f"{t.isoformat()},{v}\n" for t, v in _rain(storm))
        (folder / "rain" / f"{storm}.csv").write_text("time,rain\n" + rows)
    return folder


def _config(inp: Path, rule: str = "mean", events: bool = True) -> EnvConfig:
    return EnvConfig(
        env_type="cip",
        inp_path=str(inp),
        observations=ObservationSpec(node_depths=["J1"]),
        design_factories=[
            {
                "kind": "field_design",
                "params": {
                    "path": "subcatchment.width",
                    "ids": ["S1", "S2"],
                    "low": 100.0,
                    "high": 3000.0,
                },
            }
        ],
        reward_terms=[
            {"kind": "flooding_volume"},
            {"kind": "peak_outflow", "params": {"link_ids": ["C11"]}},
        ],
        events=[
            {
                "name": storm,
                "series": f"rain/{storm}.csv",
                "gage": "RainGage",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "weight": weight,
                "return_period": period,
            }
            for (storm, (start, end, _, _)), weight, period in zip(
                STORMS.items(), (0.5, 0.3, 0.2), (1.0, 5.0, 25.0), strict=True
            )
        ]
        if events
        else [],
        event_aggregation=rule,
    )


def _edited_model(folder: Path, storm: str) -> Path:
    """The model edited to hold one storm: its window, and the gage on the storm's series."""
    start, end, _, _ = STORMS[storm]
    text = (folder / "model.inp").read_text()
    text = re.sub(
        r"RainGage\s+VOLUME\s+0:05\s+1.0\s+TIMESERIES 2-yr",
        "RainGage         INTENSITY 0:05     1.0      TIMESERIES STORM",
        text,
    )
    for key, when in (
        ("START_DATE", f"{start:%m/%d/%Y}"),
        ("START_TIME", f"{start:%H:%M:%S}"),
        ("REPORT_START_DATE", f"{start:%m/%d/%Y}"),
        ("REPORT_START_TIME", f"{start:%H:%M:%S}"),
        ("END_DATE", f"{end:%m/%d/%Y}"),
        ("END_TIME", f"{end:%H:%M:%S}"),
    ):
        text = re.sub(rf"^{key}\s+\S+", f"{key:<20} {when}", text, flags=re.M)
    rows = "".join(f"STORM {t:%m/%d/%Y} {t:%H:%M} {v}\n" for t, v in _rain(storm))
    text = text.replace("[TIMESERIES]\n", "[TIMESERIES]\n" + rows, 1)
    path = folder / f"edited_{storm}.inp"
    path.write_text(text)
    return path


def test_each_event_runs_as_the_model_edited_to_hold_its_storm():
    folder = _folder("edited")
    scored = evaluate_decisions(_config(folder / "model.inp"), DECISIONS)
    assert scored["steps"] == 3
    for storm in STORMS:
        alone = evaluate_decisions(_config(_edited_model(folder, storm), events=False), DECISIONS)[
            "metrics"
        ]
        assert {f"{n}@{storm}": v for n, v in alone.items()} == {
            k: v for k, v in scored["metrics"].items() if k.endswith(f"@{storm}")
        }
    peaks = [scored["metrics"][f"peak_outflow@{s}"] for s in STORMS]
    assert peaks == sorted(peaks) and peaks[0] < peaks[-1]  # bigger storms, bigger peaks


@pytest.mark.parametrize("rule", ["sum", "mean", "max", "weighted_mean"])
def test_a_term_is_its_per_event_values_by_the_rule(rule):
    folder = _folder(f"rule_{rule}")
    scored = evaluate_decisions(_config(folder / "model.inp", rule), DECISIONS)
    for term in ("flooding_volume", "peak_outflow"):
        per_event = [scored["metrics"][f"{term}@{s}"] for s in STORMS]
        expected = aggregate(per_event, [0.5, 0.3, 0.2], rule)
        assert scored["metrics"][term] == expected
        assert scored["objectives"][term] == expected  # both minimized
    weighted = sum(w * v for w, v in zip([0.5, 0.3, 0.2], per_event, strict=True))
    assert aggregate(per_event, [0.5, 0.3, 0.2], "weighted_mean") == pytest.approx(weighted)


def test_an_event_set_names_its_rain_as_sidecars_and_is_validated():
    folder = _folder("validation")
    config = _config(folder / "model.inp")
    assert sidecar_files(config) == {f"rain/{s}.csv": folder / "rain" / f"{s}.csv" for s in STORMS}
    payload = config.model_dump(mode="json")
    with pytest.raises(ValidationError, match="unique"):
        EnvConfig.model_validate(payload | {"events": payload["events"][:1] * 2})
    late = dict(payload["events"][0], end=payload["events"][0]["start"])
    with pytest.raises(ValidationError, match="ends before it starts"):
        EnvConfig.model_validate(payload | {"events": [late]})
    rtc = payload | {
        "env_type": "rtc",
        "design_factories": [],
        "runtime_factories": [{"kind": "orifice_setting", "params": {"link_ids": ["C11"]}}],
    }
    with pytest.raises(ValidationError, match="takes none"):
        EnvConfig.model_validate(rtc)
    with pytest.raises(SpecError, match="No event 'huge'"):
        build_env(config, event="huge")
