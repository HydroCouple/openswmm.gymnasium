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

"""Control strategies for OpenSWMM operational optimization.

All strategies implement the L{Controller} protocol so the same env and the
same NSGA-II job can drive any of them:

  - L{MarketController} — reactive agent-based capacity market (Phase 1).
  - L{ControlCurveController} — reactive piecewise-linear breakpoint curves
    whose static parameters the optimizer tunes.
  - L{ScheduleController} — open-loop per-structure setting schedule.
  - C{MPCController} — receding-horizon re-optimization (Phase 2, forthcoming).

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

from openswmm_gymnasium.control.base import Controller
from openswmm_gymnasium.control.control_curve import (
    ControlCurveController,
    ControlCurveMetricReader,
    ControlCurvePolicy,
    CurveAsset,
    project_monotonic,
    pwl_interp,
)
from openswmm_gymnasium.control.market import (
    PID,
    MarketController,
    aggregate_prices,
    price_from_curve,
)
from openswmm_gymnasium.control.metrics import MarketMetricReader
from openswmm_gymnasium.control.schedule import ScheduleController

__all__ = [
    "PID",
    "Controller",
    "ControlCurveController",
    "ControlCurveMetricReader",
    "ControlCurvePolicy",
    "CurveAsset",
    "MarketController",
    "MarketMetricReader",
    "ScheduleController",
    "aggregate_prices",
    "price_from_curve",
    "project_monotonic",
    "pwl_interp",
]
