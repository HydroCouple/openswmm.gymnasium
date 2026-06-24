"""Control strategies for OpenSWMM operational optimization.

All strategies implement the L{Controller} protocol so the same env and the
same NSGA-II job can drive any of them:

  - L{MarketController} — reactive agent-based capacity market (Phase 1).
  - C{MPCController} — receding-horizon re-optimization (Phase 2, forthcoming).

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

from openswmm_gymnasium.control.base import Controller
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
    "MarketController",
    "MarketMetricReader",
    "ScheduleController",
    "aggregate_prices",
    "price_from_curve",
]
