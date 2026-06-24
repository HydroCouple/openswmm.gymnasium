"""Typed configuration objects for OpenSWMM control strategies.

Currently exposes the agent-based capacity-market configuration consumed by the
reactive L{openswmm_gymnasium.control.MarketController} and tuned by the
NSGA-II optimizer. This is the single source of truth for the market-config
shape; the MCP skill ships the equivalent JSON Schema for documentation.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

from openswmm_gymnasium.config.market_config import (
    Agent,
    CostCurve,
    GlobalConstraints,
    MarketConfig,
    Meta,
    Objectives,
    PIDConfig,
    RouteConstraints,
    TradeRoute,
)

__all__ = [
    "Agent",
    "CostCurve",
    "GlobalConstraints",
    "MarketConfig",
    "Meta",
    "Objectives",
    "PIDConfig",
    "RouteConstraints",
    "TradeRoute",
]
