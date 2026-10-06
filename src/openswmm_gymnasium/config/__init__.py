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

"""Typed configuration objects for OpenSWMM control strategies.

Currently exposes the agent-based capacity-market configuration consumed by the
reactive L{openswmm_gymnasium.control.MarketController} and tuned by the
NSGA-II optimizer. This is the single source of truth for the market-config
shape; the MCP skill ships the equivalent JSON Schema for documentation.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
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
