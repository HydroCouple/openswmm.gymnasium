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
openswmm_gymnasium.spaces
=========================

Action-space factories. Plan §3.

Each factory is a small class exposing:

  - A C{.name} property naming its key in the runtime/design Dict.
  - A C{.space} property returning the L{gymnasium.spaces.Space} the
    factory contributes.
  - A C{.bind(adapter)} method called once after L{SolverAdapter.open}
    to resolve symbolic IDs (link names, node names) to engine indices.
  - An C{.apply(adapter, value)} method that writes the sampled action
    value into the engine.

Runtime (RTC) factories live in L{openswmm_gymnasium.spaces.runtime}
and are applied B{every step}.

Design (CIP) factories live in L{openswmm_gymnasium.spaces.design} and
are applied B{once per episode} between L{SolverAdapter.open} and
L{SolverAdapter.initialize}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from openswmm_gymnasium.spaces.control_curve import ControlCurvePolicySpace
from openswmm_gymnasium.spaces.design import (
    DesignActionFactory,
    LIDPlacement,
    LinkDiameter,
    LinkLength,
    LinkRoughness,
    NodeMaxDepth,
    RDIIUnitHydrograph,
    StorageVolume,
    SubcatchGWOutflowCoeff,
)
from openswmm_gymnasium.spaces.policy import MarketPolicySpace, PolicyParam
from openswmm_gymnasium.spaces.runtime import NodeLateralInflow, OrificeSetting
from openswmm_gymnasium.spaces.schedule import SchedulePolicySpace

__all__ = [
    "OrificeSetting",
    "NodeLateralInflow",
    "DesignActionFactory",
    "LinkRoughness",
    "LinkLength",
    "LinkDiameter",
    "NodeMaxDepth",
    "SubcatchGWOutflowCoeff",
    "StorageVolume",
    "LIDPlacement",
    "RDIIUnitHydrograph",
    "MarketPolicySpace",
    "PolicyParam",
    "SchedulePolicySpace",
    "ControlCurvePolicySpace",
]
