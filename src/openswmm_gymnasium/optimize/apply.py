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
Write a chosen design onto a model.

L{apply_decisions} writes an evaluated design onto an opened model through the env
config's own design factories, with the values an evaluation of it used (C{float64},
clipped to each factory's range), so a model written afterwards is the design that was
scored, for every design kind.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

from typing import Any

from openswmm_gymnasium._engine import SolverAdapter
from openswmm_gymnasium.optimize.space import (
    decision_values,
    design_dimensions,
    join_decisions,
    split_vector,
)
from openswmm_gymnasium.spec.config import EnvConfig
from openswmm_gymnasium.spec.errors import ErrorCode, SpecError


def apply_decisions(
    env_config: EnvConfig, decisions: dict[str, Any], adapter: SolverAdapter
) -> list[dict[str, Any]]:
    """Write a design onto an opened, not yet initialized model.

    @param decisions: Dimension key to values (an evaluation record's C{decisions}).
    @param adapter: An adapter on the opened model (it may wrap a solver it does not own).
    @return: One C{{"kind", "label", "element_id", "value"}} per component, C{element_id}
        C{None} where a component is not one element.
    @raise SpecError: The env config searches a policy, not a design, or the decisions
        do not match its dimensions.
    @rtype: list of dict
    """
    if env_config.env_type not in ("cip", "joint"):
        raise SpecError(
            f"[{ErrorCode.VALIDATION_ERROR}] A '{env_config.env_type}' config tunes a "
            "policy, not the model; there is no design to write onto it."
        )
    dims = design_dimensions(env_config)
    vector = join_decisions(dims, decisions)
    actions, values = split_vector(dims, vector), decision_values(dims, vector)  # as evaluated
    applied: list[dict[str, Any]] = []
    for spec, dim in zip(env_config.design_factories, dims, strict=True):
        factory = spec.construct()
        factory.bind(adapter)
        factory.apply(adapter, actions[dim.key])
        for label, value in zip(dim.labels, values[dim.key], strict=True):
            _kind, sep, element = label.partition(":")
            applied.append(
                {
                    "kind": spec.kind,
                    "label": label,
                    "element_id": element if sep else None,
                    "value": float(value),
                }
            )
    return applied
