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
Search space of an optimisable env config.

The decision vector of a study is the concatenation of every design
factory's flattened action (C{cip}/C{joint} envs) or of every tunable
controller parameter (C{market}/C{schedule}/C{control_curve} envs). Each
group is one L{DesignDimension}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openswmm_gymnasium.spec.config import EnvConfig
from openswmm_gymnasium.spec.errors import ErrorCode, SpecError

#: Env types whose single step runs a whole episode under a tunable controller.
POLICY_ENV_TYPES = ("market", "schedule", "control_curve")


@dataclass(frozen=True)
class DesignDimension:
    """One design factory's (or controller parameter's) slice of the search space.

    @ivar key: Action-space key (the factory's C{name} param, or the parameter label).
    @ivar labels: Per-component labels, C{"<kind>:<element_id>"}.
    @ivar low: Lower bound: a scalar broadcast across components, or a per-component tuple.
    @ivar high: Upper bound, scalar or per-component like C{low}.

    Tuple bounds allow different physical ranges for each component, such as
    storage C{(a,b,c)} or RDII C{(R,dmax,drecov,dinit)}.
    """

    key: str
    labels: tuple[str, ...]
    low: float | tuple[float, ...]
    high: float | tuple[float, ...]

    @property
    def size(self) -> int:
        """Number of components in this dimension group.

        @rtype: int
        """
        return len(self.labels)


def _market_dimensions(env_config: EnvConfig) -> list[DesignDimension]:
    """One single-component dimension per tunable market-controller parameter."""
    from openswmm_gymnasium.config import MarketConfig
    from openswmm_gymnasium.spaces import MarketPolicySpace

    market = MarketConfig.from_dict(env_config.market_config)
    market.validate()
    bounds = (
        {k: tuple(v) for k, v in env_config.policy_bounds.items()}
        if env_config.policy_bounds
        else None
    )
    space = MarketPolicySpace.from_config(market, bounds=bounds, tune_full=env_config.tune_full)
    return [
        DesignDimension(key=p.label, labels=(p.label,), low=p.low, high=p.high)
        for p in space.params
    ]


def _schedule_dimensions(env_config: EnvConfig) -> list[DesignDimension]:
    """One single-component dimension per scheduled structure setting."""
    from openswmm_gymnasium.spaces import SchedulePolicySpace

    lo, hi = env_config.schedule_bounds or [0.0, 1.0]
    space = SchedulePolicySpace(
        env_config.structure_ids, env_config.n_points, low=float(lo), high=float(hi)
    )
    return [
        DesignDimension(
            key=label, labels=(label,), low=float(space.low[i]), high=float(space.high[i])
        )
        for i, label in enumerate(space.labels)
    ]


def _control_curve_dimensions(env_config: EnvConfig) -> list[DesignDimension]:
    """One dimension per control-curve knot setting, labelled C{control_curve/<link>/y[<k>]}."""
    from openswmm_gymnasium.spaces import ControlCurvePolicySpace

    space = ControlCurvePolicySpace.from_params(env_config.policy_factory.params)
    low, high = space.low, space.high
    return [
        DesignDimension(key=label, labels=(label,), low=float(low[i]), high=float(high[i]))
        for i, label in enumerate(space.labels)
    ]


def design_dimensions(env_config: EnvConfig) -> list[DesignDimension]:
    """Labelled search-space dimensions of an env config, in config order.

    For C{"market"}/C{"schedule"}/C{"control_curve"} configs the dimensions
    come from the controller policy space; otherwise from the design factories.

    @param env_config: Config whose factories / policy span the space.
    @type env_config: L{EnvConfig}
    @rtype: list of L{DesignDimension}
    @raise SpecError: C{VALIDATION_ERROR} when the config has no searchable
        static factory (no design_factory and no policy_factory).
    """
    if env_config.env_type == "market":
        return _market_dimensions(env_config)
    if env_config.env_type == "schedule":
        return _schedule_dimensions(env_config)
    if env_config.env_type == "control_curve":
        return _control_curve_dimensions(env_config)
    if not env_config.design_factories:
        raise SpecError(
            f"[{ErrorCode.VALIDATION_ERROR}] Optimization requires an env config "
            "with a searchable static factory: design_factories (env_type 'cip' "
            "or 'joint') or a policy_factory (env_type 'control_curve')."
        )
    dims: list[DesignDimension] = []
    for spec in env_config.design_factories:
        # Constructing the factory needs no solver (its Box is built in __init__) and gives
        # the true flat shape and per-component bounds, including factories whose
        # component count and bounds vary (storage coeffs, LID sizing+type, RDII R+IA).
        factory = spec.construct()
        space = factory.space
        lo = np.asarray(space.low, dtype=float).ravel()
        hi = np.asarray(space.high, dtype=float).ravel()
        size = int(lo.shape[0])
        params = spec.params
        ids = (
            params.get("link_ids")
            or params.get("node_ids")
            or params.get("subcatch_ids")
            or params.get("ids")
            or []
        )
        if len(ids) == size:
            labels = tuple(f"{spec.kind}:{eid}" for eid in ids)
        else:
            labels = tuple(f"{spec.kind}[{i}]" for i in range(size))
        dims.append(
            DesignDimension(
                key=factory.name,
                labels=labels,
                low=tuple(lo.tolist()),
                high=tuple(hi.tolist()),
            )
        )
    return dims


def flat_bounds(dims: list[DesignDimension]) -> tuple[np.ndarray, np.ndarray]:
    """Flat C{(low, high)} bound vectors across all dimensions.

    @rtype: tuple of two L{numpy.ndarray}
    """
    low = np.concatenate([np.broadcast_to(np.asarray(d.low, dtype=float), (d.size,)) for d in dims])
    high = np.concatenate(
        [np.broadcast_to(np.asarray(d.high, dtype=float), (d.size,)) for d in dims]
    )
    return low, high


def split_vector(dims: list[DesignDimension], vector: np.ndarray) -> dict[str, np.ndarray]:
    """Split a flat decision vector into per-dimension float32 arrays keyed by C{key}.

    @rtype: dict
    """
    payload: dict[str, np.ndarray] = {}
    offset = 0
    for dim in dims:
        payload[dim.key] = np.asarray(vector[offset : offset + dim.size], dtype=np.float32)
        offset += dim.size
    return payload


def objective_directions(env_config: EnvConfig) -> dict[str, str]:
    """Reward-term name to C{"minimize"}/C{"maximize"}, in config order.

    An empty config means the env default, a single all-nodes
    C{flooding_volume} cost.

    @rtype: dict
    """
    if not env_config.reward_terms:
        return {"flooding_volume": "minimize"}
    directions: dict[str, str] = {}
    for spec in env_config.reward_terms:
        term = spec.construct()
        directions[term.name] = term.direction
    return directions
