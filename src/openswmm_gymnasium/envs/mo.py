"""
Multi-objective Gymnasium environments.

Plan §2.1 + §5.3. Returns vector reward (one component per reward term)
instead of the scalar-summed reward the single-objective envs return.
The agent sees per-step vector reward; at episode termination,
C{info["mo_score"]} is populated with the single-point normalised
hypervolume of the episode's cumulative cost vector against a
user-supplied L{ideal_point} and L{reference_point}.

If L{mo_gymnasium} is installed, L{SwmmMORTCEnv} additionally subclasses
L{mo_gymnasium.MOEnv} so MORL-Baselines and similar agents recognise it
out of the box. When L{mo_gymnasium} is absent, the env still works as
a plain L{gymnasium.Env} returning a numpy-array reward (any
MO-Gymnasium-API-compatible consumer will accept it).

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from gymnasium import spaces

from openswmm_gymnasium.envs.base import SwmmRTCEnv
from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.rewards import RewardTerm
from openswmm_gymnasium.scoring import normalized_hypervolume
from openswmm_gymnasium.spaces.runtime import OrificeSetting

# Soft-import mo_gymnasium so the package works without it.
try:
    from mo_gymnasium import MOEnv as _MOEnvBase  # type: ignore

    _HAS_MO_GYMNASIUM = True
except ImportError:  # pragma: no cover - exercised only when missing
    _MOEnvBase = None
    _HAS_MO_GYMNASIUM = False


class SwmmMORTCEnv(SwmmRTCEnv):
    """Multi-objective RTC environment returning vector reward.

    Each component of the returned reward vector is the per-step
    contribution of one entry in C{reward_terms}, with sign flipping
    applied per term (C{minimize} → negated, C{maximize} → kept) so
    that B{higher is better} across all components — matching the
    Gymnasium convention.

    At episode termination, C{info["mo_score"]} carries the single-point
    normalised hypervolume of the episode's cumulative B{cost} vector
    (i.e. before the higher-is-better sign flip) against
    C{(ideal_point, reference_point)}. The resulting score is in
    C{[0, 1]}.

    @ivar reward_space: A L{gymnasium.spaces.Box} of shape
        C{(n_objectives,)} for MO-Gymnasium compatibility.
    """

    def __init__(
        self,
        *args: Any,
        ideal_point: Sequence[float] | None = None,
        reference_point: Sequence[float] | None = None,
        runtime_factories: Sequence[OrificeSetting] | None = None,
        observation_builder: ObservationBuilder | None = None,
        reward_terms: Sequence[RewardTerm] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        @param ideal_point: Per-objective best-case (lowest cost / most
            beneficial) value used for hypervolume normalisation. Must
            match the order of C{reward_terms}. Required.
        @type ideal_point: sequence of float
        @param reference_point: Per-objective nadir / worst-case value
            used for hypervolume normalisation. Must match the order of
            C{reward_terms}. Required.
        @type reference_point: sequence of float
        @raise ValueError: If C{reward_terms} is empty, or
            C{ideal_point} / C{reference_point} mismatch the term count
            or each other.
        """
        if not reward_terms:
            raise ValueError("SwmmMORTCEnv requires at least one reward term")
        if ideal_point is None or reference_point is None:
            raise ValueError("SwmmMORTCEnv requires both ideal_point and reference_point")

        super().__init__(
            *args,
            runtime_factories=runtime_factories,
            observation_builder=observation_builder,
            reward_terms=reward_terms,
            **kwargs,
        )

        n = len(self._reward_terms)
        if len(ideal_point) != n or len(reference_point) != n:
            raise ValueError(
                f"ideal_point/reference_point lengths must match reward_terms count ({n})"
            )

        self._ideal = np.asarray(ideal_point, dtype=np.float64)
        self._ref = np.asarray(reference_point, dtype=np.float64)
        if np.any(self._ref <= self._ideal):
            raise ValueError(
                "reference_point must be strictly greater than ideal_point in every dimension"
            )

        # Vector reward space — used by MO-Gymnasium and analogue agents.
        self.reward_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n,),
            dtype=np.float32,
        )

        self._objective_names: list[str] = [t.name for t in self._reward_terms]
        self._cumulative_cost: np.ndarray = np.zeros(n, dtype=np.float64)

    # ------------------------------------------------------------------
    # Gymnasium API overrides
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset episode state and cumulative cost accumulator."""
        obs, info = super().reset(seed=seed, options=options)
        self._cumulative_cost[:] = 0.0
        return obs, info

    def step(
        self, action: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        """Same lifecycle as L{SwmmRTCEnv.step} but returns vector reward.

        @return: Standard 5-tuple with C{reward} as a 1-D
            L{numpy.ndarray} of shape C{(n_objectives,)}.
        """
        obs, _, terminated, truncated, info = super().step(action)

        components = info["reward_components"]
        # Vector cost (positive = bad for minimize terms).
        cost_vec = np.array([components[name] for name in self._objective_names], dtype=np.float64)
        # Higher-is-better sign convention per term direction.
        signs = np.array(
            [-1.0 if t.direction == "minimize" else 1.0 for t in self._reward_terms],
            dtype=np.float64,
        )
        reward_vec = (signs * cost_vec).astype(np.float32)

        self._cumulative_cost += np.where(signs < 0, cost_vec, -cost_vec)

        if terminated:
            # Single-point HV in normalised space. Plan §5.3: result in [0, 1].
            mo_score = normalized_hypervolume(
                self._cumulative_cost.reshape(1, -1),
                self._ideal,
                self._ref,
            )
            info["mo_score"] = float(mo_score)
            info["cumulative_cost"] = self._cumulative_cost.copy()

        return obs, reward_vec, terminated, truncated, info


# ---------------------------------------------------------------------------
# MO-Gymnasium interface mixing (optional)
# ---------------------------------------------------------------------------

if _HAS_MO_GYMNASIUM:  # pragma: no cover - exercised only when installed
    # Inject MOEnv into the MRO so isinstance(env, MOEnv) is True.
    class SwmmMORTCEnv(SwmmMORTCEnv, _MOEnvBase):  # type: ignore[no-redef]
        """L{SwmmMORTCEnv} additionally subclassing L{mo_gymnasium.MOEnv}."""

        pass
