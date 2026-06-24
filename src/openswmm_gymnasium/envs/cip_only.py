"""
L{SwmmCIPEnv} — design-only Gymnasium environment.

Single-step contextual-bandit env: the agent picks a design action,
the env simulates the entire SWMM episode under that design and
returns the cumulative cost as the terminal reward. This is the form
most amenable to NSGA-II / NSGA-III / similar pareto-search algorithms
via the L{openswmm_gymnasium.scoring.adapters.platypus_adapter} (plan §5.4).

Lifecycle per env step:

  1. C{reset()} closes any prior solver and returns a zero observation
     (the agent has nothing to act on yet).
  2. C{step(action)} opens a fresh solver, applies the design action
     between L{SolverAdapter.open} and L{SolverAdapter.initialize},
     runs the full simulation, and returns the final observation,
     cumulative reward, C{terminated=True}, C{truncated=False}, and
     an C{info} dict with per-term breakdowns.

The action space is C{spaces.Dict({"design": Dict({...}),
"runtime": Dict({})})} — the runtime portion is empty per the plan §3
contract.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from openswmm_gymnasium._engine import SolverAdapter
from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.rewards import FloodingVolume, RewardTerm
from openswmm_gymnasium.spaces.design import DesignActionFactory

PathLike = str | os.PathLike

_SECONDS_PER_DAY = 86400.0


class SwmmCIPEnv(gym.Env):
    """Single-step CIP environment for design-only optimisation.

    @ivar metadata: Gymnasium metadata (no rendering; viz §5.5 consumes
        recorded trajectories).
    @ivar action_space: Dict with non-empty C{"design"} and empty
        C{"runtime"}.
    @ivar observation_space: Flat Box from the supplied observation
        builder.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        inp_path: PathLike,
        *,
        design_factories: Sequence[DesignActionFactory],
        observation_builder: ObservationBuilder | None = None,
        reward_terms: Sequence[RewardTerm] | None = None,
        rpt_path: PathLike | None = None,
        out_path: PathLike | None = None,
    ) -> None:
        """
        @param inp_path: Path to the SWMM input file driving the
            evaluation.
        @type inp_path: str or os.PathLike
        @param design_factories: One or more CIP factories whose
            combined L{gymnasium.spaces.Space} forms C{action_space["design"]}.
        @type design_factories: sequence of L{DesignActionFactory}
        @param observation_builder: Builder describing the observation
            feature set evaluated at the end of each simulation.
        @type observation_builder: L{ObservationBuilder}
        @param reward_terms: Reward terms aggregated over the simulation.
            Defaults to a single L{FloodingVolume} term.
        @type reward_terms: sequence of L{RewardTerm} or C{None}
        @param rpt_path: Optional report file path.
        @type rpt_path: str, os.PathLike, or C{None}
        @param out_path: Optional binary output file path.
        @type out_path: str, os.PathLike, or C{None}
        @raise ValueError: If C{design_factories} is empty or
            C{observation_builder} is C{None}.
        """
        super().__init__()

        if not design_factories:
            raise ValueError("design_factories must not be empty")
        if observation_builder is None:
            raise ValueError("observation_builder is required")

        self._inp_path = str(inp_path)
        self._rpt_path = None if rpt_path is None else str(rpt_path)
        self._out_path = None if out_path is None else str(out_path)
        self._design_factories: list[DesignActionFactory] = list(design_factories)
        self._observation_builder = observation_builder
        self._reward_terms: list[RewardTerm] = list(
            reward_terms if reward_terms is not None else [FloodingVolume()]
        )

        # ---- Spaces ---------------------------------------------------
        design_subspaces: dict[str, spaces.Space] = {
            f.name: f.space for f in self._design_factories
        }
        self.action_space = spaces.Dict(
            {
                "design": spaces.Dict(design_subspaces),
                "runtime": spaces.Dict({}),  # plan §3 contract; empty here
            }
        )
        self.observation_space = self._observation_builder.space()

        # ---- Per-episode state ---------------------------------------
        self._adapter: SolverAdapter | None = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the env. Returns a zero observation; the agent acts next.

        @param seed: Optional Gymnasium seed.
        @type seed: int or C{None}
        @param options: Reserved; currently ignored.
        @type options: dict or C{None}
        @return: C{(obs, info)} per Gymnasium 1.x. C{obs} is a zero
            vector matching L{observation_space}.
        @rtype: tuple
        """
        super().reset(seed=seed)
        if self._adapter is not None:
            self._adapter.close()
            self._adapter = None
        zero_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        info: dict[str, Any] = {"phase": "awaiting_design"}
        return zero_obs, info

    def step(self, action: dict[str, Any]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply the design action and run a full simulation.

        @param action: Dict matching L{action_space}. Only the
            C{"design"} subdict is used.
        @type action: dict
        @return: C{(final_obs, reward, True, False, info)} — the env is
            a single-step contextual bandit.
        @rtype: tuple
        """
        design = action["design"]

        adapter = SolverAdapter(self._inp_path, self._rpt_path, self._out_path)
        adapter.open()
        # Apply design factories BEFORE initialize so the engine picks up
        # the overridden values during data-structure setup.
        for f in self._design_factories:
            f.bind(adapter)
            f.apply(adapter, design[f.name])
        adapter.initialize()

        # Bind observation + reward against the initialized solver.
        self._observation_builder.bind(adapter)
        for term in self._reward_terms:
            term.bind(adapter)
            term.reset()

        # Run the whole simulation, accumulating reward.
        prev_elapsed_days = adapter.elapsed
        components: dict[str, float] = {t.name: 0.0 for t in self._reward_terms}
        cost = 0.0
        while adapter.is_running:
            adapter.step()
            elapsed = adapter.elapsed
            dt_seconds = (elapsed - prev_elapsed_days) * _SECONDS_PER_DAY
            prev_elapsed_days = elapsed
            # The engine resets ``elapsed`` to 0 on the final step; the negative
            # dt would flip reward-term signs, so drop it.
            if dt_seconds < 0.0:
                dt_seconds = 0.0
            for term in self._reward_terms:
                c = float(term.step(adapter, dt_seconds))
                components[term.name] += c
                if term.direction == "minimize":
                    cost += c
                else:
                    cost -= c
        reward = -cost

        final_obs = self._observation_builder.collect(adapter)

        # Wind down so .rpt / .out land on disk before the next episode.
        try:
            adapter.end()
            adapter.report()
        except Exception:
            pass
        adapter.close()

        info: dict[str, Any] = {
            "phase": "design_evaluated",
            "reward_components": components,
            "design_action": design,
        }
        return final_obs, reward, True, False, info

    def close(self) -> None:
        """Close any held solver."""
        if self._adapter is not None:
            self._adapter.close()
            self._adapter = None
