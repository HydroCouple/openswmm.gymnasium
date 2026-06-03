"""
L{SwmmRTCEnv} — minimal runtime-only Gymnasium environment.

Subclasses L{gymnasium.Env} directly (plan §2.1). The design action
portion of the Dict action space is empty; only the runtime portion
is non-trivial. This env drives one or more
L{openswmm_gymnasium.spaces.runtime} factories against an
L{openswmm.engine.Solver}, advancing the simulation
C{control_interval_steps} routing steps per C{step()}.

Per plan §0 #5, all reward terms are framed as cost-to-minimize
internally; the env negates the aggregated cost so that B{higher
reward = better outcome}, matching Gymnasium convention.

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
from openswmm.engine import EngineState

from openswmm_gymnasium._engine import SolverAdapter
from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.rewards import FloodingVolume, RewardTerm
from openswmm_gymnasium.spaces.runtime import OrificeSetting

PathLike = str | os.PathLike

# Seconds per simulation day; the engine reports C{elapsed} in days.
_SECONDS_PER_DAY = 86400.0


class SwmmRTCEnv(gym.Env):
    """Runtime-only SWMM environment for RL.

    The action space is C{spaces.Dict({"design": Dict({}),
    "runtime": Dict({...})})}, conforming to the plan §3 contract that
    every env exposes both top-level keys. C{"design"} is empty for
    this env class.

    The observation space is a flat L{gymnasium.spaces.Box} produced by
    the supplied L{ObservationBuilder}.

    @ivar metadata: Gymnasium metadata (no rendering for now; the
        Plotly viz module §5.5 consumes recorded trajectories, not
        live envs).
    @ivar action_space: Dict of C{"design"} + C{"runtime"}.
    @ivar observation_space: Flat Box.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        inp_path: PathLike,
        *,
        runtime_factories: Sequence[OrificeSetting] | None = None,
        observation_builder: ObservationBuilder | None = None,
        reward_terms: Sequence[RewardTerm] | None = None,
        control_interval_steps: int = 1,
        max_episode_steps: int | None = None,
        rpt_path: PathLike | None = None,
        out_path: PathLike | None = None,
    ) -> None:
        """
        @param inp_path: Path to the SWMM input file driving each episode.
        @type inp_path: str or os.PathLike
        @param runtime_factories: List of runtime action factories.
            Defaults to none, producing an empty runtime Dict.
        @type runtime_factories: sequence of factory or C{None}
        @param observation_builder: Builder describing the observation
            feature set. Required.
        @type observation_builder: L{ObservationBuilder}
        @param reward_terms: Reward terms to aggregate. Defaults to a
            single L{FloodingVolume} term summing over all nodes.
        @type reward_terms: sequence of L{RewardTerm} or C{None}
        @param control_interval_steps: Number of routing steps to advance
            per env C{step()}. Plan §2.2.
        @type control_interval_steps: int
        @param max_episode_steps: If not C{None}, the env reports
            C{truncated=True} after this many env steps even if the
            simulation has not ended.
        @type max_episode_steps: int or C{None}
        @param rpt_path: Optional fixed path for the C{.rpt} file.
            Defaults to a sibling of C{inp_path}.
        @type rpt_path: str, os.PathLike, or C{None}
        @param out_path: Optional fixed path for the C{.out} file.
            Defaults to a sibling of C{inp_path}.
        @type out_path: str, os.PathLike, or C{None}
        @raise ValueError: If C{observation_builder} is C{None}.
        @raise ValueError: If C{control_interval_steps < 1}.
        """
        super().__init__()

        if observation_builder is None:
            raise ValueError("observation_builder is required")
        if control_interval_steps < 1:
            raise ValueError("control_interval_steps must be >= 1")

        self._inp_path = str(inp_path)
        self._rpt_path = None if rpt_path is None else str(rpt_path)
        self._out_path = None if out_path is None else str(out_path)
        self._runtime_factories: list[OrificeSetting] = list(runtime_factories or [])
        self._observation_builder = observation_builder
        self._reward_terms: list[RewardTerm] = list(
            reward_terms if reward_terms is not None else [FloodingVolume()]
        )
        self._control_interval_steps = control_interval_steps
        self._max_episode_steps = max_episode_steps

        # ---- Spaces ---------------------------------------------------
        runtime_subspaces: dict[str, spaces.Space] = {
            f.name: f.space for f in self._runtime_factories
        }
        # Gymnasium forbids empty Dict spaces (``check_env`` rejects them),
        # so we expose only the non-empty action halves. An RTC env carries
        # just the ``"runtime"`` key; ``step`` reads it via ``action.get``.
        self.action_space = spaces.Dict({"runtime": spaces.Dict(runtime_subspaces)})
        self.observation_space = self._observation_builder.space()

        # ---- Per-episode state ---------------------------------------
        self._adapter: SolverAdapter | None = None
        self._prev_elapsed_days: float = 0.0
        self._env_step_count: int = 0
        # Unit system of the loaded model, recorded at reset(). Engine
        # getters return project units, so rewards/observations scaled by
        # physical magnitudes must not silently reuse another system's tuning.
        self._unit_system: str | None = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode.

        Closes any prior solver, opens a fresh one against C{inp_path},
        binds all factories / collectors / reward terms, and returns
        the initial observation.

        @param seed: Optional seed forwarded to L{gymnasium.Env.reset}.
        @type seed: int or C{None}
        @param options: Reserved for future use; currently ignored.
        @type options: dict or C{None}
        @return: Tuple C{(observation, info)} per Gymnasium 1.x.
        @rtype: tuple
        """
        super().reset(seed=seed)

        # Close any prior episode's solver.
        if self._adapter is not None:
            self._adapter.close()
            self._adapter = None

        # Open + initialize + start (start transitions the engine to
        # RUNNING; step() is guarded on that state).
        self._adapter = SolverAdapter(self._inp_path, self._rpt_path, self._out_path)
        self._adapter.open()
        self._adapter.initialize()
        self._adapter.start()

        # Bind all symbolic IDs against the freshly-opened engine.
        for f in self._runtime_factories:
            f.bind(self._adapter)
        self._observation_builder.bind(self._adapter)
        for term in self._reward_terms:
            term.bind(self._adapter)
            term.reset()

        self._prev_elapsed_days = self._adapter.elapsed
        self._env_step_count = 0

        # Record the model's unit system once per episode. If an episode was
        # previously run under a different system, fail loudly rather than
        # silently mis-scaling rewards/observations tuned for the old one.
        current_units = self._adapter.unit_system
        if self._unit_system is not None and current_units != self._unit_system:
            raise RuntimeError(
                f"Model unit system changed across episodes: "
                f"{self._unit_system!r} -> {current_units!r}. Observation "
                f"and reward scaling are unit-dependent; recreate the env."
            )
        self._unit_system = current_units

        obs = self._observation_builder.collect(self._adapter)
        info: dict[str, Any] = {
            "elapsed_days": self._adapter.elapsed,
            "unit_system": self._unit_system,
            "flow_units": self._adapter.flow_units,
        }
        return obs, info

    def step(self, action: dict[str, Any]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the simulation by C{control_interval_steps}.

        @param action: Dict matching L{action_space}.
        @type action: dict
        @return: Tuple C{(observation, reward, terminated, truncated,
            info)} per Gymnasium 1.x.
        @rtype: tuple
        @raise RuntimeError: If called before L{reset}.
        """
        if self._adapter is None:
            raise RuntimeError("step() called before reset()")

        # Apply runtime actions.
        runtime_action = action.get("runtime", {})
        for f in self._runtime_factories:
            f.apply(self._adapter, runtime_action[f.name])

        # Advance.
        for _ in range(self._control_interval_steps):
            if not self._adapter.is_running:
                break
            self._adapter.step()

        # Compute dt in seconds.
        elapsed_days = self._adapter.elapsed
        dt_seconds = (elapsed_days - self._prev_elapsed_days) * _SECONDS_PER_DAY
        self._prev_elapsed_days = elapsed_days

        # Compute reward.
        components: dict[str, float] = {}
        cost = 0.0
        for term in self._reward_terms:
            c = float(term.step(self._adapter, dt_seconds))
            components[term.name] = c
            if term.direction == "minimize":
                cost += c
            else:
                cost -= c
        # Gymnasium convention: higher reward = better. Internal terms
        # are costs-to-minimize, so reward = -cost. Plan §0 #5.
        reward = -cost

        self._env_step_count += 1

        # Termination / truncation.
        terminated = self._adapter.state == EngineState.ENDED or (not self._adapter.is_running)
        truncated = (
            self._max_episode_steps is not None
            and self._env_step_count >= self._max_episode_steps
            and not terminated
        )

        obs = self._observation_builder.collect(self._adapter)
        info: dict[str, Any] = {
            "elapsed_days": elapsed_days,
            "dt_seconds": dt_seconds,
            "reward_components": components,
            "env_step": self._env_step_count,
        }

        # Wind down the engine on terminal step so .rpt / .out are
        # written before the next reset() opens a fresh solver.
        if terminated:
            try:
                self._adapter.end()
                self._adapter.report()
            except Exception:
                # End-of-sim cleanup failures are non-fatal for the
                # reward signal already returned to the agent.
                pass

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the underlying solver. Safe to call multiple times."""
        if self._adapter is not None:
            self._adapter.close()
            self._adapter = None
