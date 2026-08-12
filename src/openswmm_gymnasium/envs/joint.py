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
L{SwmmJointCIPRTCEnv} — full hybrid Gymnasium environment.

Combines L{openswmm_gymnasium.envs.SwmmRTCEnv}'s per-step runtime
control with L{openswmm_gymnasium.envs.SwmmCIPEnv}'s per-episode
design action. The design action is sampled at C{reset()} time (or
supplied via C{options["design_action"]} for deterministic evaluation),
applied between L{SolverAdapter.open} and L{SolverAdapter.initialize},
and B{locked for the remainder of the episode}. The agent then drives
the RTC portion via C{step()}.

C{action["design"]} passed to C{step()} is B{ignored} — the design is
fixed at reset. This matches the plan §3 contract that the action
space always carries both keys without forcing the agent to repeat
the design every step.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
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
from openswmm_gymnasium.spaces.design import DesignActionFactory
from openswmm_gymnasium.spaces.runtime import OrificeSetting

PathLike = str | os.PathLike

_SECONDS_PER_DAY = 86400.0


class SwmmJointCIPRTCEnv(gym.Env):
    """Hybrid CIP + RTC environment.

    @ivar metadata: Gymnasium metadata.
    @ivar action_space: Dict with non-empty C{"design"} and C{"runtime"}.
    @ivar observation_space: Flat Box from the supplied observation
        builder.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        inp_path: PathLike,
        *,
        design_factories: Sequence[DesignActionFactory],
        runtime_factories: Sequence[OrificeSetting] | None = None,
        observation_builder: ObservationBuilder | None = None,
        reward_terms: Sequence[RewardTerm] | None = None,
        control_interval_steps: int = 1,
        max_episode_steps: int | None = None,
        rpt_path: PathLike | None = None,
        out_path: PathLike | None = None,
    ) -> None:
        """
        @param inp_path: Path to the SWMM input file.
        @type inp_path: str or os.PathLike
        @param design_factories: CIP factories (applied once at reset).
        @type design_factories: sequence of L{DesignActionFactory}
        @param runtime_factories: RTC factories (applied each step).
        @type runtime_factories: sequence of runtime factory or C{None}
        @param observation_builder: Builder describing the observation.
        @type observation_builder: L{ObservationBuilder}
        @param reward_terms: Reward terms aggregated each step. Defaults
            to a single L{FloodingVolume} term.
        @type reward_terms: sequence of L{RewardTerm} or C{None}
        @param control_interval_steps: Routing steps per env step.
        @type control_interval_steps: int
        @param max_episode_steps: Optional truncation horizon.
        @type max_episode_steps: int or C{None}
        @param rpt_path: Optional report file path.
        @type rpt_path: str, os.PathLike, or C{None}
        @param out_path: Optional binary output file path.
        @type out_path: str, os.PathLike, or C{None}
        @raise ValueError: If C{design_factories} is empty,
            C{observation_builder} is C{None}, or
            C{control_interval_steps < 1}.
        """
        super().__init__()

        if not design_factories:
            raise ValueError("design_factories must not be empty")
        if observation_builder is None:
            raise ValueError("observation_builder is required")
        if control_interval_steps < 1:
            raise ValueError("control_interval_steps must be >= 1")

        self._inp_path = str(inp_path)
        self._rpt_path = None if rpt_path is None else str(rpt_path)
        self._out_path = None if out_path is None else str(out_path)
        self._design_factories: list[DesignActionFactory] = list(design_factories)
        self._runtime_factories: list[OrificeSetting] = list(runtime_factories or [])
        self._observation_builder = observation_builder
        self._reward_terms: list[RewardTerm] = list(
            reward_terms if reward_terms is not None else [FloodingVolume()]
        )
        self._control_interval_steps = control_interval_steps
        self._max_episode_steps = max_episode_steps

        # ---- Spaces ---------------------------------------------------
        design_subspaces: dict[str, spaces.Space] = {
            f.name: f.space for f in self._design_factories
        }
        runtime_subspaces: dict[str, spaces.Space] = {
            f.name: f.space for f in self._runtime_factories
        }
        self.action_space = spaces.Dict(
            {
                "design": spaces.Dict(design_subspaces),
                "runtime": spaces.Dict(runtime_subspaces),
            }
        )
        self.observation_space = self._observation_builder.space()

        # ---- Per-episode state ---------------------------------------
        self._adapter: SolverAdapter | None = None
        self._prev_elapsed_days: float = 0.0
        self._env_step_count: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode under a freshly-applied design.

        If C{options["design_action"]} is supplied, that design is used
        verbatim; otherwise the design is sampled from
        C{action_space["design"]}.

        @param seed: Optional Gymnasium seed.
        @type seed: int or C{None}
        @param options: Optional dict; may include C{"design_action"}
            (a dict matching C{action_space["design"]}).
        @type options: dict or C{None}
        @return: C{(obs, info)} per Gymnasium 1.x. C{info["design_action"]}
            records the design that was applied.
        @rtype: tuple
        """
        super().reset(seed=seed)

        if self._adapter is not None:
            self._adapter.close()
            self._adapter = None

        design_action = (
            options.get("design_action")
            if options and "design_action" in options
            else self.action_space["design"].sample()
        )

        self._adapter = SolverAdapter(self._inp_path, self._rpt_path, self._out_path)
        self._adapter.open()
        for f in self._design_factories:
            f.bind(self._adapter)
            f.apply(self._adapter, design_action[f.name])
        self._adapter.initialize()
        self._adapter.start()

        # Bind runtime + observation + rewards against the initialized
        # solver.
        for f in self._runtime_factories:
            f.bind(self._adapter)
        self._observation_builder.bind(self._adapter)
        for term in self._reward_terms:
            term.bind(self._adapter)
            term.reset()

        self._prev_elapsed_days = self._adapter.elapsed
        self._env_step_count = 0

        obs = self._observation_builder.collect(self._adapter)
        info: dict[str, Any] = {
            "elapsed_days": self._adapter.elapsed,
            "design_action": design_action,
        }
        return obs, info

    def step(self, action: dict[str, Any]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the simulation by C{control_interval_steps}.

        C{action["design"]} is ignored — the design is fixed at reset.
        C{action["runtime"]} is applied via the runtime factories.

        @param action: Dict matching L{action_space}.
        @type action: dict
        @return: Gymnasium 1.x 5-tuple.
        @rtype: tuple
        @raise RuntimeError: If called before L{reset}.
        """
        if self._adapter is None:
            raise RuntimeError("step() called before reset()")

        runtime_action = action.get("runtime", {})
        for f in self._runtime_factories:
            f.apply(self._adapter, runtime_action[f.name])

        for _ in range(self._control_interval_steps):
            if not self._adapter.is_running:
                break
            self._adapter.step()

        elapsed_days = self._adapter.elapsed
        dt_seconds = (elapsed_days - self._prev_elapsed_days) * _SECONDS_PER_DAY
        self._prev_elapsed_days = elapsed_days
        # The engine resets ``elapsed`` to 0 on the final step that ends the run;
        # the resulting negative dt would flip reward-term signs, so drop it.
        if dt_seconds < 0.0:
            dt_seconds = 0.0

        components: dict[str, float] = {}
        cost = 0.0
        for term in self._reward_terms:
            c = float(term.step(self._adapter, dt_seconds))
            components[term.name] = c
            if term.direction == "minimize":
                cost += c
            else:
                cost -= c
        reward = -cost

        self._env_step_count += 1

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

        if terminated:
            try:
                self._adapter.end()
                self._adapter.report()
            except Exception:
                pass

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the underlying solver. Safe to call multiple times."""
        if self._adapter is not None:
            self._adapter.close()
            self._adapter = None
