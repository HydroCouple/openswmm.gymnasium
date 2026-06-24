"""
L{SwmmControlEnv} — single-step environment for tuning a control strategy.

Like L{openswmm_gymnasium.envs.SwmmCIPEnv}, this is a single-step contextual
bandit whose action is searched by NSGA-II — but the action is a B{controller
policy-parameter vector} (see L{openswmm_gymnasium.spaces.MarketPolicySpace}),
not a model design. On C{step()} the env rebuilds the full controller config
from the vector, runs the entire simulation B{under that controller}, and
returns the operational objective totals as C{info["reward_components"]}.

The env is B{controller-agnostic}: it drives any
L{openswmm_gymnasium.control.base.Controller}. Phase 1 ships the reactive
L{MarketController}; the Phase 2 receding-horizon C{MPCController} drops in
through the same seam.

Lifecycle per env step:

  1. C{reset()} closes any prior solver and returns a zero observation.
  2. C{step(vector)} unflattens the policy vector into a C{MarketConfig}, builds
     the controller + metric reader, opens a fresh solver, and simulates to the
     end — applying the controller's settings every C{control_interval_seconds}
     while accumulating reward terms each routing step. Returns
     C{(final_obs, reward, True, False, info)}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import numpy as np

from openswmm_gymnasium._engine import SolverAdapter
from openswmm_gymnasium.config import MarketConfig
from openswmm_gymnasium.control import MarketController, MarketMetricReader
from openswmm_gymnasium.control.base import Controller
from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.rewards import FloodingVolume, RewardTerm
from openswmm_gymnasium.spaces.policy import MarketPolicySpace

PathLike = str | os.PathLike

_SECONDS_PER_DAY = 86400.0


class SwmmControlEnv(gym.Env):
    """Single-step env that evaluates a controller policy over a full episode.

    @ivar action_space: The policy-parameter L{gymnasium.spaces.Box}.
    @ivar observation_space: Flat Box from the supplied observation builder.
    @ivar policy_space: The L{MarketPolicySpace} mapping vectors to configs.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        inp_path: PathLike,
        *,
        observation_builder: ObservationBuilder,
        reward_terms: Sequence[RewardTerm] | None = None,
        market_config: MarketConfig | None = None,
        policy_space: Any | None = None,
        controller_factory: Callable[[Any], Controller] | None = None,
        metric_reader_factory: Callable[[Any], Any] | None = None,
        control_interval_seconds: float | None = None,
        rpt_path: PathLike | None = None,
        out_path: PathLike | None = None,
    ) -> None:
        """
        Two construction modes:

          - B{Market} (reactive): pass C{market_config}; the env builds a
            L{MarketController} + L{MarketMetricReader} and a default
            L{MarketPolicySpace}. This is the Phase-1 path.
          - B{Generic}: pass C{policy_space} + C{controller_factory} (and an
            optional C{metric_reader_factory}). Each evaluation builds the
            controller from C{policy_space.unflatten(vector)}. An open-loop
            schedule uses this with C{metric_reader_factory=None}.

        @param inp_path: Path to the SWMM input file.
        @param observation_builder: Observation feature set (required).
        @param reward_terms: Operational objective terms aggregated over the
            episode. Defaults to a single L{FloodingVolume}.
        @param market_config: Base (validated) market configuration (market mode).
        @param policy_space: Search space exposing C{.space}/C{.labels}/
            C{.unflatten} (generic mode; defaults from C{market_config}).
        @param controller_factory: C{unflattened -> Controller} (generic mode).
        @param metric_reader_factory: C{unflattened -> reader} or C{None} for
            open-loop controllers.
        @param control_interval_seconds: Control interval; required in generic
            mode, else defaults to the market config's interval.
        @param rpt_path: Optional report file path.
        @param out_path: Optional binary output file path.
        @raise ValueError: On missing observation builder or an incomplete mode.
        """
        super().__init__()
        if observation_builder is None:
            raise ValueError("observation_builder is required")

        if market_config is not None:
            market_config.validate()
            self._base_config = market_config
            self.policy_space = (
                policy_space if policy_space is not None
                else MarketPolicySpace.from_config(market_config)
            )
            self._controller_factory: Callable[[Any], Controller] = (
                lambda spec: MarketController(spec)
            )
            self._metric_reader_factory: Callable[[Any], Any] | None = (
                lambda spec: MarketMetricReader(spec)
            )
            default_interval: float | None = market_config.meta.control_interval_seconds
        else:
            if policy_space is None or controller_factory is None:
                raise ValueError(
                    "SwmmControlEnv requires either market_config, or both "
                    "policy_space and controller_factory"
                )
            self._base_config = None
            self.policy_space = policy_space
            self._controller_factory = controller_factory
            self._metric_reader_factory = metric_reader_factory
            default_interval = None

        if control_interval_seconds is None and default_interval is None:
            raise ValueError(
                "control_interval_seconds is required when market_config is not given"
            )

        self._inp_path = str(inp_path)
        self._rpt_path = None if rpt_path is None else str(rpt_path)
        self._out_path = None if out_path is None else str(out_path)
        self._observation_builder = observation_builder
        self._reward_terms: list[RewardTerm] = list(
            reward_terms if reward_terms is not None else [FloodingVolume()]
        )
        self._control_interval = float(
            control_interval_seconds if control_interval_seconds is not None else default_interval
        )

        self.action_space = self.policy_space.space
        self.observation_space = self._observation_builder.space()
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
        """Reset the env; returns a zero observation (the agent acts next)."""
        super().reset(seed=seed)
        if self._adapter is not None:
            self._adapter.close()
            self._adapter = None
        zero_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return zero_obs, {"phase": "awaiting_policy"}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Evaluate a policy vector over a full simulation.

        @param action: Policy-parameter vector matching L{action_space}.
        @return: C{(final_obs, reward, True, False, info)}; C{info} carries
            C{reward_components} (per-objective totals), the applied policy, and
            the number of control updates.
        """
        spec = self.policy_space.unflatten(np.asarray(action, dtype=np.float64))
        controller: Controller = self._controller_factory(spec)
        metric_reader = (
            self._metric_reader_factory(spec) if self._metric_reader_factory is not None else None
        )

        adapter = SolverAdapter(self._inp_path, self._rpt_path, self._out_path)
        adapter.open()
        adapter.initialize()
        adapter.start()

        self._observation_builder.bind(adapter)
        if metric_reader is not None:
            metric_reader.bind(adapter)
        controller.reset()
        structure_idxs = {
            sid: adapter.links.get_index(sid) for sid in controller.structure_ids
        }
        for term in self._reward_terms:
            term.bind(adapter)
            term.reset()

        components: dict[str, float] = {t.name: 0.0 for t in self._reward_terms}
        cost = 0.0
        prev_elapsed_days = adapter.elapsed
        last_control_elapsed = 0.0
        n_control_updates = 0

        # Apply an initial control action before the first routing step.
        self._apply_control(adapter, controller, metric_reader, structure_idxs, self._control_interval)
        n_control_updates += 1

        while adapter.is_running:
            adapter.step()
            elapsed_days = adapter.elapsed
            dt_seconds = (elapsed_days - prev_elapsed_days) * _SECONDS_PER_DAY
            prev_elapsed_days = elapsed_days
            # The engine resets ``elapsed`` to 0 on the final step that ends the
            # run, which makes that step's dt hugely negative. A negative dt would
            # flip the sign of every reward-term contribution, so drop it.
            if dt_seconds < 0.0:
                dt_seconds = 0.0
            for term in self._reward_terms:
                c = float(term.step(adapter, dt_seconds))
                components[term.name] += c
                cost += c if term.direction == "minimize" else -c

            elapsed_seconds = elapsed_days * _SECONDS_PER_DAY
            if elapsed_seconds - last_control_elapsed >= self._control_interval:
                self._apply_control(
                    adapter,
                    controller,
                    metric_reader,
                    structure_idxs,
                    elapsed_seconds - last_control_elapsed,
                )
                last_control_elapsed = elapsed_seconds
                n_control_updates += 1

        reward = -cost
        final_obs = self._observation_builder.collect(adapter)

        try:
            adapter.end()
            adapter.report()
        except Exception:
            pass
        adapter.close()

        info: dict[str, Any] = {
            "phase": "policy_evaluated",
            "reward_components": components,
            "policy": dict(zip(self.policy_space.labels, [float(x) for x in np.asarray(action).ravel()])),
            "n_control_updates": n_control_updates,
        }
        return final_obs, reward, True, False, info

    def close(self) -> None:
        """Close any held solver."""
        if self._adapter is not None:
            self._adapter.close()
            self._adapter = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_control(
        adapter: SolverAdapter,
        controller: Controller,
        metric_reader: MarketMetricReader,
        structure_idxs: dict[str, int],
        dt_seconds: float,
    ) -> None:
        """Read metrics, compute settings, and push them to the engine.

        Uses C{target_setting} (the persistent runtime-control override), not
        C{controls.set_link_setting} — the latter sets C{control_setting}, which
        the engine recomputes from the target every routing step, so a pushed
        value would be clobbered before it affects routing.

        C{metric_reader} is C{None} for open-loop controllers (no metrics read).
        """
        metrics = metric_reader.read(adapter) if metric_reader is not None else {}
        settings = controller.compute_settings(metrics, dt_seconds)
        for sid, value in settings.items():
            adapter.links.set_target_setting(structure_idxs[sid], float(value))
