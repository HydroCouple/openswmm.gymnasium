"""
Action-space rescaling wrapper.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RescaleBoxActions(gym.ActionWrapper):
    """Rescale every L{gymnasium.spaces.Box} leaf of the action space.

    The wrapper presents an action space whose Box leaves all live in
    C{[src_low, src_high]}; on C{step()} the action is rescaled back
    into the env's original C{[env_low, env_high]} per-leaf bounds.
    Non-Box subspaces are left untouched.

    Useful for plugging in agents (e.g. SAC, PPO with tanh-squashed
    output) whose policy naturally emits values in a fixed unit
    interval.

    @ivar src_low: Lower bound for every Box leaf after wrapping.
    @ivar src_high: Upper bound for every Box leaf after wrapping.
    """

    def __init__(
        self,
        env: gym.Env,
        src_low: float = 0.0,
        src_high: float = 1.0,
    ) -> None:
        """
        @param env: The wrapped env.
        @type env: L{gymnasium.Env}
        @param src_low: Lower bound the wrapper exposes for Box leaves.
        @type src_low: float
        @param src_high: Upper bound the wrapper exposes for Box leaves.
        @type src_high: float
        @raise ValueError: If C{src_high <= src_low}.
        """
        super().__init__(env)
        if not (src_high > src_low):
            raise ValueError("src_high must be strictly greater than src_low")
        self.src_low = float(src_low)
        self.src_high = float(src_high)
        self._env_action_space = env.action_space
        self.action_space = self._rescale_space(env.action_space)

    # ------------------------------------------------------------------
    # ActionWrapper API
    # ------------------------------------------------------------------

    def action(self, action: Any) -> Any:
        """Translate from C{[src_low, src_high]} back into env bounds."""
        return self._rescale_value(action, self._env_action_space)

    # ------------------------------------------------------------------
    # Recursion
    # ------------------------------------------------------------------

    def _rescale_space(self, space: spaces.Space) -> spaces.Space:
        if isinstance(space, spaces.Box):
            return spaces.Box(
                low=np.full(space.shape, self.src_low, dtype=space.dtype),
                high=np.full(space.shape, self.src_high, dtype=space.dtype),
                shape=space.shape,
                dtype=space.dtype,
            )
        if isinstance(space, spaces.Dict):
            return spaces.Dict({k: self._rescale_space(v) for k, v in space.spaces.items()})
        if isinstance(space, spaces.Tuple):
            return spaces.Tuple(tuple(self._rescale_space(s) for s in space.spaces))
        return space  # Discrete / MultiBinary / MultiDiscrete — untouched

    def _rescale_value(self, value: Any, target_space: spaces.Space) -> Any:
        if isinstance(target_space, spaces.Box):
            v = np.asarray(value, dtype=np.float64)
            v = np.clip(v, self.src_low, self.src_high)
            scaled = target_space.low.astype(np.float64) + (
                (v - self.src_low) / (self.src_high - self.src_low)
            ) * (target_space.high.astype(np.float64) - target_space.low.astype(np.float64))
            return scaled.astype(target_space.dtype)
        if isinstance(target_space, spaces.Dict):
            return {k: self._rescale_value(value[k], v) for k, v in target_space.spaces.items()}
        if isinstance(target_space, spaces.Tuple):
            return tuple(
                self._rescale_value(value[i], s) for i, s in enumerate(target_space.spaces)
            )
        return value
