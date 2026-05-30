"""
Forecast-injection observation wrapper.

The forecast callable is supplied by the user — for example, a
perfect-information lookup against the rainfall time series, or a
noisy predictor with a configurable error model. The wrapper appends
the forecast vector to the env's observation each step.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ForecastObservation(gym.Wrapper):
    """Append forecast features to the observation each step.

    The user supplies a callable C{forecast_fn(env, info)} returning a
    1-D float array of length L{horizon}. The wrapped env's
    observation space is extended with that many additional features
    (bounds C{[-inf, +inf]}).

    @ivar horizon: Number of forecast features appended each step.
    """

    def __init__(
        self,
        env: gym.Env,
        forecast_fn: Callable[[gym.Env, dict[str, Any]], np.ndarray],
        horizon: int,
    ) -> None:
        """
        @param env: The wrapped env. Must have a 1-D
            L{gymnasium.spaces.Box} observation space.
        @type env: L{gymnasium.Env}
        @param forecast_fn: Callable receiving the wrapped env and the
            C{info} dict from the most recent C{reset()} or C{step()},
            returning a 1-D numpy array of length C{horizon}.
        @type forecast_fn: callable
        @param horizon: Number of forecast features.
        @type horizon: int
        @raise TypeError: If the env's observation space isn't a 1-D Box.
        @raise ValueError: If C{horizon < 1}.
        """
        super().__init__(env)
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if (
            not isinstance(env.observation_space, spaces.Box)
            or len(env.observation_space.shape) != 1
        ):
            raise TypeError("ForecastObservation requires a 1-D Box observation space")
        self.forecast_fn = forecast_fn
        self.horizon = int(horizon)
        base_size = env.observation_space.shape[0]
        new_size = base_size + self.horizon
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(new_size,),
            dtype=np.float32,
        )
        self._base_size = base_size

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._extend(obs, info), info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._extend(obs, info), reward, terminated, truncated, info

    def _extend(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        fc = np.asarray(self.forecast_fn(self.env, info), dtype=np.float32)
        if fc.shape != (self.horizon,):
            raise ValueError(f"forecast_fn returned shape {fc.shape}; expected ({self.horizon},)")
        return np.concatenate([obs.astype(np.float32), fc])
