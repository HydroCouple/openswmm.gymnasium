"""Unit tests for L{openswmm_gymnasium.wrappers.ForecastObservation}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from openswmm_gymnasium.wrappers import ForecastObservation
from tests.unit._wrapper_helpers import FlatBoxEnv


def _constant_forecast(env, info):
    return np.array([0.1, 0.2, 0.3], dtype=np.float32)


def _info_dependent_forecast(env, info):
    e = float(info.get("elapsed", 0.0))
    return np.array([e, e * 2.0], dtype=np.float32)


class TestObservationShape:
    def test_observation_space_extended(self):
        base = FlatBoxEnv(obs_size=2)
        wrapped = ForecastObservation(base, _constant_forecast, horizon=3)
        assert wrapped.observation_space.shape == (5,)

    def test_reset_returns_extended_obs(self):
        base = FlatBoxEnv(obs_size=2)
        wrapped = ForecastObservation(base, _constant_forecast, horizon=3)
        obs, _ = wrapped.reset(seed=0)
        assert obs.shape == (5,)
        np.testing.assert_allclose(obs, [0, 0, 0.1, 0.2, 0.3], atol=1e-6)


class TestForecastDynamic:
    def test_forecast_uses_step_info(self):
        base = FlatBoxEnv(obs_size=1)
        wrapped = ForecastObservation(base, _info_dependent_forecast, horizon=2)
        wrapped.reset()
        obs, _, _, _, _ = wrapped.step(base.action_space.sample())
        # FlatBoxEnv returns elapsed = step count (= 1 after first step).
        # forecast = [1.0, 2.0]; base obs = [1.0].
        np.testing.assert_allclose(obs, [1.0, 1.0, 2.0], atol=1e-6)


class TestValidation:
    def test_zero_horizon_raises(self):
        with pytest.raises(ValueError, match=">= 1"):
            ForecastObservation(FlatBoxEnv(), _constant_forecast, horizon=0)

    def test_non_box_obs_space_raises(self):
        class _DictObsEnv(gym.Env):
            action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
            observation_space = gym.spaces.Dict(
                {"x": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)}
            )

            def reset(self, **kw):
                return {"x": np.zeros(1, dtype=np.float32)}, {}

            def step(self, a):
                return {"x": np.zeros(1, dtype=np.float32)}, 0.0, True, False, {}

        with pytest.raises(TypeError, match="1-D Box"):
            ForecastObservation(_DictObsEnv(), _constant_forecast, horizon=2)

    def test_wrong_forecast_shape_raises(self):
        base = FlatBoxEnv()

        def _bad_fn(env, info):
            return np.array([0.1, 0.2, 0.3], dtype=np.float32)  # length 3, not 2

        wrapped = ForecastObservation(base, _bad_fn, horizon=2)
        with pytest.raises(ValueError, match="forecast_fn returned shape"):
            wrapped.reset()
