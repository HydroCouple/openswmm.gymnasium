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

"""Unit tests for L{openswmm_gymnasium.wrappers.ForecastObservation}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np

from openswmm_gymnasium.wrappers import ForecastObservation
from tests.unit._wrapper_helpers import FlatBoxEnv


def _constant_forecast(env, info):
    return np.array([0.1, 0.2, 0.3], dtype=np.float32)


def _info_dependent_forecast(env, info):
    e = float(info.get("elapsed", 0.0))
    return np.array([e, e * 2.0], dtype=np.float32)


class TestObservationShape(unittest.TestCase):
    def test_observation_space_extended(self):
        base = FlatBoxEnv(obs_size=2)
        wrapped = ForecastObservation(base, _constant_forecast, horizon=3)
        self.assertEqual(wrapped.observation_space.shape, (5,))

    def test_reset_returns_extended_obs(self):
        base = FlatBoxEnv(obs_size=2)
        wrapped = ForecastObservation(base, _constant_forecast, horizon=3)
        obs, _ = wrapped.reset(seed=0)
        self.assertEqual(obs.shape, (5,))
        np.testing.assert_allclose(obs, [0, 0, 0.1, 0.2, 0.3], atol=1e-6)


class TestForecastDynamic(unittest.TestCase):
    def test_forecast_uses_step_info(self):
        base = FlatBoxEnv(obs_size=1)
        wrapped = ForecastObservation(base, _info_dependent_forecast, horizon=2)
        wrapped.reset()
        obs, _, _, _, _ = wrapped.step(base.action_space.sample())
        # FlatBoxEnv returns elapsed = step count (= 1 after first step).
        # forecast = [1.0, 2.0]; base obs = [1.0].
        np.testing.assert_allclose(obs, [1.0, 1.0, 2.0], atol=1e-6)


class TestValidation(unittest.TestCase):
    def test_zero_horizon_raises(self):
        with self.assertRaisesRegex(ValueError, ">= 1"):
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

        with self.assertRaisesRegex(TypeError, "1-D Box"):
            ForecastObservation(_DictObsEnv(), _constant_forecast, horizon=2)

    def test_wrong_forecast_shape_raises(self):
        base = FlatBoxEnv()

        def _bad_fn(env, info):
            return np.array([0.1, 0.2, 0.3], dtype=np.float32)  # length 3, not 2

        wrapped = ForecastObservation(base, _bad_fn, horizon=2)
        with self.assertRaisesRegex(ValueError, "forecast_fn returned shape"):
            wrapped.reset()


if __name__ == "__main__":
    unittest.main()
