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

"""Tiny dummy envs used by the wrapper unit tests.

Pure-Python; no engine touched. Each env satisfies just enough of the
L{gymnasium.Env} contract to exercise the wrapper under test.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DictBoxEnv(gym.Env):
    """Env with action_space C{Dict({"design": Dict({"d": Box(2,5)}),
    "runtime": Dict({"r": Box(0,1)})})} and 1-D Box observation."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Dict(
            {
                "design": spaces.Dict(
                    {"d": spaces.Box(low=2.0, high=5.0, shape=(2,), dtype=np.float32)}
                ),
                "runtime": spaces.Dict(
                    {"r": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)}
                ),
            }
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.last_action: Any | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {"reset_called": True}

    def step(self, action: Any):
        self.last_action = action
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        return obs, 0.0, True, False, {"step_called": True}


class VectorRewardEnv(gym.Env):
    """Env returning a 3-D vector reward each step."""

    metadata = {"render_modes": []}
    reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self._next_reward = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self._terminate_after = 1
        self._steps = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._steps = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action: Any):
        self._steps += 1
        terminated = self._steps >= self._terminate_after
        return (
            np.zeros(1, dtype=np.float32),
            self._next_reward.copy(),
            terminated,
            False,
            {},
        )


class FlatBoxEnv(gym.Env):
    """1-D Box obs + 1-D Box action, used by forecast and record_trajectory."""

    metadata = {"render_modes": []}

    def __init__(self, obs_size: int = 2, terminate_after: int = 3) -> None:
        super().__init__()
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self._obs_size = obs_size
        self._terminate_after = terminate_after
        self._steps = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._steps = 0
        return np.zeros(self._obs_size, dtype=np.float32), {"elapsed": 0.0}

    def step(self, action: Any):
        self._steps += 1
        terminated = self._steps >= self._terminate_after
        return (
            np.full(self._obs_size, float(self._steps), dtype=np.float32),
            float(self._steps),
            terminated,
            False,
            {"elapsed": float(self._steps)},
        )
