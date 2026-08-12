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

"""Unit tests for L{openswmm_gymnasium.wrappers.MaskDesignAction} and
L{openswmm_gymnasium.wrappers.MaskRuntimeAction}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np

from openswmm_gymnasium.wrappers import MaskDesignAction, MaskRuntimeAction
from tests.unit._wrapper_helpers import DictBoxEnv


class TestMaskRuntimeAction(unittest.TestCase):
    def test_action_space_flattens_to_design(self):
        env = MaskRuntimeAction(DictBoxEnv())
        self.assertEqual(env.action_space.spaces.keys(), {"d"})

    def test_step_fills_runtime_with_midpoint(self):
        base = DictBoxEnv()
        env = MaskRuntimeAction(base)
        design_action = {"d": np.array([3.0, 4.0], dtype=np.float32)}
        env.step(design_action)
        # Runtime Box(0, 1) midpoint = 0.5.
        np.testing.assert_allclose(base.last_action["runtime"]["r"], [0.5])
        np.testing.assert_allclose(base.last_action["design"]["d"], [3.0, 4.0])


class TestMaskDesignAction(unittest.TestCase):
    def test_action_space_flattens_to_runtime(self):
        env = MaskDesignAction(DictBoxEnv())
        self.assertEqual(env.action_space.spaces.keys(), {"r"})

    def test_design_sampled_at_reset_and_frozen(self):
        base = DictBoxEnv()
        env = MaskDesignAction(base)
        _, info = env.reset(seed=0)
        sampled = info["frozen_design_action"]
        runtime_action = {"r": np.array([0.5], dtype=np.float32)}
        env.step(runtime_action)
        np.testing.assert_array_equal(base.last_action["design"]["d"], sampled["d"])

    def test_frozen_design_explicit(self):
        base = DictBoxEnv()
        frozen = {"d": np.array([2.5, 4.5], dtype=np.float32)}
        env = MaskDesignAction(base, frozen_design=frozen)
        env.reset(seed=0)
        env.step({"r": np.array([0.1], dtype=np.float32)})
        np.testing.assert_allclose(base.last_action["design"]["d"], [2.5, 4.5])

    def test_step_before_reset_raises(self):
        env = MaskDesignAction(DictBoxEnv())
        with self.assertRaisesRegex(RuntimeError, "before reset"):
            env.step({"r": np.array([0.5], dtype=np.float32)})


class TestTypeErrors(unittest.TestCase):
    def test_non_dict_action_space_raises(self):
        class _PlainEnv(gym.Env):
            action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
            observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

            def reset(self, **kw):
                return np.zeros(1, dtype=np.float32), {}

            def step(self, a):
                return np.zeros(1, dtype=np.float32), 0.0, True, False, {}

        with self.assertRaisesRegex(TypeError, "Dict action space"):
            MaskRuntimeAction(_PlainEnv())


if __name__ == "__main__":
    unittest.main()
