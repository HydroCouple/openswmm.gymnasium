"""Unit tests for L{openswmm_gymnasium.wrappers.MaskDesignAction} and
L{openswmm_gymnasium.wrappers.MaskRuntimeAction}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from openswmm_gymnasium.wrappers import MaskDesignAction, MaskRuntimeAction
from tests.unit._wrapper_helpers import DictBoxEnv


class TestMaskRuntimeAction:
    def test_action_space_flattens_to_design(self):
        env = MaskRuntimeAction(DictBoxEnv())
        assert env.action_space.spaces.keys() == {"d"}

    def test_step_fills_runtime_with_midpoint(self):
        base = DictBoxEnv()
        env = MaskRuntimeAction(base)
        design_action = {"d": np.array([3.0, 4.0], dtype=np.float32)}
        env.step(design_action)
        # Runtime Box(0, 1) midpoint = 0.5.
        np.testing.assert_allclose(base.last_action["runtime"]["r"], [0.5])
        np.testing.assert_allclose(base.last_action["design"]["d"], [3.0, 4.0])


class TestMaskDesignAction:
    def test_action_space_flattens_to_runtime(self):
        env = MaskDesignAction(DictBoxEnv())
        assert env.action_space.spaces.keys() == {"r"}

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
        with pytest.raises(RuntimeError, match="before reset"):
            env.step({"r": np.array([0.5], dtype=np.float32)})


class TestTypeErrors:
    def test_non_dict_action_space_raises(self):
        class _PlainEnv(gym.Env):
            action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
            observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

            def reset(self, **kw):
                return np.zeros(1, dtype=np.float32), {}

            def step(self, a):
                return np.zeros(1, dtype=np.float32), 0.0, True, False, {}

        with pytest.raises(TypeError, match="Dict action space"):
            MaskRuntimeAction(_PlainEnv())
