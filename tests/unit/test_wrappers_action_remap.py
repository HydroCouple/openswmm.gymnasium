"""Unit tests for L{openswmm_gymnasium.wrappers.RescaleBoxActions}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from openswmm_gymnasium.wrappers import RescaleBoxActions
from tests.unit._wrapper_helpers import DictBoxEnv


class TestSpaceRescaling:
    def test_box_leaves_rescaled_to_unit(self):
        env = RescaleBoxActions(DictBoxEnv(), src_low=0.0, src_high=1.0)
        # Every Box leaf should now have low=0, high=1.
        d = env.action_space["design"]["d"]
        r = env.action_space["runtime"]["r"]
        assert np.allclose(d.low, 0.0) and np.allclose(d.high, 1.0)
        assert np.allclose(r.low, 0.0) and np.allclose(r.high, 1.0)

    def test_dict_structure_preserved(self):
        env = RescaleBoxActions(DictBoxEnv())
        assert isinstance(env.action_space, spaces.Dict)
        assert set(env.action_space.spaces) == {"design", "runtime"}


class TestActionTranslation:
    def test_unit_action_maps_to_env_bounds(self):
        env = DictBoxEnv()
        wrapper = RescaleBoxActions(env, src_low=0.0, src_high=1.0)
        unit_action = {
            "design": {"d": np.array([0.0, 1.0], dtype=np.float32)},
            "runtime": {"r": np.array([0.5], dtype=np.float32)},
        }
        wrapper.step(unit_action)
        env_action = env.last_action
        # design.d had Box(2.0, 5.0); 0.0→2.0, 1.0→5.0
        np.testing.assert_allclose(env_action["design"]["d"], [2.0, 5.0], atol=1e-6)
        # runtime.r had Box(0.0, 1.0); 0.5→0.5
        np.testing.assert_allclose(env_action["runtime"]["r"], [0.5], atol=1e-6)

    def test_action_clipped_to_src_range(self):
        env = DictBoxEnv()
        wrapper = RescaleBoxActions(env)
        out_of_range = {
            "design": {"d": np.array([-1.0, 2.0], dtype=np.float32)},  # outside [0,1]
            "runtime": {"r": np.array([0.5], dtype=np.float32)},
        }
        wrapper.step(out_of_range)
        # -1 → clipped to 0 → env coord 2.0; 2 → clipped to 1 → env coord 5.0
        np.testing.assert_allclose(env.last_action["design"]["d"], [2.0, 5.0])


class TestInvalidBounds:
    def test_inverted_bounds_raises(self):
        with pytest.raises(ValueError, match="strictly greater"):
            RescaleBoxActions(DictBoxEnv(), src_low=1.0, src_high=0.0)
