"""Unit tests for L{openswmm_gymnasium.envs.SwmmMORTCEnv}.

Construction + registration tests run anywhere. Episode tests require
the real engine.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from openswmm_gymnasium.envs import SwmmMORTCEnv
from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.rewards import FloodingVolume, ReliabilityMargin
from openswmm_gymnasium.spaces.runtime import OrificeSetting


def _mo_env(inp_path):
    return SwmmMORTCEnv(
        inp_path,
        runtime_factories=[OrificeSetting(["C1"])],
        observation_builder=ObservationBuilder().add_node_depths(["J1"]).add_link_flows(["C1"]),
        reward_terms=[
            FloodingVolume(node_ids=["J1"]),
            ReliabilityMargin(node_ids=["J1"]),
        ],
        ideal_point=[0.0, 0.0],
        reference_point=[10.0, 10.0],
    )


class TestConstruction:
    def test_requires_ideal_and_reference(self, minimal_inp):
        with pytest.raises(ValueError, match="ideal_point and reference_point"):
            SwmmMORTCEnv(
                minimal_inp,
                runtime_factories=[OrificeSetting(["C1"])],
                observation_builder=ObservationBuilder().add_node_depths(["J1"]),
                reward_terms=[FloodingVolume(node_ids=["J1"])],
            )

    def test_requires_nonempty_reward_terms(self, minimal_inp):
        with pytest.raises(ValueError, match="at least one reward term"):
            SwmmMORTCEnv(
                minimal_inp,
                runtime_factories=[OrificeSetting(["C1"])],
                observation_builder=ObservationBuilder().add_node_depths(["J1"]),
                reward_terms=[],
                ideal_point=[0.0],
                reference_point=[1.0],
            )

    def test_ideal_reference_length_mismatch_raises(self, minimal_inp):
        with pytest.raises(ValueError, match="lengths must match"):
            SwmmMORTCEnv(
                minimal_inp,
                runtime_factories=[OrificeSetting(["C1"])],
                observation_builder=ObservationBuilder().add_node_depths(["J1"]),
                reward_terms=[FloodingVolume(node_ids=["J1"])],
                ideal_point=[0.0, 0.0],  # length 2 vs 1 term
                reference_point=[1.0, 1.0],
            )

    def test_ref_not_greater_than_ideal_raises(self, minimal_inp):
        with pytest.raises(ValueError, match="strictly greater"):
            SwmmMORTCEnv(
                minimal_inp,
                runtime_factories=[OrificeSetting(["C1"])],
                observation_builder=ObservationBuilder().add_node_depths(["J1"]),
                reward_terms=[FloodingVolume(node_ids=["J1"])],
                ideal_point=[1.0],
                reference_point=[1.0],
            )

    def test_reward_space_shape(self, minimal_inp):
        env = _mo_env(minimal_inp)
        assert isinstance(env.reward_space, spaces.Box)
        assert env.reward_space.shape == (2,)
        env.close()


class TestRegistration:
    def test_registered(self):
        import openswmm_gymnasium  # noqa: F401

        spec = gym.spec("OpenSWMM/Minimal-MORTC-v0")
        assert spec.entry_point == "openswmm_gymnasium.envs:SwmmMORTCEnv"


# ---------------------------------------------------------------------------
# Integration tier
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEpisode:
    def test_step_returns_vector_reward(self, minimal_inp):
        env = _mo_env(minimal_inp)
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(env.action_space.sample())
        assert isinstance(reward, np.ndarray)
        assert reward.shape == (2,)
        assert reward.dtype == np.float32
        env.close()

    def test_terminal_step_emits_mo_score(self, minimal_inp):
        env = _mo_env(minimal_inp)
        env.reset(seed=0)
        info = None
        while True:
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                break
        assert info is not None
        assert "mo_score" in info
        assert 0.0 <= info["mo_score"] <= 1.0
        assert "cumulative_cost" in info
        assert info["cumulative_cost"].shape == (2,)
        env.close()
