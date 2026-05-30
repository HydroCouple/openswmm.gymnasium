"""Unit tests for L{openswmm_gymnasium.envs.SwmmCIPEnv}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from openswmm_gymnasium.envs import SwmmCIPEnv
from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.spaces.design import LinkRoughness, NodeMaxDepth


class TestConstruction:
    def test_requires_design_factories(self, minimal_inp):
        with pytest.raises(ValueError, match="design_factories"):
            SwmmCIPEnv(
                minimal_inp,
                design_factories=[],
                observation_builder=ObservationBuilder().add_node_depths(["J1"]),
            )

    def test_requires_observation_builder(self, minimal_inp):
        with pytest.raises(ValueError, match="observation_builder"):
            SwmmCIPEnv(
                minimal_inp,
                design_factories=[LinkRoughness(["C1"], 0.01, 0.05)],
            )

    def test_action_space_structure(self, minimal_inp):
        env = SwmmCIPEnv(
            minimal_inp,
            design_factories=[
                LinkRoughness(["C1"], 0.01, 0.05),
                NodeMaxDepth(["J1"], 3.0, 8.0),
            ],
            observation_builder=ObservationBuilder().add_node_depths(["J1"]),
        )
        assert isinstance(env.action_space, spaces.Dict)
        assert set(env.action_space.spaces.keys()) == {"design", "runtime"}
        # Runtime is empty for CIP-only.
        assert len(env.action_space["runtime"].spaces) == 0
        # Design has one entry per factory.
        assert set(env.action_space["design"].spaces.keys()) == {
            "link_roughness",
            "node_max_depth",
        }

    def test_reset_returns_zero_obs(self, minimal_inp):
        env = SwmmCIPEnv(
            minimal_inp,
            design_factories=[LinkRoughness(["C1"], 0.01, 0.05)],
            observation_builder=ObservationBuilder().add_node_depths(["J1"]),
        )
        obs, info = env.reset(seed=0)
        assert np.array_equal(obs, np.zeros_like(obs))
        assert info["phase"] == "awaiting_design"
        env.close()


class TestRegistration:
    def test_registered_id_resolves(self):
        import openswmm_gymnasium  # noqa: F401

        spec = gym.spec("OpenSWMM/Minimal-CIP-v0")
        assert spec.entry_point == "openswmm_gymnasium.envs:SwmmCIPEnv"


# ---------------------------------------------------------------------------
# Integration tier — needs real engine
# ---------------------------------------------------------------------------


def _make_cip_env(inp_path):
    return SwmmCIPEnv(
        inp_path,
        design_factories=[
            LinkRoughness(["C1"], 0.005, 0.05),
            NodeMaxDepth(["J1"], 3.0, 8.0),
        ],
        observation_builder=ObservationBuilder().add_node_depths(["J1"]).add_link_flows(["C1"]),
    )


@pytest.mark.integration
class TestEpisode:
    def test_single_step_terminates(self, minimal_inp):
        env = _make_cip_env(minimal_inp)
        env.reset(seed=0)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert terminated is True
        assert truncated is False
        assert obs.shape == (2,)
        assert isinstance(reward, float)
        assert "reward_components" in info
        assert info["phase"] == "design_evaluated"
        env.close()

    def test_design_change_affects_simulation(self, minimal_inp):
        """Two distinct designs should produce distinct final observations.

        This is the plan §10 P3 verify clause: the .inp mutation
        round-trips through the engine and the simulation differs.
        """
        env = _make_cip_env(minimal_inp)

        env.reset(seed=0)
        low_design = {
            "link_roughness": np.array([0.005], dtype=np.float32),
            "node_max_depth": np.array([3.0], dtype=np.float32),
        }
        obs_low, reward_low, _, _, _ = env.step({"design": low_design, "runtime": {}})

        env.reset(seed=0)
        high_design = {
            "link_roughness": np.array([0.05], dtype=np.float32),
            "node_max_depth": np.array([8.0], dtype=np.float32),
        }
        obs_high, reward_high, _, _, _ = env.step({"design": high_design, "runtime": {}})

        # Different roughness drives different conduit flow rates, so
        # at least one observation component must differ.
        assert not np.allclose(obs_low, obs_high, atol=1e-6), (
            f"Expected design to change observation; got identical "
            f"obs_low={obs_low} obs_high={obs_high}"
        )
        env.close()
