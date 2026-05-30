"""Unit tests for L{openswmm_gymnasium.benchmarks.b01_twin_tank}.

Construction + registration tests run in-sandbox. Episode tests are
integration-marked (real engine required on host).

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import gymnasium as gym
import pytest
from gymnasium import spaces

import openswmm_gymnasium  # noqa: F401 — registers benchmarks
from openswmm_gymnasium.benchmarks import b01_twin_tank
from openswmm_gymnasium.envs import (
    SwmmJointCIPRTCEnv,
    SwmmMORTCEnv,
    SwmmRTCEnv,
)


class TestScenarioFile:
    def test_inp_file_exists(self):
        assert b01_twin_tank.SCENARIO_INP.exists()

    def test_inp_file_has_expected_sections(self):
        """Smoke-check the .inp file for the sections our env factories rely on."""
        content = b01_twin_tank.SCENARIO_INP.read_text()
        for section in (
            "[OPTIONS]",
            "[STORAGE]",
            "[OUTFALLS]",
            "[ORIFICES]",
            "[CONDUITS]",
            "[XSECTIONS]",
            "[TIMESERIES]",
        ):
            assert section in content, f"Missing section {section}"
        # Symbolic IDs the env factories will resolve.
        for sid in ("T1", "T2", "O1", "ORIF", "OUT", "RG", "S1"):
            assert sid in content, f"Missing symbol {sid}"


class TestRegistration:
    @pytest.mark.parametrize(
        "env_id, entry_point",
        [
            (
                "OpenSWMM/TwinTank-RTC-v0",
                "openswmm_gymnasium.benchmarks.b01_twin_tank:_make_rtc_env",
            ),
            (
                "OpenSWMM/TwinTank-Joint-v0",
                "openswmm_gymnasium.benchmarks.b01_twin_tank:_make_joint_env",
            ),
            (
                "OpenSWMM/TwinTank-MORTC-v0",
                "openswmm_gymnasium.benchmarks.b01_twin_tank:_make_mo_env",
            ),
        ],
    )
    def test_env_registered(self, env_id, entry_point):
        spec = gym.spec(env_id)
        assert spec.entry_point == entry_point


class TestMakeEnvConstruction:
    """Construction goes through enough engine surfaces (path validation,
    action space assembly) that the env can be built without C{open()}-ing
    the solver."""

    def test_rtc_returns_swmmrtcenv(self):
        env = b01_twin_tank.make_env("rtc")
        assert isinstance(env, SwmmRTCEnv)
        # Action space has the expected runtime key only.
        assert "orifice_setting" in env.action_space["runtime"].spaces
        assert len(env.action_space["design"].spaces) == 0
        env.close()

    def test_joint_returns_swmmjoint(self):
        env = b01_twin_tank.make_env("joint")
        assert isinstance(env, SwmmJointCIPRTCEnv)
        assert "node_max_depth" in env.action_space["design"].spaces
        assert "orifice_setting" in env.action_space["runtime"].spaces
        env.close()

    def test_mo_returns_swmmmortcenv(self):
        env = b01_twin_tank.make_env("mo")
        assert isinstance(env, SwmmMORTCEnv)
        assert isinstance(env.reward_space, spaces.Box)
        # Three reward terms.
        assert env.reward_space.shape == (3,)
        env.close()

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            b01_twin_tank.make_env("not_a_variant")


class TestObservationSize:
    """Observation builder produces 7 features for b01:
    2 node depths + 1 link flow + 1 link setting + 1 rainfall + 1 elapsed_frac
    (clock = 1 of 3 features when only elapsed_frac is requested)."""

    def test_obs_shape(self):
        env = b01_twin_tank.make_env("rtc")
        # 2 (depths T1,T2) + 1 (OUT flow) + 1 (ORIF setting) + 1 (RG rain)
        # + 1 (clock elapsed_frac) = 6
        assert env.observation_space.shape == (6,)
        env.close()


# ---------------------------------------------------------------------------
# Integration — needs the real engine
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEpisode:
    def test_rtc_episode_runs(self):
        env = b01_twin_tank.make_env("rtc")
        env.reset(seed=0)
        steps = 0
        while True:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            steps += 1
            if terminated or truncated:
                break
            if steps > 5000:
                pytest.fail("Episode did not terminate within safety limit")
        assert steps > 100  # 4-hr sim at 15-s routing step ≈ 960 steps
        env.close()

    def test_mo_episode_emits_score(self):
        env = b01_twin_tank.make_env("mo")
        env.reset(seed=0)
        info = None
        while True:
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                break
        assert info is not None
        assert "mo_score" in info
        assert 0.0 <= info["mo_score"] <= 1.0
        env.close()
