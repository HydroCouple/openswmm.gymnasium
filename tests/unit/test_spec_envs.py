"""Env-spec plumbing: actions from JSON, episode policies, JSON-safe values, env registry.

Moved from openswmm.mcp with the spec layer (plan G3). No engine needed.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces as gym_spaces

from openswmm_gymnasium.spec.envs import EnvManager, _Policy, build_action, json_safe
from openswmm_gymnasium.spec.errors import SpecError


def _dict_space():
    return gym_spaces.Dict(
        {
            "design": gym_spaces.Dict({}),
            "runtime": gym_spaces.Dict(
                {"orifice_setting": gym_spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)}
            ),
        }
    )


# ---------------------------------------------------------------------------
# Action / JSON plumbing (real gymnasium spaces, no engine)
# ---------------------------------------------------------------------------


def test_build_action_defaults_to_midpoint():
    action = build_action(_dict_space(), None)
    np.testing.assert_allclose(action["runtime"]["orifice_setting"], [0.5, 0.5])
    assert action["design"] == {}


def test_build_action_fills_clips_and_validates():
    space = _dict_space()
    action = build_action(space, {"runtime": {"orifice_setting": [0.2, 7.0]}})
    np.testing.assert_allclose(action["runtime"]["orifice_setting"], [0.2, 1.0])

    with pytest.raises(SpecError, match="Unknown action keys"):
        build_action(space, {"bogus": {}})
    with pytest.raises(SpecError, match="Unknown action keys"):
        build_action(space, {"runtime": {"not_a_factory": [0.1]}})
    with pytest.raises(SpecError, match="shape"):
        build_action(space, {"runtime": {"orifice_setting": [0.1, 0.2, 0.3]}})


def test_policy_kinds_and_validation():
    space = _dict_space()
    constant = _Policy(
        space, {"kind": "constant", "action": {"runtime": {"orifice_setting": [0.1, 0.9]}}}
    )
    a1 = constant.next_action()
    np.testing.assert_allclose(
        a1["runtime"]["orifice_setting"], np.array([0.1, 0.9], dtype=np.float32)
    )

    random = _Policy(space, {"kind": "random", "seed": 42})
    r1 = random.next_action()
    assert r1["runtime"]["orifice_setting"].shape == (2,)

    replay = _Policy(
        space, {"kind": "replay", "actions": [{"runtime": {"orifice_setting": [0.0, 0.0]}}]}
    )
    assert replay.next_action() is not None
    assert replay.next_action() is None
    assert replay.exhausted

    with pytest.raises(SpecError, match="Unknown policy kind"):
        _Policy(space, {"kind": "greedy"})
    with pytest.raises(SpecError, match="non-empty 'actions'"):
        _Policy(space, {"kind": "replay"})


def test_json_safe_handles_numpy():
    assert json_safe(np.float32(1.5)) == 1.5
    assert json_safe(np.array([1.0, 2.0])) == [1.0, 2.0]
    assert json_safe({"a": np.bool_(True), "b": (np.int64(3),)}) == {"a": True, "b": [3]}


# ---------------------------------------------------------------------------
# EnvManager bookkeeping errors (no engine needed for the failure paths)
# ---------------------------------------------------------------------------


def test_env_manager_get_and_close_unknown():
    manager = EnvManager()
    with pytest.raises(SpecError, match="SESSION_NOT_FOUND"):
        manager.get("nope")
    with pytest.raises(SpecError, match="SESSION_NOT_FOUND"):
        manager.close("nope")
    assert manager.list() == []
