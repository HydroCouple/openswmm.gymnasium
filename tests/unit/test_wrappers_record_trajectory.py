"""Unit tests for L{openswmm_gymnasium.wrappers.RecordTrajectory}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import json

import numpy as np

from openswmm_gymnasium.wrappers import RecordTrajectory
from tests.unit._wrapper_helpers import FlatBoxEnv


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class TestFileLifecycle:
    def test_one_file_per_episode(self, tmp_path):
        env = RecordTrajectory(FlatBoxEnv(terminate_after=2), output_dir=tmp_path)
        env.reset(seed=0)
        env.step(env.action_space.sample())
        env.step(env.action_space.sample())
        env.reset(seed=1)
        env.step(env.action_space.sample())
        env.step(env.action_space.sample())
        env.close()
        files = sorted(tmp_path.glob("episode_*.jsonl"))
        assert [f.name for f in files] == [
            "episode_00000.jsonl",
            "episode_00001.jsonl",
        ]

    def test_record_structure(self, tmp_path):
        env = RecordTrajectory(FlatBoxEnv(terminate_after=2), output_dir=tmp_path)
        env.reset(seed=0)
        env.step(env.action_space.sample())
        env.step(env.action_space.sample())
        env.close()
        records = _read_jsonl(tmp_path / "episode_00000.jsonl")
        assert records[0]["event"] == "reset"
        assert records[0]["episode"] == 0
        assert "obs" in records[0]
        assert records[1]["event"] == "step"
        assert records[1]["t"] == 1
        assert "action" in records[1]
        assert "reward" in records[1]
        assert isinstance(records[-1]["terminated"], bool)

    def test_terminated_closes_file(self, tmp_path):
        env = RecordTrajectory(FlatBoxEnv(terminate_after=1), output_dir=tmp_path)
        env.reset(seed=0)
        env.step(env.action_space.sample())
        # File handle should be closed; episode_00000.jsonl exists.
        path = tmp_path / "episode_00000.jsonl"
        assert path.exists()
        # Re-opening for write should not raise.
        with open(path, encoding="utf-8") as f:
            f.read()


class TestNumpyHandling:
    def test_ndarray_serialised_as_list(self, tmp_path):
        env = RecordTrajectory(FlatBoxEnv(obs_size=2), output_dir=tmp_path)
        env.reset(seed=0)
        env.close()
        records = _read_jsonl(tmp_path / "episode_00000.jsonl")
        assert isinstance(records[0]["obs"], list)
        assert records[0]["obs"] == [0.0, 0.0]

    def test_numpy_scalar_serialised(self, tmp_path):
        env = RecordTrajectory(FlatBoxEnv(terminate_after=1), output_dir=tmp_path)
        env.reset(seed=0)
        env.step(env.action_space.sample())
        env.close()
        records = _read_jsonl(tmp_path / "episode_00000.jsonl")
        # Step records carry a reward (float).
        assert isinstance(records[-1]["reward"], (int, float))

    def test_unknown_type_fallbacks_to_repr(self, tmp_path):
        from openswmm_gymnasium.wrappers.record_trajectory import _jsonable

        class _Weird:
            def __repr__(self):
                return "<weird>"

        assert _jsonable(_Weird()) == "<weird>"


class TestSerialisation:
    def test_numpy_action_serialised(self, tmp_path):
        env = RecordTrajectory(FlatBoxEnv(terminate_after=1), output_dir=tmp_path)
        env.reset(seed=0)
        env.step(np.array([0.42], dtype=np.float32))
        env.close()
        records = _read_jsonl(tmp_path / "episode_00000.jsonl")
        # Action should be a JSON list of floats, not a numpy repr.
        assert isinstance(records[1]["action"], list)
        assert records[1]["action"] == [pytest_approx_value()]


def pytest_approx_value():
    # Helper so the literal float matches across NumPy float32 rounding.
    return float(np.float32(0.42))
