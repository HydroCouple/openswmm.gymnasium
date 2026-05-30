"""Unit tests for L{openswmm_gymnasium.viz.trajectory}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from openswmm_gymnasium.viz.trajectory import Trajectory, TrajectoryRun


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_scalar_episode(path, n_steps=3):
    records = [
        {"event": "reset", "episode": 0, "obs": [0.0, 0.0], "info": {}},
    ]
    for t in range(1, n_steps + 1):
        records.append(
            {
                "event": "step",
                "t": t,
                "action": {"runtime": {"a": [0.5]}},
                "obs": [float(t), float(t * 2)],
                "reward": float(t),
                "terminated": t == n_steps,
                "truncated": False,
                "info": {
                    "reward_components": {"flooding": float(t), "energy": 0.5},
                },
            }
        )
    _write_jsonl(path, records)


def _make_vector_episode(path, n_steps=3):
    records = [
        {"event": "reset", "episode": 0, "obs": [0.0], "info": {}},
    ]
    for t in range(1, n_steps + 1):
        records.append(
            {
                "event": "step",
                "t": t,
                "action": {"runtime": {"a": [0.5]}},
                "obs": [float(t)],
                "reward": [float(t), float(t * 2)],
                "terminated": t == n_steps,
                "truncated": False,
                "info": {
                    "reward_components": {"a": float(t), "b": float(t * 2)},
                },
            }
        )
    _write_jsonl(path, records)


class TestTrajectoryConstruction:
    def test_load_from_jsonl(self, tmp_path):
        p = tmp_path / "ep.jsonl"
        _make_scalar_episode(p, n_steps=3)
        traj = Trajectory.from_jsonl(p)
        assert traj.n_steps == 3
        assert traj.reset_record["event"] == "reset"

    def test_missing_reset_raises(self):
        with pytest.raises(ValueError, match="reset event"):
            Trajectory(records=[{"event": "step"}])

    def test_empty_records_raises(self):
        with pytest.raises(ValueError, match="reset event"):
            Trajectory(records=[])


class TestTrajectoryArrays:
    def test_scalar_rewards(self, tmp_path):
        p = tmp_path / "ep.jsonl"
        _make_scalar_episode(p, n_steps=4)
        traj = Trajectory.from_jsonl(p)
        np.testing.assert_array_equal(traj.rewards, [1.0, 2.0, 3.0, 4.0])
        assert traj.rewards.ndim == 1

    def test_vector_rewards(self, tmp_path):
        p = tmp_path / "ep.jsonl"
        _make_vector_episode(p, n_steps=3)
        traj = Trajectory.from_jsonl(p)
        assert traj.rewards.shape == (3, 2)
        np.testing.assert_array_equal(traj.rewards[-1], [3.0, 6.0])

    def test_observations_include_reset(self, tmp_path):
        p = tmp_path / "ep.jsonl"
        _make_scalar_episode(p, n_steps=3)
        traj = Trajectory.from_jsonl(p)
        assert traj.observations.shape == (4, 2)  # 3 steps + reset
        np.testing.assert_array_equal(traj.observations[0], [0.0, 0.0])

    def test_reward_components(self, tmp_path):
        p = tmp_path / "ep.jsonl"
        _make_scalar_episode(p, n_steps=3)
        traj = Trajectory.from_jsonl(p)
        comps = traj.reward_components()
        assert set(comps.keys()) == {"flooding", "energy"}
        np.testing.assert_array_equal(comps["flooding"], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(comps["energy"], [0.5, 0.5, 0.5])

    def test_cumulative_cost_vector(self, tmp_path):
        p = tmp_path / "ep.jsonl"
        _make_scalar_episode(p, n_steps=3)
        traj = Trajectory.from_jsonl(p)
        # sorted axes: ["energy", "flooding"]; sums = [1.5, 6.0]
        np.testing.assert_array_equal(traj.cumulative_cost_vector(), [1.5, 6.0])


class TestTrajectoryRun:
    def test_from_dir(self, tmp_path):
        _make_scalar_episode(tmp_path / "episode_00000.jsonl", n_steps=2)
        _make_scalar_episode(tmp_path / "episode_00001.jsonl", n_steps=3)
        run = TrajectoryRun.from_dir(tmp_path)
        assert run.n_episodes == 2
        assert len(run) == 2

    def test_cumulative_cost_matrix(self, tmp_path):
        _make_scalar_episode(tmp_path / "episode_00000.jsonl", n_steps=2)  # f=3, e=1
        _make_scalar_episode(tmp_path / "episode_00001.jsonl", n_steps=3)  # f=6, e=1.5
        run = TrajectoryRun.from_dir(tmp_path)
        mat = run.cumulative_cost_matrix()
        assert mat.shape == (2, 2)
        # sorted axes: ["energy", "flooding"]
        np.testing.assert_array_equal(mat, [[1.0, 3.0], [1.5, 6.0]])

    def test_empty_dir(self, tmp_path):
        run = TrajectoryRun.from_dir(tmp_path)
        assert len(run) == 0
        assert run.cumulative_cost_matrix().shape == (0, 0)
