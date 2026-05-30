"""Unit tests for the figure factories.

Each test verifies that the factory returns a L{plotly.graph_objects.Figure}
with the expected number of traces / layout. Empty-input edge cases
return an empty figure (no traces) without raising.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import json

import plotly.graph_objects as go
import pytest

from openswmm_gymnasium.viz import Trajectory, TrajectoryRun
from openswmm_gymnasium.viz.figures import (
    action_timeseries,
    flooding_attribution,
    hypervolume_trace,
    network_state_heatmap,
    objective_radar,
    pareto_front,
    reward_curves,
    trajectory_replay,
)


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _scalar_episode(path, n_steps=5):
    recs = [{"event": "reset", "episode": 0, "obs": [0.0, 0.0], "info": {}}]
    for t in range(1, n_steps + 1):
        recs.append(
            {
                "event": "step",
                "t": t,
                "action": {"runtime": {"a": [float(t) / n_steps]}},
                "obs": [float(t), float(t) * 2],
                "reward": float(t),
                "terminated": t == n_steps,
                "truncated": False,
                "info": {"reward_components": {"flood": float(t), "energy": 0.5}},
            }
        )
    _write_jsonl(path, recs)


def _vector_episode(path, n_steps=5, base=1.0):
    recs = [{"event": "reset", "episode": 0, "obs": [0.0], "info": {}}]
    for t in range(1, n_steps + 1):
        recs.append(
            {
                "event": "step",
                "t": t,
                "action": {"runtime": {"a": [0.5]}},
                "obs": [float(t)],
                "reward": [base * float(t), float(t * 2)],
                "terminated": t == n_steps,
                "truncated": False,
                "info": {"reward_components": {"a": base * float(t), "b": float(t * 2)}},
            }
        )
    _write_jsonl(path, recs)


class TestRewardCurves:
    def test_scalar_returns_figure(self, tmp_path):
        p = tmp_path / "ep.jsonl"
        _scalar_episode(p, n_steps=20)
        fig = reward_curves(Trajectory.from_jsonl(p), rolling_window=5)
        assert isinstance(fig, go.Figure)
        # reward + cumulative + rolling = 3 traces
        assert len(fig.data) == 3

    def test_vector_returns_figure(self, tmp_path):
        p = tmp_path / "ep.jsonl"
        _vector_episode(p, n_steps=4)
        fig = reward_curves(Trajectory.from_jsonl(p))
        # 2 objectives × (raw + cumulative) = 4 traces
        assert len(fig.data) == 4

    def test_empty_returns_empty_figure(self):
        empty = Trajectory(records=[{"event": "reset", "episode": 0, "obs": [0.0], "info": {}}])
        fig = reward_curves(empty)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0


class TestParetoFront:
    def test_2d_returns_figure(self, tmp_path):
        for i in range(3):
            _scalar_episode(tmp_path / f"episode_0000{i}.jsonl", n_steps=2 + i)
        run = TrajectoryRun.from_dir(tmp_path)
        fig = pareto_front(run)
        assert isinstance(fig, go.Figure)
        # 2 traces: dominated + Pareto front
        assert len(fig.data) == 2

    def test_unsupported_dimension_raises(self, tmp_path):
        _scalar_episode(tmp_path / "episode_00000.jsonl", n_steps=2)
        run = TrajectoryRun.from_dir(tmp_path)
        with pytest.raises(ValueError, match="2-D or 3-D"):
            pareto_front(run, dimensions=(0,))

    def test_empty_run_returns_empty_figure(self, tmp_path):
        run = TrajectoryRun.from_dir(tmp_path)
        fig = pareto_front(run)
        assert len(fig.data) == 0


class TestHypervolumeTrace:
    def test_returns_figure_with_values_in_unit_interval(self, tmp_path):
        for i in range(3):
            _scalar_episode(tmp_path / f"episode_0000{i}.jsonl", n_steps=2 + i)
        run = TrajectoryRun.from_dir(tmp_path)
        fig = hypervolume_trace(run, ideal=[0.0, 0.0], reference=[100.0, 100.0])
        assert len(fig.data) == 1
        # y values must be in [0, 1].
        y = fig.data[0].y
        assert all(0.0 <= v <= 1.0 for v in y)

    def test_mismatch_raises(self, tmp_path):
        _scalar_episode(tmp_path / "episode_00000.jsonl", n_steps=3)
        run = TrajectoryRun.from_dir(tmp_path)
        with pytest.raises(ValueError, match="does not match"):
            hypervolume_trace(run, ideal=[0.0], reference=[100.0])


class TestActionTimeseries:
    def test_returns_heatmap(self, tmp_path):
        p = tmp_path / "ep.jsonl"
        _scalar_episode(p, n_steps=4)
        fig = action_timeseries(Trajectory.from_jsonl(p))
        assert len(fig.data) == 1
        # Heatmap is row × column; shape = (n_action_components, n_steps)
        assert fig.data[0].z.shape == (1, 4)


class TestNetworkStateHeatmap:
    def test_returns_heatmap(self, tmp_path):
        p = tmp_path / "ep.jsonl"
        _scalar_episode(p, n_steps=4)
        fig = network_state_heatmap(Trajectory.from_jsonl(p))
        assert len(fig.data) == 1
        # Without reset row: (n_features=2, n_steps=4)
        assert fig.data[0].z.shape == (2, 4)

    def test_label_mismatch_raises(self, tmp_path):
        p = tmp_path / "ep.jsonl"
        _scalar_episode(p, n_steps=3)
        with pytest.raises(ValueError, match="does not match"):
            network_state_heatmap(Trajectory.from_jsonl(p), feature_labels=["only_one"])


class TestFloodingAttribution:
    def test_stacked_area_per_component(self, tmp_path):
        p = tmp_path / "ep.jsonl"
        _scalar_episode(p, n_steps=4)
        fig = flooding_attribution(Trajectory.from_jsonl(p))
        # Two components → two traces.
        assert len(fig.data) == 2


class TestObjectiveRadar:
    def test_one_trace_per_trajectory(self, tmp_path):
        _scalar_episode(tmp_path / "a.jsonl", n_steps=3)
        _scalar_episode(tmp_path / "b.jsonl", n_steps=5)
        trajs = [
            Trajectory.from_jsonl(tmp_path / "a.jsonl"),
            Trajectory.from_jsonl(tmp_path / "b.jsonl"),
        ]
        fig = objective_radar(trajs, labels=["policy A", "policy B"])
        assert len(fig.data) == 2

    def test_label_mismatch_raises(self, tmp_path):
        _scalar_episode(tmp_path / "a.jsonl", n_steps=2)
        trajs = [Trajectory.from_jsonl(tmp_path / "a.jsonl")]
        with pytest.raises(ValueError, match="does not match"):
            objective_radar(trajs, labels=["A", "B"])  # length 2 vs 1 traj


class TestTrajectoryReplay:
    def test_returns_animated_figure(self, tmp_path):
        p = tmp_path / "ep.jsonl"
        _scalar_episode(p, n_steps=3)
        traj = Trajectory.from_jsonl(p)
        fig = trajectory_replay(
            traj,
            coords={"J1": (0.0, 0.0), "J2": (1.0, 0.0)},
            obs_index={"J1": 0, "J2": 1},
        )
        # frames = reset + 3 steps = 4
        assert len(fig.frames) == 4

    def test_key_mismatch_raises(self, tmp_path):
        p = tmp_path / "ep.jsonl"
        _scalar_episode(p, n_steps=2)
        traj = Trajectory.from_jsonl(p)
        with pytest.raises(ValueError, match="same set"):
            trajectory_replay(
                traj,
                coords={"J1": (0.0, 0.0)},
                obs_index={"J2": 0},
            )
