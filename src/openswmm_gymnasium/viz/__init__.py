"""
openswmm_gymnasium.viz
=======================

Plotly-only visualisation layer for optimisation trajectories. Plan §5.5
+ §10 P6.5.

This subpackage is gated by an optional install extra. If Plotly is not
installed, every public import here raises C{ImportError} with the
install instruction; the rest of L{openswmm_gymnasium} is unaffected.

Public API:

  - L{Trajectory} — single JSONL episode loader.
  - L{TrajectoryRun} — directory-of-JSONL multi-episode loader.
  - L{figures.reward_curves} — scalar/vector reward over time.
  - L{figures.pareto_front} — 2-D / 3-D scatter, non-dominated highlighted.
  - L{figures.hypervolume_trace} — normalised HV per episode.
  - L{figures.action_timeseries} — per-step action heatmap.
  - L{figures.network_state_heatmap} — node depths / link flows over time.
  - L{figures.flooding_attribution} — stacked area of cost-component
    contributions over time.
  - L{figures.objective_radar} — radar comparing per-objective totals
    across multiple trajectories.
  - L{figures.trajectory_replay} — animated network schematic on
    user-supplied node coordinates.

The module is post-hoc only: it consumes the JSONL trajectories already
written by L{openswmm_gymnasium.wrappers.RecordTrajectory} and does not
touch a live solver.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

try:
    import plotly  # noqa: F401
except ImportError as e:  # pragma: no cover - exercised only when missing
    raise ImportError(
        "openswmm_gymnasium.viz requires Plotly. Install with "
        "'pip install openswmm-gymnasium[viz]' (or "
        "'pip install plotly') and try again."
    ) from e

from openswmm_gymnasium.viz.trajectory import Trajectory, TrajectoryRun

__all__ = ["Trajectory", "TrajectoryRun"]
