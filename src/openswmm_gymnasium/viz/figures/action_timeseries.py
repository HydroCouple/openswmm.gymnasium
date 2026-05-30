"""
Action heatmap over time.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

from openswmm_gymnasium.viz.trajectory import Trajectory


def _flatten_action(a: Any) -> dict[str, np.ndarray]:
    """Flatten a possibly-nested action into C{<dotted-path>: ndarray}."""
    out: dict[str, np.ndarray] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        else:
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.size == 1:
                out[prefix] = arr
            else:
                for i, x in enumerate(arr):
                    out[f"{prefix}[{i}]"] = np.array([x], dtype=float)

    walk("", a)
    return out


def action_timeseries(
    trajectory: Trajectory,
    *,
    title: str = "Action timeseries",
) -> go.Figure:
    """Heatmap of per-step action components.

    Rows = action component (flattened from nested Dict, joined by C{.}).
    Columns = env step. Values = scalar component value.

    @param trajectory: Loaded episode.
    @type trajectory: L{Trajectory}
    @param title: Figure title.
    @type title: str
    @rtype: L{plotly.graph_objects.Figure}
    """
    fig = go.Figure()
    if trajectory.n_steps == 0:
        return fig.update_layout(title=title)

    per_step_flat = [_flatten_action(a) for a in trajectory.actions]
    # Union of keys across steps, sorted for stable row order.
    keys = sorted({k for d in per_step_flat for k in d})
    if not keys:
        return fig.update_layout(title=title)

    matrix = np.zeros((len(keys), len(per_step_flat)), dtype=float)
    for j, d in enumerate(per_step_flat):
        for i, k in enumerate(keys):
            if k in d:
                matrix[i, j] = float(d[k][0])

    fig.add_trace(
        go.Heatmap(
            z=matrix,
            x=np.arange(1, matrix.shape[1] + 1),
            y=keys,
            colorscale="Viridis",
            colorbar={"title": "value"},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="env step",
        yaxis_title="action component",
    )
    return fig
