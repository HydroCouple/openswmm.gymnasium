"""
Observation heatmap over time.

This figure treats the env's flat observation vector as one row per
feature × one column per step. Suitable for any 1-D Box observation
(node depths, link flows, mixed feature sets).

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go

from openswmm_gymnasium.viz.trajectory import Trajectory


def network_state_heatmap(
    trajectory: Trajectory,
    *,
    feature_labels: Sequence[str] | None = None,
    title: str = "Network state",
) -> go.Figure:
    """Heatmap of observation features over time.

    @param trajectory: Loaded episode.
    @type trajectory: L{Trajectory}
    @param feature_labels: Optional names for each observation index.
        Defaults to C{"feat[0]", "feat[1]", ...}.
    @type feature_labels: sequence of str or C{None}
    @param title: Figure title.
    @type title: str
    @rtype: L{plotly.graph_objects.Figure}
    """
    fig = go.Figure()
    obs = trajectory.observations
    if obs.size == 0:
        return fig.update_layout(title=title)

    # Drop the reset row so x-axis aligns with step indices.
    if obs.shape[0] > 1:
        obs = obs[1:]
    n_features = obs.shape[1]
    labels = list(feature_labels) if feature_labels else [f"feat[{i}]" for i in range(n_features)]
    if len(labels) != n_features:
        raise ValueError(
            f"feature_labels length ({len(labels)}) does not match observation "
            f"dimension ({n_features})"
        )

    fig.add_trace(
        go.Heatmap(
            z=obs.T,
            x=np.arange(1, obs.shape[0] + 1),
            y=labels,
            colorscale="Blues",
            colorbar={"title": "value"},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="env step",
        yaxis_title="feature",
    )
    return fig
