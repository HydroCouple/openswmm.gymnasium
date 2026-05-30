"""
Stacked-area attribution of reward-component contributions over time.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from openswmm_gymnasium.viz.trajectory import Trajectory


def flooding_attribution(
    trajectory: Trajectory,
    *,
    title: str = "Reward-component attribution",
) -> go.Figure:
    """Stacked area of per-step reward-component contributions.

    Despite the name (inherited from plan §5.5), this works for any
    reward-component breakdown — flooding, CSO, energy, etc.

    @param trajectory: Loaded episode.
    @type trajectory: L{Trajectory}
    @param title: Figure title.
    @type title: str
    @rtype: L{plotly.graph_objects.Figure}
    """
    fig = go.Figure()
    comps = trajectory.reward_components()
    if not comps:
        return fig.update_layout(title=title)

    steps = np.arange(1, trajectory.n_steps + 1)
    for name, series in comps.items():
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=series,
                mode="lines",
                stackgroup="one",
                name=name,
                hoverinfo="x+y+name",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="env step",
        yaxis_title="contribution",
        hovermode="x unified",
    )
    return fig
