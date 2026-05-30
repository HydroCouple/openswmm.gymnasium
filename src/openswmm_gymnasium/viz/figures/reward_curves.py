"""
Per-step reward and cumulative-reward line plot.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from openswmm_gymnasium.viz.trajectory import Trajectory


def reward_curves(
    trajectory: Trajectory,
    *,
    rolling_window: int = 10,
    title: str = "Reward curves",
) -> go.Figure:
    """Per-step reward + cumulative reward + rolling-mean overlay.

    For scalar reward: three traces on one axis. For vector reward
    (B{MO env}): one trace per objective, plus per-objective cumulative
    traces on a secondary axis.

    @param trajectory: Loaded episode.
    @type trajectory: L{Trajectory}
    @param rolling_window: Window size for the rolling-mean overlay.
        Set to C{0} or C{1} to disable.
    @type rolling_window: int
    @param title: Figure title.
    @type title: str
    @rtype: L{plotly.graph_objects.Figure}
    """
    r = trajectory.rewards
    fig = go.Figure()
    if r.size == 0:
        return fig.update_layout(title=title)

    if r.ndim == 1:
        steps = np.arange(1, r.size + 1)
        fig.add_trace(go.Scatter(x=steps, y=r, mode="lines", name="reward"))
        fig.add_trace(go.Scatter(x=steps, y=np.cumsum(r), mode="lines", name="cumulative"))
        if rolling_window > 1 and r.size >= rolling_window:
            kernel = np.ones(rolling_window) / rolling_window
            rm = np.convolve(r, kernel, mode="valid")
            x_rm = steps[rolling_window - 1 :]
            fig.add_trace(
                go.Scatter(x=x_rm, y=rm, mode="lines", name=f"rolling mean ({rolling_window})")
            )
    else:
        n_obj = r.shape[1]
        steps = np.arange(1, r.shape[0] + 1)
        for k in range(n_obj):
            fig.add_trace(go.Scatter(x=steps, y=r[:, k], mode="lines", name=f"reward[{k}]"))
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=np.cumsum(r[:, k]),
                    mode="lines",
                    name=f"cumulative[{k}]",
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="env step",
        yaxis_title="reward",
        hovermode="x unified",
    )
    return fig
