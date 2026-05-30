"""
Pareto-front scatter (2-D or 3-D) over a multi-episode run.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from openswmm_gymnasium.scoring import pareto_front as pf_extract
from openswmm_gymnasium.viz.trajectory import TrajectoryRun


def pareto_front(
    run: TrajectoryRun,
    *,
    dimensions: tuple[int, ...] = (0, 1),
    title: str = "Pareto front",
) -> go.Figure:
    """Scatter of cumulative cost vectors across episodes.

    Non-dominated episodes are highlighted in a distinct color.
    Supports 2-D (default) and 3-D plots; higher dimensions raise.

    @param run: Multi-episode run.
    @type run: L{TrajectoryRun}
    @param dimensions: Indices into the cumulative-cost vector to plot.
    @type dimensions: tuple of int
    @param title: Figure title.
    @type title: str
    @rtype: L{plotly.graph_objects.Figure}
    @raise ValueError: If C{len(dimensions)} not in C{{2, 3}}.
    """
    if len(dimensions) not in (2, 3):
        raise ValueError(f"pareto_front supports 2-D or 3-D plots; got {len(dimensions)} dims")
    mat = run.cumulative_cost_matrix()
    fig = go.Figure()
    if mat.size == 0:
        return fig.update_layout(title=title)

    pts = mat[:, list(dimensions)]
    nd = pf_extract(pts)
    # Build a quick "is non-dominated" mask by matching rows.
    nd_set = {tuple(row) for row in nd}
    is_nd = np.array([tuple(row) in nd_set for row in pts])

    dom = pts[~is_nd]
    front = pts[is_nd]

    if len(dimensions) == 2:
        fig.add_trace(
            go.Scatter(
                x=dom[:, 0],
                y=dom[:, 1],
                mode="markers",
                name="dominated",
                marker={"color": "lightgray", "size": 8},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=front[:, 0],
                y=front[:, 1],
                mode="markers",
                name="Pareto front",
                marker={"color": "crimson", "size": 10, "symbol": "diamond"},
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title=f"objective {dimensions[0]}",
            yaxis_title=f"objective {dimensions[1]}",
        )
    else:
        fig.add_trace(
            go.Scatter3d(
                x=dom[:, 0],
                y=dom[:, 1],
                z=dom[:, 2],
                mode="markers",
                name="dominated",
                marker={"color": "lightgray", "size": 4},
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=front[:, 0],
                y=front[:, 1],
                z=front[:, 2],
                mode="markers",
                name="Pareto front",
                marker={"color": "crimson", "size": 6, "symbol": "diamond"},
            )
        )
        fig.update_layout(
            title=title,
            scene={
                "xaxis_title": f"obj {dimensions[0]}",
                "yaxis_title": f"obj {dimensions[1]}",
                "zaxis_title": f"obj {dimensions[2]}",
            },
        )
    return fig
