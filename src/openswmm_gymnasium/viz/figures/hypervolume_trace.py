"""
Normalised single-point hypervolume per episode.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go

from openswmm_gymnasium.scoring import normalized_hypervolume
from openswmm_gymnasium.viz.trajectory import TrajectoryRun


def hypervolume_trace(
    run: TrajectoryRun,
    *,
    ideal: Sequence[float],
    reference: Sequence[float],
    title: str = "Normalized hypervolume",
) -> go.Figure:
    """Per-episode single-point normalised HV, in C{[0, 1]}.

    Each episode contributes one point — its cumulative cost vector —
    against the supplied C{ideal} and C{reference} for normalisation.

    @param run: Multi-episode run.
    @type run: L{TrajectoryRun}
    @param ideal: Per-objective best-case cost (minimisation).
    @type ideal: sequence of float
    @param reference: Per-objective nadir cost.
    @type reference: sequence of float
    @param title: Figure title.
    @type title: str
    @rtype: L{plotly.graph_objects.Figure}
    """
    fig = go.Figure()
    if len(run) == 0:
        return fig.update_layout(title=title)

    ideal_a = np.asarray(ideal, dtype=float)
    ref_a = np.asarray(reference, dtype=float)
    mat = run.cumulative_cost_matrix()
    if mat.shape[1] != ideal_a.shape[0]:
        raise ValueError(
            f"Number of cost components ({mat.shape[1]}) does not match "
            f"ideal/reference length ({ideal_a.shape[0]})"
        )

    nhv = np.array(
        [normalized_hypervolume(mat[i : i + 1, :], ideal_a, ref_a) for i in range(mat.shape[0])],
        dtype=float,
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(1, nhv.size + 1),
            y=nhv,
            mode="lines+markers",
            name="HV",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="episode",
        yaxis_title="normalised HV",
        yaxis_range=[0.0, 1.0],
    )
    return fig
