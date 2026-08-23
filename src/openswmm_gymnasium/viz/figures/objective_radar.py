# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 Caleb Buahin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Radar chart comparing per-objective episode totals across trajectories.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

from collections.abc import Sequence

import plotly.graph_objects as go

from openswmm_gymnasium.viz.trajectory import Trajectory


def objective_radar(
    trajectories: Sequence[Trajectory],
    *,
    labels: Sequence[str] | None = None,
    title: str = "Per-objective totals",
) -> go.Figure:
    """One radar trace per trajectory comparing cumulative-cost vectors.

    The set of objective axes is the B{union} of reward-component names
    across all input trajectories. Trajectories missing a component
    contribute C{0} on that axis.

    @param trajectories: Trajectories to compare. Two or more typical.
    @type trajectories: sequence of L{Trajectory}
    @param labels: Per-trajectory labels for the legend. Defaults to
        C{"policy[0]", "policy[1]", ...}.
    @type labels: sequence of str or C{None}
    @param title: Figure title.
    @type title: str
    @rtype: L{plotly.graph_objects.Figure}
    """
    fig = go.Figure()
    trajs = list(trajectories)
    if not trajs:
        return fig.update_layout(title=title)

    # Union of axis names.
    axes = sorted({k for t in trajs for k in t.reward_components()})
    if not axes:
        return fig.update_layout(title=title)

    labels = list(labels) if labels is not None else [f"policy[{i}]" for i in range(len(trajs))]
    if len(labels) != len(trajs):
        raise ValueError(
            f"labels length ({len(labels)}) does not match trajectory count ({len(trajs)})"
        )

    for label, t in zip(labels, trajs, strict=True):
        comps = t.reward_components()
        values = [float(comps[a].sum()) if a in comps else 0.0 for a in axes]
        # Close the polygon.
        values_loop = values + values[:1]
        axes_loop = axes + axes[:1]
        fig.add_trace(
            go.Scatterpolar(
                r=values_loop,
                theta=axes_loop,
                fill="toself",
                name=label,
            )
        )
    fig.update_layout(
        title=title,
        polar={"radialaxis": {"visible": True}},
    )
    return fig
