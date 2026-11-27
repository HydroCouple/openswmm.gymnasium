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
Animated network schematic — node depths colored over time.

The node coordinates and the per-node observation index (i.e. which
column of the observation vector carries each node's depth) must be
supplied by the caller, since the JSONL records do not carry network
topology.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import plotly.graph_objects as go

from openswmm_gymnasium.viz.trajectory import Trajectory


def trajectory_replay(
    trajectory: Trajectory,
    *,
    coords: Mapping[str, tuple[float, float]],
    obs_index: Mapping[str, int],
    title: str = "Trajectory replay",
) -> go.Figure:
    """Animated scatter on user-supplied node coordinates.

    @param trajectory: Loaded episode.
    @type trajectory: L{Trajectory}
    @param coords: Mapping from node ID to C{(x, y)} coordinates.
    @type coords: mapping
    @param obs_index: Mapping from node ID to its column index in
        L{Trajectory.observations}.
    @type obs_index: mapping
    @param title: Figure title.
    @type title: str
    @rtype: L{plotly.graph_objects.Figure}
    @raise ValueError: If C{coords} and C{obs_index} have mismatched keys.
    """
    if set(coords) != set(obs_index):
        raise ValueError("coords and obs_index must share the same set of node IDs")
    nodes = list(coords)
    obs = trajectory.observations
    if obs.size == 0:
        return go.Figure().update_layout(title=title)

    xs = np.array([coords[n][0] for n in nodes], dtype=float)
    ys = np.array([coords[n][1] for n in nodes], dtype=float)
    n_steps = obs.shape[0]
    depths = np.zeros((n_steps, len(nodes)), dtype=float)
    for j, n in enumerate(nodes):
        depths[:, j] = obs[:, obs_index[n]]

    vmin, vmax = float(depths.min()), float(depths.max())

    frames = [
        go.Frame(
            name=str(k),
            data=[
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers+text",
                    text=nodes,
                    textposition="top center",
                    marker={
                        "size": 18,
                        "color": depths[k],
                        "cmin": vmin,
                        "cmax": vmax,
                        "colorscale": "Blues",
                        "showscale": True,
                        "colorbar": {"title": "depth"},
                    },
                )
            ],
        )
        for k in range(n_steps)
    ]
    fig = go.Figure(
        data=frames[0].data if frames else [],
        frames=frames,
    )
    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="y",
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 100}}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
    )
    return fig
