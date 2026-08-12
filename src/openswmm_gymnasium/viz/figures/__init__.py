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
Plotly figure factories.

Each public function returns a L{plotly.graph_objects.Figure}. Plan §5.5.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from openswmm_gymnasium.viz.figures.action_timeseries import action_timeseries
from openswmm_gymnasium.viz.figures.flooding_attribution import flooding_attribution
from openswmm_gymnasium.viz.figures.hypervolume_trace import hypervolume_trace
from openswmm_gymnasium.viz.figures.network_state_heatmap import network_state_heatmap
from openswmm_gymnasium.viz.figures.objective_radar import objective_radar
from openswmm_gymnasium.viz.figures.pareto_front import pareto_front
from openswmm_gymnasium.viz.figures.reward_curves import reward_curves
from openswmm_gymnasium.viz.figures.trajectory_replay import trajectory_replay

__all__ = [
    "reward_curves",
    "pareto_front",
    "hypervolume_trace",
    "action_timeseries",
    "network_state_heatmap",
    "flooding_attribution",
    "objective_radar",
    "trajectory_replay",
]
