"""
Plotly figure factories.

Each public function returns a L{plotly.graph_objects.Figure}. Plan §5.5.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
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
