"""
openswmm_gymnasium.scoring
==========================

Multi-objective optimisation scoring utilities. Plan §5.3.

All functions operate on numpy arrays of shape C{(n_points, n_objectives)}
in B{minimisation} convention (smaller is better). The env's reward
composer is responsible for the sign-flipping bookkeeping at the
boundary (plan §0 #5); functions here always treat the input as costs.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from openswmm_gymnasium.scoring.epsilon import epsilon_indicator
from openswmm_gymnasium.scoring.hypervolume import hypervolume, normalized_hypervolume
from openswmm_gymnasium.scoring.igd import igd, igd_plus
from openswmm_gymnasium.scoring.normalization import normalize
from openswmm_gymnasium.scoring.pareto import is_dominated, pareto_front
from openswmm_gymnasium.scoring.r2 import r2_indicator
from openswmm_gymnasium.scoring.spread import spread

__all__ = [
    "pareto_front",
    "is_dominated",
    "normalize",
    "hypervolume",
    "normalized_hypervolume",
    "igd",
    "igd_plus",
    "epsilon_indicator",
    "spread",
    "r2_indicator",
]
