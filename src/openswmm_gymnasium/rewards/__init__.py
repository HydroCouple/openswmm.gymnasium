"""
openswmm_gymnasium.rewards
==========================

Reward term registry + built-in terms. Plan §5.

All built-in terms are framed as B{cost-to-minimize} (positive
contribution = bad outcome), except those flagged
C{direction="maximize"} (e.g. L{ReliabilityMargin}). The env negates
the aggregated cost so higher reward is better, per Gymnasium
convention (plan §0 #5).

Built-in terms self-register with L{RewardRegistry} at import time so
C{RewardRegistry.get("flooding_volume")} works without any explicit
imports beyond this module.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from openswmm_gymnasium.rewards.registry import RewardRegistry
from openswmm_gymnasium.rewards.terms import (
    CSOVolume,
    FloodingVolume,
    PeakOutflow,
    ReliabilityMargin,
    RewardTerm,
    SetpointSmoothness,
)

# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------

RewardRegistry.register("flooding_volume", FloodingVolume)
RewardRegistry.register("cso_volume", CSOVolume)
RewardRegistry.register("peak_outflow", PeakOutflow)
RewardRegistry.register("reliability_margin", ReliabilityMargin)
RewardRegistry.register("setpoint_smoothness", SetpointSmoothness)


__all__ = [
    "RewardTerm",
    "RewardRegistry",
    "FloodingVolume",
    "CSOVolume",
    "PeakOutflow",
    "ReliabilityMargin",
    "SetpointSmoothness",
]
