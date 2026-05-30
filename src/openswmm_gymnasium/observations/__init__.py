"""
openswmm_gymnasium.observations
================================

Observation builder and individual feature collectors. Plan §4.

The builder concatenates per-collector contributions into a single
flat L{gymnasium.spaces.Box} (the default) or a structured
L{gymnasium.spaces.Dict} (C{flatten=False}).

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from openswmm_gymnasium.observations.builder import ObservationBuilder

__all__ = ["ObservationBuilder"]
