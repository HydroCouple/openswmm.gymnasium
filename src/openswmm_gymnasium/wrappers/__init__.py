"""
openswmm_gymnasium.wrappers
============================

Composable L{gymnasium.Wrapper} subclasses. Plan §6 layout / §10 P5.

  - L{RescaleBoxActions} — recursively rescales every L{gymnasium.spaces.Box}
    leaf of a (possibly nested) action space into a unit interval, so an
    agent that emits actions in C{[0, 1]} (or any other window) is
    automatically translated into the env's physical bounds.
  - L{MaskDesignAction} — flatten a CIP+RTC env so the agent only sees
    the runtime portion; the design is sampled once at C{reset()} and
    frozen for the episode.
  - L{MaskRuntimeAction} — flatten a CIP+RTC env so the agent only sees
    the design portion; the runtime is filled with a fixed (midpoint)
    setting each step.
  - L{LinearScalarize} — convert vector reward to scalar via dot product
    with a weight vector.
  - L{TchebycheffScalarize} — convert vector reward to scalar via
    weighted-Tchebycheff distance to a utopia point.
  - L{ForecastObservation} — append features from a user-supplied
    forecast callable to the observation vector.
  - L{RecordTrajectory} — write one JSONL file per episode capturing
    actions / observations / rewards / info, consumed by the
    L{openswmm_gymnasium.viz} module (§5.5) and the regression-test
    goldens (§8.3).

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from openswmm_gymnasium.wrappers.action_mask import (
    MaskDesignAction,
    MaskRuntimeAction,
)
from openswmm_gymnasium.wrappers.action_remap import RescaleBoxActions
from openswmm_gymnasium.wrappers.forecast import ForecastObservation
from openswmm_gymnasium.wrappers.record_trajectory import RecordTrajectory
from openswmm_gymnasium.wrappers.scalarize import (
    LinearScalarize,
    TchebycheffScalarize,
)

__all__ = [
    "RescaleBoxActions",
    "MaskDesignAction",
    "MaskRuntimeAction",
    "LinearScalarize",
    "TchebycheffScalarize",
    "ForecastObservation",
    "RecordTrajectory",
]
