"""
openswmm_gymnasium.envs
========================

Concrete L{gymnasium.Env} subclasses. Plan §2.1.

  - L{SwmmRTCEnv} — runtime-only (RTC); design action portion of the
    Dict action space is empty. P1.
  - L{SwmmCIPEnv} — design-only (CIP); single-step contextual-bandit
    pattern. Runtime action portion is empty. P3.
  - L{SwmmJointCIPRTCEnv} — full hybrid. Design applied at C{reset()},
    runtime applied each C{step()}. P3.
  - L{SwmmControlEnv} — single-step env that evaluates a controller
    policy-parameter vector over a full episode (operational tuning).

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from openswmm_gymnasium.envs.base import SwmmRTCEnv
from openswmm_gymnasium.envs.cip_only import SwmmCIPEnv
from openswmm_gymnasium.envs.control import SwmmControlEnv
from openswmm_gymnasium.envs.joint import SwmmJointCIPRTCEnv
from openswmm_gymnasium.envs.mo import SwmmMORTCEnv

__all__ = [
    "SwmmRTCEnv",
    "SwmmCIPEnv",
    "SwmmJointCIPRTCEnv",
    "SwmmMORTCEnv",
    "SwmmControlEnv",
]
