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
@license: Apache-2.0
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
