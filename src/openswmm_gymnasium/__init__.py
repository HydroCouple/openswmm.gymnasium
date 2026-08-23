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
openswmm.gymnasium
==================

Farama Gymnasium environments for joint CIP + RTC optimization of SWMM
networks, backed by the handle-based, thread-safe C{openswmm.engine} v6
Python API.

See L{docs/IMPLEMENTATION_PLAN} for the authoritative architecture and
scope. Environment IDs are registered at package import time so
C{gymnasium.make("OpenSWMM/...")} works without further setup.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

import gymnasium as _gym

from openswmm_gymnasium._version import __version__

# Re-export the env classes at top level for ``from openswmm_gymnasium
# import SwmmRTCEnv`` ergonomics, but encourage callers to use
# ``gymnasium.make("OpenSWMM/...")`` for the canonical construction path.
from openswmm_gymnasium.envs import (
    SwmmCIPEnv,
    SwmmControlEnv,
    SwmmJointCIPRTCEnv,
    SwmmMORTCEnv,
    SwmmRTCEnv,
)

# ---------------------------------------------------------------------------
# Gymnasium env registration
# ---------------------------------------------------------------------------
#
# Env IDs follow the convention "OpenSWMM/<Scenario>-<Variant>-v<N>".
# Scenarios b01-b10 (plan §6A) register their own IDs in their benchmark
# modules; the entries here are the framework-level minimal envs used by
# the test suite and by external integration smoke tests.

_gym.register(
    id="OpenSWMM/Minimal-RTC-v0",
    entry_point="openswmm_gymnasium.envs:SwmmRTCEnv",
    # No default kwargs — callers must supply inp_path + observation_builder.
)
_gym.register(
    id="OpenSWMM/Minimal-CIP-v0",
    entry_point="openswmm_gymnasium.envs:SwmmCIPEnv",
)
_gym.register(
    id="OpenSWMM/Minimal-Joint-v0",
    entry_point="openswmm_gymnasium.envs:SwmmJointCIPRTCEnv",
)
_gym.register(
    id="OpenSWMM/Minimal-MORTC-v0",
    entry_point="openswmm_gymnasium.envs:SwmmMORTCEnv",
)

# Import the benchmarks package to trigger registration of all bNN scenarios.
# Kept at the end so any registration failure surfaces after the framework
# envs are already registered (so partial-registration failure modes are
# easier to diagnose).
from openswmm_gymnasium import benchmarks  # noqa: F401, E402

__all__ = [
    "__version__",
    "SwmmRTCEnv",
    "SwmmCIPEnv",
    "SwmmJointCIPRTCEnv",
    "SwmmMORTCEnv",
    "SwmmControlEnv",
]
