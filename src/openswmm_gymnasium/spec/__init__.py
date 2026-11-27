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

"""Declarative environment specs: build any env from plain JSON.

  - L{config<openswmm_gymnasium.spec.config>} -- Pydantic models
    (L{EnvConfig<openswmm_gymnasium.spec.config.EnvConfig>}) and
    L{build_env<openswmm_gymnasium.spec.config.build_env>}.
  - L{registry<openswmm_gymnasium.spec.registry>} -- kind names mapped to
    gym classes, each with a params schema.
  - L{envs<openswmm_gymnasium.spec.envs>} -- a thread-safe registry of open
    envs, episode runner and JSON-safe conversion.

Needs the C{spec} extra (C{pip install openswmm.gymnasium[spec]}), which adds
pydantic; the rest of the package does not import this module.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

try:
    import pydantic  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise ImportError(
        "openswmm_gymnasium.spec needs pydantic: pip install 'openswmm.gymnasium[spec]'"
    ) from exc

from openswmm_gymnasium.spec.config import (
    ActionFactorySpec,
    CellFieldSpec,
    EnvConfig,
    ObservationSpec,
    RewardTermSpec,
    WrapperSpec,
    build_env,
)
from openswmm_gymnasium.spec.envs import EnvManager, build_action, json_safe, run_episode
from openswmm_gymnasium.spec.errors import ErrorCode, SpecError
from openswmm_gymnasium.spec.registry import KindSpec, get_kind, list_kinds, resolve_kind

__all__ = [
    "ActionFactorySpec",
    "CellFieldSpec",
    "EnvConfig",
    "EnvManager",
    "ErrorCode",
    "KindSpec",
    "ObservationSpec",
    "RewardTermSpec",
    "SpecError",
    "WrapperSpec",
    "build_action",
    "build_env",
    "get_kind",
    "json_safe",
    "list_kinds",
    "resolve_kind",
    "run_episode",
]
