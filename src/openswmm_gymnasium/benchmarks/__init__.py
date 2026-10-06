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
openswmm_gymnasium.benchmarks
==============================

10 contrived multi-objective benchmark scenarios. Plan §6A.

Each scenario lives in its own subpackage exposing:

  - A C{make_env(variant, mo)} factory.
  - Registered L{gymnasium} IDs of the form
    C{OpenSWMM/<Scenario>-<Variant>[-MO]-v0}.

Importing this package triggers Gymnasium registration for every
scenario subpackage it contains. The first scenario to land is
L{openswmm_gymnasium.benchmarks.b01_twin_tank} — the architectural
proof; b02-b10 land in subsequent phases.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from openswmm_gymnasium.benchmarks import b01_twin_tank  # noqa: F401 — triggers registration

__all__ = ["b01_twin_tank"]
