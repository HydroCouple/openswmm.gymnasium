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

"""Errors raised by the environment spec layer.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import re


class ErrorCode:
    """Codes prefixed to every L{SpecError} message, e.g. C{"[VALIDATION_ERROR] ..."}."""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
    ENGINE_ERROR = "ENGINE_ERROR"
    MAX_SESSIONS_REACHED = "MAX_SESSIONS_REACHED"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"


class SpecError(Exception):
    """An invalid environment spec, or an env-manager request that cannot be served.

    The message starts with its code in brackets (C{"[SESSION_NOT_FOUND] No open
    env 'x'."}), so callers such as the MCP server can pass it on unchanged.
    Deliberately not a C{ValueError}: pydantic would wrap one raised inside a
    validator into a C{ValidationError} and lose the code.
    """

    @property
    def code(self) -> str:
        """The bracketed code at the start of the message."""
        m = re.match(r"\[(\w+)\]", str(self))
        return m.group(1) if m else ErrorCode.VALIDATION_ERROR
