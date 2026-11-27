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
Class-level reward-term registry.

Built-in terms register themselves at import time; users can register
custom terms via L{RewardRegistry.register}. The registry is purely
a name→class lookup; instantiation of a term is the caller's job
(terms typically take constructor arguments like C{node_ids}).

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

from typing import ClassVar


class RewardRegistry:
    """Singleton-style registry of reward-term classes.

    @cvar _registry: Backing dict mapping term name to class.
    """

    _registry: ClassVar[dict[str, type]] = {}

    @classmethod
    def register(cls, name: str, term_cls: type) -> None:
        """Register a reward-term class under C{name}.

        @param name: Unique short identifier (typically matches the
            instance's default C{name} attribute).
        @type name: str
        @param term_cls: The reward-term class to register.
        @type term_cls: type
        @raise ValueError: If C{name} is already registered to a
            different class.
        """
        existing = cls._registry.get(name)
        if existing is not None and existing is not term_cls:
            raise ValueError(
                f"Reward term {name!r} already registered to {existing!r}; "
                f"refusing to replace with {term_cls!r}"
            )
        cls._registry[name] = term_cls

    @classmethod
    def get(cls, name: str) -> type:
        """Look up a registered term class by name.

        @param name: Registered identifier.
        @type name: str
        @return: The class previously passed to L{register}.
        @rtype: type
        @raise KeyError: If C{name} is not registered.
        """
        try:
            return cls._registry[name]
        except KeyError as e:
            raise KeyError(
                f"Reward term {name!r} not registered; known: {sorted(cls._registry)}"
            ) from e

    @classmethod
    def names(cls) -> list[str]:
        """List all registered names in sorted order.

        @rtype: list[str]
        """
        return sorted(cls._registry)

    @classmethod
    def clear(cls) -> None:
        """Drop all registrations. Intended for tests only."""
        cls._registry.clear()
