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
Inverted Generational Distance (IGD) and IGD+.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def igd(approximation: ArrayLike, reference_front: ArrayLike) -> float:
    """Inverted Generational Distance.

    For each point in C{reference_front}, compute the Euclidean
    distance to the closest point in C{approximation}; IGD is the
    mean of those distances. Lower is better.

    @param approximation: 2-D array, shape C{(n, d)}.
    @type approximation: array_like
    @param reference_front: 2-D array, shape C{(m, d)}.
    @type reference_front: array_like
    @return: Mean nearest-neighbour distance, or C{inf} if either input
        is empty.
    @rtype: float
    """
    a = np.asarray(approximation, dtype=float)
    r = np.asarray(reference_front, dtype=float)
    if a.size == 0 or r.size == 0:
        return float("inf")
    diff = r[:, None, :] - a[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    return float(dists.min(axis=1).mean())


def igd_plus(approximation: ArrayLike, reference_front: ArrayLike) -> float:
    """IGD+ — distance is computed only along dominated dimensions.

    For minimisation, the IGD+ distance from reference point C{r} to
    approximation point C{a} replaces C{(a - r)} with C{max(0, a - r)}
    before computing the Euclidean norm. The result is zero whenever
    C{a} weakly dominates C{r}, which makes IGD+ Pareto-compliant in a
    way standard IGD is not.

    @rtype: float
    """
    a = np.asarray(approximation, dtype=float)
    r = np.asarray(reference_front, dtype=float)
    if a.size == 0 or r.size == 0:
        return float("inf")
    diff = a[None, :, :] - r[:, None, :]
    diff_clamped = np.maximum(0.0, diff)
    dists = np.sqrt(np.sum(diff_clamped * diff_clamped, axis=2))
    return float(dists.min(axis=1).mean())
