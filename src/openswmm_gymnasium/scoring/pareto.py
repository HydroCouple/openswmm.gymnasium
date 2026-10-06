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
Pareto-front extraction for minimisation problems.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def is_dominated(point: ArrayLike, others: ArrayLike) -> bool:
    """Return whether C{point} is dominated by at least one row of C{others}.

    For minimisation: row C{j} dominates C{point} iff C{j[d] <= point[d]}
    for every dimension C{d} and C{j[d] < point[d]} for at least one C{d}.

    @param point: 1-D array of length C{d}.
    @type point: array_like
    @param others: 2-D array of shape C{(n, d)}.
    @type others: array_like
    @return: C{True} if some row of C{others} strictly dominates C{point}.
    @rtype: bool
    """
    p = np.asarray(point, dtype=float)
    o = np.asarray(others, dtype=float)
    if o.ndim != 2:
        raise ValueError("others must be a 2-D array of shape (n, d)")
    le = np.all(o <= p, axis=1)
    lt = np.any(o < p, axis=1)
    return bool(np.any(le & lt))


def pareto_front(points: ArrayLike) -> NDArray[np.float64]:
    """Return the non-dominated subset of C{points} (minimisation).

    @param points: 2-D array of shape C{(n, d)}.
    @type points: array_like
    @return: Subset of rows of C{points} that are non-dominated, in
        the original ordering.
    @rtype: numpy.ndarray
    """
    p = np.asarray(points, dtype=float)
    if p.size == 0:
        return p.reshape(0, p.shape[1] if p.ndim == 2 else 0)
    if p.ndim != 2:
        raise ValueError("points must be a 2-D array of shape (n, d)")
    n = p.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        # Row i is dominated if any other row j satisfies j <= i with at
        # least one strict.
        le = np.all(p <= p[i], axis=1)
        lt = np.any(p < p[i], axis=1)
        dom_by = le & lt
        dom_by[i] = False
        if np.any(dom_by):
            dominated[i] = True
    return p[~dominated]
