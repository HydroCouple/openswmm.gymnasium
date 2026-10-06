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
Min-max normalisation against L{ideal} and L{reference} (nadir) points.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def normalize(
    points: ArrayLike,
    ideal: ArrayLike,
    reference: ArrayLike,
) -> NDArray[np.float64]:
    """Return C{(points - ideal) / (reference - ideal)}.

    @param points: Array of shape C{(d,)} or C{(n, d)}.
    @type points: array_like
    @param ideal: 1-D array of length C{d} giving the per-dimension
        best-possible value (lower in minimisation).
    @type ideal: array_like
    @param reference: 1-D array of length C{d} giving the per-dimension
        worst-case (nadir) value used as the hypervolume reference.
    @type reference: array_like
    @return: Same shape as C{points}, values typically in C{[0, 1]} but
        B{not clipped} — callers do their own clipping if required.
    @rtype: numpy.ndarray
    @raise ValueError: If C{reference[d] <= ideal[d]} for any C{d}.
    """
    p = np.asarray(points, dtype=float)
    ideal_a = np.asarray(ideal, dtype=float)
    ref_a = np.asarray(reference, dtype=float)
    span = ref_a - ideal_a
    if np.any(span <= 0):
        raise ValueError(
            f"reference must be strictly greater than ideal in every "
            f"dimension; got ideal={ideal_a}, reference={ref_a}"
        )
    return (p - ideal_a) / span
