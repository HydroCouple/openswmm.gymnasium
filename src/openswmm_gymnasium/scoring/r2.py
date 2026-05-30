"""
R2 indicator with weighted Tchebycheff utility.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def r2_indicator(
    front: ArrayLike,
    weights: ArrayLike,
    reference: ArrayLike,
) -> float:
    """R2 indicator over a set of weight vectors.

    For each weight vector C{w}, compute the minimum over the front of
    the weighted-Tchebycheff utility
    C{max_d w_d * |p_d - reference_d|}; the R2 indicator is the mean of
    those minima over all weight vectors. Lower is better.

    @param front: 2-D array C{(n, d)}.
    @type front: array_like
    @param weights: 2-D array C{(k, d)} of weight vectors.
    @type weights: array_like
    @param reference: 1-D array C{(d,)} of the utopia / ideal point.
    @type reference: array_like
    @return: Mean weighted-Tchebycheff utility.
    @rtype: float
    """
    p = np.asarray(front, dtype=float)
    w = np.asarray(weights, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if p.size == 0 or w.size == 0:
        return 0.0
    diff = np.abs(p[None, :, :] - ref[None, None, :])  # (k, n, d)
    weighted = w[:, None, :] * diff  # (k, n, d)
    util_per_wp = weighted.max(axis=2)  # (k, n)
    min_per_w = util_per_wp.min(axis=1)  # (k,)
    return float(min_per_w.mean())
