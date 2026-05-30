"""
Additive epsilon (ε) indicator.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def epsilon_indicator(
    front_a: ArrayLike,
    front_b: ArrayLike,
) -> float:
    """Additive ε-indicator C{I_ε+(A, B)}.

    Smallest C{ε} such that for every B{b ∈ B} there exists B{a ∈ A}
    with C{a_d - ε <= b_d} for every dimension C{d}.
    Equivalently C{ε = max_b min_a max_d (a_d - b_d)}.

    @param front_a: 2-D array C{(n, d)} — typically the approximation.
    @type front_a: array_like
    @param front_b: 2-D array C{(m, d)} — typically the reference front.
    @type front_b: array_like
    @return: The ε value, or C{inf} if either input is empty.
    @rtype: float
    """
    a = np.asarray(front_a, dtype=float)
    b = np.asarray(front_b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("inf")
    diff = a[None, :, :] - b[:, None, :]  # (m, n, d)
    per_pair_max = diff.max(axis=2)  # (m, n)
    per_b_min = per_pair_max.min(axis=1)  # (m,)
    return float(per_b_min.max())
