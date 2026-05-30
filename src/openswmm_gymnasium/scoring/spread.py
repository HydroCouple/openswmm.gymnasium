"""
Schott's spacing metric — a simple front-distribution measure.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def spread(front: ArrayLike) -> float:
    """Standard deviation of nearest-neighbour Euclidean distances.

    For a front with fewer than two points, returns C{0.0}.

    @param front: 2-D array C{(n, d)}.
    @type front: array_like
    @return: Stddev of per-point nearest-neighbour distances. Lower
        means more uniform spacing.
    @rtype: float
    """
    p = np.asarray(front, dtype=float)
    if p.size == 0 or len(p) < 2:
        return 0.0
    # Pairwise distances; mask diagonal with +inf, take per-row min.
    diff = p[:, None, :] - p[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    nn = dists.min(axis=1)
    return float(nn.std())
