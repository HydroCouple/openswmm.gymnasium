"""
Hypervolume indicator.

Plan §5.3: 2-D exact closed-form; Monte-Carlo for higher dimensions.
We deliberately B{do not} vendor C{pymoo} / C{platypus-opt}'s HV
implementations; the metrics live here so the package has no required
dependency on a third-party MOO library.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from openswmm_gymnasium.scoring.normalization import normalize
from openswmm_gymnasium.scoring.pareto import pareto_front


def hypervolume(
    points: ArrayLike,
    reference: ArrayLike,
    *,
    method: str = "auto",
    mc_samples: int = 100_000,
    rng: np.random.Generator | None = None,
) -> float:
    """Hypervolume dominated by C{points} w.r.t. C{reference}.

    For C{d <= 2} an exact closed-form is used (sweep). For C{d >= 3}
    Monte-Carlo is used. Setting C{method="mc"} forces Monte-Carlo even
    in low dimensions (useful for cross-checking).

    @param points: 2-D array of shape C{(n, d)} in minimisation
        convention.
    @type points: array_like
    @param reference: 1-D array of length C{d} giving the worst-case
        (nadir) reference point. Points not strictly dominating
        C{reference} contribute nothing.
    @type reference: array_like
    @param method: C{"auto"}, C{"exact"} (only valid for C{d <= 2}), or
        C{"mc"}.
    @type method: str
    @param mc_samples: Number of Monte-Carlo samples when C{method} is
        C{"mc"} or auto-dispatched for C{d >= 3}.
    @type mc_samples: int
    @param rng: Optional pre-seeded generator for reproducible MC.
    @type rng: numpy.random.Generator or C{None}
    @return: Hypervolume value, C{0.0} for empty / fully dominated
        inputs.
    @rtype: float
    @raise ValueError: If C{method} is unknown or C{"exact"} is
        requested for C{d > 2}.
    """
    p = np.asarray(points, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if p.size == 0:
        return 0.0
    if p.ndim != 2:
        raise ValueError("points must be a 2-D array of shape (n, d)")
    d = p.shape[1]
    if ref.shape != (d,):
        raise ValueError(f"reference shape {ref.shape} != ({d},)")

    # Discard rows not strictly dominating the reference.
    strict = np.all(p < ref, axis=1)
    p = p[strict]
    if p.size == 0:
        return 0.0

    # Take the non-dominated subset only.
    p = pareto_front(p)

    if method == "exact" and d > 2:
        raise ValueError(
            f"method='exact' only supports d <= 2 (got d={d}); use method='auto' or method='mc'."
        )

    if d == 1:
        return float(ref[0] - p[:, 0].min())

    if d == 2 and method != "mc":
        return _hv_2d_exact(p, ref)

    return _hv_monte_carlo(p, ref, mc_samples=mc_samples, rng=rng)


def normalized_hypervolume(
    points: ArrayLike,
    ideal: ArrayLike,
    reference: ArrayLike,
    *,
    method: str = "auto",
    mc_samples: int = 100_000,
    rng: np.random.Generator | None = None,
) -> float:
    """Hypervolume in normalised C{[0, 1]^d} space.

    Normalises C{points} via L{normalize} so the reference becomes
    C{(1, ..., 1)} and the ideal becomes C{(0, ..., 0)}; the returned
    value is therefore in C{[0, 1]} regardless of the original
    objective scales.

    @rtype: float
    """
    norm = normalize(points, ideal, reference)
    one_ref = np.ones(np.asarray(reference).shape[0], dtype=float)
    return hypervolume(norm, one_ref, method=method, mc_samples=mc_samples, rng=rng)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _hv_2d_exact(p: np.ndarray, ref: np.ndarray) -> float:
    """2-D HV by sweeping the Pareto front sorted on the first axis."""
    s = p[np.argsort(p[:, 0])]
    n = len(s)
    area = 0.0
    for i in range(n):
        next_f1 = float(s[i + 1, 0]) if i + 1 < n else float(ref[0])
        width = next_f1 - float(s[i, 0])
        height = float(ref[1]) - float(s[i, 1])
        if width > 0 and height > 0:
            area += width * height
    return area


def _hv_monte_carlo(
    p: np.ndarray,
    ref: np.ndarray,
    *,
    mc_samples: int,
    rng: np.random.Generator | None,
) -> float:
    """Monte-Carlo HV via uniform sampling in C{[min(p, axis=0), ref]}."""
    rng = rng if rng is not None else np.random.default_rng()
    low = p.min(axis=0)
    box_vol = float(np.prod(ref - low))
    if box_vol <= 0:
        return 0.0
    samples = rng.uniform(low=low, high=ref, size=(mc_samples, p.shape[1]))
    dominated = np.zeros(mc_samples, dtype=bool)
    for fp in p:
        dominated |= np.all(samples >= fp, axis=1)
    return float(box_vol * dominated.mean())
