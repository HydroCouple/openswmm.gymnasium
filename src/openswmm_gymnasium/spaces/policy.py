"""Policy-parameter search space for controller tuning.

Where design factories (L{openswmm_gymnasium.spaces.design}) search *model*
parameters, this space searches *controller* parameters: the tunable scalars of
a L{openswmm_gymnasium.config.MarketConfig} — cost-curve C{onset}/C{steepness}/
C{ceiling} and per-route PID gains — flattened into one bounded vector. The
NSGA-II optimizer searches the vector; L{MarketPolicySpace.unflatten} rebuilds a
full C{MarketConfig} (all non-tuned fields preserved from the base) for the env
to run under L{openswmm_gymnasium.control.MarketController}.

The mapping is total and reversible: C{flatten} reads the addressed scalars in a
stable order, C{unflatten} writes them back (clipped to bounds), so
C{flatten(base)} -> C{unflatten} reproduces the base on the tuned fields.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from openswmm_gymnasium.config import MarketConfig

# Default search bounds per tunable field (overridable in from_config).
_DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "onset": (0.0, 1.0),
    "ceiling": (0.0, 1.0),
    "steepness": (1.0, 40.0),
    "full": (0.0, 1.0),
    "kp": (0.0, 5.0),
    "ki": (0.0, 2.0),
    "kd": (0.0, 1.0),
}


@dataclass(frozen=True)
class PolicyParam:
    """One tunable scalar: a human label, an address, and search bounds.

    @ivar label: Stable, human-readable identifier (e.g. C{"pid:GATE.kp"}).
    @ivar address: C{("curve", curve_name, field)} or
        C{("pid", structure_link_id, field)}.
    @ivar low: Lower search bound.
    @ivar high: Upper search bound.
    """

    label: str
    address: tuple[str, str, str]
    low: float
    high: float


class MarketPolicySpace:
    """A bounded vector space over a market config's tunable controller params.

    @ivar params: The ordered L{PolicyParam} list defining the vector layout.
    """

    def __init__(self, base: MarketConfig, params: list[PolicyParam]) -> None:
        """
        @param base: The configuration supplying every non-tuned field.
        @type base: L{MarketConfig}
        @param params: Ordered tunable parameters (vector layout).
        @type params: list of L{PolicyParam}
        @raise ValueError: If C{params} is empty or has duplicate labels.
        """
        if not params:
            raise ValueError("MarketPolicySpace requires at least one PolicyParam")
        labels = [p.label for p in params]
        if len(set(labels)) != len(labels):
            raise ValueError("MarketPolicySpace param labels must be unique")
        self._base = base
        self.params: list[PolicyParam] = list(params)
        self._low = np.array([p.low for p in params], dtype=np.float32)
        self._high = np.array([p.high for p in params], dtype=np.float32)

    # -- Auto-discovery ------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        base: MarketConfig,
        *,
        bounds: dict[str, tuple[float, float]] | None = None,
        tune_full: bool = False,
    ) -> MarketPolicySpace:
        """Build a space over the standard tunables of *base*.

        Tunes, per cost curve, C{onset} and C{ceiling} plus C{steepness}
        (logistic) or C{full} (piecewise-linear, only when C{tune_full}); and,
        per trade route, the PID gains C{kp}/C{ki}/C{kd}.

        @param base: The base configuration.
        @param bounds: Optional per-field C{(low, high)} overrides merged over
            the defaults (keys: onset, ceiling, steepness, full, kp, ki, kd).
        @param tune_full: Also tune piecewise-linear C{full} knees.
        @rtype: L{MarketPolicySpace}
        """
        b = dict(_DEFAULT_BOUNDS)
        if bounds:
            b.update(bounds)

        params: list[PolicyParam] = []
        for name, curve in base.cost_curves.items():
            params.append(PolicyParam(f"curve:{name}.onset", ("curve", name, "onset"), *b["onset"]))
            params.append(
                PolicyParam(f"curve:{name}.ceiling", ("curve", name, "ceiling"), *b["ceiling"])
            )
            if curve.type == "logistic":
                params.append(
                    PolicyParam(f"curve:{name}.steepness", ("curve", name, "steepness"), *b["steepness"])
                )
            elif tune_full:
                params.append(PolicyParam(f"curve:{name}.full", ("curve", name, "full"), *b["full"]))

        for route in base.trade_routes:
            link = route.structure_link_id
            for field in ("kp", "ki", "kd"):
                params.append(PolicyParam(f"pid:{link}.{field}", ("pid", link, field), *b[field]))

        return cls(base, params)

    # -- Vector layout -------------------------------------------------------

    @property
    def labels(self) -> list[str]:
        """The vector's parameter labels, in order."""
        return [p.label for p in self.params]

    @property
    def low(self) -> np.ndarray:
        """Lower bounds (float32, shape C{(n,)})."""
        return self._low.copy()

    @property
    def high(self) -> np.ndarray:
        """Upper bounds (float32, shape C{(n,)})."""
        return self._high.copy()

    @property
    def space(self) -> spaces.Box:
        """The C{gymnasium.spaces.Box} over the flat decision vector."""
        return spaces.Box(low=self._low, high=self._high, shape=(len(self.params),), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.params)

    # -- Flatten / unflatten -------------------------------------------------

    def flatten(self, config: MarketConfig) -> np.ndarray:
        """Read the tuned scalars from *config* into a vector (this layout)."""
        return np.array(
            [_get_address(config, p.address) for p in self.params], dtype=np.float32
        )

    def unflatten(self, vector: np.ndarray) -> MarketConfig:
        """Build a full C{MarketConfig} from a decision *vector*.

        Every non-tuned field comes from the base config; tuned scalars are
        overwritten with the (bound-clipped) vector values. The result is a deep
        copy, so the base is never mutated.

        @param vector: Decision vector, length C{len(self)}.
        @raise ValueError: If C{vector} length mismatches the layout.
        @rtype: L{MarketConfig}
        """
        vec = np.asarray(vector, dtype=np.float64).ravel()
        if vec.shape[0] != len(self.params):
            raise ValueError(
                f"vector length {vec.shape[0]} != param count {len(self.params)}"
            )
        clipped = np.clip(vec, self._low.astype(np.float64), self._high.astype(np.float64))
        out = copy.deepcopy(self._base)
        for p, value in zip(self.params, clipped, strict=True):
            _set_address(out, p.address, float(value))
        return out


# =============================================================================
# Address get/set helpers
# =============================================================================


def _get_address(config: MarketConfig, address: tuple[str, str, str]) -> float:
    kind, key, field = address
    if kind == "curve":
        return float(getattr(config.cost_curves[key], field))
    if kind == "pid":
        return float(getattr(_route_pid(config, key), field))
    raise ValueError(f"unknown policy address kind: {kind!r}")  # pragma: no cover


def _set_address(config: MarketConfig, address: tuple[str, str, str], value: float) -> None:
    kind, key, field = address
    if kind == "curve":
        setattr(config.cost_curves[key], field, value)
    elif kind == "pid":
        setattr(_route_pid(config, key), field, value)
    else:  # pragma: no cover
        raise ValueError(f"unknown policy address kind: {kind!r}")


def _route_pid(config: MarketConfig, structure_link_id: str):
    for route in config.trade_routes:
        if route.structure_link_id == structure_link_id:
            return route.pid
    raise KeyError(f"no trade route with structure_link_id {structure_link_id!r}")
