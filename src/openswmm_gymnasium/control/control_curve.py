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

"""Reactive piecewise-linear (PWL) control-curve strategy.

A L{ControlCurveController} maps a single observed state per controlled link
(typically the downstream node's fractional depth) through a fixed
piecewise-linear B{breakpoint curve} to that link's C{[0,1]} setting. The curve
breakpoints (the C{y} value at each C{x} knot) are B{static per episode} — they
are the searchable decision variables — yet the controller is B{reactive at
runtime}: every control step it reads live state and looks the setting up on the
curve. This is the seam the optimizer tunes (one decision vector per episode,
exactly like a design factory), evaluated by a closed-loop episode.

The strategy is split into two pure, engine-testable pieces, mirroring the
market controller:

  - L{ControlCurveMetricReader} turns live solver state into the per-link
    C{x} value the curve is indexed by (raw, or normalized to a fraction of
    the observed node's full depth when C{x_normalized}).
  - L{ControlCurveController} consumes that C{{link_id: x}} mapping and returns
    C{{link_id: setting}} via PWL interpolation, optional per-step rate
    limiting, and an optional multi-step control cadence.

Both consume a L{ControlCurvePolicy} — the decoded, ready-to-apply policy that
L{openswmm_gymnasium.spaces.control_curve.ControlCurvePolicySpace.unflatten}
produces from a decision vector (with monotonic projection already applied).

B{Determinism}: no RNG anywhere; identical params produce an identical
trajectory (the optimizer seed only affects sampling, never the controller).

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from openswmm_gymnasium._engine import SolverAdapter

# Supported observation attributes -> SolverAdapter node getter name. The
# divisor for ``x_normalized`` is always the node's max (rim) depth.
_NODE_OBS_GETTERS: dict[str, str] = {
    "depthN": "get_depth",
    "depth": "get_depth",
    "headN": "get_head",
    "volumeN": "get_volume",
    "inflowN": "get_inflow",
}

#: Valid monotonicity constraints for a curve.
MONOTONIC_MODES = ("none", "nonincreasing", "nondecreasing")


# =============================================================================
# Decoded policy data model
# =============================================================================


@dataclass(frozen=True)
class CurveAsset:
    """One controlled link and the breakpoint curve driving it.

    @ivar link_id: Controlled link ID (orifice / weir / pump / conduit).
    @ivar obs_node: Observed node feeding the curve's C{x} axis.
    @ivar obs_attr: Observation attribute (see L{_NODE_OBS_GETTERS}).
    @ivar x_knots: Monotonically increasing knot positions (length >= 2).
    @ivar y_values: Setting at each knot, in C{[0,1]}, same length as
        C{x_knots}; monotonic projection (if any) is already applied.
    """

    link_id: str
    obs_node: str
    obs_attr: str
    x_knots: tuple[float, ...]
    y_values: tuple[float, ...]


@dataclass(frozen=True)
class ControlCurvePolicy:
    """A fully decoded control-curve policy, ready to install on an env.

    @ivar assets: Per-link curves, in stable actuation order.
    @ivar x_normalized: Whether C{x} is a fraction of observed full depth.
    @ivar rate_limit: Max C{|Δsetting|} per control step, or C{None} for off.
    @ivar control_interval_steps: Recompute the curve every N control calls
        (>= 1); the previous setting is held in between.
    """

    assets: tuple[CurveAsset, ...]
    x_normalized: bool
    rate_limit: float | None
    control_interval_steps: int

    @property
    def link_ids(self) -> list[str]:
        """Controlled link IDs in actuation order.

        @rtype: list of str
        """
        return [a.link_id for a in self.assets]


# =============================================================================
# Pure helpers
# =============================================================================


def pwl_interp(
    x: float, x_knots: Sequence[float], y_values: Sequence[float]
) -> float:
    """Piecewise-linear interpolation with flat extrapolation, clamped to [0,1].

    C{x} is first clamped to C{[x_knots[0], x_knots[-1]]} (flat extrapolation
    beyond the ends), linearly interpolated between the bracketing knots, then
    the result is clamped to C{[0, 1]}.

    @param x: Query position.
    @type x: float
    @param x_knots: Strictly increasing knot positions (length >= 2).
    @type x_knots: sequence of float
    @param y_values: Setting at each knot, same length as C{x_knots}.
    @type y_values: sequence of float
    @return: Interpolated setting in C{[0, 1]}.
    @rtype: float
    """
    n = len(x_knots)
    if x <= x_knots[0]:
        y = y_values[0]
    elif x >= x_knots[-1]:
        y = y_values[-1]
    else:
        # Find the bracketing segment [x_knots[i], x_knots[i+1]].
        i = 0
        while i < n - 1 and x > x_knots[i + 1]:
            i += 1
        x0, x1 = x_knots[i], x_knots[i + 1]
        y0, y1 = y_values[i], y_values[i + 1]
        span = x1 - x0
        frac = 0.0 if span <= 0.0 else (x - x0) / span
        y = y0 + frac * (y1 - y0)
    return 0.0 if y < 0.0 else 1.0 if y > 1.0 else float(y)


def project_monotonic(y_values: Sequence[float], mode: str) -> tuple[float, ...]:
    """Project a setting vector onto the requested monotonicity constraint.

    Uses a deterministic, idempotent running clamp:

      - C{"nonincreasing"} — running minimum (each value capped by its
        predecessor).
      - C{"nondecreasing"} — running maximum (each value floored by its
        predecessor).
      - C{"none"} — returned unchanged.

    Applied at decode time so MOEAs that ignore constraints still produce
    feasible curves. Re-applying the projection is a no-op.

    @param y_values: Raw per-knot settings.
    @type y_values: sequence of float
    @param mode: One of L{MONOTONIC_MODES}.
    @type mode: str
    @return: The projected settings.
    @rtype: tuple of float
    @raise ValueError: If C{mode} is not a valid monotonicity mode.
    """
    if mode not in MONOTONIC_MODES:
        raise ValueError(
            f"monotonic must be one of {MONOTONIC_MODES}, got {mode!r}"
        )
    vals = [float(v) for v in y_values]
    if mode == "none" or not vals:
        return tuple(vals)
    out = [vals[0]]
    if mode == "nonincreasing":
        for v in vals[1:]:
            out.append(min(out[-1], v))
    else:  # nondecreasing
        for v in vals[1:]:
            out.append(max(out[-1], v))
    return tuple(out)


# =============================================================================
# Metric reader
# =============================================================================


@dataclass
class _ObsProbe:
    """Resolved per-asset read plan: getter, engine index, divisor."""

    link_id: str
    getter_name: str
    idx: int
    divisor: float  # full (rim) depth when normalizing; else 1.0


class ControlCurveMetricReader:
    """Turn live solver state into the C{{link_id: x}} the curves index by.

    Construct from a L{ControlCurvePolicy}, L{bind} once against the
    open/initialized adapter (resolves observed-node indices and caches the
    normalization divisor), then call L{read} each control step. The reader
    depends only on the policy's B{static} asset definitions — the searched
    C{y_values} are irrelevant to it.
    """

    def __init__(self, policy: ControlCurvePolicy) -> None:
        """
        @param policy: The decoded control-curve policy.
        @type policy: L{ControlCurvePolicy}
        """
        self._assets = list(policy.assets)
        self._x_normalized = bool(policy.x_normalized)
        self._probes: list[_ObsProbe] | None = None

    def bind(self, adapter: SolverAdapter) -> None:
        """Resolve observed-node indices and cache normalization divisors.

        @param adapter: Adapter wrapping the open, initialized solver.
        @type adapter: L{SolverAdapter}
        @raise ValueError: If an asset's C{obs_attr} is unsupported.
        @raise Exception: From the engine when an C{obs_node} ID is unknown.
        """
        probes: list[_ObsProbe] = []
        for a in self._assets:
            getter = _NODE_OBS_GETTERS.get(a.obs_attr)
            if getter is None:
                raise ValueError(
                    f"asset {a.link_id!r}: unsupported obs_attr {a.obs_attr!r}; "
                    f"valid: {sorted(_NODE_OBS_GETTERS)}"
                )
            idx = adapter.nodes.get_index(a.obs_node)
            divisor = 1.0
            if self._x_normalized:
                divisor = float(adapter.nodes.get_max_depth(idx))
                if divisor <= 0.0:
                    # Degenerate geometry: fall back to raw value.
                    divisor = 1.0
            probes.append(_ObsProbe(a.link_id, getter, idx, divisor))
        self._probes = probes

    def read(self, adapter: SolverAdapter) -> dict[str, float]:
        """Return C{{link_id: x}} for the current solver state.

        @param adapter: Adapter wrapping the running solver.
        @type adapter: L{SolverAdapter}
        @return: The per-link curve index value (normalized when requested).
        @rtype: dict of str to float
        """
        assert self._probes is not None, "bind() before read()"
        out: dict[str, float] = {}
        for p in self._probes:
            raw = float(getattr(adapter.nodes, p.getter_name)(p.idx))
            out[p.link_id] = raw / p.divisor
        return out


# =============================================================================
# Controller
# =============================================================================


class ControlCurveController:
    """Reactive PWL breakpoint controller.

    Implements L{openswmm_gymnasium.control.base.Controller}. Each control step
    it reads the per-link C{x} value from C{metrics}, interpolates the setting
    on that link's curve, optionally rate-limits the change, and returns the
    new settings. With C{control_interval_steps > 1} it recomputes only every
    Nth call and holds the previous setting in between.

    @ivar _policy: The decoded policy driving the controller.
    """

    def __init__(self, policy: ControlCurvePolicy) -> None:
        """
        @param policy: The decoded control-curve policy.
        @type policy: L{ControlCurvePolicy}
        """
        self._assets = list(policy.assets)
        self._rate_limit = policy.rate_limit
        self._interval = max(1, int(policy.control_interval_steps))
        self._structure_ids = [a.link_id for a in self._assets]
        self._prev: dict[str, float] = {}
        self._call = 0

    @property
    def structure_ids(self) -> list[str]:
        """Controlled link IDs, in actuation order.

        @rtype: list of str
        """
        return list(self._structure_ids)

    def reset(self) -> None:
        """Clear cross-step state at the start of an episode."""
        self._prev = {}
        self._call = 0

    def compute_settings(
        self, metrics: Mapping[str, float], dt_seconds: float
    ) -> dict[str, float]:
        """Return C{{link_id: setting}} for the current control step.

        @param metrics: C{{link_id: x}} from L{ControlCurveMetricReader.read}.
            A missing link defaults to C{x = 0.0}.
        @type metrics: mapping of str to float
        @param dt_seconds: Control interval in seconds (unused; the curve is a
            pure function of state).
        @type dt_seconds: float
        @return: A setting in C{[0,1]} for every controlled link.
        @rtype: dict of str to float
        """
        recompute = (self._call % self._interval) == 0
        self._call += 1

        out: dict[str, float] = {}
        for a in self._assets:
            if not recompute and a.link_id in self._prev:
                out[a.link_id] = self._prev[a.link_id]
                continue
            x = float(metrics.get(a.link_id, 0.0))
            raw = pwl_interp(x, a.x_knots, a.y_values)
            if self._rate_limit is not None and a.link_id in self._prev:
                prev = self._prev[a.link_id]
                delta = raw - prev
                lim = float(self._rate_limit)
                if delta > lim:
                    raw = prev + lim
                elif delta < -lim:
                    raw = prev - lim
            setting = 0.0 if raw < 0.0 else 1.0 if raw > 1.0 else float(raw)
            out[a.link_id] = setting
            self._prev[a.link_id] = setting
        return out
