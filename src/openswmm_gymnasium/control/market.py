"""Reactive agent-based capacity-market controller.

Promotes the skill's C{control_loop.py} reference to real, tested code. Pricing
agents map their normalized stress to a price in [0,1] via cost curves; each
trade route's PID throttles its structure on the buyer-minus-seller cost
differential. Pump cycle limits (C{min_cycle_seconds}, C{max_starts_per_hour}),
which the reference left as a NOTE, are enforced here.

The controller is engine-agnostic (see L{openswmm_gymnasium.control.base}): it
consumes a C{{agent_id: stress}} mapping and returns C{{structure_id: setting}}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from openswmm_gymnasium.config import CostCurve, MarketConfig, PIDConfig, TradeRoute

# A setting at or below this magnitude counts as "off" for cycle tracking.
_ON_EPS = 1e-6


# =============================================================================
# Pure pricing primitives
# =============================================================================


def price_from_curve(metric: float, curve: CostCurve) -> float:
    """Return a normalized price in [0,1] for a stress *metric* in [0,1].

    Both curve shapes are monotonic non-decreasing in the metric, so a
    more-stressed agent never prices below a less-stressed one. The logistic
    shape is centred on C{onset} (price = midpoint of [floor, ceiling] there).

    @param metric: Stress metric, clamped into [0,1].
    @param curve: The curve parameters.
    @rtype: float
    """
    m = _clamp(metric, 0.0, 1.0)
    if curve.type == "piecewise_linear":
        if m <= curve.onset:
            frac = 0.0
        elif m >= curve.full or curve.full <= curve.onset:
            frac = 1.0
        else:
            frac = (m - curve.onset) / (curve.full - curve.onset)
    elif curve.type == "logistic":
        frac = 1.0 / (1.0 + math.exp(-curve.steepness * (m - curve.onset)))
    else:  # pragma: no cover - validate() prevents this
        raise ValueError(f"unknown curve type: {curve.type!r}")
    return _clamp(curve.floor + (curve.ceiling - curve.floor) * frac, 0.0, 1.0)


def aggregate_prices(prices: list[float], how: str) -> float:
    """Combine one side's agent prices into a single side price."""
    if not prices:
        return 0.0
    return max(prices) if how == "max" else sum(prices) / len(prices)


@dataclass
class PID:
    """Direct-acting PID: output rises with the cost differential.

    Conditional anti-windup: the integral only accumulates when doing so does
    not push a saturated output further into saturation.
    """

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    setpoint: float = 0.0
    out_min: float = 0.0
    out_max: float = 1.0
    _integral: float = field(default=0.0, repr=False)
    _prev_error: float | None = field(default=None, repr=False)

    @classmethod
    def from_config(cls, cfg: PIDConfig) -> PID:
        """Build a PID from a L{PIDConfig}."""
        return cls(
            kp=cfg.kp,
            ki=cfg.ki,
            kd=cfg.kd,
            setpoint=cfg.setpoint,
            out_min=cfg.out_min,
            out_max=cfg.out_max,
        )

    def reset(self) -> None:
        """Clear the integrator and derivative memory."""
        self._integral = 0.0
        self._prev_error = None

    def update(self, measurement: float, dt: float) -> float:
        """Advance by *dt* seconds; return the new setting in [out_min, out_max].

        @param measurement: The cost differential (buyer_price - seller_price).
        @param dt: Timestep in seconds.
        """
        error = measurement - self.setpoint
        derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / dt

        candidate_i = self._integral + error * dt
        raw = self.kp * error + self.ki * candidate_i + self.kd * derivative
        output = _clamp(raw, self.out_min, self.out_max)

        # Conditional-integration anti-windup: freeze the integrator while the
        # output is saturated and the error would drive it *further* into
        # saturation; otherwise accumulate. (The skill's control_loop.py
        # reference had these two saturated-branch sign tests inverted, which
        # defeated the anti-windup — corrected here.)
        if (
            self.out_min < raw < self.out_max
            or (raw >= self.out_max and error < 0)
            or (raw <= self.out_min and error > 0)
        ):
            self._integral = candidate_i

        self._prev_error = error
        return output


# =============================================================================
# Per-route runtime state
# =============================================================================


@dataclass
class _RouteState:
    """Mutable per-route state: PID + cycle-constraint bookkeeping."""

    pid: PID
    on: bool = False
    last_switch_t: float = float("-inf")
    last_setting: float = 0.0
    starts: list[float] = field(default_factory=list)

    def reset(self) -> None:
        self.pid.reset()
        self.on = False
        self.last_switch_t = float("-inf")
        self.last_setting = 0.0
        self.starts = []


# =============================================================================
# Controller
# =============================================================================


class MarketController:
    """Reactive agent-based capacity-market controller.

    Implements L{openswmm_gymnasium.control.base.Controller}. After each
    L{compute_settings} call, L{last_detail} holds per-route buyer/seller/diff/
    setting values for tracing and report figures.

    @ivar last_detail: C{{structure_id: {"buyer", "seller", "diff", "setting"}}}
        from the most recent step (empty before the first step).
    """

    def __init__(self, config: MarketConfig) -> None:
        """
        @param config: A validated market configuration.
        @type config: L{MarketConfig}
        """
        self._config = config
        self._agent_curve: dict[str, CostCurve] = {
            a.id: config.cost_curves[a.curve] for a in config.agents
        }
        self._routes: list[TradeRoute] = list(config.trade_routes)
        self._state: dict[str, _RouteState] = {
            r.structure_link_id: _RouteState(PID.from_config(r.pid)) for r in self._routes
        }
        self._t = 0.0
        self.last_detail: dict[str, dict[str, float]] = {}

    @property
    def structure_ids(self) -> list[str]:
        return [r.structure_link_id for r in self._routes]

    def reset(self) -> None:
        for st in self._state.values():
            st.reset()
        self._t = 0.0
        self.last_detail = {}

    def compute_settings(
        self, metrics: Mapping[str, float], dt_seconds: float
    ) -> dict[str, float]:
        # Price every agent once per step from its curve.
        prices = {
            aid: price_from_curve(float(metrics.get(aid, 0.0)), curve)
            for aid, curve in self._agent_curve.items()
        }

        settings: dict[str, float] = {}
        detail: dict[str, dict[str, float]] = {}
        for route in self._routes:
            how = route.aggregate
            buyer = aggregate_prices([prices.get(a, 0.0) for a in route.buyer_agents], how)
            seller = aggregate_prices([prices.get(a, 0.0) for a in route.seller_agents], how)
            diff = buyer - seller

            link = route.structure_link_id
            st = self._state[link]
            c = route.constraints

            setting = st.pid.update(diff, dt_seconds)
            setting = _clamp(setting, c.setting_min, c.setting_max)
            setting = self._apply_cycle_limits(route, st, setting)

            settings[link] = setting
            detail[link] = {"buyer": buyer, "seller": seller, "diff": diff, "setting": setting}

        self.last_detail = detail
        self._t += dt_seconds
        return settings

    # -- Cycle-constraint enforcement ----------------------------------------

    def _apply_cycle_limits(
        self, route: TradeRoute, st: _RouteState, setting: float
    ) -> float:
        """Gate on/off transitions by min dwell time and start-rate limits.

        With neither C{min_cycle_seconds} nor C{max_starts_per_hour} set this is
        a near no-op (it just records the applied setting). When a switch would
        violate a limit it is blocked and the route holds its previous setting.
        """
        c = route.constraints
        if c.min_cycle_seconds is None and c.max_starts_per_hour is None:
            st.last_setting = setting
            return setting

        desired_on = setting > _ON_EPS
        if desired_on != st.on:  # an on/off switch is requested this step
            if c.max_starts_per_hour is not None:
                st.starts = [s for s in st.starts if self._t - s < 3600.0]
            blocked = False
            if c.min_cycle_seconds is not None and (self._t - st.last_switch_t) < c.min_cycle_seconds:
                blocked = True
            if (
                desired_on
                and c.max_starts_per_hour is not None
                and len(st.starts) >= c.max_starts_per_hour
            ):
                blocked = True

            if blocked:
                st.last_setting = st.last_setting  # hold previous command
                return st.last_setting
            st.on = desired_on
            st.last_switch_t = self._t
            if desired_on and c.max_starts_per_hour is not None:
                st.starts.append(self._t)

        st.last_setting = setting
        return setting


# =============================================================================
# Helpers
# =============================================================================


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x
