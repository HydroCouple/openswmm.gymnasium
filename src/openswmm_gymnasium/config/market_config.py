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

"""Typed agent-based capacity-market configuration.

A faithful, dependency-free port of the market-config JSON Schema shipped with
the OpenSWMM MCP C{operational-optimization} skill
(C{skills/operational-optimization/references/market_config.schema.json}). This
module is the single source of truth for the configuration shape: the reactive
L{openswmm_gymnasium.control.MarketController} consumes it, and the NSGA-II
optimizer treats a subset of its numeric fields (cost-curve onset/steepness/
ceiling, PID gains) as the decision vector.

Design notes:

  - Enums are kept as plain strings (matching the JSON) and checked in
    L{MarketConfig.validate}, rather than Python C{Enum}s, to keep round-trips
    byte-faithful and dependency-free.
  - L{MarketConfig.from_dict} fills schema defaults so downstream code never
    sees a missing optional; L{MarketConfig.to_dict} emits the canonical filled
    form, so C{from_dict(to_dict(cfg)) == cfg}.
  - Validation is hand-rolled (no C{jsonschema} dependency) and raises
    C{ValueError} with an actionable message on the first problem found.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# Allowed enum vocabularies (mirror the JSON Schema).
CURVE_TYPES = ("piecewise_linear", "logistic")
ELEMENT_TYPES = ("node", "link")
ROLES = ("buyer", "seller", "both")
COMMODITIES = ("conveyance", "storage", "treatment")
STRESS_METRICS = (
    "filling_ratio",
    "freeboard_fraction",
    "storage_fill",
    "treatment_utilization",
    "spill_imminence",
)
AGGREGATES = ("max", "mean")


# =============================================================================
# Leaf objects
# =============================================================================


@dataclass
class Meta:
    """Run-level metadata (C{meta} block of the schema).

    @ivar model_path: Path to the C{.inp} model.
    @ivar output_dir: User-reviewable folder for config, logs, and figures.
    @ivar control_interval_seconds: How often prices are recomputed and
        settings reapplied; should be a multiple of the routing step.
    @ivar treatment_tag: Node tag marking treatment-connected outfalls; all
        other outfalls are treated as uncontrolled.
    @ivar currency_note: Free text documenting the normalized currency.
    """

    model_path: str
    output_dir: str
    control_interval_seconds: float
    treatment_tag: str = "TREATED"
    currency_note: str | None = None


@dataclass
class Objectives:
    """Relative weights for the four minimized objectives.

    Used only for tuning / scalar baselining, never by the reactive loop.
    """

    flooding: float = 1.0
    uncontrolled_discharge: float = 1.0
    storage_underutilization: float = 1.0
    pumping_energy: float = 1.0


@dataclass
class CostCurve:
    """A named curve mapping a stress metric in [0,1] to a price in [0,1].

    @ivar type: One of L{CURVE_TYPES}.
    @ivar onset: Metric value below which price stays at C{floor}
        (piecewise_linear) or the sigmoid midpoint (logistic).
    @ivar full: C{piecewise_linear} only — metric at which price hits C{ceiling}.
    @ivar steepness: C{logistic} only — slope of the sigmoid.
    @ivar floor: Minimum price.
    @ivar ceiling: Maximum price.
    """

    type: str
    onset: float
    full: float = 1.0
    steepness: float = 10.0
    floor: float = 0.0
    ceiling: float = 1.0


@dataclass
class Agent:
    """A pricing agent attached to a model element.

    @ivar id: Model element ID (node or link).
    @ivar element_type: One of L{ELEMENT_TYPES}.
    @ivar commodity: One of L{COMMODITIES}.
    @ivar stress_metric: One of L{STRESS_METRICS}.
    @ivar curve: Key into C{MarketConfig.cost_curves}.
    @ivar role: One of L{ROLES}; C{"both"} lets the agent switch by live stress.
    """

    id: str
    element_type: str
    commodity: str
    stress_metric: str
    curve: str
    role: str = "both"


@dataclass
class PIDConfig:
    """PID gains and output limits for a trade route.

    The controller is direct-acting: the output (structure setting) rises with
    the buyer-minus-seller cost differential.
    """

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    setpoint: float = 0.0
    out_min: float = 0.0
    out_max: float = 1.0


@dataclass
class RouteConstraints:
    """Per-route actuator limits.

    @ivar min_cycle_seconds: Minimum on/off dwell for pumps (C{None} = none).
    @ivar max_starts_per_hour: Pump start-rate cap (C{None} = none).
    """

    setting_min: float = 0.0
    setting_max: float = 1.0
    min_cycle_seconds: float | None = None
    max_starts_per_hour: float | None = None


@dataclass
class TradeRoute:
    """A controllable structure mediating one buyer/seller trade.

    @ivar structure_link_id: Controllable pump/gate/orifice/weir link ID.
    @ivar buyer_agents: Agent IDs whose stress sets the buyer price.
    @ivar seller_agents: Agent IDs whose spare capacity sets the seller price.
    @ivar aggregate: How to combine each side's agents (L{AGGREGATES}).
    @ivar pid: The route's PID configuration.
    @ivar constraints: The route's actuator constraints.
    """

    structure_link_id: str
    buyer_agents: list[str]
    seller_agents: list[str]
    pid: PIDConfig
    aggregate: str = "max"
    constraints: RouteConstraints = field(default_factory=RouteConstraints)


@dataclass
class GlobalConstraints:
    """Global hard limits enforced regardless of price."""

    min_velocity: float | None = None
    max_hgl_freeboard: float | None = None


# =============================================================================
# Root object
# =============================================================================


@dataclass
class MarketConfig:
    """The complete agent-based capacity-market configuration.

    Construct via L{from_dict} / L{load} (which fill defaults), then call
    L{validate} before use. L{to_dict} / L{save} emit the canonical filled form.
    """

    meta: Meta
    cost_curves: dict[str, CostCurve]
    agents: list[Agent]
    trade_routes: list[TradeRoute]
    objectives: Objectives = field(default_factory=Objectives)
    constraints: GlobalConstraints = field(default_factory=GlobalConstraints)

    # -- Construction --------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> MarketConfig:
        """Build a config from a plain dict, filling schema defaults.

        @param data: Parsed JSON object matching the market-config schema.
        @raise ValueError: If a required block or key is missing.
        """
        try:
            meta_d = data["meta"]
            meta = Meta(
                model_path=meta_d["model_path"],
                output_dir=meta_d["output_dir"],
                control_interval_seconds=meta_d["control_interval_seconds"],
                treatment_tag=meta_d.get("treatment_tag", "TREATED"),
                currency_note=meta_d.get("currency_note"),
            )

            curves = {
                name: CostCurve(
                    type=c["type"],
                    onset=c["onset"],
                    full=c.get("full", 1.0),
                    steepness=c.get("steepness", 10.0),
                    floor=c.get("floor", 0.0),
                    ceiling=c.get("ceiling", 1.0),
                )
                for name, c in data["cost_curves"].items()
            }

            agents = [
                Agent(
                    id=a["id"],
                    element_type=a["element_type"],
                    commodity=a["commodity"],
                    stress_metric=a["stress_metric"],
                    curve=a["curve"],
                    role=a.get("role", "both"),
                )
                for a in data["agents"]
            ]

            routes = []
            for r in data["trade_routes"]:
                p = r["pid"]
                pid = PIDConfig(
                    kp=p["kp"],
                    ki=p.get("ki", 0.0),
                    kd=p.get("kd", 0.0),
                    setpoint=p.get("setpoint", 0.0),
                    out_min=p.get("out_min", 0.0),
                    out_max=p.get("out_max", 1.0),
                )
                c = r.get("constraints", {})
                constraints = RouteConstraints(
                    setting_min=c.get("setting_min", 0.0),
                    setting_max=c.get("setting_max", 1.0),
                    min_cycle_seconds=c.get("min_cycle_seconds"),
                    max_starts_per_hour=c.get("max_starts_per_hour"),
                )
                routes.append(
                    TradeRoute(
                        structure_link_id=r["structure_link_id"],
                        buyer_agents=list(r["buyer_agents"]),
                        seller_agents=list(r["seller_agents"]),
                        pid=pid,
                        aggregate=r.get("aggregate", "max"),
                        constraints=constraints,
                    )
                )
        except KeyError as exc:
            raise ValueError(f"market config missing required key: {exc}") from exc

        obj_d = data.get("objectives", {})
        objectives = Objectives(
            flooding=obj_d.get("flooding", 1.0),
            uncontrolled_discharge=obj_d.get("uncontrolled_discharge", 1.0),
            storage_underutilization=obj_d.get("storage_underutilization", 1.0),
            pumping_energy=obj_d.get("pumping_energy", 1.0),
        )
        con_d = data.get("constraints", {})
        constraints = GlobalConstraints(
            min_velocity=con_d.get("min_velocity"),
            max_hgl_freeboard=con_d.get("max_hgl_freeboard"),
        )

        return cls(
            meta=meta,
            cost_curves=curves,
            agents=agents,
            trade_routes=routes,
            objectives=objectives,
            constraints=constraints,
        )

    @classmethod
    def load(cls, path: str) -> MarketConfig:
        """Load and parse a config from a JSON file (does not validate)."""
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        """Emit the canonical filled form (round-trips through L{from_dict})."""
        meta = {
            "model_path": self.meta.model_path,
            "output_dir": self.meta.output_dir,
            "treatment_tag": self.meta.treatment_tag,
            "control_interval_seconds": self.meta.control_interval_seconds,
        }
        if self.meta.currency_note is not None:
            meta["currency_note"] = self.meta.currency_note

        out: dict = {
            "meta": meta,
            "objectives": {
                "flooding": self.objectives.flooding,
                "uncontrolled_discharge": self.objectives.uncontrolled_discharge,
                "storage_underutilization": self.objectives.storage_underutilization,
                "pumping_energy": self.objectives.pumping_energy,
            },
            "cost_curves": {
                name: {
                    "type": c.type,
                    "onset": c.onset,
                    "full": c.full,
                    "steepness": c.steepness,
                    "floor": c.floor,
                    "ceiling": c.ceiling,
                }
                for name, c in self.cost_curves.items()
            },
            "agents": [
                {
                    "id": a.id,
                    "element_type": a.element_type,
                    "role": a.role,
                    "commodity": a.commodity,
                    "stress_metric": a.stress_metric,
                    "curve": a.curve,
                }
                for a in self.agents
            ],
            "trade_routes": [
                {
                    "structure_link_id": r.structure_link_id,
                    "buyer_agents": list(r.buyer_agents),
                    "seller_agents": list(r.seller_agents),
                    "aggregate": r.aggregate,
                    "pid": {
                        "kp": r.pid.kp,
                        "ki": r.pid.ki,
                        "kd": r.pid.kd,
                        "setpoint": r.pid.setpoint,
                        "out_min": r.pid.out_min,
                        "out_max": r.pid.out_max,
                    },
                    "constraints": _route_constraints_to_dict(r.constraints),
                }
                for r in self.trade_routes
            ],
        }

        glob = _global_constraints_to_dict(self.constraints)
        if glob:
            out["constraints"] = glob
        return out

    def save(self, path: str) -> None:
        """Write the canonical form to a JSON file (indent=2)."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    # -- Validation ----------------------------------------------------------

    def validate(self) -> None:
        """Check enums, bounds, and referential integrity.

        @raise ValueError: On the first problem found, with an actionable
            message (which field, which allowed values).
        """
        if self.meta.control_interval_seconds <= 0:
            raise ValueError("meta.control_interval_seconds must be > 0")

        if not self.cost_curves:
            raise ValueError("cost_curves must define at least one curve")
        for name, c in self.cost_curves.items():
            _require_enum(c.type, CURVE_TYPES, f"cost_curves[{name!r}].type")
            _require_unit(c.onset, f"cost_curves[{name!r}].onset")
            _require_unit(c.floor, f"cost_curves[{name!r}].floor")
            _require_unit(c.ceiling, f"cost_curves[{name!r}].ceiling")
            if c.floor > c.ceiling:
                raise ValueError(f"cost_curves[{name!r}]: floor must be <= ceiling")
            if c.type == "piecewise_linear":
                _require_unit(c.full, f"cost_curves[{name!r}].full")
            elif c.steepness <= 0:
                raise ValueError(f"cost_curves[{name!r}].steepness must be > 0")

        if not self.agents:
            raise ValueError("agents must define at least one agent")
        agent_ids: set[str] = set()
        for i, a in enumerate(self.agents):
            _require_enum(a.element_type, ELEMENT_TYPES, f"agents[{i}].element_type")
            _require_enum(a.role, ROLES, f"agents[{i}].role")
            _require_enum(a.commodity, COMMODITIES, f"agents[{i}].commodity")
            _require_enum(a.stress_metric, STRESS_METRICS, f"agents[{i}].stress_metric")
            if a.curve not in self.cost_curves:
                raise ValueError(
                    f"agents[{i}].curve {a.curve!r} not found in cost_curves "
                    f"({sorted(self.cost_curves)})"
                )
            agent_ids.add(a.id)

        if not self.trade_routes:
            raise ValueError("trade_routes must define at least one route")
        for i, r in enumerate(self.trade_routes):
            if not r.structure_link_id:
                raise ValueError(f"trade_routes[{i}].structure_link_id is empty")
            _require_enum(r.aggregate, AGGREGATES, f"trade_routes[{i}].aggregate")
            if not r.buyer_agents:
                raise ValueError(f"trade_routes[{i}].buyer_agents is empty")
            if not r.seller_agents:
                raise ValueError(f"trade_routes[{i}].seller_agents is empty")
            for side in ("buyer_agents", "seller_agents"):
                for aid in getattr(r, side):
                    if aid not in agent_ids:
                        raise ValueError(
                            f"trade_routes[{i}].{side} references unknown agent "
                            f"{aid!r}"
                        )
            _require_unit(r.constraints.setting_min, f"trade_routes[{i}].constraints.setting_min")
            _require_unit(r.constraints.setting_max, f"trade_routes[{i}].constraints.setting_max")
            if r.constraints.setting_min > r.constraints.setting_max:
                raise ValueError(
                    f"trade_routes[{i}].constraints: setting_min must be <= setting_max"
                )


# =============================================================================
# Helpers
# =============================================================================


def _route_constraints_to_dict(c: RouteConstraints) -> dict:
    d: dict = {"setting_min": c.setting_min, "setting_max": c.setting_max}
    if c.min_cycle_seconds is not None:
        d["min_cycle_seconds"] = c.min_cycle_seconds
    if c.max_starts_per_hour is not None:
        d["max_starts_per_hour"] = c.max_starts_per_hour
    return d


def _global_constraints_to_dict(c: GlobalConstraints) -> dict:
    d: dict = {}
    if c.min_velocity is not None:
        d["min_velocity"] = c.min_velocity
    if c.max_hgl_freeboard is not None:
        d["max_hgl_freeboard"] = c.max_hgl_freeboard
    return d


def _require_enum(value: str, allowed: tuple[str, ...], where: str) -> None:
    if value not in allowed:
        raise ValueError(f"{where} must be one of {allowed}, got {value!r}")


def _require_unit(value: float, where: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{where} must be in [0, 1], got {value}")
