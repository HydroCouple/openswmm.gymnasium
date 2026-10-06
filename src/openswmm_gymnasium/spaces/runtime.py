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
Runtime (RTC) action factories.

Each factory exposes a L{gymnasium.spaces.Box}-or-similar over a set of
controllable elements and translates sampled values into persistent
runtime overrides via the link C{target_setting} (or related) each step.

Ships three actuators:

  - L{OrificeSetting} — per-link C{[0, 1]} control setting.
  - L{NodeLateralInflow} — per-node controllable lateral inflow.
  - L{HeatSourceTemperatureSetpoint} — per-pathway inlet temperature in
    degC. The engine documents heat source writes as B{live} (they take
    effect on the next routing step), which is what makes a heat
    configuration call a legitimate runtime actuator rather than a
    pre-initialize-only edit.

B{No water-age runtime twin is shipped}, deliberately. Water-age source
writes are live in exactly the same way, so one would be mechanically
trivial — but it would not be a meaningful control. Heat has a physical
actuator behind it (a discharge whose temperature a plant genuinely
regulates, and which genuinely alters downstream water temperature at
step resolution). A source's assigned water age is a bookkeeping label on
inflowing water, not a quantity any operator can move during an event;
per-step modulation of it would let an agent chase reward by rewriting
its own accounting rather than by operating the network. Water age is
therefore exposed as a once-per-episode design/scenario input only
(L{openswmm_gymnasium.spaces.design.WaterAgeSourceAge}).

Plan §3.2.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from gymnasium import spaces

from openswmm_gymnasium._engine import SolverAdapter, element_kind, field_entry


class OrificeSetting:
    """Box action over the control setting of one or more links.

    The setting is a real number in C{[0, 1]}, where C{0} = fully closed
    and C{1} = fully open. Applied via the link's C{target_setting} (the
    persistent runtime-control override; C{Controls.set_link_setting}
    sets C{control_setting}, which the engine recomputes each routing
    step and so would not stick without a control rule).

    Although the name reflects the primary use case, the underlying
    engine call accepts any controllable link (orifices, weirs, pumps,
    and conduits with the C{LINK_OFFSETS} option set appropriately).

    @ivar _link_ids: Symbolic link IDs supplied at construction.
    @type _link_ids: list[str]
    @ivar _name: Action-space key under which this factory's component
        appears in the env's runtime Dict.
    @type _name: str
    @ivar _link_idxs: Engine link indices, resolved by L{bind}.
    @type _link_idxs: list[int] or C{None}
    """

    #: Catalog paths this component needs from the engine (see require_for).
    requires: tuple[str, ...] = ("link.target_setting",)

    def __init__(
        self,
        link_ids: Sequence[str],
        name: str = "orifice_setting",
    ) -> None:
        """
        @param link_ids: Symbolic IDs of links to control.
        @type link_ids: sequence of str
        @param name: Action-space key for this factory.
        @type name: str
        """
        if not link_ids:
            raise ValueError("OrificeSetting requires at least one link_id")
        self._link_ids: list[str] = list(link_ids)
        self._name = name
        self._link_idxs: list[int] | None = None

    @property
    def name(self) -> str:
        """Action-space key for this factory.

        @rtype: str
        """
        return self._name

    @property
    def space(self) -> spaces.Box:
        """Per-link setting in C{[0, 1]}.

        @rtype: L{gymnasium.spaces.Box}
        """
        n = len(self._link_ids)
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n,),
            dtype=np.float32,
        )

    def bind(self, adapter: SolverAdapter) -> None:
        """Resolve symbolic link IDs to engine indices.

        Must be called after L{SolverAdapter.open} (so the engine knows
        about the model's links) and before L{apply}.

        @param adapter: Adapter wrapping the open solver.
        @type adapter: L{SolverAdapter}
        """
        self._link_idxs = [adapter.links.get_index(lid) for lid in self._link_ids]

    def apply(self, adapter: SolverAdapter, value: np.ndarray) -> None:
        """Push the sampled action value into the engine.

        @param adapter: Adapter wrapping the running solver.
        @type adapter: L{SolverAdapter}
        @param value: 1-D float array of length C{len(link_ids)}.
        @type value: numpy.ndarray
        @raise RuntimeError: If L{bind} was not called first.
        """
        if self._link_idxs is None:
            raise RuntimeError("OrificeSetting.bind() must be called before apply()")
        # Defensive clip — agents may sample outside [0,1] under
        # numerical noise; the engine refuses values outside its expected
        # range and we'd rather silently clamp.
        clipped = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
        for idx, v in zip(self._link_idxs, clipped, strict=True):
            adapter.links.set_target_setting(idx, float(v))


class NodeLateralInflow:
    """Box action injecting a controllable lateral inflow at one or more nodes.

    Each component is a flow rate in C{[0, max_inflow]} (project flow units),
    applied via C{adapter.nodes.set_lateral_inflow} (the engine's
    ``swmm_node_set_lateral_inflow``). Useful for controllable sources,
    pumped diversions, or adversarial inflow scenarios in RL tasks.

    @ivar _node_ids: Symbolic node IDs supplied at construction.
    @ivar _max_inflow: Upper bound of each component's action range.
    """

    #: Catalog paths this component needs from the engine (see require_for).
    requires: tuple[str, ...] = ("node.lateral_inflow",)

    def __init__(
        self,
        node_ids: Sequence[str],
        max_inflow: float,
        name: str = "node_lateral_inflow",
    ) -> None:
        """
        @param node_ids: Symbolic IDs of nodes to drive.
        @type node_ids: sequence of str
        @param max_inflow: Upper bound of the per-node inflow action
            (project flow units). Must be positive.
        @type max_inflow: float
        @param name: Action-space key for this factory.
        @type name: str
        """
        if not node_ids:
            raise ValueError("NodeLateralInflow requires at least one node_id")
        if max_inflow <= 0.0:
            raise ValueError("NodeLateralInflow requires max_inflow > 0")
        self._node_ids: list[str] = list(node_ids)
        self._max_inflow = float(max_inflow)
        self._name = name
        self._node_idxs: list[int] | None = None

    @property
    def name(self) -> str:
        """Action-space key for this factory.

        @rtype: str
        """
        return self._name

    @property
    def space(self) -> spaces.Box:
        """Per-node inflow in C{[0, max_inflow]}.

        @rtype: L{gymnasium.spaces.Box}
        """
        n = len(self._node_ids)
        return spaces.Box(
            low=0.0,
            high=self._max_inflow,
            shape=(n,),
            dtype=np.float32,
        )

    def bind(self, adapter: SolverAdapter) -> None:
        """Resolve symbolic node IDs to engine indices.

        @param adapter: Adapter wrapping the open solver.
        @type adapter: L{SolverAdapter}
        """
        self._node_idxs = [adapter.nodes.get_index(nid) for nid in self._node_ids]

    def apply(self, adapter: SolverAdapter, value: np.ndarray) -> None:
        """Push the sampled inflow values into the engine.

        @param adapter: Adapter wrapping the running solver.
        @type adapter: L{SolverAdapter}
        @param value: 1-D float array of length C{len(node_ids)}.
        @type value: numpy.ndarray
        @raise RuntimeError: If L{bind} was not called first.
        """
        if self._node_idxs is None:
            raise RuntimeError("NodeLateralInflow.bind() must be called before apply()")
        clipped = np.clip(np.asarray(value, dtype=np.float32), 0.0, self._max_inflow)
        for idx, v in zip(self._node_idxs, clipped, strict=True):
            adapter.nodes.set_lateral_inflow(idx, float(v))


class FieldSetpoint:
    """Box action writing any writable numeric engine field every control step.

    C{path} is a catalog path of an element field such as
    C{"link.target_setting"}, C{"node.lateral_inflow"} or
    C{"subcatchment.rain_scale_factor"} (see L{openswmm.engine.catalog});
    each component is that field's value for one element, in C{[low, high]}
    and the field's own units. Prefer the named factories where one exists:
    they encode engine semantics (for example that C{control_setting} is
    recomputed from C{target_setting} every step) that a raw field write
    does not. Some fields (link roughness, node maximum depth, ...) can
    only be written before the simulation starts; writing one of those
    during an episode raises the engine's lifecycle error.

    @ivar _path: Catalog field path.
    @ivar _ids: Symbolic element IDs.
    """

    def __init__(
        self,
        path: str,
        ids: Sequence[str],
        low: float,
        high: float,
        name: str | None = None,
    ) -> None:
        """
        @param path: Catalog path of a writable numeric element field.
        @param ids: Element IDs to drive.
        @param low: Lower bound of every component.
        @param high: Upper bound of every component.
        @param name: Action-space key; defaults to the path.
        @raise ValueError: Unknown or read-only path, no ids, or C{low >= high}.
        """
        entry = field_entry(path)
        if entry.get("access") != "rw":
            raise ValueError(f"FieldSetpoint: {path!r} is read-only")
        if not ids:
            raise ValueError("FieldSetpoint requires at least one id")
        if not low < high:
            raise ValueError("FieldSetpoint requires low < high")
        self._path = path
        self.requires = (path,)
        self._kind = element_kind(path)
        self._ids = list(ids)
        self._low, self._high = float(low), float(high)
        self._name = name or path
        self._idxs: list[int] | None = None

    @property
    def name(self) -> str:
        """Action-space key for this factory.

        @rtype: str
        """
        return self._name

    @property
    def space(self) -> spaces.Box:
        """Per-element value in C{[low, high]}.

        @rtype: L{gymnasium.spaces.Box}
        """
        return spaces.Box(low=self._low, high=self._high, shape=(len(self._ids),), dtype=np.float32)

    def bind(self, adapter: SolverAdapter) -> None:
        """Resolve symbolic IDs to engine indices."""
        self._idxs = [adapter.index(self._kind, i) for i in self._ids]

    def apply(self, adapter: SolverAdapter, value: np.ndarray) -> None:
        """Write the clipped values to the engine.

        @raise RuntimeError: If L{bind} was not called first.
        """
        if self._idxs is None:
            raise RuntimeError("FieldSetpoint.bind() must be called before apply()")
        clipped = np.clip(np.asarray(value, dtype=np.float32), self._low, self._high)
        for idx, v in zip(self._idxs, clipped, strict=True):
            adapter.write(self._path, idx, float(v))


#: Engine refusal range for a heat source temperature, degrees Celsius.
#: Mirrors L{openswmm_gymnasium.spaces.design._HEAT_TEMP_MIN_C} / C{_MAX_C};
#: kept local so the two action modules stay independent.
_HEAT_TEMP_MIN_C = -50.0
_HEAT_TEMP_MAX_C = 100.0


class HeatSourceTemperatureSetpoint:
    """Box action over the inlet temperature of one or more heat sources.

    Each component is a temperature in degrees Celsius, applied B{every env
    step} to a heat-source pathway (C{"DWF"}, C{"EXTERNAL_INFLOW"},
    C{"RAINFALL"}, C{"GW"}, C{"RDII"}, C{"IFACE"}, C{"INITIAL_STATE"}) via
    C{adapter.heat.set_source_temp}. The engine documents these writes as
    B{live}: the new value is picked up on the next routing step, which is
    what makes this a runtime actuator rather than a pre-initialize edit.

    The obvious use is thermal-discharge control — modulating the
    temperature of a regulated effluent (C{EXTERNAL_INFLOW}) or of
    dry-weather flow over an event, subject to a downstream thermal
    objective.

    B{No observation counterpart exists.} The C API exposes no per-node or
    per-link water-temperature getter, so an agent driving this actuator
    cannot observe the temperature field it is acting on. Only two
    heat-related scalars are readable at all — C{adapter.heat.current_shortwave}
    and C{adapter.heat.current_cloud_fraction} — and both are forcing, not
    state. Treat this as an open-loop / feed-forward actuator (or close the
    loop through a proxy such as flow split) until the engine grows a
    temperature reader.

    B{Bounds.} Default C{[-50, 100]} degC, the engine's own refusal range.
    The engine B{refuses} rather than clamps an out-of-range write, and a
    refused write does not take effect, so a Box wider than the engine range
    could hand it a sampled action it silently declines — leaving the
    previous step's temperature in force with nothing logged. Narrowing is
    free; widening is rejected at construction.

    B{Units.} Degrees Celsius regardless of the model's unit system,
    unconverted.

    B{Requires} C{[OPTIONS] HEAT_TRANSPORT YES}; L{bind} raises with that
    message otherwise.

    @ivar _sources: Heat-source pathway names supplied at construction.
    @type _sources: list[str]
    @ivar _name: Action-space key for this factory.
    @type _name: str
    """

    #: Catalog paths this component needs from the engine (see require_for).
    requires: tuple[str, ...] = ("heat",)

    def __init__(
        self,
        sources: Sequence[str],
        low: float = _HEAT_TEMP_MIN_C,
        high: float = _HEAT_TEMP_MAX_C,
        name: str = "heat_source_temperature_setpoint",
    ) -> None:
        """
        @param sources: Heat-source pathway names to drive.
        @type sources: sequence of str
        @param low: Lower bound in degC (default: the engine minimum C{-50}).
        @type low: float
        @param high: Upper bound in degC (default: the engine maximum C{100}).
        @type high: float
        @param name: Action-space key for this factory.
        @type name: str
        @raise ValueError: If C{sources} is empty, C{high <= low}, or the
            range extends beyond the engine's C{[-50, 100]}.
        """
        if not sources:
            raise ValueError(
                "HeatSourceTemperatureSetpoint requires at least one source"
            )
        self._sources: list[str] = [str(s).upper() for s in sources]
        self._low = float(low)
        self._high = float(high)
        if not (self._high > self._low):
            raise ValueError(
                f"high ({self._high}) must be strictly greater than low ({self._low})"
            )
        if self._low < _HEAT_TEMP_MIN_C or self._high > _HEAT_TEMP_MAX_C:
            raise ValueError(
                f"HeatSourceTemperatureSetpoint bounds [{self._low}, "
                f"{self._high}] degC extend beyond the engine's refusal range "
                f"[{_HEAT_TEMP_MIN_C}, {_HEAT_TEMP_MAX_C}]. The engine refuses "
                "(does not clamp) an out-of-range temperature and a refused "
                "write does not take effect, so a sampled action outside that "
                "range would silently leave the temperature unchanged."
            )
        self._name = name
        self._bound = False

    @property
    def name(self) -> str:
        """Action-space key for this factory.

        @rtype: str
        """
        return self._name

    @property
    def space(self) -> spaces.Box:
        """Per-source temperature in C{[low, high]} degC.

        @rtype: L{gymnasium.spaces.Box}
        """
        return spaces.Box(
            low=self._low,
            high=self._high,
            shape=(len(self._sources),),
            dtype=np.float32,
        )

    def bind(self, adapter: SolverAdapter) -> None:
        """Verify heat transport is on and every named pathway is valid.

        Heat sources are addressed by name rather than index, so there is
        nothing to resolve; C{bind} validates instead.

        @param adapter: Adapter wrapping the open solver.
        @type adapter: L{SolverAdapter}
        @raise RuntimeError: If the engine build has no heat module, or the
            open model does not enable C{HEAT_TRANSPORT}.
        @raise ValueError: If a name is not a heat-source pathway.
        """
        known = set(adapter.heat.source_kinds())
        unknown = [s for s in self._sources if s not in known]
        if unknown:
            raise ValueError(
                f"HeatSourceTemperatureSetpoint: unknown heat source(s) "
                f"{unknown!r}. Valid pathways: {sorted(known)!r}"
            )
        self._bound = True

    def apply(self, adapter: SolverAdapter, value: np.ndarray) -> None:
        """Push the sampled temperatures into the engine.

        @param adapter: Adapter wrapping the running solver.
        @type adapter: L{SolverAdapter}
        @param value: 1-D float array of length C{len(sources)}, degC.
        @type value: numpy.ndarray
        @raise RuntimeError: If L{bind} was not called first.
        """
        if not self._bound:
            raise RuntimeError(
                "HeatSourceTemperatureSetpoint.bind() must be called before apply()"
            )
        # Defensive clip — agents may sample outside the declared bounds
        # under numerical noise. Here the clip does more than tidy up: the
        # engine REFUSES a temperature outside [-50, 100] instead of
        # clamping it, and a refused write does not take effect, so an
        # unclipped stray value would leave the previous step's setpoint
        # silently in force rather than erroring.
        clipped = np.clip(np.asarray(value, dtype=np.float32), self._low, self._high)
        for source, v in zip(self._sources, clipped, strict=True):
            adapter.heat.set_source_temp(source, float(v))
