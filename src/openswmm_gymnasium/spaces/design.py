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
Design (CIP) action factories.

Plan §3.1. Each factory exposes a L{gymnasium.spaces.Box} (or
L{gymnasium.spaces.Discrete} / L{gymnasium.spaces.MultiBinary} in
later phases) over a static-attribute action vector and writes the
sampled value into the engine through the appropriate
L{openswmm.engine} setter. Design factories are applied **once per
episode**, between L{SolverAdapter.open} and L{SolverAdapter.initialize}.

P3 ships four representative continuous factories that exercise the
pre-initialize edit path:

  - L{LinkRoughness} — sets Manning's C{n}.
  - L{LinkLength}    — sets conduit length.
  - L{LinkDiameter}  — resizes the cross-section to a target rise (full
    depth), scaling every length-dimensioned geometry parameter of the
    shape so the section stays geometrically similar.
  - L{NodeMaxDepth}  — sets node L{max_depth} (useful as a proxy for
    storage volume on tank-like nodes).

Four further §3.1-family factories size the assets modelers most want to
explore:

  - L{SubcatchGWOutflowCoeff} — the C{[GROUNDWATER]} outflow coefficient
    C{a1} per subcatchment, a design lever over baseflow; every other
    groundwater parameter is preserved.
  - L{StorageVolume} — sizes detention/retention storage via either the
    FUNCTIONAL surface-area relation or a TABULAR depth-area curve
    (scalar footprint multiplier, or the raw C{(a,b,c)} triple).
  - L{LIDPlacement}  — sizes green-infrastructure / nature-based solutions
    and selects among candidate LID control types per subcatchment.
  - L{RDIIUnitHydrograph} — sizes RDII response by editing unit-hydrograph
    R fractions (and optionally initial abstraction), preserving T and K.

Three B{process-configuration} factories reach the engine's heat-transport,
water-age and multi-species reaction surfaces. These differ in kind from
everything above: they do not resize an asset, they set the B{boundary
conditions and rate constants of a transport process}, which makes them
calibration / inverse-problem handles at least as much as CIP levers.

  - L{ReactionCoefficientValue} — C{[REACTION_COEFFICIENTS]} PARAMETER
    values. The highest-value of the three: it turns a multi-species
    reaction model into a directly calibratable / optimisable surface.
  - L{HeatSourceTemperature} — global inlet temperature per heat-source
    pathway (degC).
  - L{WaterAgeSourceAge} — global source age per water-age pathway (hours;
    B{negative values are legal} — they extract age-volume).

The remaining §3.1 factories (C{OutfallStage}, C{WeirCrestElev},
C{OrificeMaxOpening}, C{PumpCurveChoice}, C{ControlRuleSelection}) land in
subsequent phases as benchmark scenarios that need them come online. Two
further engine surfaces are reachable but deliberately B{not} wrapped yet:
C{InitialQuality} (per-element initial temperature / age / species rows —
an initial-condition randomiser rather than a design dimension, so it
belongs with hot-start seeding) and C{ProcessComponents} (registration of
external process plugins by config path — a model-assembly concern with no
continuous search space).

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from gymnasium import spaces

from openswmm_gymnasium._engine import SolverAdapter

# =============================================================================
# Protocol
# =============================================================================


@runtime_checkable
class DesignActionFactory(Protocol):
    """Interface every CIP factory must satisfy.

    Same shape as L{openswmm_gymnasium.spaces.runtime.OrificeSetting} —
    the difference is B{when} the env calls L{apply}: design factories
    are applied once at L{gymnasium.Env.reset}, between
    L{SolverAdapter.open} and L{SolverAdapter.initialize}.
    """

    name: str

    @property
    def space(self) -> spaces.Space: ...

    def bind(self, adapter: SolverAdapter) -> None: ...

    def apply(self, adapter: SolverAdapter, value: np.ndarray) -> None: ...


# =============================================================================
# Shared helpers
# =============================================================================


def _make_box(low: float, high: float, n: int) -> spaces.Box:
    if not (high > low):
        raise ValueError(f"high ({high}) must be strictly greater than low ({low})")
    return spaces.Box(
        low=np.full((n,), low, dtype=np.float32),
        high=np.full((n,), high, dtype=np.float32),
        shape=(n,),
        dtype=np.float32,
    )


def _make_box_bounds(low: Sequence[float], high: Sequence[float]) -> spaces.Box:
    """Build a flat C{Box} from explicit per-component bound vectors.

    Unlike L{_make_box} (uniform bounds), this supports design dimensions
    whose components carry B{different} physical ranges — e.g. a storage
    functional triple C{(a, b, c)} or an RDII C{(R, dmax, drecov, dinit)}
    vector. Each component may be degenerate (C{high == low}) to pin a
    coordinate; at least one component must be non-degenerate so the space
    is actually searchable.

    @param low: Per-component lower bounds.
    @param high: Per-component upper bounds (C{>=} the matching C{low}).
    @raise ValueError: If lengths differ, any C{high < low}, or every
        component is degenerate.
    """
    lo = np.asarray(low, dtype=np.float32)
    hi = np.asarray(high, dtype=np.float32)
    if lo.shape != hi.shape or lo.ndim != 1:
        raise ValueError("low and high must be 1-D sequences of equal length")
    if lo.size == 0:
        raise ValueError("bound vectors must be non-empty")
    if np.any(hi < lo):
        bad = int(np.argmax(hi < lo))
        raise ValueError(
            f"high[{bad}] ({hi[bad]}) must be >= low[{bad}] ({lo[bad]})"
        )
    if not np.any(hi > lo):
        raise ValueError("at least one component must have high > low (searchable)")
    return spaces.Box(low=lo, high=hi, shape=lo.shape, dtype=np.float32)


def _validate_link_ids(name: str, link_ids: Sequence[str]) -> list[str]:
    if not link_ids:
        raise ValueError(f"{name} requires at least one link_id")
    return list(link_ids)


def _validate_node_ids(name: str, node_ids: Sequence[str]) -> list[str]:
    if not node_ids:
        raise ValueError(f"{name} requires at least one node_id")
    return list(node_ids)


def _validate_subcatch_ids(name: str, subcatch_ids: Sequence[str]) -> list[str]:
    if not subcatch_ids:
        raise ValueError(f"{name} requires at least one subcatch_id")
    return list(subcatch_ids)


# =============================================================================
# Factories
# =============================================================================


class LinkRoughness:
    """Manning's C{n} for each link.

    @ivar name: Action-space key, default C{"link_roughness"}.
    """

    #: Catalog paths this component needs from the engine (see require_for).
    requires: tuple[str, ...] = ("link.roughness",)

    def __init__(
        self,
        link_ids: Sequence[str],
        low: float,
        high: float,
        name: str = "link_roughness",
    ) -> None:
        """
        @param link_ids: Symbolic link IDs to control.
        @type link_ids: sequence of str
        @param low: Lower bound on Manning's C{n}.
        @type low: float
        @param high: Upper bound on Manning's C{n}.
        @type high: float
        @param name: Action-space key.
        @type name: str
        @raise ValueError: If C{link_ids} is empty or C{high <= low}.
        """
        self._link_ids = _validate_link_ids("LinkRoughness", link_ids)
        self._low = float(low)
        self._high = float(high)
        self.name = name
        self._idxs: list[int] | None = None
        self._box = _make_box(self._low, self._high, len(self._link_ids))

    @property
    def space(self) -> spaces.Box:
        return self._box

    def bind(self, adapter: SolverAdapter) -> None:
        self._idxs = [adapter.links.get_index(lid) for lid in self._link_ids]

    def apply(self, adapter: SolverAdapter, value: np.ndarray) -> None:
        if self._idxs is None:
            raise RuntimeError("LinkRoughness.bind() must be called before apply()")
        clipped = np.clip(np.asarray(value, dtype=np.float32), self._low, self._high)
        for idx, v in zip(self._idxs, clipped, strict=True):
            adapter.links.set_roughness(idx, float(v))


class LinkLength:
    """Conduit length for each link.

    @ivar name: Action-space key, default C{"link_length"}.
    """

    #: Catalog paths this component needs from the engine (see require_for).
    requires: tuple[str, ...] = ("link.length",)

    def __init__(
        self,
        link_ids: Sequence[str],
        low: float,
        high: float,
        name: str = "link_length",
    ) -> None:
        """
        @param link_ids: Symbolic link IDs to control.
        @type link_ids: sequence of str
        @param low: Lower bound on length (project units).
        @type low: float
        @param high: Upper bound on length.
        @type high: float
        @param name: Action-space key.
        @type name: str
        @raise ValueError: If C{link_ids} is empty or C{high <= low}.
        """
        self._link_ids = _validate_link_ids("LinkLength", link_ids)
        self._low = float(low)
        self._high = float(high)
        self.name = name
        self._idxs: list[int] | None = None
        self._box = _make_box(self._low, self._high, len(self._link_ids))

    @property
    def space(self) -> spaces.Box:
        return self._box

    def bind(self, adapter: SolverAdapter) -> None:
        self._idxs = [adapter.links.get_index(lid) for lid in self._link_ids]

    def apply(self, adapter: SolverAdapter, value: np.ndarray) -> None:
        if self._idxs is None:
            raise RuntimeError("LinkLength.bind() must be called before apply()")
        clipped = np.clip(np.asarray(value, dtype=np.float32), self._low, self._high)
        for idx, v in zip(self._idxs, clipped, strict=True):
            adapter.links.set_length(idx, float(v))


# Which of the four cross-section geometry parameters carry a B{length}
# dimension, per shape, and may therefore be scaled when the section is
# resized. Keyed by L{openswmm.engine.XSectShape} member B{name} (the
# ordinals were renumbered in engine 6.0, the names were not) and derived
# from the engine's own C{CrossSection.geom_labels} table. Slot i is
# C{geom(i+1)}.
#
# Everything omitted is dimensionless or an index: side slopes
# (TRAPEZOIDAL), the POWER exponent, RECT_OPEN's sides-removed flag,
# FORCE_MAIN's roughness coefficient, and the transect / shape-curve /
# street table indices. Scaling any of those would change the shape, not
# its size. A shape mapping to an empty tuple has no scalable dimension at
# all — its geometry lives in a transect or street table — so it cannot be
# sized by this factory.
_RESIZABLE_GEOM_SLOTS: dict[str, tuple[int, ...]] = {
    "CIRCULAR": (0,),                 # diameter
    "FILLED_CIRCULAR": (0, 1),        # diameter, filled depth
    "RECT_CLOSED": (0, 1),            # height, width
    "RECT_OPEN": (0, 1),              # height, width
    "TRAPEZOIDAL": (0, 1),            # height, bottom width
    "TRIANGULAR": (0, 1),             # height, top width
    "PARABOLIC": (0, 1),              # height, top width
    "POWER": (0, 1),                  # height, top width
    "MODBASKETHANDLE": (0, 1, 2),     # height, bottom width, top radius
    "EGGSHAPED": (0,),                # height
    "HORSESHOE": (0,),                # height
    "GOTHIC": (0,),                   # height
    "CATENARY": (0,),                 # height
    "SEMIELLIPTICAL": (0,),           # height
    "BASKETHANDLE": (0,),             # height
    "SEMICIRCULAR": (0,),             # height
    "RECT_TRIANG": (0, 1, 2),         # height, top width, triangle height
    "RECT_ROUND": (0, 1, 2),          # height, top width, bottom radius
    "HORIZ_ELLIPSE": (0, 1),          # height, width
    "VERT_ELLIPSE": (0, 1),           # height, width
    "ARCH": (0, 1),                   # height, width
    "CUSTOM": (0,),                   # height (the shape curve is an index)
    "FORCE_MAIN": (0,),               # diameter
    "IRREGULAR": (),                  # transect-defined — not sizable
    "STREET_XSECT": (),               # street-table-defined — not sizable
    "DUMMY": (),                      # no geometry
}


class LinkDiameter:
    """Shape-aware cross-section sizing for each link.

    The action value is the section's B{rise} — its full depth, in project
    length units. For a L{CIRCULAR} conduit that is exactly the diameter, so
    the factory reads as "pipe diameter" on the shape it is named for; for
    every other shape it is the true full depth reported by the engine's
    analytic cross-section geometry (L{openswmm.engine.XSectionGeometry}),
    B{not} C{geom1}.

    B{What is searched.} One scalar per link. On L{apply} the factory
    computes C{f = target_rise / baseline_rise} and multiplies B{every
    length-dimensioned geometry parameter of the shape} by C{f} — height and
    width for a box culvert, height and bottom width for a trapezoid, height
    and top width and bottom radius for a rect-round, and so on. The section
    is therefore resized B{similarly}: aspect ratio, shape, and hydraulic
    character are preserved and only the scale changes.

    B{What is not searched.} The shape code itself; dimensionless parameters
    (trapezoid side slopes, the POWER exponent, RECT_OPEN's sides-removed
    flag, FORCE_MAIN's roughness coefficient); and index-valued parameters
    (transect, shape-curve and street table references). Independent control
    of, say, a box culvert's width and height is deliberately B{not} offered
    — that would need a two-component-per-link space and a second bound
    vector. Use L{LinkRoughness} / L{LinkLength} alongside this factory to
    vary the other conduit dimensions.

    Shapes whose geometry lives entirely in a transect or street table
    (C{IRREGULAR}, C{STREET_XSECT}) and C{DUMMY} sections have no scalable
    dimension; binding one raises L{ValueError} rather than silently
    resizing nothing.

    @ivar name: Action-space key, default C{"link_diameter"}.
    """

    #: Catalog paths this component needs from the engine (see require_for).
    requires: tuple[str, ...] = ("link.xsect", "xsect")

    def __init__(
        self,
        link_ids: Sequence[str],
        low: float,
        high: float,
        name: str = "link_diameter",
    ) -> None:
        """
        @param link_ids: Symbolic link IDs to control.
        @type link_ids: sequence of str
        @param low: Lower bound on the section rise (full depth).
        @type low: float
        @param high: Upper bound on the section rise.
        @type high: float
        @param name: Action-space key.
        @type name: str
        @raise ValueError: If C{link_ids} is empty or C{high <= low}.
        """
        self._link_ids = _validate_link_ids("LinkDiameter", link_ids)
        self._low = float(low)
        self._high = float(high)
        self.name = name
        self._idxs: list[int] | None = None
        # Per link: (shape_code, geoms, resizable_slots, baseline_rise).
        self._baseline: list[tuple[int, list[float], tuple[int, ...], float]] | None = None
        self._box = _make_box(self._low, self._high, len(self._link_ids))

    @property
    def space(self) -> spaces.Box:
        return self._box

    def bind(self, adapter: SolverAdapter) -> None:
        """Resolve indices and cache each section's baseline geometry.

        @raise ValueError: If a target link's shape has no scalable
            dimension, or its baseline rise is non-positive.
        """
        self._idxs = [adapter.links.get_index(lid) for lid in self._link_ids]
        baseline: list[tuple[int, list[float], tuple[int, ...], float]] = []
        for lid, idx in zip(self._link_ids, self._idxs, strict=True):
            shape_code, *geoms = adapter.links.get_xsect(idx)
            shape_name = adapter.links.get_xsect_shape_name(idx)
            slots = _RESIZABLE_GEOM_SLOTS.get(shape_name)
            if not slots:
                raise ValueError(
                    f"LinkDiameter cannot size link {lid!r}: cross-section "
                    f"shape {shape_name} has no scalable length dimension "
                    "(its geometry comes from a transect / street table). "
                    "Remove it from link_ids."
                )
            rise = float(adapter.links.get_full_depth(idx))
            if rise <= 0.0:
                raise ValueError(
                    f"LinkDiameter cannot size link {lid!r}: the engine "
                    f"reports a full depth of {rise} for its {shape_name} "
                    "cross-section."
                )
            baseline.append((int(shape_code), [float(g) for g in geoms], slots, rise))
        self._baseline = baseline

    def apply(self, adapter: SolverAdapter, value: np.ndarray) -> None:
        if self._idxs is None or self._baseline is None:
            raise RuntimeError("LinkDiameter.bind() must be called before apply()")
        clipped = np.clip(np.asarray(value, dtype=np.float32), self._low, self._high)
        for idx, (shape_code, geoms, slots, rise), v in zip(
            self._idxs, self._baseline, clipped, strict=True
        ):
            factor = float(v) / rise
            scaled = list(geoms)
            for s in slots:
                scaled[s] = geoms[s] * factor
            adapter.links.set_xsect(idx, shape_code, *scaled)


class NodeMaxDepth:
    """Maximum allowable depth at each node.

    Used as a CIP proxy for storage capacity: tank-like nodes whose
    L{max_depth} grows can hold more water before surcharging /
    flooding. The factory delegates to L{openswmm.engine.Nodes.set_max_depth}.

    @ivar name: Action-space key, default C{"node_max_depth"}.
    """

    #: Catalog paths this component needs from the engine (see require_for).
    requires: tuple[str, ...] = ("node.max_depth",)

    def __init__(
        self,
        node_ids: Sequence[str],
        low: float,
        high: float,
        name: str = "node_max_depth",
    ) -> None:
        """
        @param node_ids: Symbolic node IDs to control.
        @type node_ids: sequence of str
        @param low: Lower bound on max_depth.
        @type low: float
        @param high: Upper bound on max_depth.
        @type high: float
        @param name: Action-space key.
        @type name: str
        @raise ValueError: If C{node_ids} is empty or C{high <= low}.
        """
        self._node_ids = _validate_node_ids("NodeMaxDepth", node_ids)
        self._low = float(low)
        self._high = float(high)
        self.name = name
        self._idxs: list[int] | None = None
        self._box = _make_box(self._low, self._high, len(self._node_ids))

    @property
    def space(self) -> spaces.Box:
        return self._box

    def bind(self, adapter: SolverAdapter) -> None:
        self._idxs = [adapter.nodes.get_index(nid) for nid in self._node_ids]

    def apply(self, adapter: SolverAdapter, value: np.ndarray) -> None:
        if self._idxs is None:
            raise RuntimeError("NodeMaxDepth.bind() must be called before apply()")
        clipped = np.clip(np.asarray(value, dtype=np.float32), self._low, self._high)
        for idx, v in zip(self._idxs, clipped, strict=True):
            adapter.nodes.set_max_depth(idx, float(v))


class SubcatchGWOutflowCoeff:
    """Groundwater outflow coefficient (C{a1}) for each subcatchment.

    A CIP / green-infrastructure design lever over baseflow: the
    ``[GROUNDWATER]`` outflow coefficient ``a1`` scales how readily the
    saturated zone discharges to the receiving node. The remaining
    groundwater parameters (surface elevation, exponents, surface-water
    terms, tailwater, threshold) are B{preserved} — the factory reads the
    current parameter tuple at L{bind} time and rewrites only ``a1`` on
    L{apply}, via the engine's :meth:`Subcatchment.set_gw_params`.

    Each target subcatchment must already have an aquifer assigned; reading
    ``gw_params`` on an aquifer-less subcatchment raises in the engine.

    @ivar name: Action-space key, default C{"subcatch_gw_outflow_coeff"}.
    """

    #: Catalog paths this component needs from the engine (see require_for).
    requires: tuple[str, ...] = ("subcatchment.gw_params",)

    def __init__(
        self,
        subcatch_ids: Sequence[str],
        low: float,
        high: float,
        name: str = "subcatch_gw_outflow_coeff",
    ) -> None:
        """
        @param subcatch_ids: Symbolic subcatchment IDs to control.
        @type subcatch_ids: sequence of str
        @param low: Lower bound on the outflow coefficient C{a1}.
        @type low: float
        @param high: Upper bound on C{a1}.
        @type high: float
        @param name: Action-space key.
        @type name: str
        @raise ValueError: If C{subcatch_ids} is empty or C{high <= low}.
        """
        self._subcatch_ids = _validate_subcatch_ids("SubcatchGWOutflowCoeff", subcatch_ids)
        self._low = float(low)
        self._high = float(high)
        self.name = name
        self._idxs: list[int] | None = None
        self._cached_params: list[tuple] | None = None
        self._box = _make_box(self._low, self._high, len(self._subcatch_ids))

    @property
    def space(self) -> spaces.Box:
        return self._box

    def bind(self, adapter: SolverAdapter) -> None:
        self._idxs = [adapter.subcatchments.get_index(s) for s in self._subcatch_ids]
        # Cache the full parameter tuple; only a1 is overwritten in apply.
        self._cached_params = [
            tuple(adapter.subcatchments.get_gw_params(idx)) for idx in self._idxs
        ]

    def apply(self, adapter: SolverAdapter, value: np.ndarray) -> None:
        if self._idxs is None or self._cached_params is None:
            raise RuntimeError(
                "SubcatchGWOutflowCoeff.bind() must be called before apply()"
            )
        clipped = np.clip(np.asarray(value, dtype=np.float32), self._low, self._high)
        for idx, params, v in zip(self._idxs, self._cached_params, clipped, strict=True):
            surf_elev, _a1, b1, a2, b2, a3, tw, hstar = params
            adapter.subcatchments.set_gw_params(
                idx, float(surf_elev), float(v), float(b1),
                float(a2), float(b2), float(a3), float(tw), float(hstar),
            )


class StorageVolume:
    """Storage sizing for each storage node (plan §3.1).

    Sizes detention / retention storage assets by editing the node's
    depth–area relation. Both of the engine's tabular storage
    representations are supported, resolved per node at L{bind} from
    L{openswmm.engine.StorageView.shape}:

      - B{FUNCTIONAL} — the power-law relation C{Area = a * Depth^b + c},
        edited through L{openswmm.engine.StorageView.functional}.
      - B{TABULAR} — the node's depth–area curve, edited through
        L{openswmm.engine.Tables} (the curve's areas are rewritten, its
        depths are left alone).

    Two modes let a config trade interpretability for expressiveness:

      - C{mode="scalar"} (default): one footprint B{multiplier} per node in
        C{[low, high]}. On L{apply} the baseline surface areas read at
        L{bind} are scaled by the multiplier — for FUNCTIONAL by scaling
        C{a} and C{c} and leaving the exponent C{b} untouched, for TABULAR
        by scaling every curve ordinate. Either way a factor C{f} scales
        stored volume by C{f} at every depth. Works on both shapes, so a
        mixed set of nodes can be searched with one action vector.
      - C{mode="coeffs"}: the raw C{(a, b, c)} triple per node, searched
        directly within per-coefficient bounds. C{low}/C{high} are length-3
        sequences C{(a, b, c)} applied to every node. Most expressive; lets
        the search reshape the depth–area curve, not just scale it.
        B{FUNCTIONAL nodes only} — there is no C{(a, b, c)} to search on a
        tabular node, and binding one in this mode raises.

    Each target must be a STORAGE node whose shape is FUNCTIONAL or
    TABULAR. The purely geometric shapes (C{CYLINDRICAL}, C{CONICAL},
    C{PARABOLOID}, C{PYRAMIDAL}) are rejected at L{bind} with a message
    naming the shape.

    B{Note on shared curves.} A TABULAR node's curve is rewritten B{in
    place} in the open model (the C{.inp} on disk is never touched, and
    each episode re-opens from it). Two target nodes sharing one curve is
    rejected at L{bind}; a curve shared with a node B{outside} the target
    set is not detected, and that node is resized too. Give each searchable
    basin its own curve.

    @ivar name: Action-space key, default C{"storage_volume"}.
    """

    #: Catalog paths this component needs from the engine (see require_for).
    requires: tuple[str, ...] = ("node.storage", "tables")

    def __init__(
        self,
        node_ids: Sequence[str],
        low: float | Sequence[float],
        high: float | Sequence[float],
        mode: str = "scalar",
        name: str = "storage_volume",
    ) -> None:
        """
        @param node_ids: Symbolic STORAGE node IDs to size.
        @param low: Lower bound(s). C{mode="scalar"}: a scalar multiplier
            floor. C{mode="coeffs"}: a length-3 C{(a, b, c)} sequence.
        @param high: Upper bound(s), matching C{low}'s shape.
        @param mode: C{"scalar"} (footprint multiplier) or C{"coeffs"}
            (raw functional triple).
        @param name: Action-space key.
        @raise ValueError: On empty C{node_ids}, bad C{mode}, or invalid bounds.
        """
        self._node_ids = _validate_node_ids("StorageVolume", node_ids)
        if mode not in ("scalar", "coeffs"):
            raise ValueError(f"mode must be 'scalar' or 'coeffs', got {mode!r}")
        self._mode = mode
        self.name = name
        n = len(self._node_ids)
        self._idxs: list[int] | None = None
        # Per node, one of:
        #   ("FUNCTIONAL", (a, b, c))
        #   ("TABULAR", (curve_idx, [(depth, area), ...]))
        self._baseline: list[tuple[str, object]] | None = None

        if mode == "scalar":
            self._low = float(low)  # type: ignore[arg-type]
            self._high = float(high)  # type: ignore[arg-type]
            self._box = _make_box(self._low, self._high, n)
        else:  # coeffs
            lo = np.asarray(low, dtype=np.float32)
            hi = np.asarray(high, dtype=np.float32)
            if lo.shape != (3,) or hi.shape != (3,):
                raise ValueError(
                    "coeffs mode requires length-3 (a, b, c) low/high sequences"
                )
            # Node-major tiling: [a0,b0,c0, a1,b1,c1, ...].
            self._box = _make_box_bounds(np.tile(lo, n), np.tile(hi, n))
            self._coeff_low = lo
            self._coeff_high = hi

    @property
    def space(self) -> spaces.Box:
        return self._box

    def bind(self, adapter: SolverAdapter) -> None:
        """Resolve indices and cache each node's baseline depth–area relation.

        @raise ValueError: If a target node's storage shape is neither
            FUNCTIONAL nor TABULAR, if C{mode="coeffs"} is used with a
            TABULAR node, or if two targets share one storage curve.
        """
        self._idxs = [adapter.nodes.get_index(nid) for nid in self._node_ids]
        baseline: list[tuple[str, object]] = []
        seen_curves: dict[int, str] = {}
        for nid, idx in zip(self._node_ids, self._idxs, strict=True):
            shape = adapter.nodes.get_storage_shape(idx)
            if shape == "FUNCTIONAL":
                baseline.append(
                    ("FUNCTIONAL", tuple(adapter.nodes.get_storage_functional(idx)))
                )
            elif shape == "TABULAR":
                if self._mode == "coeffs":
                    raise ValueError(
                        f"StorageVolume(mode='coeffs') cannot size node {nid!r}: "
                        "its storage shape is TABULAR, which has no (a, b, c) "
                        "triple. Use mode='scalar'."
                    )
                curve_idx = int(adapter.nodes.get_storage_curve(idx))
                prior = seen_curves.get(curve_idx)
                if prior is not None:
                    raise ValueError(
                        f"StorageVolume: nodes {prior!r} and {nid!r} share "
                        f"storage curve index {curve_idx}; sizing them "
                        "independently would rewrite the same curve twice. "
                        "Give each searchable basin its own curve."
                    )
                seen_curves[curve_idx] = nid
                points = [
                    (float(x), float(y))
                    for x, y in adapter.tables.get_curve_points(curve_idx)
                ]
                baseline.append(("TABULAR", (curve_idx, points)))
            else:
                raise ValueError(
                    f"StorageVolume cannot size node {nid!r}: storage shape "
                    f"{shape} is neither FUNCTIONAL nor TABULAR."
                )
        self._baseline = baseline

    def apply(self, adapter: SolverAdapter, value: np.ndarray) -> None:
        if self._idxs is None or self._baseline is None:
            raise RuntimeError("StorageVolume.bind() must be called before apply()")
        arr = np.asarray(value, dtype=np.float32)
        if self._mode == "scalar":
            factors = np.clip(arr, self._low, self._high)
            for idx, (shape, base), f in zip(
                self._idxs, self._baseline, factors, strict=True
            ):
                if shape == "FUNCTIONAL":
                    a0, b0, c0 = base  # type: ignore[misc]
                    adapter.nodes.set_storage_functional(
                        idx, float(a0) * float(f), float(b0), float(c0) * float(f)
                    )
                else:  # TABULAR — scale the curve's areas, keep its depths.
                    curve_idx, points = base  # type: ignore[misc]
                    adapter.tables.set_curve_points(
                        curve_idx, [(d, a * float(f)) for d, a in points]
                    )
        else:  # coeffs — FUNCTIONAL only, enforced at bind.
            triples = arr.reshape(len(self._idxs), 3)
            triples = np.clip(triples, self._coeff_low, self._coeff_high)
            for idx, (a, b, c) in zip(self._idxs, triples, strict=True):
                adapter.nodes.set_storage_functional(idx, float(a), float(b), float(c))


class LIDPlacement:
    """Green-infrastructure sizing + type selection per subcatchment (plan §3.1).

    Explores nature-based solutions by placing a sized LID (Low-Impact
    Development) usage on each target subcatchment and, when several
    candidate LID controls are offered, B{choosing which type}. The
    candidate controls are LID definitions that already exist in the model
    (a modeler-defined palette — e.g. a bioretention profile, a
    permeable-pavement profile, a green-roof profile); the factory selects
    among them and sizes the placed area, rather than inventing process
    layers. All placement happens through
    L{openswmm.engine.LIDs.usage_add} in the OPENED (pre-initialize) state.

    Per subcatchment the action carries two components, subcatchment-major
    C{[type_i, area_i]}:

      - B{type}: a continuous index in C{[0, n_controls]} floored and clamped
        to pick one of C{lid_controls} (continuous relaxation of a discrete
        choice, so gradient-free MOEA search over all types stays a plain
        C{Box}).
      - B{area}: placed LID area per subcatchment in C{[area_low, area_high]}
        (project area units).

    C{number}, C{width}, C{init_sat}, and C{from_imperv} are fixed
    placement parameters (not searched).

    @ivar name: Action-space key, default C{"lid_placement"}.
    """

    #: Catalog paths this component needs from the engine (see require_for).
    requires: tuple[str, ...] = ("infrastructure.lids",)

    def __init__(
        self,
        subcatch_ids: Sequence[str],
        lid_controls: Sequence[str],
        area_low: float,
        area_high: float,
        number: int = 1,
        width: float = 0.0,
        init_sat: float = 0.0,
        from_imperv: float = 0.0,
        name: str = "lid_placement",
    ) -> None:
        """
        @param subcatch_ids: Subcatchments to place GI on.
        @param lid_controls: Existing LID-control IDs to choose among
            (the selectable "types"); must be defined in the model.
        @param area_low: Lower bound on placed LID area.
        @param area_high: Upper bound on placed LID area.
        @param number: Number of replicate LID units per placement.
        @param width: Overland-flow width of each unit (0 = engine default).
        @param init_sat: Initial saturation fraction of the LID (0-100).
        @param from_imperv: Percent of upstream impervious runoff routed
            onto the LID.
        @param name: Action-space key.
        @raise ValueError: On empty IDs or invalid area bounds.
        """
        self._subcatch_ids = _validate_subcatch_ids("LIDPlacement", subcatch_ids)
        if not lid_controls:
            raise ValueError("LIDPlacement requires at least one lid_control")
        self._lid_controls = list(lid_controls)
        self._area_low = float(area_low)
        self._area_high = float(area_high)
        if not (self._area_high > self._area_low):
            raise ValueError(
                f"area_high ({area_high}) must be strictly greater than "
                f"area_low ({area_low})"
            )
        self._number = int(number)
        self._width = float(width)
        self._init_sat = float(init_sat)
        self._from_imperv = float(from_imperv)
        self.name = name

        n = len(self._subcatch_ids)
        m = len(self._lid_controls)
        # Subcatchment-major [type, area] bounds. type in [0, m] (floor+clamp
        # -> 0..m-1); area in [area_low, area_high].
        low = np.empty(2 * n, dtype=np.float32)
        high = np.empty(2 * n, dtype=np.float32)
        low[0::2] = 0.0
        high[0::2] = float(m)
        low[1::2] = self._area_low
        high[1::2] = self._area_high
        self._box = _make_box_bounds(low, high)
        self._control_idxs: list[int] | None = None

    @property
    def space(self) -> spaces.Box:
        return self._box

    def bind(self, adapter: SolverAdapter) -> None:
        # Validate subcatchments resolve, and map each candidate control id
        # to its engine index (usage_add needs an integer lid index).
        for sid in self._subcatch_ids:
            adapter.subcatchments.get_index(sid)
        self._control_idxs = [
            adapter.infrastructure.get_lid_index(cid) for cid in self._lid_controls
        ]

    def apply(self, adapter: SolverAdapter, value: np.ndarray) -> None:
        if self._control_idxs is None:
            raise RuntimeError("LIDPlacement.bind() must be called before apply()")
        m = len(self._lid_controls)
        pairs = np.asarray(value, dtype=np.float32).reshape(len(self._subcatch_ids), 2)
        for sid, (type_raw, area_raw) in zip(self._subcatch_ids, pairs, strict=True):
            t = int(np.clip(np.floor(type_raw), 0, m - 1))
            area = float(np.clip(area_raw, self._area_low, self._area_high))
            adapter.infrastructure.lid_usage_add(
                sid,
                self._control_idxs[t],
                self._number,
                area,
                self._width,
                self._init_sat,
                self._from_imperv,
            )


class RDIIUnitHydrograph:
    """RDII unit-hydrograph R + initial-abstraction sizing (RTK triangles).

    Explores rainfall-derived infiltration/inflow response by editing the
    R fraction — and optionally the initial-abstraction terms — of one or
    more existing unit-hydrograph entries, identified by
    C{(uh_name, month, response)}. The triangle's T and K are B{preserved}
    (read at L{bind}), so the search varies only the response volume (R)
    and the initial-abstraction depths, not the hydrograph timing. Edits go
    through L{openswmm.engine.Inflows.set_hydrograph_rtk} /
    C{set_hydrograph_ia} in the OPENED (pre-initialize) state.

    Per target the action is C{[R]} (default) or C{[R, dmax, drecov, dinit]}
    (when C{include_ia=True}), target-major.

    @ivar name: Action-space key, default C{"rdii_unit_hydrograph"}.
    """

    #: Catalog paths this component needs from the engine (see require_for).
    requires: tuple[str, ...] = ("inflows",)

    def __init__(
        self,
        targets: Sequence[tuple[str, int, int]],
        r_low: float,
        r_high: float,
        include_ia: bool = False,
        ia_low: Sequence[float] = (0.0, 0.0, 0.0),
        ia_high: Sequence[float] = (0.0, 0.0, 0.0),
        name: str = "rdii_unit_hydrograph",
    ) -> None:
        """
        @param targets: Unit-hydrograph entries to size, each a
            C{(uh_name, month, response)} triple. C{month} is 0/-1 for the
            all-months row or 1-12; C{response} is 0/1/2 (short/medium/long).
        @param r_low: Lower bound on the R fraction.
        @param r_high: Upper bound on the R fraction.
        @param include_ia: Also search initial abstraction (C{dmax},
            C{drecov}, C{dinit}). When C{False} (default) only R is searched.
        @param ia_low: Length-3 lower bounds C{(dmax, drecov, dinit)}
            (used only when C{include_ia}).
        @param ia_high: Length-3 upper bounds C{(dmax, drecov, dinit)}.
        @param name: Action-space key.
        @raise ValueError: On empty targets or invalid bounds.
        """
        if not targets:
            raise ValueError("RDIIUnitHydrograph requires at least one target")
        self._targets = [
            (str(uh), int(month), int(response)) for (uh, month, response) in targets
        ]
        self._r_low = float(r_low)
        self._r_high = float(r_high)
        self._include_ia = bool(include_ia)
        self.name = name
        self._stride = 4 if include_ia else 1

        per_low = [self._r_low]
        per_high = [self._r_high]
        if include_ia:
            ia_lo = np.asarray(ia_low, dtype=np.float32)
            ia_hi = np.asarray(ia_high, dtype=np.float32)
            if ia_lo.shape != (3,) or ia_hi.shape != (3,):
                raise ValueError("ia_low/ia_high must be length-3 (dmax, drecov, dinit)")
            per_low.extend(ia_lo.tolist())
            per_high.extend(ia_hi.tolist())
            self._ia_low = ia_lo
            self._ia_high = ia_hi

        n = len(self._targets)
        self._box = _make_box_bounds(
            np.tile(per_low, n).astype(np.float32),
            np.tile(per_high, n).astype(np.float32),
        )
        # Cached (t, k) per target, preserved across apply.
        self._tk: list[tuple[float, float]] | None = None

    @property
    def space(self) -> spaces.Box:
        return self._box

    def bind(self, adapter: SolverAdapter) -> None:
        # Resolve each target to its existing hydrograph entry so T and K
        # can be preserved. Entries are addressed by value, so scan.
        count = adapter.inflows.hydrograph_count()
        by_key: dict[tuple[str, int, int], tuple[float, float]] = {}
        for i in range(count):
            e = adapter.inflows.get_hydrograph(i)
            by_key[(str(e.uh_name), int(e.month), int(e.response))] = (
                float(e.t),
                float(e.k),
            )
        tk: list[tuple[float, float]] = []
        for key in self._targets:
            if key not in by_key:
                raise ValueError(
                    f"RDIIUnitHydrograph target {key} not found among "
                    f"{count} hydrograph entries; the unit hydrograph must "
                    "already be defined in the model"
                )
            tk.append(by_key[key])
        self._tk = tk

    def apply(self, adapter: SolverAdapter, value: np.ndarray) -> None:
        if self._tk is None:
            raise RuntimeError("RDIIUnitHydrograph.bind() must be called before apply()")
        rows = np.asarray(value, dtype=np.float32).reshape(len(self._targets), self._stride)
        for (uh, month, response), (t, k), row in zip(
            self._targets, self._tk, rows, strict=True
        ):
            r = float(np.clip(row[0], self._r_low, self._r_high))
            adapter.inflows.set_hydrograph_rtk(uh, month, response, r, float(t), float(k))
            if self._include_ia:
                ia = np.clip(row[1:4], self._ia_low, self._ia_high)
                adapter.inflows.set_hydrograph_ia(
                    uh, month, response, float(ia[0]), float(ia[1]), float(ia[2])
                )


# =============================================================================
# Process-configuration factories (heat / water age / reactions)
# =============================================================================
#
# The engine's heat, water-age and reaction modules are almost entirely
# *configuration* surfaces: the C API exposes no per-node or per-link
# temperature, water-age or species-concentration getter, so none of them can
# be observed from a running solver. That asymmetry is deliberate in this
# package — they appear here as searchable inputs (and, for heat, as a runtime
# actuator) and nowhere in ``observations/``.


class ReactionCoefficientValue:
    """Reaction-coefficient values from C{[REACTION_COEFFICIENTS]}.

    Searches over the rate constants, half-saturation constants, yields and
    stoichiometric factors of a multi-species reaction model — the terms that
    appear by name inside the model's own pipe/tank reaction expressions. One
    scalar per named coefficient, written through
    L{openswmm.engine.Reactions} in the OPENED (pre-initialize) state.

    B{Why this one matters.} Reaction coefficients are exactly the quantities
    a water-quality modeller normally fits by hand against observed
    concentrations. Exposing them as a design space turns calibration into an
    ordinary optimisation problem over the same env machinery that does CIP
    sizing: pair this factory with an observed-vs-simulated objective and the
    search B{is} a calibration run; pair it with a treatment-performance
    objective and it is a process-design run instead.

    B{PARAMETER vs CONSTANT.} The engine distinguishes the two
    (C{is_param}). A CONSTANT is declared by the modeller as fixed, and
    writing one is meaningless-at-best; L{bind} B{refuses} any CONSTANT
    target with a L{ValueError} naming it, rather than writing values the
    model was never meant to vary. Pass only PARAMETER coefficients.

    B{Units.} Whatever the model's reaction expressions assume — the engine
    does not declare coefficient units and this factory does not guess.
    Values are written B{unconverted}, so the bounds must be given in the
    model's own terms.

    @ivar name: Action-space key, default C{"reaction_coefficient_value"}.
    """

    #: Catalog paths this component needs from the engine (see require_for).
    requires: tuple[str, ...] = ("reactions",)

    def __init__(
        self,
        coefficient_ids: Sequence[str],
        low: float | Sequence[float],
        high: float | Sequence[float],
        name: str = "reaction_coefficient_value",
    ) -> None:
        """
        @param coefficient_ids: Coefficient names as declared in
            C{[REACTION_COEFFICIENTS]}. Each must be a PARAMETER.
        @type coefficient_ids: sequence of str
        @param low: Lower bound — a scalar applied to every coefficient, or
            one bound per coefficient (they rarely share a range).
        @type low: float or sequence of float
        @param high: Upper bound, matching C{low}'s shape.
        @type high: float or sequence of float
        @param name: Action-space key.
        @type name: str
        @raise ValueError: If C{coefficient_ids} is empty, or the bounds are
            mis-shaped / not searchable.
        """
        if not coefficient_ids:
            raise ValueError(
                "ReactionCoefficientValue requires at least one coefficient_id"
            )
        self._coefficient_ids = [str(c) for c in coefficient_ids]
        n = len(self._coefficient_ids)

        scalar_low = isinstance(low, (int, float))
        scalar_high = isinstance(high, (int, float))
        if scalar_low != scalar_high:
            raise ValueError(
                "ReactionCoefficientValue: low and high must both be scalars "
                "or both be per-coefficient sequences"
            )
        if scalar_low:
            self._low = np.full((n,), float(low), dtype=np.float32)
            self._high = np.full((n,), float(high), dtype=np.float32)
        else:
            self._low = np.asarray(low, dtype=np.float32)
            self._high = np.asarray(high, dtype=np.float32)
            if self._low.shape != (n,) or self._high.shape != (n,):
                raise ValueError(
                    "ReactionCoefficientValue: per-coefficient low/high must "
                    f"each have one entry per coefficient_id ({n})"
                )
        self.name = name
        self._bound: bool = False
        self._box = _make_box_bounds(self._low, self._high)

    @property
    def space(self) -> spaces.Box:
        return self._box

    def bind(self, adapter: SolverAdapter) -> None:
        """Verify every target exists and is a searchable PARAMETER.

        Coefficients are addressed by name end-to-end (the engine's
        C{get_index} is not needed and index stability across edits is not
        assumed), so C{bind} validates rather than resolving.

        @raise ValueError: If a coefficient is undeclared, or is a CONSTANT
            rather than a PARAMETER.
        """
        declared = set(adapter.reactions.coefficient_names())
        missing = [c for c in self._coefficient_ids if c not in declared]
        if missing:
            raise ValueError(
                f"ReactionCoefficientValue: coefficient(s) {missing!r} are not "
                "declared in [REACTION_COEFFICIENTS]. Declared: "
                f"{sorted(declared)!r}"
            )
        constants = [
            c
            for c in self._coefficient_ids
            if not adapter.reactions.is_coefficient_param(c)
        ]
        if constants:
            raise ValueError(
                f"ReactionCoefficientValue: coefficient(s) {constants!r} are "
                "declared CONSTANT, not PARAMETER. The model marks them as "
                "fixed, so searching over them would write values it was "
                "never meant to vary. Re-declare them as PARAMETER, or drop "
                "them from coefficient_ids."
            )
        self._bound = True

    def apply(self, adapter: SolverAdapter, value: np.ndarray) -> None:
        if not self._bound:
            raise RuntimeError(
                "ReactionCoefficientValue.bind() must be called before apply()"
            )
        clipped = np.clip(np.asarray(value, dtype=np.float32), self._low, self._high)
        for cid, v in zip(self._coefficient_ids, clipped, strict=True):
            adapter.reactions.set_coefficient(cid, float(v))


#: Engine refusal range for a heat source temperature, degrees Celsius. The
#: engine's parser REFUSES (does not clamp) anything outside it, and a refused
#: write does not take effect — so the default action bounds are pinned here
#: rather than left open. An agent sampling at the edge of its own Box must
#: never be able to produce a mid-episode engine refusal.
_HEAT_TEMP_MIN_C = -50.0
_HEAT_TEMP_MAX_C = 100.0


class HeatSourceTemperature:
    """Global inlet temperature per heat-source pathway, degrees Celsius.

    Sets the temperature of the water B{entering} the network along each of
    the engine's source pathways — C{"DWF"}, C{"EXTERNAL_INFLOW"},
    C{"RAINFALL"}, C{"GW"}, C{"RDII"}, C{"IFACE"}, C{"INITIAL_STATE"} — via
    L{openswmm.engine.Heat}. Pathways are named by string; the engine's
    C{HeatSourceKind} enum never enters a config.

    B{Design vs. runtime.} Heat source writes are documented LIVE (they take
    effect on the next routing step), so the same engine call backs both this
    once-per-episode design factory and the per-step runtime actuator
    L{openswmm_gymnasium.spaces.runtime.HeatSourceTemperatureSetpoint}. Use
    this one for a fixed thermal boundary condition per episode (a design /
    scenario variable); use the runtime twin when the agent should modulate
    an inlet temperature over the event.

    B{Bounds.} C{low}/C{high} default to the engine's own refusal range
    C{[-50, 100]} degC. The engine B{refuses} rather than clamps an
    out-of-range write, and a refused write does not take effect — so a
    factory whose Box extended past that range could hand the engine a
    sampled action it silently declines, leaving the previous episode's
    temperature in place. Narrow the bounds freely; widening them past the
    engine range is rejected at construction.

    B{Units.} Degrees Celsius B{regardless of the model's unit system} —
    heat is the one place SWMM does not follow US/SI. Values are written
    unconverted.

    B{Requires} C{[OPTIONS] HEAT_TRANSPORT YES}; L{bind} raises with that
    message otherwise (via L{SolverAdapter.heat}) rather than storing
    temperatures the model will never route.

    @ivar name: Action-space key, default C{"heat_source_temperature"}.
    """

    #: Catalog paths this component needs from the engine (see require_for).
    requires: tuple[str, ...] = ("heat",)

    def __init__(
        self,
        sources: Sequence[str],
        low: float = _HEAT_TEMP_MIN_C,
        high: float = _HEAT_TEMP_MAX_C,
        name: str = "heat_source_temperature",
    ) -> None:
        """
        @param sources: Heat-source pathway names, e.g. C{("DWF",
            "EXTERNAL_INFLOW")}.
        @type sources: sequence of str
        @param low: Lower bound in degC. Defaults to the engine minimum
            C{-50}.
        @type low: float
        @param high: Upper bound in degC. Defaults to the engine maximum
            C{100}.
        @type high: float
        @param name: Action-space key.
        @type name: str
        @raise ValueError: If C{sources} is empty, C{high <= low}, or the
            range extends beyond the engine's C{[-50, 100]}.
        """
        if not sources:
            raise ValueError("HeatSourceTemperature requires at least one source")
        self._sources = [str(s).upper() for s in sources]
        self._low = float(low)
        self._high = float(high)
        if self._low < _HEAT_TEMP_MIN_C or self._high > _HEAT_TEMP_MAX_C:
            raise ValueError(
                f"HeatSourceTemperature bounds [{self._low}, {self._high}] degC "
                f"extend beyond the engine's refusal range "
                f"[{_HEAT_TEMP_MIN_C}, {_HEAT_TEMP_MAX_C}]. The engine refuses "
                "(does not clamp) an out-of-range temperature and a refused "
                "write does not take effect, so a sampled action outside that "
                "range would silently leave the temperature unchanged."
            )
        self.name = name
        self._bound = False
        self._box = _make_box(self._low, self._high, len(self._sources))

    @property
    def space(self) -> spaces.Box:
        return self._box

    def bind(self, adapter: SolverAdapter) -> None:
        """Verify heat transport is on and every named pathway is valid.

        @raise RuntimeError: If the engine build has no heat module, or the
            open model does not enable C{HEAT_TRANSPORT}.
        @raise ValueError: If a name is not a heat-source pathway.
        """
        known = set(adapter.heat.source_kinds())
        unknown = [s for s in self._sources if s not in known]
        if unknown:
            raise ValueError(
                f"HeatSourceTemperature: unknown heat source(s) {unknown!r}. "
                f"Valid pathways: {sorted(known)!r}"
            )
        self._bound = True

    def apply(self, adapter: SolverAdapter, value: np.ndarray) -> None:
        if not self._bound:
            raise RuntimeError(
                "HeatSourceTemperature.bind() must be called before apply()"
            )
        clipped = np.clip(np.asarray(value, dtype=np.float32), self._low, self._high)
        for source, v in zip(self._sources, clipped, strict=True):
            adapter.heat.set_source_temp(source, float(v))


class WaterAgeSourceAge:
    """Global source age per water-age pathway, in hours.

    Sets the age the water entering along each pathway — C{"DWF"},
    C{"EXTERNAL_INFLOW"}, C{"RAINFALL"}, C{"GW"}, C{"RDII"}, C{"IFACE"},
    C{"INITIAL_STATE"} — is credited with, via L{openswmm.engine.WaterAge}.
    Pathways are named by string; the engine's C{WaterAgeSource} enum never
    enters a config.

    B{Negative values are legal and meaningful, and are NOT clamped here.}
    A negative source age B{extracts} age-volume from the pathway (the
    engine clamps the resulting age at zero, not the input). That makes a
    negative bound a real modelling choice — e.g. searching over how much a
    fresh-water injection should reset the age of a stagnant branch — so
    this factory's default low bound is B{not} zero and callers must
    deliberately opt into a non-negative range if that is what they want.
    Silently flooring at zero would delete half the physically meaningful
    search space.

    B{Units.} Hours, the config file's own unit, written unconverted.

    B{Requires} C{[OPTIONS] WATER_AGE YES}; L{bind} raises with that message
    otherwise (via L{SolverAdapter.water_age}) rather than storing ages the
    model will never transport.

    @ivar name: Action-space key, default C{"water_age_source_age"}.
    """

    #: Catalog paths this component needs from the engine (see require_for).
    requires: tuple[str, ...] = ("water_age",)

    def __init__(
        self,
        sources: Sequence[str],
        low: float,
        high: float,
        name: str = "water_age_source_age",
    ) -> None:
        """
        @param sources: Water-age pathway names, e.g. C{("DWF", "RDII")}.
        @type sources: sequence of str
        @param low: Lower bound in hours. B{May be negative} — see the class
            docstring; there is no default, because defaulting either way
            would make a modelling decision on the caller's behalf.
        @type low: float
        @param high: Upper bound in hours.
        @type high: float
        @param name: Action-space key.
        @type name: str
        @raise ValueError: If C{sources} is empty or C{high <= low}.
        """
        if not sources:
            raise ValueError("WaterAgeSourceAge requires at least one source")
        self._sources = [str(s).upper() for s in sources]
        self._low = float(low)
        self._high = float(high)
        self.name = name
        self._bound = False
        # No non-negativity check: negative hours are a legal engine input.
        self._box = _make_box(self._low, self._high, len(self._sources))

    @property
    def space(self) -> spaces.Box:
        return self._box

    def bind(self, adapter: SolverAdapter) -> None:
        """Verify water age is on and every named pathway is valid.

        @raise RuntimeError: If the engine build has no water-age module, or
            the open model does not enable C{WATER_AGE}.
        @raise ValueError: If a name is not a water-age pathway.
        """
        known = set(adapter.water_age.source_pathways())
        unknown = [s for s in self._sources if s not in known]
        if unknown:
            raise ValueError(
                f"WaterAgeSourceAge: unknown water-age source(s) {unknown!r}. "
                f"Valid pathways: {sorted(known)!r}"
            )
        self._bound = True

    def apply(self, adapter: SolverAdapter, value: np.ndarray) -> None:
        if not self._bound:
            raise RuntimeError("WaterAgeSourceAge.bind() must be called before apply()")
        clipped = np.clip(np.asarray(value, dtype=np.float32), self._low, self._high)
        for source, v in zip(self._sources, clipped, strict=True):
            adapter.water_age.set_source_age(source, float(v))
