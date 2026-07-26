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
  - L{LinkDiameter}  — sets cross-section C{geom1} (e.g. diameter for
    L{CIRCULAR}); shape is preserved by reading the current C{xsect}
    and rewriting only the first geometry parameter.
  - L{NodeMaxDepth}  — sets node L{max_depth} (useful as a proxy for
    storage volume on tank-like nodes).

Three further §3.1-family factories size the assets modelers most want to
explore:

  - L{StorageVolume} — sizes detention/retention storage via the FUNCTIONAL
    surface-area relation (scalar footprint multiplier or raw C{(a,b,c)}).
  - L{LIDPlacement}  — sizes green-infrastructure / nature-based solutions
    and selects among candidate LID control types per subcatchment.
  - L{RDIIUnitHydrograph} — sizes RDII response by editing unit-hydrograph
    R fractions (and optionally initial abstraction), preserving T and K.

The remaining §3.1 factories (C{OutfallStage}, C{WeirCrestElev},
C{OrificeMaxOpening}, C{PumpCurveChoice}, C{ControlRuleSelection}) land in
subsequent phases as benchmark scenarios that need them come online.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
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


class LinkDiameter:
    """Cross-section primary geometry parameter (C{geom1}) for each link.

    For a L{CIRCULAR} conduit this is the diameter; for other shapes
    it is the first dimension per the engine's
    L{openswmm.engine.CrossSection.geom_labels}. The shape itself is
    B{preserved} — the factory reads the existing cross-section at
    L{bind} time and rewrites only C{geom1} on L{apply}, leaving
    C{shape}, C{geom2}, C{geom3}, C{geom4} unchanged.

    @ivar name: Action-space key, default C{"link_diameter"}.
    """

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
        @param low: Lower bound on C{geom1}.
        @type low: float
        @param high: Upper bound on C{geom1}.
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
        self._cached_xsects: list[tuple[int, float, float, float, float]] | None = None
        self._box = _make_box(self._low, self._high, len(self._link_ids))

    @property
    def space(self) -> spaces.Box:
        return self._box

    def bind(self, adapter: SolverAdapter) -> None:
        self._idxs = [adapter.links.get_index(lid) for lid in self._link_ids]
        # Cache shape + ancillary geom; we only overwrite geom1 in apply.
        self._cached_xsects = [tuple(adapter.links.get_xsect(idx)) for idx in self._idxs]

    def apply(self, adapter: SolverAdapter, value: np.ndarray) -> None:
        if self._idxs is None or self._cached_xsects is None:
            raise RuntimeError("LinkDiameter.bind() must be called before apply()")
        clipped = np.clip(np.asarray(value, dtype=np.float32), self._low, self._high)
        for idx, xsect, v in zip(self._idxs, self._cached_xsects, clipped, strict=True):
            shape, _g1, g2, g3, g4 = xsect
            adapter.links.set_xsect(idx, int(shape), float(v), float(g2), float(g3), float(g4))


class NodeMaxDepth:
    """Maximum allowable depth at each node.

    Used as a CIP proxy for storage capacity: tank-like nodes whose
    L{max_depth} grows can hold more water before surcharging /
    flooding. The factory delegates to L{openswmm.engine.Nodes.set_max_depth}.

    @ivar name: Action-space key, default C{"node_max_depth"}.
    """

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
    """Functional-storage sizing for each storage node (plan §3.1).

    Sizes detention / retention storage assets by editing the FUNCTIONAL
    surface-area relation C{Area = a * Depth^b + c} through
    L{openswmm.engine.StorageView.functional}. Two modes let a config trade
    interpretability for expressiveness:

      - C{mode="scalar"} (default): one footprint B{multiplier} per node in
        C{[low, high]}. On L{apply} the baseline C{a} and C{c} coefficients
        (read at L{bind}) are scaled by the multiplier, leaving the exponent
        C{b} untouched — so a factor C{f} scales stored volume by C{f} at
        every depth. This is the simplest single-knob "how big is the tank"
        design variable.
      - C{mode="coeffs"}: the raw C{(a, b, c)} triple per node, searched
        directly within per-coefficient bounds. C{low}/C{high} are length-3
        sequences C{(a, b, c)} applied to every node. Most expressive; lets
        the search reshape the depth–area curve, not just scale it.

    Each target must be a STORAGE node whose shape is FUNCTIONAL; reading
    C{functional} on a non-storage / tabular node raises in the engine.

    @ivar name: Action-space key, default C{"storage_volume"}.
    """

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
        self._baseline: list[tuple[float, float, float]] | None = None

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
        self._idxs = [adapter.nodes.get_index(nid) for nid in self._node_ids]
        # Baseline functional triple; scalar mode scales it, coeffs mode
        # overwrites it (cached anyway so bind is uniform across modes).
        self._baseline = [
            tuple(adapter.nodes.get_storage_functional(idx)) for idx in self._idxs
        ]

    def apply(self, adapter: SolverAdapter, value: np.ndarray) -> None:
        if self._idxs is None or self._baseline is None:
            raise RuntimeError("StorageVolume.bind() must be called before apply()")
        arr = np.asarray(value, dtype=np.float32)
        if self._mode == "scalar":
            factors = np.clip(arr, self._low, self._high)
            for idx, base, f in zip(self._idxs, self._baseline, factors, strict=True):
                a0, b0, c0 = base
                adapter.nodes.set_storage_functional(
                    idx, float(a0) * float(f), float(b0), float(c0) * float(f)
                )
        else:  # coeffs
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
