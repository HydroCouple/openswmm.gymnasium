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

The remaining six §3.1 factories (C{StorageVolume}, C{OutfallStage},
C{WeirCrestElev}, C{OrificeMaxOpening}, C{PumpCurveChoice},
C{LIDPlacement}, C{ControlRuleSelection}) land in subsequent phases as
benchmark scenarios that need them come online.

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


def _validate_link_ids(name: str, link_ids: Sequence[str]) -> list[str]:
    if not link_ids:
        raise ValueError(f"{name} requires at least one link_id")
    return list(link_ids)


def _validate_node_ids(name: str, node_ids: Sequence[str]) -> list[str]:
    if not node_ids:
        raise ValueError(f"{name} requires at least one node_id")
    return list(node_ids)


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
