"""
Runtime (RTC) action factories.

Each factory exposes a L{gymnasium.spaces.Box}-or-similar over a set of
controllable elements and translates sampled values into persistent
runtime overrides via the link C{target_setting} (or related) each step.

Plan §3.2.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from gymnasium import spaces

from openswmm_gymnasium._engine import SolverAdapter


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
    applied via L{SolverAdapter.set_lateral_inflow} (the engine's
    ``swmm_node_set_lateral_inflow``). Useful for controllable sources,
    pumped diversions, or adversarial inflow scenarios in RL tasks.

    @ivar _node_ids: Symbolic node IDs supplied at construction.
    @ivar _max_inflow: Upper bound of each component's action range.
    """

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
            adapter.set_lateral_inflow(idx, float(v))
