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

"""Engine-aware reader that turns live solver state into agent stress metrics.

The L{openswmm_gymnasium.control.MarketController} is pure: it consumes a
C{{agent_id: stress in [0,1]}} mapping. This reader produces that mapping from a
running L{SolverAdapter}, normalizing each agent's raw state to [0,1] per its
C{stress_metric}.

Stress-metric convention (IMPORTANT — a deliberate modelling choice):
each metric is normalized so that it B{rises toward 1.0 as the element nears its
operating limit} (fuller pipe, less freeboard, fuller storage, closer to spill).
This makes every cost curve monotonic-increasing in stress for B{both} buyers
(more stressed -> willing to pay more) and sellers (less spare -> ask more),
matching the market design and the schema's increasing curves.

First-cut normalizations (refine per metric as needed):

  - node metrics (C{freeboard_fraction}, C{storage_fill}, C{spill_imminence},
    C{treatment_utilization}) -> C{clamp(depth / max_depth, 0, 1)} — fraction of
    rim depth reached. (These collapse to one depth-fill proxy for now; the
    distinctions, e.g. true storage volume fill vs. depth, are a documented
    follow-up.)
  - link metric (C{filling_ratio}) -> C{clamp(depth / full_depth, 0, 1)}, where
    C{full_depth} is the section's true rise as reported by the engine's
    analytic cross-section geometry (L{openswmm.engine.XSectionGeometry}).
    Exact for every shape, including box culverts, arches and irregular
    (transect) sections.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

from dataclasses import dataclass

from openswmm_gymnasium._engine import SolverAdapter
from openswmm_gymnasium.config import MarketConfig

_NODE_METRICS = frozenset(
    {"freeboard_fraction", "storage_fill", "spill_imminence", "treatment_utilization"}
)
_LINK_METRICS = frozenset({"filling_ratio"})


@dataclass
class _AgentProbe:
    """Resolved per-agent read plan: engine index + normalization divisor."""

    agent_id: str
    element_type: str
    idx: int
    divisor: float  # max_depth (node) or full_depth (link); >0


class MarketMetricReader:
    """Compute C{{agent_id: stress in [0,1]}} from a running solver.

    Construct from a L{MarketConfig}, L{bind} once against the open/initialized
    adapter (resolves indices and caches the normalization divisors), then call
    L{read} each control step.
    """

    def __init__(self, config: MarketConfig) -> None:
        """
        @param config: The market configuration whose agents are read.
        @type config: L{MarketConfig}
        """
        self._agents = list(config.agents)
        self._probes: list[_AgentProbe] | None = None

    def bind(self, adapter: SolverAdapter) -> None:
        """Resolve engine indices and cache per-agent normalization divisors.

        @param adapter: Adapter wrapping the open, initialized solver.
        @raise ValueError: If an agent's C{stress_metric} is unsupported for its
            C{element_type}.
        """
        probes: list[_AgentProbe] = []
        for a in self._agents:
            if a.element_type == "node" and a.stress_metric in _NODE_METRICS:
                idx = adapter.nodes.get_index(a.id)
                divisor = float(adapter.nodes.get_max_depth(idx))
            elif a.element_type == "link" and a.stress_metric in _LINK_METRICS:
                idx = adapter.links.get_index(a.id)
                divisor = float(adapter.links.get_full_depth(idx))
            else:
                raise ValueError(
                    f"agent {a.id!r}: stress_metric {a.stress_metric!r} is not "
                    f"supported for element_type {a.element_type!r}"
                )
            if divisor <= 0.0:
                divisor = 1.0  # degenerate geometry; avoid div-by-zero
            probes.append(_AgentProbe(a.id, a.element_type, idx, divisor))
        self._probes = probes

    def read(self, adapter: SolverAdapter) -> dict[str, float]:
        """Return C{{agent_id: stress in [0,1]}} for the current solver state."""
        assert self._probes is not None, "bind() before read()"
        out: dict[str, float] = {}
        node_depth = adapter.nodes.get_depth
        link_depth = adapter.links.get_depth
        for p in self._probes:
            raw = node_depth(p.idx) if p.element_type == "node" else link_depth(p.idx)
            stress = float(raw) / p.divisor
            out[p.agent_id] = 0.0 if stress < 0.0 else 1.0 if stress > 1.0 else stress
        return out
