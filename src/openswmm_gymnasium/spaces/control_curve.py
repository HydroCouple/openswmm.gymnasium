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

"""Search space over reactive PWL control-curve breakpoints.

The decision vector is a flat list of per-knot C{y} settings — one block per
controlled link, knot-major — that L{ControlCurvePolicySpace.unflatten}
decodes (applying any monotonic projection) into a
L{ControlCurvePolicy<openswmm_gymnasium.control.control_curve.ControlCurvePolicy>}
for a
L{ControlCurveController<openswmm_gymnasium.control.control_curve.ControlCurveController>}.
The C{x} knot positions are B{fixed} (not searched) for optimization stability
and curve interpretability; only the settings move.

Layout is asset-major: all C{len(x_knots)} settings of the first asset, then the
second, and so on — matching C{labels} order C{control_curve/<link_id>/y[<k>]}.
This makes a job's decision vector self-describing: each label decodes straight
back to a per-asset C{(x_knots, y_values)} curve.

The neutral curve (all settings at C{y_high}, the default C{y_init}) reproduces
the uncontrolled baseline for passive-open structures — a flat C{y = 1.0} curve
is state-independent and equivalent to never touching the link.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from gymnasium import spaces

from openswmm_gymnasium.control.control_curve import (
    MONOTONIC_MODES,
    ControlCurvePolicy,
    CurveAsset,
    project_monotonic,
)


class _AssetSpec:
    """Validated static definition of one controllable asset's curve."""

    __slots__ = (
        "link_id",
        "obs_node",
        "obs_attr",
        "x_knots",
        "y_low",
        "y_high",
        "y_init",
        "monotonic",
    )

    def __init__(self, raw: Mapping[str, Any]) -> None:
        if not isinstance(raw, Mapping):
            raise ValueError("each asset must be a mapping")
        link_id = raw.get("link_id")
        obs_node = raw.get("obs_node")
        if not link_id or not isinstance(link_id, str):
            raise ValueError("asset requires a non-empty string link_id")
        if not obs_node or not isinstance(obs_node, str):
            raise ValueError(f"asset {link_id!r} requires a non-empty obs_node")

        x_knots = [float(x) for x in raw.get("x_knots", [])]
        if len(x_knots) < 2:
            raise ValueError(
                f"asset {link_id!r}: x_knots must have at least 2 entries"
            )
        if any(x_knots[i + 1] <= x_knots[i] for i in range(len(x_knots) - 1)):
            raise ValueError(
                f"asset {link_id!r}: x_knots must be strictly increasing"
            )

        y_low = float(raw.get("y_low", 0.0))
        y_high = float(raw.get("y_high", 1.0))
        if y_low > y_high:
            raise ValueError(
                f"asset {link_id!r}: y_low ({y_low}) must be <= y_high ({y_high})"
            )

        monotonic = str(raw.get("monotonic", "none"))
        if monotonic not in MONOTONIC_MODES:
            raise ValueError(
                f"asset {link_id!r}: monotonic must be one of {MONOTONIC_MODES}, "
                f"got {monotonic!r}"
            )

        y_init_raw = raw.get("y_init")
        if y_init_raw is None:
            y_init = [y_high] * len(x_knots)
        else:
            y_init = [float(v) for v in y_init_raw]
            if len(y_init) != len(x_knots):
                raise ValueError(
                    f"asset {link_id!r}: y_init has {len(y_init)} entries, "
                    f"expected {len(x_knots)} (one per knot)"
                )

        self.link_id = link_id
        self.obs_node = obs_node
        self.obs_attr = str(raw.get("obs_attr", "depthN"))
        self.x_knots = tuple(x_knots)
        self.y_low = y_low
        self.y_high = y_high
        self.y_init = tuple(y_init)
        self.monotonic = monotonic


class ControlCurvePolicySpace:
    """A bounded vector space over per-asset PWL breakpoint settings.

    @ivar assets: Validated per-asset curve definitions, in actuation order.
    @ivar labels: Per-dimension labels C{control_curve/<link_id>/y[<k>]}.
    @ivar x_normalized: Whether C{x} is a fraction of observed full depth.
    @ivar rate_limit: Max C{|Δsetting|} per control step, or C{None}.
    @ivar control_interval_steps: Recompute cadence (>= 1).
    """

    def __init__(
        self,
        assets: Sequence[Mapping[str, Any]],
        *,
        x_normalized: bool = True,
        rate_limit: float | None = None,
        control_interval_steps: int = 1,
    ) -> None:
        """
        @param assets: One mapping per controllable asset (required, non-empty).
            Keys: C{link_id}, C{obs_node}, C{obs_attr}, C{x_knots}, C{y_low},
            C{y_high}, C{y_init}, C{monotonic}.
        @type assets: sequence of mapping
        @param x_normalized: Index curves by C{depth / full_depth} (default).
        @type x_normalized: bool
        @param rate_limit: Max C{|Δsetting|} per control step, or C{None}.
        @type rate_limit: float or C{None}
        @param control_interval_steps: Recompute the curve every N control
            steps (>= 1).
        @type control_interval_steps: int
        @raise ValueError: On empty assets, a malformed asset, a non-positive
            C{rate_limit}, or C{control_interval_steps < 1}.
        """
        if not assets:
            raise ValueError("ControlCurvePolicySpace requires at least one asset")
        if rate_limit is not None and float(rate_limit) <= 0.0:
            raise ValueError("rate_limit must be > 0 when set")
        if int(control_interval_steps) < 1:
            raise ValueError("control_interval_steps must be >= 1")

        self.assets: list[_AssetSpec] = [_AssetSpec(a) for a in assets]
        seen: set[str] = set()
        for a in self.assets:
            if a.link_id in seen:
                raise ValueError(f"duplicate link_id {a.link_id!r} in assets")
            seen.add(a.link_id)

        self.x_normalized = bool(x_normalized)
        self.rate_limit = None if rate_limit is None else float(rate_limit)
        self.control_interval_steps = int(control_interval_steps)

        self.labels: list[str] = [
            f"control_curve/{a.link_id}/y[{k}]"
            for a in self.assets
            for k in range(len(a.x_knots))
        ]
        self._low_v = np.concatenate(
            [np.full(len(a.x_knots), a.y_low, dtype=np.float32) for a in self.assets]
        )
        self._high_v = np.concatenate(
            [np.full(len(a.x_knots), a.y_high, dtype=np.float32) for a in self.assets]
        )

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> ControlCurvePolicySpace:
        """Build from the declarative C{control_curve} params block.

        @param params: Mapping with C{assets} plus the optional top-level
            C{x_normalized}, C{rate_limit}, C{control_interval_steps}.
        @type params: mapping
        @rtype: L{ControlCurvePolicySpace}
        """
        return cls(
            params.get("assets", []),
            x_normalized=bool(params.get("x_normalized", True)),
            rate_limit=params.get("rate_limit"),
            control_interval_steps=int(params.get("control_interval_steps", 1)),
        )

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def low(self) -> np.ndarray:
        """Per-dimension lower bounds (float32)."""
        return self._low_v.copy()

    @property
    def high(self) -> np.ndarray:
        """Per-dimension upper bounds (float32)."""
        return self._high_v.copy()

    @property
    def space(self) -> spaces.Box:
        """The C{gymnasium.spaces.Box} over the flat breakpoint vector."""
        return spaces.Box(
            low=self._low_v, high=self._high_v, shape=(len(self),), dtype=np.float32
        )

    def default_vector(self) -> np.ndarray:
        """Return the neutral seed vector (each asset's C{y_init}).

        @rtype: L{numpy.ndarray}
        """
        return np.concatenate(
            [np.asarray(a.y_init, dtype=np.float32) for a in self.assets]
        )

    def unflatten(self, vector: np.ndarray) -> ControlCurvePolicy:
        """Decode a flat decision *vector* into a L{ControlCurvePolicy}.

        Per-asset settings are clipped to C{[y_low, y_high]} and then projected
        onto the asset's monotonicity constraint (deterministic, idempotent).

        @param vector: Flat vector in C{labels} order.
        @type vector: L{numpy.ndarray}
        @return: The decoded, ready-to-apply policy.
        @rtype: L{ControlCurvePolicy}
        @raise ValueError: If C{vector} length mismatches the layout.
        """
        vec = np.asarray(vector, dtype=np.float64).ravel()
        if vec.shape[0] != len(self):
            raise ValueError(f"vector length {vec.shape[0]} != {len(self)}")

        decoded: list[CurveAsset] = []
        offset = 0
        for a in self.assets:
            n = len(a.x_knots)
            block = np.clip(vec[offset : offset + n], a.y_low, a.y_high)
            offset += n
            y_values = project_monotonic([float(v) for v in block], a.monotonic)
            decoded.append(
                CurveAsset(
                    link_id=a.link_id,
                    obs_node=a.obs_node,
                    obs_attr=a.obs_attr,
                    x_knots=a.x_knots,
                    y_values=y_values,
                )
            )
        return ControlCurvePolicy(
            assets=tuple(decoded),
            x_normalized=self.x_normalized,
            rate_limit=self.rate_limit,
            control_interval_steps=self.control_interval_steps,
        )

    def decode_curves(self, vector: np.ndarray) -> list[dict[str, Any]]:
        """Decode *vector* into human-readable per-asset curves.

        @param vector: Flat decision vector in C{labels} order.
        @type vector: L{numpy.ndarray}
        @return: One dict per asset with C{link_id}, C{obs_node}, C{obs_attr},
            C{x_knots}, and the applied (projected) C{y_values}.
        @rtype: list of dict
        """
        policy = self.unflatten(vector)
        return [
            {
                "link_id": a.link_id,
                "obs_node": a.obs_node,
                "obs_attr": a.obs_attr,
                "x_knots": list(a.x_knots),
                "y_values": list(a.y_values),
            }
            for a in policy.assets
        ]
