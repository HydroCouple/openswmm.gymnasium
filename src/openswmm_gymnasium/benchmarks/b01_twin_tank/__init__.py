"""
b01 TwinTank benchmark scenario.

Plan §6A: two storage tanks in series with a controllable side orifice
between them, draining through a conduit to an outfall. A 2-hour
triangular design storm hits the upstream tank's subcatchment.

Competing objectives (all framed as cost-to-minimise for the env's
internal aggregation; sign flipping handled per term direction):

  - C{flooding_volume} — overflow at C{T1} (upstream).
  - C{peak_outflow} — running peak of flow in C{OUT} conduit
    (downstream sensitive receiver).
  - C{setpoint_smoothness} — L2 of Δsetting on C{ORIF} between steps.

Three Gymnasium IDs are registered:

  - C{OpenSWMM/TwinTank-RTC-v0}   — runtime control of the orifice
    setting; design fixed at the C{.inp} defaults.
  - C{OpenSWMM/TwinTank-Joint-v0} — CIP (tank max-depths) + RTC.
  - C{OpenSWMM/TwinTank-MORTC-v0} — multi-objective RTC variant
    returning vector reward and per-episode normalised HV in
    C{info["mo_score"]}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import gymnasium as gym

from openswmm_gymnasium.envs import (
    SwmmJointCIPRTCEnv,
    SwmmMORTCEnv,
    SwmmRTCEnv,
)
from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.rewards import (
    FloodingVolume,
    PeakOutflow,
    SetpointSmoothness,
)
from openswmm_gymnasium.spaces.design import NodeMaxDepth
from openswmm_gymnasium.spaces.runtime import OrificeSetting

# Symbolic IDs in the scenario.inp — keep in sync with the INP file.
SCENARIO_INP = (Path(__file__).parent / "scenario.inp").resolve()
NODES = ("T1", "T2", "O1")
LINKS = ("ORIF", "OUT")
SUBCATCHMENTS = ("S1",)
GAGES = ("RG",)
CONTROLLED_LINK = "ORIF"
PEAK_LINK = "OUT"
FLOOD_NODE = "T1"

# Reference / ideal points for normalised HV. Tunable per scenario;
# these are deliberately loose so a default run lands in [0, 1].
MO_IDEAL = [0.0, 0.0, 0.0]  # zero flooding, zero peak, zero churn
MO_REF = [1.0e5, 50.0, 100.0]  # generous nadir; refine after evaluation runs


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------


def _observation_builder() -> ObservationBuilder:
    """The B{recommended} default observation for this scenario."""
    return (
        ObservationBuilder()
        .add_node_depths(["T1", "T2"])
        .add_link_flows(["OUT"])
        .add_link_settings([CONTROLLED_LINK])
        .add_rainfall(["RG"])
        .add_clock(features=["elapsed_frac"])
    )


def _reward_terms() -> list:
    return [
        FloodingVolume(node_ids=[FLOOD_NODE]),
        PeakOutflow(link_ids=[PEAK_LINK]),
        SetpointSmoothness(link_ids=[CONTROLLED_LINK]),
    ]


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_rtc_env(**overrides: Any) -> SwmmRTCEnv:
    return SwmmRTCEnv(
        SCENARIO_INP,
        runtime_factories=[OrificeSetting([CONTROLLED_LINK])],
        observation_builder=_observation_builder(),
        reward_terms=_reward_terms(),
        **overrides,
    )


def _make_joint_env(**overrides: Any) -> SwmmJointCIPRTCEnv:
    return SwmmJointCIPRTCEnv(
        SCENARIO_INP,
        design_factories=[NodeMaxDepth(node_ids=["T1", "T2"], low=6.0, high=16.0)],
        runtime_factories=[OrificeSetting([CONTROLLED_LINK])],
        observation_builder=_observation_builder(),
        reward_terms=_reward_terms(),
        **overrides,
    )


def _make_mo_env(**overrides: Any) -> SwmmMORTCEnv:
    return SwmmMORTCEnv(
        SCENARIO_INP,
        runtime_factories=[OrificeSetting([CONTROLLED_LINK])],
        observation_builder=_observation_builder(),
        reward_terms=_reward_terms(),
        ideal_point=MO_IDEAL,
        reference_point=MO_REF,
        **overrides,
    )


def make_env(
    variant: Literal["rtc", "joint", "mo"] = "rtc",
    **overrides: Any,
) -> gym.Env:
    """Construct one of the registered TwinTank env variants.

    @param variant: C{"rtc"}, C{"joint"}, or C{"mo"}.
    @type variant: str
    @param overrides: Keyword arguments forwarded to the env constructor.
    @type overrides: dict
    @return: The constructed env.
    @rtype: L{gymnasium.Env}
    @raise ValueError: If C{variant} is unknown.
    """
    if variant == "rtc":
        return _make_rtc_env(**overrides)
    if variant == "joint":
        return _make_joint_env(**overrides)
    if variant == "mo":
        return _make_mo_env(**overrides)
    raise ValueError(f"Unknown variant {variant!r}; expected 'rtc', 'joint', or 'mo'")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

gym.register(
    id="OpenSWMM/TwinTank-RTC-v0",
    entry_point="openswmm_gymnasium.benchmarks.b01_twin_tank:_make_rtc_env",
)
gym.register(
    id="OpenSWMM/TwinTank-Joint-v0",
    entry_point="openswmm_gymnasium.benchmarks.b01_twin_tank:_make_joint_env",
)
gym.register(
    id="OpenSWMM/TwinTank-MORTC-v0",
    entry_point="openswmm_gymnasium.benchmarks.b01_twin_tank:_make_mo_env",
)


__all__ = [
    "make_env",
    "SCENARIO_INP",
    "NODES",
    "LINKS",
    "CONTROLLED_LINK",
    "PEAK_LINK",
    "FLOOD_NODE",
    "MO_IDEAL",
    "MO_REF",
]
