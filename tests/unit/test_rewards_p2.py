"""Unit tests for the P2 reward terms.

L{FloodingVolume} P1 coverage stays in C{test_rewards_terms.py}; here
we focus on the four new terms.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import pytest

from openswmm_gymnasium.rewards import (
    CSOVolume,
    PeakOutflow,
    ReliabilityMargin,
    RewardTerm,
    SetpointSmoothness,
)

# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------


class TestConstruction:
    @pytest.mark.parametrize(
        "cls, kwargs, expected_dir",
        [
            (CSOVolume, {"node_ids": ["J1"]}, "minimize"),
            (PeakOutflow, {"link_ids": ["C1"]}, "minimize"),
            (ReliabilityMargin, {}, "maximize"),
            (SetpointSmoothness, {"link_ids": ["C1"]}, "minimize"),
        ],
    )
    def test_direction(self, cls, kwargs, expected_dir):
        assert cls(**kwargs).direction == expected_dir

    @pytest.mark.parametrize(
        "cls, name, kwargs",
        [
            (CSOVolume, "cso_volume", {"node_ids": ["J1"]}),
            (PeakOutflow, "peak_outflow", {"link_ids": ["C1"]}),
            (ReliabilityMargin, "reliability_margin", {}),
            (SetpointSmoothness, "setpoint_smoothness", {"link_ids": ["C1"]}),
        ],
    )
    def test_default_name(self, cls, name, kwargs):
        assert cls(**kwargs).name == name

    @pytest.mark.parametrize(
        "cls, kwargs",
        [
            (CSOVolume, {"node_ids": []}),
            (PeakOutflow, {"link_ids": []}),
            (SetpointSmoothness, {"link_ids": []}),
        ],
    )
    def test_empty_ids_raise(self, cls, kwargs):
        with pytest.raises(ValueError, match="at least one"):
            cls(**kwargs)


# -----------------------------------------------------------------------------
# Protocol conformance
# -----------------------------------------------------------------------------


class TestProtocolConformance:
    @pytest.mark.parametrize(
        "term",
        [
            CSOVolume(node_ids=["J1"]),
            PeakOutflow(link_ids=["C1"]),
            ReliabilityMargin(),
            SetpointSmoothness(link_ids=["C1"]),
        ],
    )
    def test_satisfies_reward_term(self, term):
        assert isinstance(term, RewardTerm)


# -----------------------------------------------------------------------------
# Stateful behaviour: PeakOutflow and SetpointSmoothness need engine state
# we don't have in-sandbox. Verify the *logic* against scripted stubs.
# -----------------------------------------------------------------------------


class _StubLinks:
    def __init__(self, flow_series=None, setting_series=None):
        self._flow_series = flow_series or []
        self._setting_series = setting_series or []
        self._flow_step = 0
        self._setting_step = 0

    def get_index(self, lid):
        return 0  # only one link in stubs

    def get_flow(self, idx):
        v = self._flow_series[self._flow_step]
        # advance only when all reads for this step are done; tests
        # call with one idx per step, so always advance.
        self._flow_step += 1
        return v

    def get_control_setting(self, idx):
        v = self._setting_series[self._setting_step]
        self._setting_step += 1
        return v


class _StubAdapter:
    def __init__(self, links):
        self.links = links
        self.nodes = None


class TestPeakOutflowLogic:
    def test_running_max_increment(self):
        # Flows over 4 steps: 1.0 -> 3.0 -> 2.5 -> 4.0
        # Increments    :    +1.0 +2.0  +0.0  +1.0  (total = 4.0)
        links = _StubLinks(flow_series=[1.0, 3.0, 2.5, 4.0])
        adapter = _StubAdapter(links)
        term = PeakOutflow(link_ids=["C1"])
        term.bind(adapter)
        term.reset()
        increments = [term.step(adapter, dt_seconds=15.0) for _ in range(4)]
        assert increments == pytest.approx([1.0, 2.0, 0.0, 1.0])
        assert sum(increments) == pytest.approx(4.0)  # equals peak observed


class TestSetpointSmoothnessLogic:
    def test_first_step_returns_zero(self):
        links = _StubLinks(setting_series=[0.5])
        adapter = _StubAdapter(links)
        term = SetpointSmoothness(link_ids=["C1"])
        term.bind(adapter)
        term.reset()
        assert term.step(adapter, dt_seconds=15.0) == 0.0

    def test_delta_squared(self):
        # Settings: 0.5 -> 0.8 -> 0.3
        # Δ²      :       0.09   0.25 (total = 0.34)
        links = _StubLinks(setting_series=[0.5, 0.8, 0.3])
        adapter = _StubAdapter(links)
        term = SetpointSmoothness(link_ids=["C1"])
        term.bind(adapter)
        term.reset()
        s = [term.step(adapter, dt_seconds=15.0) for _ in range(3)]
        assert s == pytest.approx([0.0, 0.09, 0.25])


# -----------------------------------------------------------------------------
# Integration tier — real engine required
# -----------------------------------------------------------------------------


@pytest.mark.integration
class TestIntegration:
    def test_cso_volume_zero_on_dry_episode(self, solver_adapter):
        adapter = solver_adapter(open=True)
        term = CSOVolume(node_ids=["J1"])
        term.bind(adapter)
        term.reset()
        total = 0.0
        while adapter.is_running:
            adapter.step()
            total += term.step(adapter, dt_seconds=15.0)
        assert total == pytest.approx(0.0, abs=1e-9)
        adapter.close()

    def test_reliability_margin_within_max_depth(self, solver_adapter):
        adapter = solver_adapter(open=True)
        term = ReliabilityMargin(node_ids=["J1"])
        term.bind(adapter)
        term.reset()
        # Read max depth from the fixture; freeboard must be in [0, max_depth].
        max_d = float(adapter.nodes.get_max_depth(adapter.nodes.get_index("J1")))
        observed = []
        while adapter.is_running:
            adapter.step()
            observed.append(term.step(adapter, dt_seconds=15.0))
        assert all(0.0 <= v <= max_d + 1e-6 for v in observed)
        adapter.close()
