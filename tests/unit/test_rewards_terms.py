"""Unit tests for L{openswmm_gymnasium.rewards.FloodingVolume}.

Per plan §8.0 the integration tier (RUNNING engine) drives the term
against the minimal fixture. The construction tier (this file's
TestConstruction class) needs only a stub.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import pytest

from openswmm_gymnasium.rewards import FloodingVolume, RewardTerm


class TestConstruction:
    def test_default_name_and_direction(self):
        term = FloodingVolume()
        assert term.name == "flooding_volume"
        assert term.direction == "minimize"

    def test_custom_name(self):
        term = FloodingVolume(name="overflow")
        assert term.name == "overflow"

    def test_node_ids_passed_through(self):
        term = FloodingVolume(node_ids=["J1", "J2"])
        # Internal but worth a sanity check — we expose no public getter.
        assert term._node_ids == ["J1", "J2"]

    def test_step_before_bind_raises(self):
        term = FloodingVolume()

        class _Stub:
            pass

        with pytest.raises(AssertionError, match="bind"):
            term.step(_Stub(), dt_seconds=15.0)


class TestProtocolConformance:
    def test_flooding_volume_is_reward_term(self):
        """L{FloodingVolume} must satisfy the L{RewardTerm} protocol."""
        assert isinstance(FloodingVolume(), RewardTerm)


# -----------------------------------------------------------------------------
# Integration tier — needs the real engine running
# -----------------------------------------------------------------------------


@pytest.mark.integration
class TestIntegration:
    def test_zero_flooding_on_minimal_dry_episode(self, solver_adapter):
        """On the minimal fixture the inflow drains via C1; no flooding."""
        adapter = solver_adapter(open=True)
        term = FloodingVolume()
        term.bind(adapter)
        term.reset()
        # Drive a few steps; FloodingVolume should be zero.
        total = 0.0
        while adapter.is_running:
            adapter.step()
            total += term.step(adapter, dt_seconds=15.0)
        assert total == pytest.approx(0.0, abs=1e-9)
        adapter.close()
