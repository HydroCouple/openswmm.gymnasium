"""Unit tests for L{openswmm_gymnasium.rewards.RewardRegistry}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import pytest

from openswmm_gymnasium.rewards import (
    CSOVolume,
    FloodingVolume,
    PeakOutflow,
    ReliabilityMargin,
    RewardRegistry,
    SetpointSmoothness,
)


class TestBuiltInRegistrations:
    def test_all_five_builtins_registered(self):
        names = RewardRegistry.names()
        assert {
            "flooding_volume",
            "cso_volume",
            "peak_outflow",
            "reliability_margin",
            "setpoint_smoothness",
        }.issubset(set(names))

    def test_get_returns_correct_class(self):
        assert RewardRegistry.get("flooding_volume") is FloodingVolume
        assert RewardRegistry.get("cso_volume") is CSOVolume
        assert RewardRegistry.get("peak_outflow") is PeakOutflow
        assert RewardRegistry.get("reliability_margin") is ReliabilityMargin
        assert RewardRegistry.get("setpoint_smoothness") is SetpointSmoothness


class TestRegistration:
    def test_unknown_name_raises(self):
        with pytest.raises(KeyError, match="not registered"):
            RewardRegistry.get("does_not_exist")

    def test_idempotent_re_register_same_class(self):
        # Built-ins already registered; re-registering the same class
        # is a no-op.
        RewardRegistry.register("flooding_volume", FloodingVolume)

    def test_conflicting_re_register_raises(self):
        class Imposter:
            pass

        with pytest.raises(ValueError, match="already registered"):
            RewardRegistry.register("flooding_volume", Imposter)
