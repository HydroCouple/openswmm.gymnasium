"""Unit tests for the P2 collectors added to L{ObservationBuilder}.

P1 collectors (node depths, link flows) are covered in
C{test_observations_builder.py}. Here we focus on the new ones plus the
clock collector's pure-math feature computation.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from gymnasium import spaces

from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.observations.builder import _ClockCollector


class TestNewCollectorsSize:
    """Each new collector should contribute the right number of features."""

    @pytest.mark.parametrize(
        "method, ids, expected_size",
        [
            ("add_node_heads", ["J1", "J2"], 2),
            ("add_node_inflows", ["J1"], 1),
            ("add_node_overflows", ["J1", "J2", "J3"], 3),
            ("add_link_depths", ["C1"], 1),
            ("add_link_settings", ["C1", "C2"], 2),
            ("add_subcatch_runoff", ["S1"], 1),
            ("add_rainfall", ["RainGage"], 1),
        ],
    )
    def test_size(self, method, ids, expected_size):
        b = ObservationBuilder()
        getattr(b, method)(ids)
        assert b.space().shape == (expected_size,)
        assert b.space().dtype == np.float32

    @pytest.mark.parametrize(
        "method",
        [
            "add_node_heads",
            "add_node_inflows",
            "add_node_overflows",
            "add_link_depths",
            "add_link_settings",
            "add_subcatch_runoff",
            "add_rainfall",
        ],
    )
    def test_empty_ids_raises(self, method):
        with pytest.raises(ValueError, match="at least one"):
            getattr(ObservationBuilder(), method)([])


class TestClock:
    def test_default_features(self):
        b = ObservationBuilder().add_clock()
        assert b.space().shape == (3,)

    def test_feature_subset(self):
        b = ObservationBuilder().add_clock(features=["elapsed_frac"])
        assert b.space().shape == (1,)

    def test_unknown_feature_raises(self):
        with pytest.raises(ValueError, match="Unknown clock features"):
            ObservationBuilder().add_clock(features=["not_a_real_feature"])

    def test_collect_against_stub_adapter(self):
        """Clock math against a stub — no engine needed.

        The collector reads C{adapter.current_time / start_time /
        end_time}, which are pure floats. We can satisfy that without
        the engine.
        """
        c = _ClockCollector()

        class _StubAdapter:
            # 6:00 AM on day 100 → fractional part 0.25 → angle = π/2
            # → sin=1, cos≈0
            current_time = 100.25
            start_time = 100.0
            end_time = 101.0

        c.bind(_StubAdapter())
        out = c.collect(_StubAdapter())
        assert out.shape == (3,)
        assert out[0] == pytest.approx(math.sin(math.pi / 2), abs=1e-6)  # hour_sin
        assert out[1] == pytest.approx(math.cos(math.pi / 2), abs=1e-6)  # hour_cos
        assert out[2] == pytest.approx(0.25, abs=1e-6)  # elapsed_frac

    def test_collect_clamps_elapsed_frac_at_one(self):
        c = _ClockCollector(features=["elapsed_frac"])

        class _StubAdapter:
            current_time = 102.0  # past end
            start_time = 100.0
            end_time = 101.0

        c.bind(_StubAdapter())
        out = c.collect(_StubAdapter())
        assert out[0] == pytest.approx(1.0)


class TestTenFeatureChain:
    """The whole P2 collector set composes into a 10+ -feature observation."""

    def test_ten_features(self):
        b = (
            ObservationBuilder()
            .add_node_depths(["J1"])
            .add_node_heads(["J1"])
            .add_node_inflows(["J1"])
            .add_node_overflows(["J1"])
            .add_link_flows(["C1"])
            .add_link_depths(["C1"])
            .add_link_settings(["C1"])
            .add_subcatch_runoff(["S1"])
            .add_rainfall(["RainGage"])
            .add_clock(features=["elapsed_frac"])
        )
        s = b.space()
        assert isinstance(s, spaces.Box)
        assert s.shape == (10,)
