"""Unit tests for L{openswmm_gymnasium.observations.ObservationBuilder}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from openswmm_gymnasium.observations import ObservationBuilder


class TestConstruction:
    def test_empty_builder_space_raises(self):
        b = ObservationBuilder()
        with pytest.raises(ValueError, match="empty"):
            b.space()

    def test_chainable(self):
        b = ObservationBuilder()
        ret = b.add_node_depths(["J1"]).add_link_flows(["C1"])
        assert ret is b

    def test_node_depths_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            ObservationBuilder().add_node_depths([])

    def test_link_flows_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            ObservationBuilder().add_link_flows([])


class TestSpace:
    def test_single_node_box_shape(self):
        b = ObservationBuilder().add_node_depths(["J1"])
        s = b.space()
        assert isinstance(s, spaces.Box)
        assert s.shape == (1,)
        assert s.dtype == np.float32

    def test_concatenation_size(self):
        b = ObservationBuilder().add_node_depths(["J1", "J2", "J3"]).add_link_flows(["C1", "C2"])
        s = b.space()
        assert s.shape == (5,)

    def test_bounds_infinite(self):
        b = ObservationBuilder().add_node_depths(["J1"])
        s = b.space()
        assert np.isneginf(s.low).all()
        assert np.isposinf(s.high).all()
