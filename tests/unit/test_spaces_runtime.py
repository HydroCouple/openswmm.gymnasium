"""Unit tests for L{openswmm_gymnasium.spaces.runtime.OrificeSetting}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from openswmm_gymnasium.spaces.runtime import OrificeSetting


class TestConstruction:
    def test_name_default(self):
        f = OrificeSetting(["C1"])
        assert f.name == "orifice_setting"

    def test_name_custom(self):
        f = OrificeSetting(["C1"], name="gate")
        assert f.name == "gate"

    def test_empty_link_ids_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            OrificeSetting([])


class TestSpace:
    def test_box_shape_matches_links(self):
        f = OrificeSetting(["C1", "C2", "C3"])
        s = f.space
        assert isinstance(s, spaces.Box)
        assert s.shape == (3,)
        assert s.dtype == np.float32

    def test_bounds_are_unit_interval(self):
        f = OrificeSetting(["C1"])
        s = f.space
        assert np.array_equal(s.low, np.zeros(1, dtype=np.float32))
        assert np.array_equal(s.high, np.ones(1, dtype=np.float32))

    def test_sample_in_bounds(self):
        f = OrificeSetting(["C1", "C2"])
        for _ in range(10):
            sample = f.space.sample()
            assert sample.shape == (2,)
            assert (sample >= 0.0).all() and (sample <= 1.0).all()


class TestApplyRequiresBind:
    def test_apply_before_bind_raises(self, minimal_inp):
        # No adapter touched - pure error-path test.
        f = OrificeSetting(["C1"])

        class _Stub:
            """Bare object; apply() should never reach it."""

        with pytest.raises(RuntimeError, match="bind"):
            f.apply(_Stub(), np.array([0.5], dtype=np.float32))
