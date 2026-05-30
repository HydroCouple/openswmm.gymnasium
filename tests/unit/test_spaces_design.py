"""Unit tests for L{openswmm_gymnasium.spaces.design} factories.

Engine-touching behaviour (the actual setter calls) is exercised by
the CIP env integration tests; here we focus on space construction,
validation, and bind-before-apply enforcement.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from openswmm_gymnasium.spaces.design import (
    DesignActionFactory,
    LinkDiameter,
    LinkLength,
    LinkRoughness,
    NodeMaxDepth,
)

# -----------------------------------------------------------------------------
# Parametrized construction tests
# -----------------------------------------------------------------------------


_BOX_FACTORIES = [
    (LinkRoughness, "link_roughness", "link_ids", ["C1"]),
    (LinkLength, "link_length", "link_ids", ["C1"]),
    (LinkDiameter, "link_diameter", "link_ids", ["C1"]),
    (NodeMaxDepth, "node_max_depth", "node_ids", ["J1"]),
]


class TestConstruction:
    @pytest.mark.parametrize("cls, name, kwarg, ids", _BOX_FACTORIES)
    def test_default_name(self, cls, name, kwarg, ids):
        f = (
            cls(ids, low=0.01, high=0.05, **{kwarg: ids})
            if False
            else cls(ids, low=0.01, high=0.05)
        )
        assert f.name == name

    @pytest.mark.parametrize("cls, name, kwarg, ids", _BOX_FACTORIES)
    def test_empty_ids_raises(self, cls, name, kwarg, ids):
        with pytest.raises(ValueError, match="at least one"):
            cls([], low=0.0, high=1.0)

    @pytest.mark.parametrize("cls, name, kwarg, ids", _BOX_FACTORIES)
    def test_invalid_bounds_raise(self, cls, name, kwarg, ids):
        with pytest.raises(ValueError, match="strictly greater"):
            cls(ids, low=1.0, high=1.0)
        with pytest.raises(ValueError, match="strictly greater"):
            cls(ids, low=2.0, high=1.0)


class TestSpace:
    @pytest.mark.parametrize("cls, name, kwarg, ids", _BOX_FACTORIES)
    def test_box_shape(self, cls, name, kwarg, ids):
        f = cls(ids * 3, low=0.0, high=1.0)
        s = f.space
        assert isinstance(s, spaces.Box)
        assert s.shape == (3,)
        assert s.dtype == np.float32

    @pytest.mark.parametrize("cls, name, kwarg, ids", _BOX_FACTORIES)
    def test_bounds_propagate(self, cls, name, kwarg, ids):
        f = cls(ids, low=0.25, high=0.75)
        s = f.space
        assert np.allclose(s.low, [0.25])
        assert np.allclose(s.high, [0.75])

    @pytest.mark.parametrize("cls, name, kwarg, ids", _BOX_FACTORIES)
    def test_sample_in_bounds(self, cls, name, kwarg, ids):
        f = cls(ids, low=0.1, high=0.9)
        for _ in range(20):
            v = f.space.sample()
            assert v.shape == (1,)
            assert (v >= 0.1).all() and (v <= 0.9).all()


class TestApplyRequiresBind:
    @pytest.mark.parametrize("cls, name, kwarg, ids", _BOX_FACTORIES)
    def test_apply_before_bind_raises(self, cls, name, kwarg, ids):
        f = cls(ids, low=0.0, high=1.0)

        class _Stub:
            pass

        with pytest.raises(RuntimeError, match="bind"):
            f.apply(_Stub(), np.array([0.5], dtype=np.float32))


class TestProtocolConformance:
    @pytest.mark.parametrize("cls, name, kwarg, ids", _BOX_FACTORIES)
    def test_satisfies_factory_protocol(self, cls, name, kwarg, ids):
        f = cls(ids, low=0.0, high=1.0)
        assert isinstance(f, DesignActionFactory)
