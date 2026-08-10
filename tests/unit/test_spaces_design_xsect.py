"""G2 — shape-aware cross-section sizing and exact filling ratios.

L{LinkDiameter} used to rewrite C{geom1} alone, which resized a box
culvert by its "diameter" and left the width untouched; L{MarketMetricReader}
used C{geom1} as the full depth, which is only the rise for shapes whose
first geometry parameter happens to be the height. Both now go through the
engine's analytic cross-section geometry.

Decode behaviour is exercised against a fake adapter that records the
engine setter calls, so these run without the compiled engine. End-to-end
engine behaviour is covered by the CIP env integration tests.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from openswmm_gymnasium.config.market_config import Agent
from openswmm_gymnasium.control.metrics import MarketMetricReader
from openswmm_gymnasium.spaces.design import LinkDiameter

# Fixtures: (shape_name, shape_code, (g1, g2, g3, g4), full_depth).
# ``full_depth`` is what the engine's XSectionGeometry reports — note it
# differs from geom1 for FILLED_CIRCULAR (rise = diameter - filled depth).
_SECTIONS = {
    "C_CIRC": ("CIRCULAR", 0, (2.0, 0.0, 0.0, 0.0), 2.0),
    "C_BOX": ("RECT_CLOSED", 2, (4.0, 8.0, 0.0, 0.0), 4.0),
    "C_TRAP": ("TRAPEZOIDAL", 4, (5.0, 10.0, 2.0, 3.0), 5.0),
    "C_FILLED": ("FILLED_CIRCULAR", 1, (4.0, 1.0, 0.0, 0.0), 3.0),
    "C_IRREG": ("IRREGULAR", 21, (3.0, 0.0, 0.0, 0.0), 6.0),
    "C_DEGEN": ("DUMMY", 25, (0.0, 0.0, 0.0, 0.0), 0.0),
}


class _FakeLinks:
    """Minimal stand-in with the real ``_LinksCompat`` attribute layout."""

    def __init__(self) -> None:
        self._order = list(_SECTIONS)
        self.applied: list[tuple] = []
        self.depth: dict[int, float] = {}

    def get_index(self, link_id: str) -> int:
        return self._order.index(link_id)

    def _section(self, idx: int):
        return _SECTIONS[self._order[idx]]

    def get_xsect(self, idx: int) -> tuple:
        _name, code, geoms, _rise = self._section(idx)
        return (code, *geoms)

    def get_xsect_shape_name(self, idx: int) -> str:
        return self._section(idx)[0]

    def get_full_depth(self, idx: int) -> float:
        return self._section(idx)[3]

    def set_xsect(self, idx, shape, g1, g2, g3, g4) -> None:
        self.applied.append(
            (idx, shape, round(g1, 6), round(g2, 6), round(g3, 6), round(g4, 6))
        )

    def get_depth(self, idx: int) -> float:
        return self.depth[idx]


class _FakeNodes:
    """Present but unused — L{MarketMetricReader.read} looks it up eagerly."""

    def get_depth(self, idx: int) -> float:  # pragma: no cover - no node agents
        raise AssertionError("no node agents in these fixtures")


class _FakeAdapter:
    def __init__(self) -> None:
        self.links = _FakeLinks()
        self.nodes = _FakeNodes()


class TestCircularUnchanged(unittest.TestCase):
    """On a CIRCULAR pipe rise == diameter == geom1, so nothing changes."""

    def test_circular_sets_the_diameter_directly(self):
        a = _FakeAdapter()
        f = LinkDiameter(["C_CIRC"], low=1.0, high=6.0)
        f.bind(a)
        f.apply(a, np.array([3.0], dtype=np.float32))
        self.assertEqual(a.links.applied, [(0, 0, 3.0, 0.0, 0.0, 0.0)])


class TestShapeAwareScaling(unittest.TestCase):
    def test_box_culvert_scales_width_with_height(self):
        """The regression this phase exists for: geom2 must move too."""
        a = _FakeAdapter()
        f = LinkDiameter(["C_BOX"], low=1.0, high=8.0)
        f.bind(a)
        f.apply(a, np.array([2.0], dtype=np.float32))
        # rise 4 -> 2 halves the section: 8 ft wide becomes 4 ft wide.
        self.assertEqual(a.links.applied, [(1, 2, 2.0, 4.0, 0.0, 0.0)])

    def test_trapezoid_scales_bottom_width_but_not_side_slopes(self):
        a = _FakeAdapter()
        f = LinkDiameter(["C_TRAP"], low=1.0, high=20.0)
        f.bind(a)
        f.apply(a, np.array([10.0], dtype=np.float32))
        # Side slopes are dimensionless (run/rise) and must survive verbatim.
        self.assertEqual(a.links.applied, [(2, 4, 10.0, 20.0, 2.0, 3.0)])

    def test_value_is_the_rise_not_geom1(self):
        """FILLED_CIRCULAR: rise = diameter - filled depth, so geom1 != rise."""
        a = _FakeAdapter()
        f = LinkDiameter(["C_FILLED"], low=1.0, high=10.0)
        f.bind(a)
        f.apply(a, np.array([6.0], dtype=np.float32))
        # Baseline rise 3.0 -> factor 2.0: diameter 4->8, filled depth 1->2,
        # so the new rise is 8 - 2 == 6, the value asked for.
        applied = a.links.applied[0]
        self.assertEqual(applied, (3, 1, 8.0, 2.0, 0.0, 0.0))
        self.assertAlmostEqual(applied[2] - applied[3], 6.0)

    def test_multiple_links_of_different_shapes(self):
        a = _FakeAdapter()
        f = LinkDiameter(["C_CIRC", "C_BOX"], low=1.0, high=8.0)
        f.bind(a)
        f.apply(a, np.array([4.0, 8.0], dtype=np.float32))
        self.assertEqual(
            a.links.applied,
            [(0, 0, 4.0, 0.0, 0.0, 0.0), (1, 2, 8.0, 16.0, 0.0, 0.0)],
        )

    def test_values_are_clipped_to_the_action_range(self):
        a = _FakeAdapter()
        f = LinkDiameter(["C_BOX"], low=2.0, high=6.0)
        f.bind(a)
        f.apply(a, np.array([99.0], dtype=np.float32))
        self.assertEqual(a.links.applied, [(1, 2, 6.0, 12.0, 0.0, 0.0)])


class TestUnsizableShapes(unittest.TestCase):
    def test_irregular_section_is_rejected_at_bind(self):
        f = LinkDiameter(["C_IRREG"], low=1.0, high=6.0)
        with self.assertRaisesRegex(ValueError, "IRREGULAR"):
            f.bind(_FakeAdapter())

    def test_zero_rise_is_rejected_at_bind(self):
        f = LinkDiameter(["C_DEGEN"], low=1.0, high=6.0)
        # DUMMY has no scalable dimension, so it trips the shape check first.
        with self.assertRaisesRegex(ValueError, "DUMMY"):
            f.bind(_FakeAdapter())

    def test_apply_before_bind_raises(self):
        f = LinkDiameter(["C_CIRC"], low=1.0, high=6.0)
        with self.assertRaisesRegex(RuntimeError, "bind"):
            f.apply(_FakeAdapter(), np.array([2.0], dtype=np.float32))


class TestFillingRatioIsExact(unittest.TestCase):
    """C{filling_ratio} must normalise by the true rise, not C{geom1}."""

    def _reader(self, link_id: str) -> MarketMetricReader:
        # MarketMetricReader only reads ``config.agents``.
        agent = Agent(
            id=link_id,
            element_type="link",
            commodity="conveyance",
            stress_metric="filling_ratio",
            curve="c1",
            role="seller",
        )
        return MarketMetricReader(SimpleNamespace(agents=[agent]))

    def test_filled_circular_uses_the_rise(self):
        a = _FakeAdapter()
        reader = self._reader("C_FILLED")
        reader.bind(a)
        # Rise is 3.0 (geom1 4.0 would report 0.375 instead).
        a.links.depth[a.links.get_index("C_FILLED")] = 1.5
        self.assertAlmostEqual(reader.read(a)["C_FILLED"], 0.5)

    def test_stress_is_clamped_to_unit_interval(self):
        a = _FakeAdapter()
        reader = self._reader("C_BOX")
        reader.bind(a)
        idx = a.links.get_index("C_BOX")
        a.links.depth[idx] = 99.0
        self.assertEqual(reader.read(a)["C_BOX"], 1.0)
        a.links.depth[idx] = -1.0
        self.assertEqual(reader.read(a)["C_BOX"], 0.0)


if __name__ == "__main__":
    unittest.main()
