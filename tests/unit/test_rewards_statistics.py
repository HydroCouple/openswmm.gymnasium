"""G5.1 — reward terms that read the engine's cumulative statistics.

L{FloodingVolume} and L{PeakOutflow} no longer re-integrate a sampled rate
in Python; they difference the engine's cumulative
C{swmm_node_get_stat_vol_flooded} / C{swmm_link_get_stat_max_flow}
accumulators, which advance every routing step rather than only at env-step
boundaries. The arithmetic that turns a cumulative statistic into a
per-step contribution is what these tests pin.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import unittest

from openswmm_gymnasium.rewards import CSOVolume, FloodingVolume, PeakOutflow


class _FakeStatistics:
    """Scriptable stand-in for L{_StatisticsCompat} (cumulative values)."""

    def __init__(self) -> None:
        self.flooded: dict[int, float] = {0: 0.0, 1: 0.0}
        self.max_flow: dict[int, float] = {0: 0.0, 1: 0.0}

    def node_vol_flooded(self, idx: int) -> float:
        return self.flooded[idx]

    def link_max_flow(self, idx: int) -> float:
        return self.max_flow[idx]


class _FakeNodes:
    _IDS = {"J1": 0, "J2": 1}

    def count(self) -> int:
        return len(self._IDS)

    def get_index(self, node_id: str) -> int:
        return self._IDS[node_id]


class _FakeLinks:
    _IDS = {"C1": 0, "C2": 1}

    def get_index(self, link_id: str) -> int:
        return self._IDS[link_id]


class _FakeAdapter:
    def __init__(self) -> None:
        self.nodes = _FakeNodes()
        self.links = _FakeLinks()
        self.statistics = _FakeStatistics()


class TestFloodingVolume(unittest.TestCase):
    def test_first_step_returns_the_whole_reading(self):
        a = _FakeAdapter()
        term = FloodingVolume(["J1"])
        term.bind(a)
        term.reset()
        a.statistics.flooded[0] = 25.0
        self.assertAlmostEqual(term.step(a, 300.0), 25.0)

    def test_later_steps_return_the_increment_only(self):
        a = _FakeAdapter()
        term = FloodingVolume(["J1"])
        term.bind(a)
        term.reset()
        a.statistics.flooded[0] = 25.0
        term.step(a, 300.0)
        a.statistics.flooded[0] = 40.0
        self.assertAlmostEqual(term.step(a, 300.0), 15.0)

    def test_a_dry_step_contributes_zero(self):
        a = _FakeAdapter()
        term = FloodingVolume(["J1"])
        term.bind(a)
        term.reset()
        a.statistics.flooded[0] = 25.0
        term.step(a, 300.0)
        self.assertEqual(term.step(a, 300.0), 0.0)

    def test_cumulative_reward_equals_the_final_statistic(self):
        a = _FakeAdapter()
        term = FloodingVolume(["J1"])
        term.bind(a)
        term.reset()
        total = 0.0
        for reading in (5.0, 12.0, 12.0, 30.0):
            a.statistics.flooded[0] = reading
            total += term.step(a, 60.0)
        self.assertAlmostEqual(total, 30.0)

    def test_nodes_are_summed(self):
        a = _FakeAdapter()
        term = FloodingVolume(["J1", "J2"])
        term.bind(a)
        term.reset()
        a.statistics.flooded[0] = 5.0
        a.statistics.flooded[1] = 7.0
        self.assertAlmostEqual(term.step(a, 60.0), 12.0)

    def test_none_node_ids_covers_every_node(self):
        a = _FakeAdapter()
        term = FloodingVolume()
        term.bind(a)
        term.reset()
        a.statistics.flooded[1] = 9.0
        self.assertAlmostEqual(term.step(a, 60.0), 9.0)

    def test_reset_drops_the_previous_episode_baseline(self):
        a = _FakeAdapter()
        term = FloodingVolume(["J1"])
        term.bind(a)
        term.reset()
        a.statistics.flooded[0] = 25.0
        term.step(a, 60.0)
        # New episode: the engine's accumulator restarts from zero.
        term.reset()
        a.statistics.flooded[0] = 4.0
        self.assertAlmostEqual(term.step(a, 60.0), 4.0)

    def test_contribution_is_independent_of_dt(self):
        """The engine already integrated; dt must not scale the result again."""
        a = _FakeAdapter()
        term = FloodingVolume(["J1"])
        term.bind(a)
        term.reset()
        a.statistics.flooded[0] = 10.0
        self.assertAlmostEqual(term.step(a, 1.0), 10.0)

    def test_cso_volume_inherits_the_statistic_path(self):
        a = _FakeAdapter()
        term = CSOVolume(["J2"])
        term.bind(a)
        term.reset()
        a.statistics.flooded[1] = 3.5
        self.assertAlmostEqual(term.step(a, 60.0), 3.5)


class TestPeakOutflow(unittest.TestCase):
    def test_new_peak_contributes_the_delta(self):
        a = _FakeAdapter()
        term = PeakOutflow(["C1"])
        term.bind(a)
        term.reset()
        a.statistics.max_flow[0] = 4.0
        self.assertAlmostEqual(term.step(a, 60.0), 4.0)
        a.statistics.max_flow[0] = 6.5
        self.assertAlmostEqual(term.step(a, 60.0), 2.5)

    def test_no_new_peak_contributes_zero(self):
        a = _FakeAdapter()
        term = PeakOutflow(["C1"])
        term.bind(a)
        term.reset()
        a.statistics.max_flow[0] = 4.0
        term.step(a, 60.0)
        self.assertEqual(term.step(a, 60.0), 0.0)

    def test_cumulative_equals_the_peak_summed_over_links(self):
        a = _FakeAdapter()
        term = PeakOutflow(["C1", "C2"])
        term.bind(a)
        term.reset()
        total = 0.0
        for f0, f1 in ((1.0, 0.5), (3.0, 0.5), (3.0, 2.0)):
            a.statistics.max_flow[0] = f0
            a.statistics.max_flow[1] = f1
            total += term.step(a, 60.0)
        self.assertAlmostEqual(total, 5.0)


if __name__ == "__main__":
    unittest.main()
