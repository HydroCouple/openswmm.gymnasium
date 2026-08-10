"""G4 — pollutant-concentration observations and the C{TSSLoad} reward.

Both were deferred in-tree (``builder.py`` "the pollutant phase",
``terms.py`` "land in subsequent phases") pending the engine's
C{Pollutants} / C{Quality} bindings, which now exist.

Collector wiring, bulk/scalar parity and the load arithmetic run against a
fake adapter so they need no compiled engine; end-to-end quality routing is
an engine-touching concern covered by the env integration tests.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import unittest

import numpy as np

from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.rewards import RewardRegistry, TSSLoad

_NODE_IDS = ["J1", "J2", "J3"]
_LINK_IDS = ["C1", "C2"]
# Whole-network concentration arrays, per pollutant index.
_NODE_CONC = {0: np.array([10.0, 20.0, 30.0]), 1: np.array([1.0, 2.0, 3.0])}
_LINK_CONC = {0: np.array([40.0, 50.0]), 1: np.array([4.0, 5.0])}
_LINK_FLOW = {0: 2.0, 1: -3.0}


class _FakePollutants:
    _IDS = {"TSS": 0, "BOD": 1}

    def get_index(self, pollutant_id: str) -> int:
        try:
            return self._IDS[pollutant_id]
        except KeyError as exc:
            raise KeyError(f"undefined pollutant {pollutant_id!r}") from exc


class _FakeNodes:
    def __init__(self) -> None:
        self.bulk_reads = 0

    def get_index(self, node_id: str) -> int:
        return _NODE_IDS.index(node_id)

    def get_quality(self, idx: int, pollutant: int) -> float:
        return float(_NODE_CONC[pollutant][idx])

    def qualities(self, pollutant: int):
        self.bulk_reads += 1
        return _NODE_CONC[pollutant]


class _FakeLinks:
    def get_index(self, link_id: str) -> int:
        return _LINK_IDS.index(link_id)

    def get_flow(self, idx: int) -> float:
        return _LINK_FLOW[idx]

    def get_quality(self, idx: int, pollutant: int) -> float:
        return float(_LINK_CONC[pollutant][idx])

    def qualities(self, pollutant: int):
        return _LINK_CONC[pollutant]


class _FakeAdapter:
    def __init__(self) -> None:
        self.nodes = _FakeNodes()
        self.links = _FakeLinks()
        self.pollutants = _FakePollutants()


# -----------------------------------------------------------------------------
# Observation collectors
# -----------------------------------------------------------------------------


class TestPollutantCollectors(unittest.TestCase):
    def test_node_concentrations_are_gathered_in_id_order(self):
        b = ObservationBuilder().add_pollutant_concentration(["J3", "J1"], "TSS")
        self.assertEqual(b.space().shape, (2,))
        a = _FakeAdapter()
        b.bind(a)
        np.testing.assert_allclose(b.collect(a), [30.0, 10.0])

    def test_link_concentrations_are_gathered_in_id_order(self):
        b = ObservationBuilder().add_link_pollutant_concentration(["C2"], "TSS")
        a = _FakeAdapter()
        b.bind(a)
        np.testing.assert_allclose(b.collect(a), [50.0])

    def test_one_collector_per_pollutant_composes(self):
        b = (
            ObservationBuilder()
            .add_pollutant_concentration(["J1"], "TSS")
            .add_pollutant_concentration(["J1"], "BOD")
        )
        self.assertEqual(b.space().shape, (2,))
        a = _FakeAdapter()
        b.bind(a)
        np.testing.assert_allclose(b.collect(a), [10.0, 1.0])

    def test_collect_uses_the_bulk_engine_read(self):
        b = ObservationBuilder().add_pollutant_concentration(_NODE_IDS, "TSS")
        a = _FakeAdapter()
        b.bind(a)
        b.collect(a)
        self.assertEqual(a.nodes.bulk_reads, 1)

    def test_output_is_float32(self):
        b = ObservationBuilder().add_pollutant_concentration(["J1"], "TSS")
        a = _FakeAdapter()
        b.bind(a)
        self.assertEqual(b.collect(a).dtype, np.float32)

    def test_unknown_pollutant_fails_at_bind(self):
        b = ObservationBuilder().add_pollutant_concentration(["J1"], "NOPE")
        with self.assertRaises(KeyError):
            b.bind(_FakeAdapter())

    def test_empty_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            ObservationBuilder().add_pollutant_concentration([], "TSS")

    def test_blank_pollutant_rejected(self):
        with self.assertRaisesRegex(ValueError, "pollutant"):
            ObservationBuilder().add_pollutant_concentration(["J1"], "")


# -----------------------------------------------------------------------------
# TSSLoad reward term
# -----------------------------------------------------------------------------


class TestTSSLoad(unittest.TestCase):
    def test_load_is_flow_times_concentration_times_dt(self):
        term = TSSLoad(["C1"])
        a = _FakeAdapter()
        term.bind(a)
        term.reset()
        # C1: flow 2.0, TSS 40.0, dt 60 s.
        self.assertAlmostEqual(term.step(a, 60.0), 2.0 * 40.0 * 60.0)

    def test_reversed_flow_contributes_nothing(self):
        # C2 carries -3.0 (backflow); only positive flow is a discharge.
        term = TSSLoad(["C2"])
        a = _FakeAdapter()
        term.bind(a)
        term.reset()
        self.assertEqual(term.step(a, 60.0), 0.0)

    def test_links_are_summed(self):
        term = TSSLoad(_LINK_IDS)
        a = _FakeAdapter()
        term.bind(a)
        term.reset()
        self.assertAlmostEqual(term.step(a, 1.0), 2.0 * 40.0)

    def test_works_for_any_declared_pollutant(self):
        term = TSSLoad(["C1"], pollutant="BOD")
        a = _FakeAdapter()
        term.bind(a)
        term.reset()
        self.assertAlmostEqual(term.step(a, 1.0), 2.0 * 4.0)

    def test_direction_and_default_name(self):
        term = TSSLoad(["C1"])
        self.assertEqual(term.direction, "minimize")
        self.assertEqual(term.name, "tss_load")

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "at least one link_id"):
            TSSLoad([])
        with self.assertRaisesRegex(ValueError, "pollutant"):
            TSSLoad(["C1"], pollutant="")

    def test_registered_under_tss_load(self):
        self.assertIs(RewardRegistry.get("tss_load"), TSSLoad)


if __name__ == "__main__":
    unittest.main()
