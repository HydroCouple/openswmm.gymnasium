# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 Caleb Buahin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""G4 — pollutant-concentration observations and the C{TSSLoad} reward.

Both were deferred in-tree (``builder.py`` "the pollutant phase",
``terms.py`` "land in subsequent phases") pending the engine's
C{Pollutants} / C{Quality} bindings, which now exist.

Collector wiring, bulk/scalar parity and the load arithmetic run against a
fake adapter so they need no compiled engine; end-to-end quality routing is
an engine-touching concern covered by the env integration tests.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.rewards import RewardRegistry, TSSLoad

_NODE_IDS = ["J1", "J2", "J3"]
_LINK_IDS = ["C1", "C2"]
# Whole-network concentration arrays, per pollutant index.
_NODE_CONC = {0: np.array([10.0, 20.0, 30.0]), 1: np.array([1.0, 2.0, 3.0])}
_LINK_CONC = {0: np.array([40.0, 50.0]), 1: np.array([4.0, 5.0])}
_LINK_FLOW = {0: 2.0, 1: -3.0}


_POLLUTANT_IDS = {"TSS": 0, "BOD": 1}


class _FakeElement:
    """One C{solver.nodes[idx]} / C{solver.links[idx]} with a C{quality} reader."""

    def __init__(self, conc: dict[int, np.ndarray], idx: int) -> None:
        self._conc = conc
        self._idx = idx

    def quality(self, pollutant: int) -> float:
        return float(self._conc[pollutant][self._idx])


class _FakeCollection:
    def __init__(self, conc: dict[int, np.ndarray]) -> None:
        self._conc = conc

    def __getitem__(self, idx: int) -> _FakeElement:
        return _FakeElement(self._conc, idx)


class _FakeAdapter:
    """The adapter surface the collectors and C{TSSLoad} use.

    C{index} / C{read} / C{call} by catalog path, plus C{solver.nodes} /
    C{solver.links} for the per-element C{quality} reader.
    """

    _IDS = {"node": _NODE_IDS, "link": _LINK_IDS}
    _BULK = {"nodes.qualities": _NODE_CONC, "links.qualities": _LINK_CONC}

    def __init__(self) -> None:
        self.bulk_reads: dict[str, int] = {}
        self.solver = SimpleNamespace(
            nodes=_FakeCollection(_NODE_CONC), links=_FakeCollection(_LINK_CONC)
        )

    def index(self, kind: str, element_id: str) -> int:
        if kind == "pollutant":
            try:
                return _POLLUTANT_IDS[element_id]
            except KeyError as exc:
                raise KeyError(f"undefined pollutant {element_id!r}") from exc
        return self._IDS[kind].index(element_id)

    def read(self, path: str, idx: int) -> float:
        assert path == "link.flow", path
        return _LINK_FLOW[idx]

    def call(self, path: str, pollutant: int):
        self.bulk_reads[path] = self.bulk_reads.get(path, 0) + 1
        return self._BULK[path][pollutant]


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
        self.assertEqual(a.bulk_reads, {"nodes.qualities": 1})

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
