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

"""Newer engine surfaces observed through the existing public API (plan G-4).

Per plan §8.0 -- no engine mocks. Each surface is driven on the real engine,
and no collector was added under C{src/}:

  - B{LID node layers}: per-layer moisture of a storage-node LID, a custom
    collector over C{infrastructure.lids.node_profile} (C{get_layers} is the
    authored stack, read at bind); C{add_field("node.volume")} carries the
    retained water. Fixture C{tests/data/lid_node.inp}.
  - B{2D groundwater ledgers}: C{groundwater2d.ledger} (water) and
    C{groundwater2d.species_ledger} (transported species), a custom collector.
    C{groundwater2d.transport} itself is the authoring view (sources,
    parameters, boundaries) and has no ledger reader. Fixture
    C{tests/data/groundwater_2d.inp}.
  - B{Coupling frames}: a frame (C{coupling.apply_frame}) is an input; what it
    delivered is observed through the receipts
    (C{coupling.surface.source_receipt}, C{water_totals}), a custom collector.
    Fixture C{tests/data/minimal_2d.inp}.
  - B{Flow tracing}: a standalone C{FlowTracer} rescans the output file the
    running engine is writing (C{prepare_live}) and traces from a seed node,
    giving C{trace.averages} and passage fractions once per completed
    reporting period, a custom collector. Fixture
    C{tests/data/site_drainage_example.inp}.
  - B{Heat snapshots} are B{not} observations: C{heat.snapshot()} saves and
    restores pre-run heat settings and is refused once the run is initialized,
    i.e. before any collector binds. The test shows the supported use (around
    a design factory, before initialize). Fixture: C{minimal.inp} with
    C{HEAT_TRANSPORT YES} added.

None of these is one numeric value per element (C{add_field}) or per mesh
cell (C{add_cell_field}); L{TestGenericRoutesRefuseTheseSurfaces} pins that.
Custom collectors join an L{ObservationBuilder} through
L{ObservationBuilder.add_collector}.

Every file a test writes lands in C{tests/_output/observations_surfaces/}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import shutil
import unittest
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import openswmm.engine as _engine

from openswmm_gymnasium._engine import optional_module, require_for
from openswmm_gymnasium.envs import SwmmRTCEnv
from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.spaces.design import HeatSourceTemperature

from ._base import BaseEngineTest

_DATA_DIR = (Path(__file__).resolve().parents[1] / "data").resolve()
_OUTPUT = Path(__file__).resolve().parents[1] / "_output" / "observations_surfaces"

_HAS_2D = bool(getattr(_engine, "HAS_2D", False))


# =============================================================================
# The extension route: custom collectors (ObservationBuilder.add_collector)
# =============================================================================


class _LidLayerMoisture:
    """Volumetric moisture of chosen authored layers of a storage-node LID.

    C{infrastructure.lids.node_profile(node)} lists the runtime sublayers, each
    with its one-based authored C{layer}, C{bottom}/C{top} heights above the
    storage invert and C{moisture}; this reports the thickness-weighted mean
    per requested layer. C{get_layers} (the authored stack) validates the
    layer numbers at bind.
    """

    requires = (
        "infrastructure.lids.node_assignment",
        "infrastructure.lids.get_id",
        "infrastructure.lids.get_layers",
        "infrastructure.lids.node_profile",
    )

    def __init__(self, node: str, layers: Sequence[int]) -> None:
        self._node = node
        self._layers = [int(k) for k in layers]

    @property
    def size(self) -> int:
        return len(self._layers)

    def units(self, unit_system, flow_units) -> list[str]:
        return ["fraction"] * self.size

    def bind(self, adapter) -> None:
        assignment = adapter.call("infrastructure.lids.node_assignment", self._node)
        if assignment is None:
            raise ValueError(f"node {self._node!r} has no LID control")
        control = adapter.call("infrastructure.lids.get_id", assignment[0])
        stack = adapter.call("infrastructure.lids.get_layers", control)
        n = sum(layer.kind != _engine.LidNodeLayerKind.BOTTOM for layer in stack)
        bad = [k for k in self._layers if not 1 <= k <= n]
        if bad:
            raise ValueError(f"LID control {control!r} has layers 1..{n}; got {bad}")

    def collect(self, adapter) -> np.ndarray:
        rows = adapter.call("infrastructure.lids.node_profile", self._node)
        out = np.empty(self.size, dtype=np.float32)
        for i, k in enumerate(self._layers):
            cells = [r for r in rows if r["layer"] == k]
            height = np.array([r["top"] - r["bottom"] for r in cells])
            moisture = np.array([r["moisture"] for r in cells])
            out[i] = moisture @ height / height.sum()
        return out


class _GroundwaterLedger:
    """Terms of the 2D aquifer's water ledger and of one species' ledger.

    C{groundwater2d.ledger(term)} is in m3 (storage terms instantaneous, the
    rest cumulative since start); C{groundwater2d.species_ledger(species, term)}
    is cumulative, in the species' mass units. Terms are given by
    C{GroundwaterLedger} / C{GroundwaterSpeciesLedger} member name.
    """

    requires = ("surface2d", "groundwater2d.ledger", "groundwater2d.species_ledger")

    def __init__(
        self,
        water: Sequence[str] = (),
        species: str | None = None,
        species_terms: Sequence[str] = (),
    ) -> None:
        self._water = [_engine.GroundwaterLedger[t] for t in water]
        self._species_terms = [_engine.GroundwaterSpeciesLedger[t] for t in species_terms]
        if self._species_terms and species is None:
            raise ValueError("species_terms need a species")
        self._species = species
        self._species_idx: int | None = None

    @property
    def size(self) -> int:
        return len(self._water) + len(self._species_terms)

    def units(self, unit_system, flow_units) -> list[str]:
        return ["m3"] * len(self._water) + [f"{self._species} mass"] * len(self._species_terms)

    def bind(self, adapter) -> None:
        optional_module(adapter, "surface2d")  # 2D build, active surface
        gw = adapter.solver.groundwater2d
        if not gw.active:
            raise RuntimeError("the open model configures no 2D aquifer ([2D_AQUIFER])")
        if self._species_terms:
            species = list(gw.species)
            if self._species not in species:
                raise ValueError(f"{self._species!r} is not transported; one of {species}")
            self._species_idx = species.index(self._species)

    def collect(self, adapter) -> np.ndarray:
        values = [adapter.call("groundwater2d.ledger", t) for t in self._water]
        values += [
            adapter.call("groundwater2d.species_ledger", self._species_idx, t)
            for t in self._species_terms
        ]
        return np.asarray(values, dtype=np.float32)


class _CouplingReceipts:
    """Water one runtime surface source has delivered, per cell and in total.

    C{applied_m3} from C{coupling.surface.source_receipt(source_id, cell)} for
    each cell, then the gross accepted inflow C{water_totals["external_in"]}
    (m3 since start). A cell no frame has given this source reads 0.
    """

    requires = ("surface2d", "coupling.surface.source_receipt", "coupling.surface.water_totals")

    def __init__(self, source_id: str, cells: Sequence[int]) -> None:
        self._source_id = source_id
        self._cells = [int(c) for c in cells]

    @property
    def size(self) -> int:
        return len(self._cells) + 1

    def units(self, unit_system, flow_units) -> list[str]:
        return ["m3"] * self.size  # the coupling API is SI

    def bind(self, adapter) -> None:
        n = int(optional_module(adapter, "surface2d").n_cells)
        bad = [c for c in self._cells if not 0 <= c < n]
        if bad:
            raise ValueError(f"cells out of range [0, {n}): {bad}")

    def _applied(self, adapter, cell: int) -> float:
        try:
            receipt = adapter.call("coupling.surface.source_receipt", self._source_id, cell)
        except _engine.BadIndexError:  # no frame has set this source on this cell
            return 0.0
        return receipt.applied_m3

    def collect(self, adapter) -> np.ndarray:
        applied = [self._applied(adapter, c) for c in self._cells]
        total = adapter.solver.coupling.surface.water_totals["external_in"]
        return np.asarray([*applied, total], dtype=np.float32)


class _TraceFractions:
    """Flow-tracing passage fractions at chosen nodes, from the growing output.

    C{bind} builds a C{FlowTracer} over the model topology (ids, types,
    endpoints, link lengths in m). Each C{collect} rescans the reporting
    periods the running engine has written to C{out_path} so far
    (C{prepare_live}) and traces from C{seed}: upstream, the fraction of the
    seed's flow that passed each node; downstream, the fraction of the seed's
    flow that reaches it. Averages cover all completed periods, so values move
    once per reporting step, and read 0 before the first period is written.
    """

    requires = ("trace", "trace.prepare_live", "trace.estimate")

    def __init__(self, out_path, seed: str, nodes: Sequence[str], direction=None) -> None:
        self._out = str(out_path)
        self._seed = seed
        self._nodes = list(nodes)
        self._direction = direction if direction is not None else _engine.TraceDirection.UPSTREAM
        self.tracer = None
        self._idx: list[int] = []

    @property
    def size(self) -> int:
        return len(self._nodes)

    def units(self, unit_system, flow_units) -> list[str]:
        return ["fraction"] * self.size

    def bind(self, adapter) -> None:
        self.close()  # a new episode writes a new output file
        if not Path(self._out).exists():
            raise ValueError(f"{self._out} is not this run's output file (pass the env's out_path)")
        to_m = 0.3048 if adapter.unit_system == "US" else 1.0
        nodes = [_engine.TraceNode(n.id, n.type) for n in adapter.solver.nodes]
        links = [
            _engine.TraceLink(
                lk.id, lk.from_node.index, lk.to_node.index, lk.type, lk.length * to_m
            )
            for lk in adapter.solver.links
        ]
        self.tracer = _engine.FlowTracer(nodes, links)
        self._idx = [self.tracer.node_ids.index(n) for n in self._nodes]

    def collect(self, adapter) -> np.ndarray:
        try:
            self.tracer.prepare_live(self._out)
        except _engine.TraceError as exc:
            if exc.code != _engine.TraceStatus.IO:
                raise
            return np.zeros(self.size, dtype=np.float32)  # no complete period yet
        result = self.tracer.estimate(self._seed, self._direction)
        return np.asarray([result.nodes[i].ratio for i in self._idx], dtype=np.float32)

    def close(self) -> None:
        if self.tracer is not None:
            self.tracer.close()
            self.tracer = None


# =============================================================================
# Test bases
# =============================================================================


class BaseSurfaceTest(BaseEngineTest):
    """L{BaseEngineTest} whose files land in C{tests/_output/observations_surfaces/}.

    Each test gets C{<Class>.<test>/}, emptied when the test starts and kept
    afterwards for review, in place of the temp directory, with the class's
    C{fixture} copied in as C{self.inp} (report C{self.rpt}, output
    C{self.out}). Adapter construction and teardown come from
    L{BaseEngineTest}.
    """

    fixture = "minimal.inp"

    def setUp(self) -> None:
        unittest.TestCase.setUp(self)  # not BaseTestCase: no temp directory
        self.tmp_path = _OUTPUT / f"{type(self).__name__}.{self._testMethodName}"
        shutil.rmtree(self.tmp_path, ignore_errors=True)
        self.tmp_path.mkdir(parents=True)
        self.inp = self.copy_fixture(self.fixture)
        self.rpt = self.inp.with_suffix(".rpt")
        self.out = self.inp.with_suffix(".out")
        # The names BaseEngineTest.make_adapter reads.
        self.minimal_inp, self.minimal_rpt, self.minimal_out = self.inp, self.rpt, self.out
        self._adapters: list = []
        self.addCleanup(self._close_adapters)

    def make_adapter(self, open: bool = True, *, inp: Path | None = None):
        """L{BaseEngineTest.make_adapter}, with report and output named after C{inp}.

        So adapters over two fixtures in one test never write the same files.
        """
        if inp is not None:
            self.minimal_rpt, self.minimal_out = inp.with_suffix(".rpt"), inp.with_suffix(".out")
        try:
            return super().make_adapter(open, inp=inp)
        finally:
            self.minimal_rpt, self.minimal_out = self.rpt, self.out

    def copy_fixture(self, name: str) -> Path:
        """Copy C{tests/data/<name>} into this test's output directory."""
        dest = self.tmp_path / name
        shutil.copy(_DATA_DIR / name, dest)
        return dest

    def run_steps(self, adapter, n: int) -> None:
        for _ in range(n):
            if not adapter.is_running:
                break
            adapter.step()


class TestGenericRoutesRefuseTheseSurfaces(unittest.TestCase):
    """Why each surface needs a custom collector: the generic routes refuse it."""

    def test_add_field_refuses_methods_and_non_numeric_properties(self):
        for path in (
            "trace.averages",  # a standalone handle's tuple of snapshots
            "coupling.surface.water_totals",  # a dict on a service
            "groundwater2d.ledger",  # a method
            "infrastructure.lids.node_profile",  # a method
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                ObservationBuilder().add_field(path, ["J1"])

    def test_add_cell_field_refuses_global_and_non_2d_methods(self):
        for path, args in (
            ("groundwater2d.ledger", {"term": "STORAGE"}),  # one value, not per cell
            ("coupling.surface.source_receipts", {}),  # not a 2D-service path
            ("infrastructure.lids.node_profile", {"node": "S"}),  # not a 2D-service path
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                ObservationBuilder().add_cell_field(path, [0], **args)

    def test_add_collector_names_what_a_collector_lacks(self):
        with self.assertRaisesRegex(TypeError, "requires.*collect"):
            ObservationBuilder().add_collector(object())


# =============================================================================
# LID node layers
# =============================================================================

# tests/data/lid_node.inp: Stack = SURFACE 6 in, MEDIA 12 in, AGGREGATE 6 in on a
# 100 ft2 storage node at 10 % initial saturation.
_LID_THICKNESS_FT = np.array([0.5, 1.0, 0.5])
_LID_AREA_FT2 = 100.0


@unittest.skipUnless(
    hasattr(_engine, "LidNodeLayerKind"), "installed openswmm.engine has no storage-node LIDs"
)
class TestLidNodeLayers(BaseSurfaceTest):
    fixture = "lid_node.inp"

    def _builder(self, layers=(1, 2, 3)) -> ObservationBuilder:
        return (
            ObservationBuilder()
            .add_field("node.volume", ["S"])
            .add_collector(_LidLayerMoisture("S", layers))
        )

    def test_get_layers_is_the_authored_stack(self):
        adapter = self.make_adapter()
        stack = adapter.call("infrastructure.lids.get_layers", "Stack")
        kinds = [layer.kind.name for layer in stack]
        self.assertEqual(kinds, ["SURFACE", "MEDIA", "AGGREGATE"])
        self.assertEqual(stack[1].params[0], 12.0)  # media thickness, in

    def test_initial_moisture_accounts_for_the_node_volume(self):
        adapter = self.make_adapter()
        builder = self._builder()
        builder.bind(adapter)
        obs = builder.collect(adapter)
        self.assertEqual(obs.shape, (4,))
        self.assertEqual(obs.dtype, np.float32)
        volume, moisture = obs[0], obs[1:]
        # 10 % saturation: wilting point + 10 % of the way to porosity.
        np.testing.assert_allclose(moisture, [0.0, 0.08 + 0.1 * 0.37, 0.1 * 0.4], rtol=1e-6)
        # A dry node's volume is exactly the moisture its layers retain.
        self.assertAlmostEqual(
            float(_LID_AREA_FT2 * moisture @ _LID_THICKNESS_FT), float(volume), places=4
        )

    def test_media_wets_and_aggregate_drains_during_the_episode(self):
        adapter = self.make_adapter()
        builder = self._builder((2, 3))
        builder.bind(adapter)
        rows = []
        for _ in range(6):
            self.run_steps(adapter, 100)
            rows.append(builder.collect(adapter))
        media, aggregate = np.array(rows)[:, 1], np.array(rows)[:, 2]
        self.assertTrue(np.all(np.diff(media) > 0), media)  # inflow wets the media
        self.assertTrue(np.all((media >= 0.08) & (media <= 0.45)), media)  # wp..porosity
        self.assertTrue(np.all(np.diff(aggregate) < 0), aggregate)  # the drain empties it
        self.assertTrue(np.all((aggregate >= 0.0) & (aggregate <= 0.4)), aggregate)

    def test_bind_refuses_a_node_without_lid_and_a_missing_layer(self):
        adapter = self.make_adapter()
        with self.assertRaisesRegex(ValueError, "no LID control"):
            _LidLayerMoisture("O", [1]).bind(adapter)
        with self.assertRaisesRegex(ValueError, r"layers 1\.\.3"):
            _LidLayerMoisture("S", [4]).bind(adapter)

    def test_rtc_env_observes_the_layers(self):
        builder = self._builder((2, 3))
        env = SwmmRTCEnv(
            self.inp,
            observation_builder=builder,
            control_interval_steps=60,
            rpt_path=self.rpt,
            out_path=self.out,
        )
        self.addCleanup(env.close)
        self.assertIn("infrastructure.lids.node_profile", builder.requires())
        obs, info = env.reset()
        self.assertIn(obs, env.observation_space)
        self.assertEqual(info["observation_units"][1:], ["fraction", "fraction"])
        obs2, *_ = env.step({"runtime": {}})
        self.assertGreater(obs2[1], obs[1])


# =============================================================================
# 2D groundwater ledgers
# =============================================================================

_WELL_M3_S = 1e-6  # tests/data/groundwater_2d.inp, [GW_SOURCES] W
_WELL_DYE = 3.0
_WATER_TERMS = ("INIT_STORAGE", "STORAGE", "INFIL_IN", "SOURCE_IN", "DEEP")
_DYE_TERMS = ("INIT", "STORAGE", "SOURCE_IN", "DEEP_OUT")


@unittest.skipUnless(
    _HAS_2D and hasattr(_engine, "Groundwater"), "installed openswmm.engine has no 2D groundwater"
)
class TestGroundwaterLedgers(BaseSurfaceTest):
    fixture = "groundwater_2d.inp"

    def _ledger(self) -> _GroundwaterLedger:
        return _GroundwaterLedger(_WATER_TERMS, "Dye", _DYE_TERMS)

    def test_ledgers_close_during_the_episode(self):
        adapter = self.make_adapter()
        builder = ObservationBuilder().add_collector(self._ledger())
        builder.bind(adapter)
        self.assertEqual(builder.units("SI", "CMS"), ["m3"] * 5 + ["Dye mass"] * 4)
        for _ in range(4):
            self.run_steps(adapter, 150)
            seconds = adapter.elapsed * 86400.0
            init, storage, infil, source, deep, d_init, d_storage, d_source, d_deep = (
                builder.collect(adapter).astype(np.float64)
            )
            with self.subTest(seconds=seconds):
                # Water: the only fluxes here are infiltration, the well and deep loss.
                self.assertAlmostEqual(storage - init, infil + source - deep, places=6)
                self.assertAlmostEqual(source, _WELL_M3_S * seconds, delta=1e-9)
                # Dye enters only with the well water (the ponded water carries none).
                self.assertAlmostEqual(d_source, _WELL_DYE * source, delta=1e-8)
                self.assertAlmostEqual(d_storage - d_init, d_source - d_deep, delta=2e-5)
        # What the aquifer gained by infiltration is what the 2D surface lost.
        surface = adapter.call("surface2d.get_mass_balance")
        self.assertAlmostEqual(
            surface["init_storage"] - surface["final_storage"], float(infil), places=6
        )

    def test_transport_view_is_configuration(self):
        adapter = self.make_adapter()
        (source,) = adapter.solver.groundwater2d.transport.sources
        self.assertEqual((source.name, source.flow_timeseries), ("W", "WELL"))
        self.assertEqual(adapter.call("groundwater2d.transport.source_scale", 0), 1.0)

    def test_bind_refuses_unknown_terms_species_and_models_without_aquifer(self):
        with self.assertRaises(KeyError):
            _GroundwaterLedger(["NOT_A_TERM"])
        adapter = self.make_adapter()
        with self.assertRaisesRegex(ValueError, "not transported"):
            _GroundwaterLedger(species="TSS", species_terms=["STORAGE"]).bind(adapter)
        surface_only = self.make_adapter(inp=self.copy_fixture("minimal_2d.inp"))
        with self.assertRaisesRegex(RuntimeError, "no 2D aquifer"):
            _GroundwaterLedger(["STORAGE"]).bind(surface_only)
        one_d = self.make_adapter(inp=self.copy_fixture("minimal.inp"))
        with self.assertRaisesRegex(RuntimeError, "no active 2D surface"):
            _GroundwaterLedger(["STORAGE"]).bind(one_d)


# =============================================================================
# Coupling frames
# =============================================================================

_Q = 1e-3  # m3/s per source cell


@unittest.skipUnless(
    _HAS_2D and hasattr(_engine, "CouplingFrame"), "installed openswmm.engine has no 2D coupling"
)
class TestCouplingReceipts(BaseSurfaceTest):
    fixture = "minimal_2d.inp"

    def _adapter_and_builder(self):
        adapter = self.make_adapter()
        builder = ObservationBuilder().add_collector(_CouplingReceipts("agent", [0, 1]))
        builder.bind(adapter)
        return adapter, builder

    def _frame(self, *sources, clears=()):
        return _engine.CouplingFrame(
            sources=tuple(_engine.RuntimeSource("agent", "surface", c, q) for c, q in sources),
            clears=tuple(_engine.RuntimeClear("agent", "surface", c) for c in clears),
        )

    def test_receipts_read_zero_before_any_frame(self):
        adapter, builder = self._adapter_and_builder()
        np.testing.assert_array_equal(builder.collect(adapter), [0.0, 0.0, 0.0])

    def test_receipts_track_the_volume_a_frame_delivers(self):
        adapter, builder = self._adapter_and_builder()
        adapter.call("coupling.apply_frame", self._frame((0, _Q), (1, 2 * _Q)))
        self.run_steps(adapter, 20)
        t = adapter.elapsed * 86400.0
        obs = builder.collect(adapter)
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_allclose(obs, [_Q * t, 2 * _Q * t, 3 * _Q * t], rtol=1e-6)

    def test_a_rejected_frame_changes_nothing(self):
        adapter, builder = self._adapter_and_builder()
        adapter.call("coupling.apply_frame", self._frame((0, _Q)))
        self.run_steps(adapter, 5)
        with self.assertRaises(_engine.BadIndexError):  # cell 99 does not exist
            adapter.call("coupling.apply_frame", self._frame((1, _Q), (99, _Q)))
        self.run_steps(adapter, 5)
        t = adapter.elapsed * 86400.0
        np.testing.assert_allclose(builder.collect(adapter), [_Q * t, 0.0, _Q * t], rtol=1e-6)

    def test_a_cleared_source_stops_delivering(self):
        adapter, builder = self._adapter_and_builder()
        adapter.call("coupling.apply_frame", self._frame((0, _Q)))
        self.run_steps(adapter, 10)
        adapter.call("coupling.apply_frame", self._frame(clears=[0]))
        delivered = builder.collect(adapter)
        self.run_steps(adapter, 10)
        np.testing.assert_array_equal(builder.collect(adapter), delivered)
        self.assertGreater(delivered[0], 0.0)

    def test_bind_refuses_bad_cells_and_1d_models(self):
        adapter = self.make_adapter()
        n = adapter.solver.surface2d.n_cells
        with self.assertRaisesRegex(ValueError, "out of range"):
            _CouplingReceipts("agent", [n]).bind(adapter)
        one_d = self.make_adapter(inp=self.copy_fixture("minimal.inp"))
        with self.assertRaisesRegex(RuntimeError, "no active 2D surface"):
            _CouplingReceipts("agent", [0]).bind(one_d)


# =============================================================================
# Flow tracing
# =============================================================================

# Main stem of tests/data/site_drainage_example.inp, upstream to downstream; O1
# drains J11. In this dendritic network the share of O1's flow that passed a
# node can only grow downstream.
_STEM = ["J1", "J5", "J6", "J8", "J9", "J10", "J11"]


@unittest.skipUnless(hasattr(_engine, "FlowTracer"), "installed openswmm.engine has no FlowTracer")
class TestTraceFractions(BaseSurfaceTest):
    fixture = "site_drainage_example.inp"

    def _collector(self, nodes=_STEM) -> _TraceFractions:
        collector = _TraceFractions(self.out, "O1", nodes)
        self.addCleanup(collector.close)
        return collector

    def test_fractions_read_zero_before_the_first_reporting_period(self):
        adapter = self.make_adapter()
        builder = ObservationBuilder().add_collector(self._collector())
        builder.bind(adapter)
        np.testing.assert_array_equal(builder.collect(adapter), np.zeros(len(_STEM)))

    def test_upstream_fractions_of_outfall_flow_during_the_run(self):
        adapter = self.make_adapter()
        builder = ObservationBuilder().add_collector(self._collector())
        builder.bind(adapter)
        self.run_steps(adapter, 1200)  # about an hour of the storm
        obs = builder.collect(adapter)
        self.assertEqual(obs.dtype, np.float32)
        self.assertAlmostEqual(float(obs[-1]), 1.0, places=6)  # all of O1's flow left J11
        self.assertTrue(np.all(obs > 0.0), obs)
        self.assertTrue(np.all(np.diff(obs) >= -1e-6), obs)

    def test_fractions_agree_with_the_tracer_averages(self):
        adapter = self.make_adapter()
        collector = self._collector(["J2", "J10"])
        collector.bind(adapter)
        self.run_steps(adapter, 1200)
        j2, j10 = collector.collect(adapter)
        nodes, links = collector.tracer.averages  # SI hydraulic means, topology order
        info = collector.tracer.info
        self.assertEqual((len(nodes), len(links)), (12, 11))
        # Every 1-minute reporting period completed so far (the last may still be buffered).
        minutes = int(adapter.elapsed * 1440.0)
        self.assertIn(info.periods, (minutes - 1, minutes))
        flow = {lk: links[i].net_flow_m3s for i, lk in enumerate(collector.tracer.link_ids)}
        # J2 reaches O1 only through C2 and J10 only through C10, so their shares
        # of O1's flow follow the averaged link flows (to within storage change).
        self.assertAlmostEqual(float(j2), flow["C2"] / flow["C11"], delta=0.01)
        self.assertAlmostEqual(float(j10), flow["C10"] / flow["C11"], delta=0.01)
        self.assertLess(j2 + j10, 1.0)  # S6 drains straight into J11

    def test_rtc_env_observes_live_fractions_each_episode(self):
        builder = (
            ObservationBuilder()
            .add_node_depths(["J11"])
            .add_collector(self._collector(["J10", "J11"]))
        )
        env = SwmmRTCEnv(
            self.inp,
            observation_builder=builder,
            control_interval_steps=100,
            rpt_path=self.rpt,
            out_path=self.out,
        )
        self.addCleanup(env.close)
        require_for(builder)  # the env made the same check at construction
        self.assertIn("trace.prepare_live", builder.requires())
        for _episode in range(2):  # every reset rebinds to the fresh output file
            obs, info = env.reset()
            np.testing.assert_array_equal(obs[1:], [0.0, 0.0])
            self.assertEqual(info["observation_units"][1:], ["fraction", "fraction"])
            for _ in range(6):
                obs, *_ = env.step({"runtime": {}})
            self.assertIn(obs, env.observation_space)
            self.assertTrue(0.0 < obs[1] < 1.0, obs)
            self.assertAlmostEqual(float(obs[2]), 1.0, places=6)


# =============================================================================
# Heat snapshots: pre-run save/restore, not an observation
# =============================================================================


@unittest.skipUnless(
    hasattr(_engine, "Heat") and hasattr(_engine, "HeatSnapshot"),
    "installed openswmm.engine has no Heat / HeatSnapshot",
)
class TestHeatSnapshot(BaseSurfaceTest):
    """C{heat.snapshot()} captures pre-run heat settings; C{restore()} puts them back.

    Both are refused once the model is initialized. Envs bind collectors after
    C{start()}, so no collector can use a snapshot; the window it serves is the
    one design factories write in (after open, before initialize).
    """

    def setUp(self) -> None:
        super().setUp()
        text = self.inp.read_text().replace(
            "FLOW_ROUTING         DYNWAVE",
            "FLOW_ROUTING         DYNWAVE\nHEAT_TRANSPORT       YES",
        )
        self.inp.write_text(text)
        self.assertIn("HEAT_TRANSPORT", text)

    def _opened(self):
        adapter = self.make_adapter(open=False)
        adapter.open()
        return adapter, optional_module(adapter, "heat")

    def test_snapshot_restores_what_a_design_factory_wrote(self):
        adapter, heat = self._opened()
        dwf = _engine.HeatSourceKind.DWF
        authored = heat.sources[dwf]
        factory = HeatSourceTemperature(["DWF"], 0.0, 40.0)
        factory.bind(adapter)
        with heat.snapshot() as saved:
            factory.apply(adapter, np.array([30.0], dtype=np.float32))
            self.assertEqual(heat.sources[dwf], 30.0)
            saved.restore()
            self.assertEqual(heat.sources[dwf], authored)
        adapter.initialize()  # the restored model still runs
        adapter.start()
        self.run_steps(adapter, 5)
        self.assertGreater(adapter.elapsed, 0.0)

    def test_snapshot_is_refused_once_the_run_is_initialized(self):
        adapter, heat = self._opened()
        saved = heat.snapshot()
        self.addCleanup(saved.close)
        adapter.initialize()
        with self.assertRaises(_engine.LifecycleError):
            saved.restore()
        adapter.start()  # where a collector would bind
        with self.assertRaises(_engine.LifecycleError):
            heat.snapshot()


if __name__ == "__main__":
    unittest.main()
