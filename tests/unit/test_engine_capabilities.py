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

"""G5.2 — the engine capability probe.

C{pyproject.toml} pins C{openswmm>=6.0.0.dev2} with no upper bound and the
C API is still moving, so the adapter probes for the specific symbols it
calls instead of asserting a version. The probe is a pure function over a
namespace object, which is what these tests drive — no engine required.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from openswmm_gymnasium._engine import solver_adapter
from openswmm_gymnasium._engine.solver_adapter import (
    CORE_REQUIREMENTS,
    EngineCapabilityError,
    SolverAdapter,
    _missing_engine_capabilities,
    require_engine_capabilities,
    require_for,
)
from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.rewards import FloodingVolume, TSSLoad
from openswmm_gymnasium.spaces import LinkRoughness, StorageVolume


def _complete_engine() -> SimpleNamespace:
    """A stand-in engine namespace with every core requirement."""
    solver = type("Solver", (), {name: None for name in CORE_REQUIREMENTS})
    return SimpleNamespace(__version__="6.0.0.test", Solver=solver)


def _real_engine_without(*class_names: str) -> SimpleNamespace:
    """Every class of the installed engine, minus C{class_names} (a partial build)."""
    import importlib
    import pkgutil

    import openswmm.engine as engine

    names: dict[str, object] = {"__version__": "6.0.0.partial"}
    for info in pkgutil.iter_modules(engine.__path__):
        if info.name.startswith("_"):
            module = importlib.import_module(f"openswmm.engine.{info.name}")
            names.update({k: v for k, v in vars(module).items() if isinstance(v, type)})
    for name in class_names:
        names.pop(name, None)
    return SimpleNamespace(**names)


class TestCoreProbe(unittest.TestCase):
    def test_complete_engine_reports_nothing_missing(self):
        self.assertEqual(_missing_engine_capabilities(_complete_engine()), [])
        require_engine_capabilities(_complete_engine())

    def test_missing_core_attribute_is_named(self):
        module = _complete_engine()
        delattr(module.Solver, "set_lenient_open")
        self.assertEqual(
            _missing_engine_capabilities(module),
            ["set_lenient_open (openswmm.engine.Solver.set_lenient_open)"],
        )

    def test_error_names_every_missing_symbol_and_the_remedy(self):
        with self.assertRaises(EngineCapabilityError) as ctx:
            require_engine_capabilities(_complete_engine(), ("tables", "statistics"))
        message = str(ctx.exception)
        self.assertIn("tables (openswmm.engine.Tables)", message)
        self.assertIn("statistics (openswmm.engine.Statistics)", message)
        self.assertIn("6.0.0.test", message)
        self.assertIn("Upgrade", message)

    def test_optional_surfaces_are_not_core(self):
        """A build without 2D, heat, reactions ... still passes the core probe."""
        for path in (
            "surface2d",
            "heat",
            "water_age",
            "reactions",
            "tables",
            "statistics",
            "pollutants",
            "xsect",
        ):
            with self.subTest(path=path):
                self.assertNotIn(path, CORE_REQUIREMENTS)
        require_engine_capabilities(_complete_engine())


class TestRequireFor(unittest.TestCase):
    """G2: an env checks exactly what its configured components declare."""

    def test_components_declare_the_surfaces_they_use(self):
        self.assertEqual(FloodingVolume.requires, ("statistics",))
        self.assertIn("pollutants", TSSLoad.requires)
        self.assertIn("tables", StorageVolume.requires)
        builder = (
            ObservationBuilder()
            .add_field("link.stats.max_flow", ["C1"])
            .add_pollutant_concentration(["J1"], "TSS")
        )
        self.assertIn("link.stats.max_flow", builder.requires())
        self.assertIn("pollutants", builder.requires())

    def test_a_partial_build_fails_only_the_components_that_need_it(self):
        engine = _real_engine_without("Tables")
        hydraulic = (
            ObservationBuilder().add_node_depths(["J1"]),
            FloodingVolume(),
            LinkRoughness(["C1"], 0.01, 0.02),
        )
        require_for(*hydraulic, module=engine)
        storage = StorageVolume(["SU1"], 0.5, 2.0)
        with self.assertRaises(EngineCapabilityError) as ctx:
            require_for(*hydraulic, storage, module=engine)
        self.assertIn("tables (openswmm.engine.Tables)", str(ctx.exception))

    def test_every_declared_path_is_in_the_catalog(self):
        import inspect

        from openswmm.engine import catalog

        from openswmm_gymnasium import rewards, spaces
        from openswmm_gymnasium.control import metrics

        declared = {
            (cls.__name__, path)
            for module in (rewards, spaces, metrics)
            for _, cls in inspect.getmembers(module, inspect.isclass)
            for path in getattr(cls, "requires", ())
            if isinstance(getattr(cls, "requires", ()), tuple)
        }
        self.assertTrue(declared)
        for owner, path in sorted(declared):
            with self.subTest(owner=owner, path=path):
                if path not in catalog.targets():
                    catalog.lookup(path)  # raises KeyError naming the path

    def test_components_without_requirements_pass(self):
        require_for(object(), module=_complete_engine())


class TestOptionalAccessorProbes(unittest.TestCase):
    """Each optional accessor names its own missing module, and the remedy.

    Drives the guards against a stand-in solver so no engine build is
    needed. Only the B{build} tier is exercised here; the B{model} tier
    (C{[OPTIONS] HEAT_TRANSPORT} / C{WATER_AGE} off) needs a real open
    model and lives in C{test_process_surfaces.py}.
    """

    def _adapter_over(self, module) -> SolverAdapter:
        """Build an adapter whose engine namespace is *module*.

        Bypasses C{__init__} so no real solver is constructed: the build-tier
        probes read only C{_engine} and the cached-accessor slots.
        """
        adapter = SolverAdapter.__new__(SolverAdapter)
        adapter._solver = SimpleNamespace()
        adapter._heat = None
        adapter._water_age = None
        adapter._reactions = None
        return adapter

    def _patch_engine(self, module) -> None:
        original = solver_adapter._engine
        solver_adapter._engine = module
        self.addCleanup(setattr, solver_adapter, "_engine", original)

    def test_heat_accessor_names_the_missing_module(self):
        self._patch_engine(_complete_engine())
        with self.assertRaises(RuntimeError) as ctx:
            _ = self._adapter_over(None).heat
        message = str(ctx.exception)
        self.assertIn("Heat", message)
        self.assertIn("rebuild", message.lower())

    def test_water_age_accessor_names_the_missing_module(self):
        self._patch_engine(_complete_engine())
        with self.assertRaises(RuntimeError) as ctx:
            _ = self._adapter_over(None).water_age
        message = str(ctx.exception)
        self.assertIn("WaterAge", message)
        self.assertIn("rebuild", message.lower())

    def test_reactions_accessor_names_the_missing_module(self):
        self._patch_engine(_complete_engine())
        with self.assertRaises(RuntimeError) as ctx:
            _ = self._adapter_over(None).reactions
        message = str(ctx.exception)
        self.assertIn("Reactions", message)
        self.assertIn("rebuild", message.lower())

    def test_heat_accessor_refuses_a_model_with_heat_transport_off(self):
        """Build tier passes, model tier does not: the module exists but
        the open model never routes heat, so writes would be silent."""
        module = _complete_engine()
        module.Heat = lambda solver: SimpleNamespace(enabled=False)
        self._patch_engine(module)
        with self.assertRaises(RuntimeError) as ctx:
            _ = self._adapter_over(None).heat
        self.assertIn("HEAT_TRANSPORT", str(ctx.exception))

    def test_water_age_accessor_refuses_a_model_with_water_age_off(self):
        module = _complete_engine()
        module.WaterAge = lambda solver: SimpleNamespace(enabled=False)
        self._patch_engine(module)
        with self.assertRaises(RuntimeError) as ctx:
            _ = self._adapter_over(None).water_age
        self.assertIn("WATER_AGE", str(ctx.exception))

    def test_heat_accessor_is_cached_once_both_tiers_pass(self):
        module = _complete_engine()
        constructions = []

        def _heat(solver):
            constructions.append(solver)
            return SimpleNamespace(enabled=True)

        module.Heat = _heat
        self._patch_engine(module)
        adapter = self._adapter_over(None)
        first = adapter.heat
        self.assertIs(adapter.heat, first)
        self.assertEqual(len(constructions), 1)

    def test_reactions_has_no_model_tier_gate(self):
        """There is no [OPTIONS] flag for reactions, so an empty
        coefficient table must not be treated as a misconfiguration."""
        module = _complete_engine()
        module.Reactions = lambda solver: SimpleNamespace(species=[], coefficients=[], terms=[])
        self._patch_engine(module)
        self.assertEqual(self._adapter_over(None).reactions.coefficient_names(), [])


if __name__ == "__main__":
    unittest.main()
