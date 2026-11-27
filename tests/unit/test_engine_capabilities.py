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

C{pyproject.toml} pins C{openswmm>=6.0.0a4.dev1} with no upper bound and the
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

from openswmm_gymnasium._engine import requirements
from openswmm_gymnasium._engine.requirements import (
    CORE_REQUIREMENTS,
    EngineCapabilityError,
    _missing_engine_capabilities,
    optional_module,
    require_engine_capabilities,
    require_for,
)
from openswmm_gymnasium._engine.solver_adapter import SolverAdapter
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


class TestOptionalModuleProbes(unittest.TestCase):
    """Each optional module names its own missing build, and the remedy.

    Drives L{optional_module} against a stand-in engine namespace and
    solver so no engine build is needed. The B{model} tier against a real
    open model (C{[OPTIONS] HEAT_TRANSPORT} / C{WATER_AGE} off) lives in
    C{test_process_surfaces.py}.
    """

    def _adapter_over(self, **modules) -> SolverAdapter:
        """Build an adapter whose solver carries only C{modules}.

        Bypasses C{__init__} so no real solver is constructed: the guard
        reads only the engine namespace and C{adapter.solver.<name>}.
        """
        adapter = SolverAdapter.__new__(SolverAdapter)
        adapter._solver = SimpleNamespace(**modules)
        return adapter

    def _patch_engine(self, module) -> None:
        original = requirements._engine
        requirements._engine = module
        self.addCleanup(setattr, requirements, "_engine", original)

    def test_heat_guard_names_the_missing_module(self):
        self._patch_engine(_complete_engine())
        with self.assertRaises(RuntimeError) as ctx:
            optional_module(self._adapter_over(), "heat")
        message = str(ctx.exception)
        self.assertIn("Heat", message)
        self.assertIn("rebuild", message.lower())

    def test_water_age_guard_names_the_missing_module(self):
        self._patch_engine(_complete_engine())
        with self.assertRaises(RuntimeError) as ctx:
            optional_module(self._adapter_over(), "water_age")
        message = str(ctx.exception)
        self.assertIn("WaterAge", message)
        self.assertIn("rebuild", message.lower())

    def test_reactions_guard_names_the_missing_module(self):
        self._patch_engine(_complete_engine())
        with self.assertRaises(RuntimeError) as ctx:
            optional_module(self._adapter_over(), "reactions")
        message = str(ctx.exception)
        self.assertIn("Reactions", message)
        self.assertIn("rebuild", message.lower())

    def test_surface2d_guard_names_the_missing_build(self):
        """A build without 2D raises ImportError from C{solver.surface2d}."""

        class _No2DSolver:
            @property
            def surface2d(self):
                raise ImportError("built without 2D")

        adapter = SolverAdapter.__new__(SolverAdapter)
        adapter._solver = _No2DSolver()
        with self.assertRaises(RuntimeError) as ctx:
            optional_module(adapter, "surface2d")
        self.assertIn("OPENSWMM_BUILD_2D=OFF", str(ctx.exception))

    def test_heat_guard_refuses_a_model_with_heat_transport_off(self):
        """Build tier passes, model tier does not: the module exists but
        the open model never routes heat, so writes would be silent."""
        module = _complete_engine()
        module.Heat = object
        self._patch_engine(module)
        with self.assertRaises(RuntimeError) as ctx:
            optional_module(self._adapter_over(heat=SimpleNamespace(enabled=False)), "heat")
        self.assertIn("HEAT_TRANSPORT", str(ctx.exception))

    def test_water_age_guard_refuses_a_model_with_water_age_off(self):
        module = _complete_engine()
        module.WaterAge = object
        self._patch_engine(module)
        adapter = self._adapter_over(water_age=SimpleNamespace(enabled=False))
        with self.assertRaises(RuntimeError) as ctx:
            optional_module(adapter, "water_age")
        self.assertIn("WATER_AGE", str(ctx.exception))

    def test_surface2d_guard_refuses_an_inactive_surface(self):
        adapter = self._adapter_over(surface2d=SimpleNamespace(is_active=False))
        with self.assertRaises(RuntimeError) as ctx:
            optional_module(adapter, "surface2d")
        self.assertIn("no active 2D surface", str(ctx.exception))

    def test_heat_guard_returns_the_solvers_own_module_once_both_tiers_pass(self):
        """No wrapper is built per call: every call hands back C{solver.heat}."""
        module = _complete_engine()
        module.Heat = object
        self._patch_engine(module)
        heat = SimpleNamespace(enabled=True)
        adapter = self._adapter_over(heat=heat)
        first = optional_module(adapter, "heat")
        self.assertIs(first, heat)
        self.assertIs(optional_module(adapter, "heat"), first)

    def test_reactions_has_no_model_tier_gate(self):
        """There is no [OPTIONS] flag for reactions, so an empty
        coefficient table must not be treated as a misconfiguration."""
        module = _complete_engine()
        module.Reactions = object
        self._patch_engine(module)
        adapter = self._adapter_over(reactions=SimpleNamespace(coefficients=[]))
        self.assertEqual(list(optional_module(adapter, "reactions").coefficients), [])


if __name__ == "__main__":
    unittest.main()
