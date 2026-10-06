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

"""Process-configuration surfaces: heat, water age, reactions, slot share.

Covers the action factories and the reward term that consume the engine's
new heat-transport / water-age / multi-species-reaction modules and the
link Preissmann-slot readers.

The engine exposes B{no} per-node or per-link temperature, water-age or
species-concentration getter, so none of these surfaces has an observation
counterpart and none is tested as one. What is tested is the three things
that can go wrong silently:

  1. A factory writing to a module the B{model} never runs — configured
     and stored, never routed, no error, no reward movement. Guarded by
     C{SolverAdapter.heat} / C{.water_age} (two-tier: build then model).
  2. L{SurchargeSlotShare} reading a hard 0.0 because the model uses a
     non-finite-volume router, which is indistinguishable from a genuinely
     unpressurized network.
  3. L{ReactionCoefficientValue} writing a CONSTANT coefficient the model
     declared fixed.

Factory / term logic runs against hand-rolled fakes at module scope (never
C{unittest.mock}), so it needs no compiled engine. The two-tier guards
themselves need a real solver and live in L{TestOptionalSurfaceGuards},
which is skipped when the installed engine lacks the optional modules —
the first use of that idiom in this suite, mirroring the accessor-level
probes in C{solver_adapter.py}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest

import numpy as np
import openswmm.engine as _engine

from openswmm_gymnasium.rewards import RewardRegistry, SurchargeSlotShare
from openswmm_gymnasium.spaces.design import (
    HeatSourceTemperature,
    ReactionCoefficientValue,
    WaterAgeSourceAge,
)
from openswmm_gymnasium.spaces.runtime import HeatSourceTemperatureSetpoint
from tests.unit._base import BaseEngineTest

_LINK_IDS = ["C1", "C2"]
#: Per-link run-level slot shares the fake reports, one tuple per read.
_SOURCE_PATHWAYS = (
    "RAINFALL",
    "DWF",
    "GW",
    "RDII",
    "EXTERNAL_INFLOW",
    "IFACE",
    "INITIAL_STATE",
)


class _FakeLinks:
    """Link collection reporting scripted Preissmann-slot shares."""

    def __init__(self, shares: list[list[float]]) -> None:
        # shares[step][link_idx]; the last row repeats once exhausted.
        self._shares = shares
        self._step = -1
        self.reads = 0

    def get_index(self, link_id: str) -> int:
        return _LINK_IDS.index(link_id)

    def advance(self) -> None:
        self._step += 1

    def slot_share(self, idx: int) -> float:
        self.reads += 1
        row = self._shares[min(self._step, len(self._shares) - 1)]
        return float(row[idx])


class _FakeHeat:
    """Heat module recording every source-temperature write."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.writes: list[tuple[str, float]] = []

    def source_kinds(self) -> list[str]:
        return list(_SOURCE_PATHWAYS)

    def set_source_temp(self, source: str, temp_c: float) -> None:
        self.writes.append((source, float(temp_c)))


class _FakeWaterAge:
    """Water-age module recording every source-age write."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.writes: list[tuple[str, float]] = []

    def source_pathways(self) -> list[str]:
        return list(_SOURCE_PATHWAYS)

    def set_source_age(self, source: str, hours: float) -> None:
        self.writes.append((source, float(hours)))


class _FakeReactions:
    """Reaction module with two PARAMETERs and one CONSTANT."""

    _PARAMS = {"kd": True, "ks": True, "yield_fixed": False}

    def __init__(self) -> None:
        self.values: dict[str, float] = {"kd": 0.1, "ks": 2.0, "yield_fixed": 0.5}

    def coefficient_names(self) -> list[str]:
        return list(self._PARAMS)

    def is_coefficient_param(self, name: str) -> bool:
        return self._PARAMS[name]

    def set_coefficient(self, name: str, value: float) -> None:
        self.values[name] = float(value)


class _FakeAdapter:
    """Minimal stand-in for L{SolverAdapter}.

    C{options} drives L{get_option}; a key mapped to C{None} raises, which
    is how an engine that will not report the router is simulated.
    """

    def __init__(
        self,
        *,
        shares: list[list[float]] | None = None,
        options: dict[str, str | None] | None = None,
        heat_enabled: bool = True,
        water_age_enabled: bool = True,
    ) -> None:
        self.links = _FakeLinks(shares or [[0.0, 0.0]])
        self._options = options if options is not None else {"FLOW_ROUTING": "FV"}
        self.heat = _FakeHeat(heat_enabled)
        self.water_age = _FakeWaterAge(water_age_enabled)
        self.reactions = _FakeReactions()

    def get_option(self, name: str) -> str:
        value = self._options.get(name)
        if value is None:
            raise KeyError(name)
        return value


# =============================================================================
# SurchargeSlotShare
# =============================================================================


class TestSurchargeSlotShare(unittest.TestCase):
    def test_registered_under_its_kind_name(self):
        self.assertIs(RewardRegistry.get("surcharge_slot_share"), SurchargeSlotShare)

    def test_direction_is_minimize(self):
        self.assertEqual(SurchargeSlotShare.direction, "minimize")

    def test_empty_link_ids_rejected(self):
        with self.assertRaises(ValueError):
            SurchargeSlotShare([])

    def test_bind_accepts_finite_volume(self):
        term = SurchargeSlotShare(_LINK_IDS)
        term.bind(_FakeAdapter(options={"FLOW_ROUTING": "FV"}))

    def test_bind_refuses_dynamic_wave(self):
        term = SurchargeSlotShare(_LINK_IDS)
        with self.assertRaises(ValueError) as ctx:
            term.bind(_FakeAdapter(options={"FLOW_ROUTING": "DYNWAVE"}))
        message = str(ctx.exception)
        self.assertIn("DYNWAVE", message)
        self.assertIn("FV", message)

    def test_bind_refuses_kinematic_and_steady_too(self):
        """Not only dynamic wave: every non-FV router reads a hard 0.0."""
        for token in ("KINWAVE", "STEADY"):
            with self.subTest(routing=token):
                term = SurchargeSlotShare(_LINK_IDS)
                with self.assertRaises(ValueError):
                    term.bind(_FakeAdapter(options={"FLOW_ROUTING": token}))

    def test_bind_falls_back_to_the_alternate_option_key(self):
        term = SurchargeSlotShare(_LINK_IDS)
        with self.assertRaises(ValueError):
            term.bind(
                _FakeAdapter(options={"FLOW_ROUTING": None, "ROUTING_MODEL": "DYNWAVE"})
            )

    def test_bind_proceeds_when_the_router_cannot_be_detected(self):
        """An engine that reports neither key must not block the term."""
        term = SurchargeSlotShare(_LINK_IDS)
        term.bind(_FakeAdapter(options={"FLOW_ROUTING": None, "ROUTING_MODEL": None}))

    def test_step_returns_the_mean_share_across_links(self):
        adapter = _FakeAdapter(shares=[[0.2, 0.4]])
        term = SurchargeSlotShare(_LINK_IDS)
        term.bind(adapter)
        term.reset()
        adapter.links.advance()
        self.assertAlmostEqual(term.step(adapter, 60.0), 0.3, places=6)

    def test_contribution_is_the_running_max_increment(self):
        adapter = _FakeAdapter(shares=[[0.2, 0.2], [0.5, 0.5], [0.3, 0.3]])
        term = SurchargeSlotShare(_LINK_IDS)
        term.bind(adapter)
        term.reset()
        contributions = []
        for _ in range(3):
            adapter.links.advance()
            contributions.append(term.step(adapter, 60.0))
        # 0.2 then +0.3 to reach the 0.5 peak, then nothing as it recedes.
        self.assertAlmostEqual(contributions[0], 0.2, places=6)
        self.assertAlmostEqual(contributions[1], 0.3, places=6)
        self.assertAlmostEqual(contributions[2], 0.0, places=6)
        # Cumulative equals the peak mean share, and stays inside [0, 1].
        self.assertAlmostEqual(sum(contributions), 0.5, places=6)

    def test_contribution_never_negative_as_the_share_recedes(self):
        adapter = _FakeAdapter(shares=[[0.9, 0.9], [0.1, 0.1]])
        term = SurchargeSlotShare(_LINK_IDS)
        term.bind(adapter)
        term.reset()
        adapter.links.advance()
        term.step(adapter, 60.0)
        adapter.links.advance()
        self.assertEqual(term.step(adapter, 60.0), 0.0)

    def test_reset_drops_the_previous_episode_envelope(self):
        adapter = _FakeAdapter(shares=[[0.6, 0.6]])
        term = SurchargeSlotShare(_LINK_IDS)
        term.bind(adapter)
        term.reset()
        adapter.links.advance()
        first = term.step(adapter, 60.0)
        term.reset()
        self.assertAlmostEqual(term.step(adapter, 60.0), first, places=6)

    def test_dt_seconds_is_unused(self):
        """The engine already time-integrated; multiplying by dt would
        double-count."""
        adapter = _FakeAdapter(shares=[[0.4, 0.4]])
        term = SurchargeSlotShare(_LINK_IDS)
        term.bind(adapter)
        term.reset()
        adapter.links.advance()
        self.assertAlmostEqual(term.step(adapter, 3600.0), 0.4, places=6)


# =============================================================================
# ReactionCoefficientValue
# =============================================================================


class TestReactionCoefficientValue(unittest.TestCase):
    def test_empty_coefficient_ids_rejected(self):
        with self.assertRaises(ValueError):
            ReactionCoefficientValue([], 0.0, 1.0)

    def test_mixed_scalar_and_vector_bounds_rejected(self):
        with self.assertRaises(ValueError):
            ReactionCoefficientValue(["kd"], 0.0, [1.0])

    def test_vector_bounds_must_match_the_coefficient_count(self):
        with self.assertRaises(ValueError):
            ReactionCoefficientValue(["kd", "ks"], [0.0], [1.0])

    def test_scalar_bounds_broadcast_over_every_coefficient(self):
        factory = ReactionCoefficientValue(["kd", "ks"], 0.0, 5.0)
        self.assertEqual(factory.space.shape, (2,))
        np.testing.assert_allclose(factory.space.low, [0.0, 0.0])
        np.testing.assert_allclose(factory.space.high, [5.0, 5.0])

    def test_per_coefficient_bounds_are_kept_distinct(self):
        factory = ReactionCoefficientValue(["kd", "ks"], [0.0, 1.0], [0.5, 9.0])
        np.testing.assert_allclose(factory.space.low, [0.0, 1.0])
        np.testing.assert_allclose(factory.space.high, [0.5, 9.0])

    def test_apply_before_bind_raises(self):
        factory = ReactionCoefficientValue(["kd"], 0.0, 1.0)
        with self.assertRaises(RuntimeError):
            factory.apply(_FakeAdapter(), np.array([0.5], dtype=np.float32))

    def test_bind_rejects_an_undeclared_coefficient(self):
        factory = ReactionCoefficientValue(["not_declared"], 0.0, 1.0)
        with self.assertRaises(ValueError) as ctx:
            factory.bind(_FakeAdapter())
        self.assertIn("not_declared", str(ctx.exception))

    def test_bind_rejects_a_constant_coefficient(self):
        """A CONSTANT is declared fixed; searching it would write values
        the model was never meant to vary."""
        factory = ReactionCoefficientValue(["yield_fixed"], 0.0, 1.0)
        with self.assertRaises(ValueError) as ctx:
            factory.bind(_FakeAdapter())
        message = str(ctx.exception)
        self.assertIn("yield_fixed", message)
        self.assertIn("CONSTANT", message)

    def test_bind_accepts_parameters(self):
        ReactionCoefficientValue(["kd", "ks"], 0.0, 1.0).bind(_FakeAdapter())

    def test_apply_writes_every_coefficient(self):
        adapter = _FakeAdapter()
        factory = ReactionCoefficientValue(["kd", "ks"], 0.0, 10.0)
        factory.bind(adapter)
        factory.apply(adapter, np.array([1.5, 7.25], dtype=np.float32))
        self.assertAlmostEqual(adapter.reactions.values["kd"], 1.5, places=5)
        self.assertAlmostEqual(adapter.reactions.values["ks"], 7.25, places=5)

    def test_apply_clips_to_the_declared_bounds(self):
        adapter = _FakeAdapter()
        factory = ReactionCoefficientValue(["kd", "ks"], [0.0, 1.0], [1.0, 2.0])
        factory.bind(adapter)
        factory.apply(adapter, np.array([-5.0, 99.0], dtype=np.float32))
        self.assertAlmostEqual(adapter.reactions.values["kd"], 0.0, places=5)
        self.assertAlmostEqual(adapter.reactions.values["ks"], 2.0, places=5)


# =============================================================================
# HeatSourceTemperature (design) / HeatSourceTemperatureSetpoint (runtime)
# =============================================================================


class TestHeatSourceTemperature(unittest.TestCase):
    def test_empty_sources_rejected(self):
        with self.assertRaises(ValueError):
            HeatSourceTemperature([])

    def test_defaults_to_the_engine_refusal_range(self):
        factory = HeatSourceTemperature(["DWF"])
        np.testing.assert_allclose(factory.space.low, [-50.0])
        np.testing.assert_allclose(factory.space.high, [100.0])

    def test_bounds_beyond_the_engine_range_are_rejected(self):
        """The engine REFUSES (does not clamp) an out-of-range write, so a
        wider Box could produce a silent no-op mid-episode."""
        for low, high in ((-60.0, 10.0), (0.0, 150.0)):
            with self.subTest(low=low, high=high):
                with self.assertRaises(ValueError) as ctx:
                    HeatSourceTemperature(["DWF"], low, high)
                self.assertIn("refus", str(ctx.exception).lower())

    def test_narrowing_the_bounds_is_allowed(self):
        factory = HeatSourceTemperature(["DWF"], 5.0, 25.0)
        np.testing.assert_allclose(factory.space.low, [5.0])
        np.testing.assert_allclose(factory.space.high, [25.0])

    def test_source_names_are_case_insensitive(self):
        factory = HeatSourceTemperature(["dwf", "external_inflow"])
        factory.bind(_FakeAdapter())

    def test_bind_rejects_an_unknown_pathway(self):
        factory = HeatSourceTemperature(["NOT_A_SOURCE"])
        with self.assertRaises(ValueError) as ctx:
            factory.bind(_FakeAdapter())
        self.assertIn("NOT_A_SOURCE", str(ctx.exception))

    def test_apply_before_bind_raises(self):
        factory = HeatSourceTemperature(["DWF"])
        with self.assertRaises(RuntimeError):
            factory.apply(_FakeAdapter(), np.array([10.0], dtype=np.float32))

    def test_apply_writes_each_source(self):
        adapter = _FakeAdapter()
        factory = HeatSourceTemperature(["DWF", "EXTERNAL_INFLOW"], 0.0, 40.0)
        factory.bind(adapter)
        factory.apply(adapter, np.array([12.0, 21.0], dtype=np.float32))
        self.assertEqual(
            adapter.heat.writes, [("DWF", 12.0), ("EXTERNAL_INFLOW", 21.0)]
        )

    def test_apply_clips_to_the_declared_bounds(self):
        adapter = _FakeAdapter()
        factory = HeatSourceTemperature(["DWF"], 5.0, 25.0)
        factory.bind(adapter)
        factory.apply(adapter, np.array([99.0], dtype=np.float32))
        self.assertEqual(adapter.heat.writes, [("DWF", 25.0)])


class TestHeatSourceTemperatureSetpoint(unittest.TestCase):
    def test_empty_sources_rejected(self):
        with self.assertRaises(ValueError):
            HeatSourceTemperatureSetpoint([])

    def test_name_defaults_and_is_overridable(self):
        self.assertEqual(
            HeatSourceTemperatureSetpoint(["DWF"]).name,
            "heat_source_temperature_setpoint",
        )
        self.assertEqual(
            HeatSourceTemperatureSetpoint(["DWF"], name="effluent_temp").name,
            "effluent_temp",
        )

    def test_bounds_beyond_the_engine_range_are_rejected(self):
        with self.assertRaises(ValueError):
            HeatSourceTemperatureSetpoint(["DWF"], -50.0, 120.0)

    def test_degenerate_bounds_rejected(self):
        with self.assertRaises(ValueError):
            HeatSourceTemperatureSetpoint(["DWF"], 10.0, 10.0)

    def test_space_shape_matches_the_source_count(self):
        self.assertEqual(
            HeatSourceTemperatureSetpoint(["DWF", "GW"], 0.0, 30.0).space.shape, (2,)
        )

    def test_apply_before_bind_raises(self):
        factory = HeatSourceTemperatureSetpoint(["DWF"])
        with self.assertRaises(RuntimeError):
            factory.apply(_FakeAdapter(), np.array([10.0], dtype=np.float32))

    def test_bind_rejects_an_unknown_pathway(self):
        with self.assertRaises(ValueError):
            HeatSourceTemperatureSetpoint(["NOPE"]).bind(_FakeAdapter())

    def test_apply_clips_defensively(self):
        """An unclipped stray value would be REFUSED by the engine, leaving
        the previous step's setpoint silently in force."""
        adapter = _FakeAdapter()
        factory = HeatSourceTemperatureSetpoint(["DWF"], 10.0, 20.0)
        factory.bind(adapter)
        factory.apply(adapter, np.array([-999.0], dtype=np.float32))
        self.assertEqual(adapter.heat.writes, [("DWF", 10.0)])

    def test_repeated_apply_is_the_per_step_contract(self):
        adapter = _FakeAdapter()
        factory = HeatSourceTemperatureSetpoint(["DWF"], 0.0, 30.0)
        factory.bind(adapter)
        for temp in (10.0, 15.0, 20.0):
            factory.apply(adapter, np.array([temp], dtype=np.float32))
        self.assertEqual(
            adapter.heat.writes, [("DWF", 10.0), ("DWF", 15.0), ("DWF", 20.0)]
        )


# =============================================================================
# WaterAgeSourceAge
# =============================================================================


class TestWaterAgeSourceAge(unittest.TestCase):
    def test_empty_sources_rejected(self):
        with self.assertRaises(ValueError):
            WaterAgeSourceAge([], 0.0, 1.0)

    def test_negative_low_bound_is_legal_and_preserved(self):
        """A negative source age extracts age-volume; flooring it at zero
        would delete half the physically meaningful search space."""
        factory = WaterAgeSourceAge(["DWF"], -12.0, 24.0)
        np.testing.assert_allclose(factory.space.low, [-12.0])
        np.testing.assert_allclose(factory.space.high, [24.0])

    def test_apply_writes_a_negative_age_unclamped(self):
        adapter = _FakeAdapter()
        factory = WaterAgeSourceAge(["DWF"], -12.0, 24.0)
        factory.bind(adapter)
        factory.apply(adapter, np.array([-6.0], dtype=np.float32))
        self.assertEqual(adapter.water_age.writes, [("DWF", -6.0)])

    def test_apply_clips_to_the_declared_bounds(self):
        adapter = _FakeAdapter()
        factory = WaterAgeSourceAge(["DWF"], -12.0, 24.0)
        factory.bind(adapter)
        factory.apply(adapter, np.array([-99.0], dtype=np.float32))
        self.assertEqual(adapter.water_age.writes, [("DWF", -12.0)])

    def test_degenerate_bounds_rejected(self):
        with self.assertRaises(ValueError):
            WaterAgeSourceAge(["DWF"], 1.0, 1.0)

    def test_bind_rejects_an_unknown_pathway(self):
        with self.assertRaises(ValueError):
            WaterAgeSourceAge(["NOPE"], 0.0, 1.0).bind(_FakeAdapter())

    def test_apply_before_bind_raises(self):
        factory = WaterAgeSourceAge(["DWF"], 0.0, 1.0)
        with self.assertRaises(RuntimeError):
            factory.apply(_FakeAdapter(), np.array([0.5], dtype=np.float32))

    def test_source_names_are_case_insensitive(self):
        WaterAgeSourceAge(["dwf"], 0.0, 1.0).bind(_FakeAdapter())


# =============================================================================
# Two-tier optional-surface guards (engine-touching)
# =============================================================================


@unittest.skipUnless(
    hasattr(_engine, "Heat") and hasattr(_engine, "WaterAge"),
    "installed openswmm.engine has no Heat / WaterAge module",
)
class TestOptionalSurfaceGuards(BaseEngineTest):
    """The model tier of the C{heat} / C{water_age} guards.

    C{tests/data/minimal.inp} is a plain hydraulic model: it enables
    neither C{HEAT_TRANSPORT} nor C{WATER_AGE}, which is exactly the case
    the model-tier check exists for. Writing source temperatures into such
    a model stores them and routes nothing, with no error anywhere — so
    the accessor must refuse rather than hand back a usable object.
    """

    def test_heat_accessor_refuses_a_model_without_heat_transport(self):
        adapter = self.make_adapter()
        with self.assertRaises(RuntimeError) as ctx:
            _ = adapter.heat
        self.assertIn("HEAT_TRANSPORT", str(ctx.exception))

    def test_water_age_accessor_refuses_a_model_without_water_age(self):
        adapter = self.make_adapter()
        with self.assertRaises(RuntimeError) as ctx:
            _ = adapter.water_age
        self.assertIn("WATER_AGE", str(ctx.exception))

    def test_heat_factory_bind_surfaces_the_model_tier_error(self):
        adapter = self.make_adapter()
        with self.assertRaises(RuntimeError):
            HeatSourceTemperature(["DWF"]).bind(adapter)

    def test_water_age_factory_bind_surfaces_the_model_tier_error(self):
        adapter = self.make_adapter()
        with self.assertRaises(RuntimeError):
            WaterAgeSourceAge(["DWF"], 0.0, 24.0).bind(adapter)

    def test_reactions_accessor_has_no_model_tier_gate(self):
        """Unlike heat / water age there is no [OPTIONS] flag for
        reactions, so a model with no species is a legitimate (empty)
        state rather than a misconfiguration."""
        adapter = self.make_adapter()
        self.assertEqual(adapter.reactions.coefficient_names(), [])

    def test_slot_share_term_refuses_the_dynamic_wave_fixture(self):
        """minimal.inp is FLOW_ROUTING DYNWAVE, so the guard must fire
        against the real engine's option reader, not just the fake."""
        adapter = self.make_adapter()
        with self.assertRaises(ValueError) as ctx:
            SurchargeSlotShare(["C1"]).bind(adapter)
        self.assertIn("FV", str(ctx.exception))

    def test_get_option_reads_the_router(self):
        adapter = self.make_adapter()
        self.assertEqual(adapter.get_option("FLOW_ROUTING").upper(), "DYNWAVE")


if __name__ == "__main__":
    unittest.main()
