"""P0.2 — unit-system recording at env reset().

The engine returns every quantity in the units declared in the C{.inp}
file (project units), so observation/reward scaling is unit-dependent. The
env records the model's unit system at C{reset()}, surfaces it in C{info},
and refuses to silently continue if the system changes across episodes.

Integration tier: drives a real engine over C{minimal.inp} (a CFS / US
model). B{No engine mocks.}

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

from tests.unit._base import BaseEngineTest

from openswmm_gymnasium.envs import SwmmRTCEnv
from openswmm_gymnasium.observations import ObservationBuilder


def _make_minimal_env(inp_path):
    return SwmmRTCEnv(
        inp_path,
        control_interval_steps=1,
        observation_builder=ObservationBuilder().add_node_depths(["J1"]).add_link_flows(["C1"]),
    )


class TestUnitRecording(BaseEngineTest):
    def test_reset_info_reports_units(self):
        env = _make_minimal_env(self.minimal_inp)
        try:
            _obs, info = env.reset(seed=0)
            self.assertEqual(info["flow_units"], "CFS")
            self.assertEqual(info["unit_system"], "US")
        finally:
            env.close()

    def test_adapter_exposes_units(self):
        adapter = self.make_adapter(open=True)
        self.assertEqual(adapter.flow_units, "CFS")
        self.assertEqual(adapter.unit_system, "US")

    def test_repeated_reset_same_system_ok(self):
        env = _make_minimal_env(self.minimal_inp)
        try:
            _o1, i1 = env.reset(seed=0)
            _o2, i2 = env.reset(seed=1)
            self.assertEqual(i1["unit_system"], i2["unit_system"])
        finally:
            env.close()
