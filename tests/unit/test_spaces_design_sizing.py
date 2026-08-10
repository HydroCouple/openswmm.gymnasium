"""Unit tests for the storage / green-infrastructure / RDII sizing factories.

Covers L{StorageVolume}, L{LIDPlacement}, and L{RDIIUnitHydrograph} — the
"explore designs more exhaustively" family (plan §3.1). Space construction,
bound validation, and B{bind/apply decode} are exercised against a mock
adapter that records the engine setter calls, so these run without the
compiled engine. End-to-end engine behaviour is covered by the CIP env
integration tests.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import unittest

import numpy as np

from openswmm_gymnasium.spaces.design import (
    DesignActionFactory,
    LIDPlacement,
    RDIIUnitHydrograph,
    StorageVolume,
)


# -----------------------------------------------------------------------------
# Mock adapter recording engine setter calls
# -----------------------------------------------------------------------------


class _HG:
    """Stand-in for the engine's HydrographEntry NamedTuple."""

    def __init__(self, uh_name, month, response, t, k):
        self.uh_name = uh_name
        self.month = month
        self.response = response
        self.r = 0.0
        self.t = t
        self.k = k
        self.dmax = 0.0
        self.drecov = 0.0
        self.dinit = 0.0


class _Nodes:
    def __init__(self, calls, baseline, shapes=None, curves=None):
        self._calls = calls
        self._baseline = baseline
        self._ids = {"T1": 0, "T2": 1, "T3": 2, "T4": 3, "C1": 4}
        # Default: every node in the legacy fixtures is FUNCTIONAL.
        self._shapes = shapes if shapes is not None else {}
        self._curves = curves if curves is not None else {}

    def get_index(self, nid):
        return self._ids[nid]

    def get_storage_shape(self, idx):
        return self._shapes.get(idx, "FUNCTIONAL")

    def get_storage_curve(self, idx):
        return self._curves[idx]

    def get_storage_functional(self, idx):
        return self._baseline[idx]

    def set_storage_functional(self, idx, a, b, c):
        self._calls.append(("storage", idx, round(a, 4), round(b, 4), round(c, 4)))


class _Tables:
    """Stand-in for L{_TablesCompat} over a couple of depth-area curves."""

    def __init__(self, calls, curves):
        self._calls = calls
        self._curves = curves

    def get_curve_points(self, idx):
        return self._curves[idx]

    def set_curve_points(self, idx, points):
        self._calls.append(
            ("curve", idx, [(round(x, 4), round(y, 4)) for x, y in points])
        )


class _Subs:
    def get_index(self, sid):
        return {"S1": 0, "S2": 1}[sid]


class _Infra:
    def __init__(self, calls):
        self._calls = calls
        self._ctrl = {"BIO": 10, "PAVE": 11, "ROOF": 12}

    def get_lid_index(self, cid):
        return self._ctrl[cid]

    def lid_usage_add(self, sid, lid, number, area, width, init_sat, from_imperv):
        self._calls.append(("lid", sid, lid, number, round(area, 3), width))


class _Inflows:
    def __init__(self, calls, entries):
        self._calls = calls
        self._entries = entries

    def hydrograph_count(self):
        return len(self._entries)

    def get_hydrograph(self, i):
        return self._entries[i]

    def set_hydrograph_rtk(self, uh, month, response, r, t, k):
        self._calls.append(("rtk", uh, month, response, round(r, 4), t, k))

    def set_hydrograph_ia(self, uh, month, response, dmax, drecov, dinit):
        self._calls.append(
            ("ia", uh, month, response, round(dmax, 3), round(drecov, 3), round(dinit, 3))
        )


class MockAdapter:
    """Records the engine setter calls the factories make."""

    def __init__(self):
        self.calls: list[tuple] = []
        # T1/T2 are FUNCTIONAL; T3/T4 are TABULAR on curves 7 and 8; C1 is a
        # geometric (CYLINDRICAL) storage unit that cannot be sized.
        self.nodes = _Nodes(
            self.calls,
            {0: (10000.0, 0.0, 0.0), 1: (5000.0, 0.5, 100.0)},
            shapes={2: "TABULAR", 3: "TABULAR", 4: "CYLINDRICAL"},
            curves={2: 7, 3: 8},
        )
        self.tables = _Tables(
            self.calls,
            {
                7: [(0.0, 1000.0), (5.0, 2000.0)],
                8: [(0.0, 400.0), (2.0, 800.0)],
            },
        )
        self.subcatchments = _Subs()
        self.infrastructure = _Infra(self.calls)
        self.inflows = _Inflows(
            self.calls,
            [_HG("SanSewer", -1, 0, 2.0, 3.0), _HG("SanSewer", -1, 1, 4.0, 5.0)],
        )


# -----------------------------------------------------------------------------
# StorageVolume
# -----------------------------------------------------------------------------


class TestStorageVolume(unittest.TestCase):
    def test_scalar_space_and_apply_scales_footprint(self):
        f = StorageVolume(["T1", "T2"], low=0.5, high=3.0, mode="scalar")
        self.assertEqual(f.name, "storage_volume")
        self.assertEqual(f.space.shape, (2,))
        a = MockAdapter()
        f.bind(a)
        # Second value (10.0) is clipped to the multiplier ceiling 3.0.
        f.apply(a, np.array([2.0, 10.0], dtype=np.float32))
        # a and c scale by the factor; exponent b is preserved.
        self.assertEqual(
            a.calls,
            [("storage", 0, 20000.0, 0.0, 0.0), ("storage", 1, 15000.0, 0.5, 300.0)],
        )

    def test_coeffs_space_and_apply_sets_triple(self):
        f = StorageVolume(
            ["T1", "T2"], low=[100.0, 0.0, 0.0], high=[9000.0, 2.0, 500.0], mode="coeffs"
        )
        self.assertEqual(f.space.shape, (6,))
        a = MockAdapter()
        f.bind(a)
        f.apply(a, np.array([1000.0, 1.0, 50.0, 2000.0, 1.5, 60.0], dtype=np.float32))
        self.assertEqual(
            a.calls,
            [("storage", 0, 1000.0, 1.0, 50.0), ("storage", 1, 2000.0, 1.5, 60.0)],
        )

    def test_scalar_scales_a_tabular_storage_curve(self):
        """G3 — a TABULAR node is sized directly, not via the NodeMaxDepth proxy."""
        f = StorageVolume(["T3"], low=0.5, high=3.0, mode="scalar")
        a = MockAdapter()
        f.bind(a)
        f.apply(a, np.array([2.0], dtype=np.float32))
        # Depths untouched; areas doubled.
        self.assertEqual(
            a.calls, [("curve", 7, [(0.0, 2000.0), (5.0, 4000.0)])]
        )

    def test_scalar_mixes_functional_and_tabular_nodes(self):
        f = StorageVolume(["T1", "T3"], low=0.5, high=3.0, mode="scalar")
        a = MockAdapter()
        f.bind(a)
        f.apply(a, np.array([2.0, 0.5], dtype=np.float32))
        self.assertEqual(
            a.calls,
            [
                ("storage", 0, 20000.0, 0.0, 0.0),
                ("curve", 7, [(0.0, 500.0), (5.0, 1000.0)]),
            ],
        )

    def test_tabular_multiplier_is_clipped(self):
        f = StorageVolume(["T3"], low=0.5, high=3.0, mode="scalar")
        a = MockAdapter()
        f.bind(a)
        f.apply(a, np.array([99.0], dtype=np.float32))
        self.assertEqual(a.calls, [("curve", 7, [(0.0, 3000.0), (5.0, 6000.0)])])

    def test_coeffs_mode_rejects_a_tabular_node(self):
        f = StorageVolume(
            ["T3"], low=[100.0, 0.0, 0.0], high=[9000.0, 2.0, 500.0], mode="coeffs"
        )
        with self.assertRaisesRegex(ValueError, "TABULAR"):
            f.bind(MockAdapter())

    def test_geometric_storage_shape_is_rejected(self):
        f = StorageVolume(["C1"], low=0.5, high=3.0, mode="scalar")
        with self.assertRaisesRegex(ValueError, "CYLINDRICAL"):
            f.bind(MockAdapter())

    def test_two_nodes_sharing_one_curve_are_rejected(self):
        a = MockAdapter()
        a.nodes._curves[3] = 7  # point T4 at T3's curve
        f = StorageVolume(["T3", "T4"], low=0.5, high=3.0, mode="scalar")
        with self.assertRaisesRegex(ValueError, "share"):
            f.bind(a)

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "at least one node"):
            StorageVolume([], low=0.5, high=3.0)
        with self.assertRaisesRegex(ValueError, "mode"):
            StorageVolume(["T1"], low=0.5, high=3.0, mode="bogus")
        with self.assertRaisesRegex(ValueError, "length-3"):
            StorageVolume(["T1"], low=[1, 2], high=[3, 4], mode="coeffs")
        with self.assertRaisesRegex(ValueError, "strictly greater"):
            StorageVolume(["T1"], low=1.0, high=1.0, mode="scalar")

    def test_apply_before_bind_raises(self):
        f = StorageVolume(["T1"], low=0.5, high=3.0)
        with self.assertRaisesRegex(RuntimeError, "bind"):
            f.apply(MockAdapter(), np.array([1.0], dtype=np.float32))

    def test_protocol_conformance(self):
        self.assertIsInstance(StorageVolume(["T1"], 0.5, 3.0), DesignActionFactory)


# -----------------------------------------------------------------------------
# LIDPlacement
# -----------------------------------------------------------------------------


class TestLIDPlacement(unittest.TestCase):
    def test_space_layout(self):
        f = LIDPlacement(["S1", "S2"], ["BIO", "PAVE", "ROOF"], area_low=100.0, area_high=2000.0)
        self.assertEqual(f.space.shape, (4,))
        # Subcatchment-major [type, area]: type in [0, n_controls], area in [lo, hi].
        self.assertTrue(np.allclose(f.space.low, [0.0, 100.0, 0.0, 100.0]))
        self.assertTrue(np.allclose(f.space.high, [3.0, 2000.0, 3.0, 2000.0]))

    def test_apply_selects_type_and_sizes_area(self):
        f = LIDPlacement(
            ["S1", "S2"], ["BIO", "PAVE", "ROOF"], area_low=100.0, area_high=2000.0,
            number=2, width=5.0,
        )
        a = MockAdapter()
        f.bind(a)
        # S1: type 2.7 -> floor 2 -> ROOF(12), area 500.
        # S2: type 0.1 -> BIO(10), area 5000 -> clipped to 2000.
        f.apply(a, np.array([2.7, 500.0, 0.1, 5000.0], dtype=np.float32))
        self.assertEqual(
            a.calls,
            [("lid", "S1", 12, 2, 500.0, 5.0), ("lid", "S2", 10, 2, 2000.0, 5.0)],
        )

    def test_type_index_clamps_at_ceiling(self):
        f = LIDPlacement(["S1"], ["BIO", "PAVE"], area_low=1.0, area_high=10.0)
        a = MockAdapter()
        f.bind(a)
        # type 2.0 == n_controls -> clamped to last valid index (1 -> PAVE=11).
        f.apply(a, np.array([2.0, 5.0], dtype=np.float32))
        self.assertEqual(a.calls, [("lid", "S1", 11, 1, 5.0, 0.0)])

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "at least one subcatch"):
            LIDPlacement([], ["BIO"], 1.0, 2.0)
        with self.assertRaisesRegex(ValueError, "lid_control"):
            LIDPlacement(["S1"], [], 1.0, 2.0)
        with self.assertRaisesRegex(ValueError, "strictly greater"):
            LIDPlacement(["S1"], ["BIO"], area_low=2.0, area_high=1.0)

    def test_apply_before_bind_raises(self):
        f = LIDPlacement(["S1"], ["BIO"], 1.0, 2.0)
        with self.assertRaisesRegex(RuntimeError, "bind"):
            f.apply(MockAdapter(), np.array([0.0, 1.5], dtype=np.float32))


# -----------------------------------------------------------------------------
# RDIIUnitHydrograph
# -----------------------------------------------------------------------------


class TestRDIIUnitHydrograph(unittest.TestCase):
    def test_r_only_space_and_apply_preserves_tk(self):
        f = RDIIUnitHydrograph([("SanSewer", -1, 0)], r_low=0.0, r_high=0.5)
        self.assertEqual(f.space.shape, (1,))
        a = MockAdapter()
        f.bind(a)
        f.apply(a, np.array([0.2], dtype=np.float32))
        # T=2.0, K=3.0 read at bind are preserved; only R changes.
        self.assertEqual(a.calls, [("rtk", "SanSewer", -1, 0, 0.2, 2.0, 3.0)])

    def test_r_plus_ia_space_and_apply(self):
        f = RDIIUnitHydrograph(
            [("SanSewer", -1, 0), ("SanSewer", -1, 1)],
            r_low=0.0, r_high=0.5,
            include_ia=True, ia_low=(0.0, 0.0, 0.0), ia_high=(0.5, 2.0, 0.2),
        )
        self.assertEqual(f.space.shape, (8,))
        a = MockAdapter()
        f.bind(a)
        # Second target's R (0.9) is clipped to 0.5.
        f.apply(
            a,
            np.array([0.1, 0.05, 1.0, 0.02, 0.9, 0.3, 3.0, 0.1], dtype=np.float32),
        )
        self.assertEqual(
            a.calls,
            [
                ("rtk", "SanSewer", -1, 0, 0.1, 2.0, 3.0),
                ("ia", "SanSewer", -1, 0, 0.05, 1.0, 0.02),
                ("rtk", "SanSewer", -1, 1, 0.5, 4.0, 5.0),
                ("ia", "SanSewer", -1, 1, 0.3, 2.0, 0.1),
            ],
        )

    def test_bind_rejects_unknown_target(self):
        f = RDIIUnitHydrograph([("MISSING", -1, 0)], r_low=0.0, r_high=0.5)
        with self.assertRaisesRegex(ValueError, "not found"):
            f.bind(MockAdapter())

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "at least one target"):
            RDIIUnitHydrograph([], r_low=0.0, r_high=0.5)
        with self.assertRaisesRegex(ValueError, "high"):
            # r_low == r_high -> no searchable component.
            RDIIUnitHydrograph([("U", -1, 0)], r_low=0.5, r_high=0.5)
        with self.assertRaisesRegex(ValueError, "length-3"):
            RDIIUnitHydrograph(
                [("U", -1, 0)], 0.0, 0.5, include_ia=True, ia_low=(0.0, 0.0), ia_high=(1.0, 1.0)
            )

    def test_apply_before_bind_raises(self):
        f = RDIIUnitHydrograph([("SanSewer", -1, 0)], r_low=0.0, r_high=0.5)
        with self.assertRaisesRegex(RuntimeError, "bind"):
            f.apply(MockAdapter(), np.array([0.2], dtype=np.float32))


# -----------------------------------------------------------------------------
# Integration tier — real engine (StorageVolume against the b01 twin-tank
# scenario, whose T1/T2 are FUNCTIONAL storage). LID / RDII integration tests
# need fixtures with [LID_CONTROLS] / [HYDROGRAPHS]; add alongside as those
# benchmark scenarios come online.
# -----------------------------------------------------------------------------


class TestStorageVolumeEpisode(unittest.TestCase):
    def _make_env(self, mode, low, high):
        from openswmm_gymnasium.benchmarks import b01_twin_tank
        from openswmm_gymnasium.envs import SwmmCIPEnv
        from openswmm_gymnasium.observations import ObservationBuilder

        return SwmmCIPEnv(
            b01_twin_tank.SCENARIO_INP,
            design_factories=[
                StorageVolume(["T1", "T2"], low=low, high=high, mode=mode)
            ],
            observation_builder=ObservationBuilder().add_node_depths(["T1", "T2"]),
        )

    def test_scalar_storage_episode_runs(self):
        env = self._make_env("scalar", 0.5, 2.0)
        try:
            env.reset(seed=0)
            action = {
                "design": {"storage_volume": np.array([1.5, 1.5], dtype=np.float32)},
                "runtime": {},
            }
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertIs(terminated, True)
            self.assertIs(truncated, False)
            self.assertEqual(obs.shape, (2,))
            self.assertTrue(np.isfinite(reward))
            self.assertEqual(info["phase"], "design_evaluated")
        finally:
            env.close()

    def test_coeffs_storage_episode_runs(self):
        # Directly rewrite the functional (a, b, c) triple of each tank.
        env = self._make_env("coeffs", [2000.0, 0.0, 0.0], [20000.0, 1.0, 500.0])
        try:
            env.reset(seed=0)
            action = {
                "design": {
                    "storage_volume": np.array(
                        [8000.0, 0.0, 0.0, 8000.0, 0.0, 0.0], dtype=np.float32
                    )
                },
                "runtime": {},
            }
            _obs, reward, terminated, _trunc, _info = env.step(action)
            self.assertIs(terminated, True)
            self.assertTrue(np.isfinite(reward))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
