"""Unit tests for L{openswmm_gymnasium.config.MarketConfig}.

Pure-logic tier (no engine): construction, default-filling, canonical
round-trip (C{from_dict(to_dict(cfg)) == cfg}), and validation rejections.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import copy
import unittest

from openswmm_gymnasium.config import MarketConfig


def _valid_dict() -> dict:
    """A minimal but complete, schema-valid market config dict."""
    return {
        "meta": {
            "model_path": "/m.inp",
            "output_dir": "/out",
            "control_interval_seconds": 300,
        },
        "cost_curves": {
            "buyer": {"type": "logistic", "onset": 0.85, "steepness": 20},
            "seller": {"type": "piecewise_linear", "onset": 0.6, "full": 1.0},
        },
        "agents": [
            {
                "id": "J1",
                "element_type": "node",
                "commodity": "conveyance",
                "stress_metric": "freeboard_fraction",
                "curve": "buyer",
            },
            {
                "id": "ST1",
                "element_type": "node",
                "role": "seller",
                "commodity": "storage",
                "stress_metric": "storage_fill",
                "curve": "seller",
            },
        ],
        "trade_routes": [
            {
                "structure_link_id": "GATE1",
                "buyer_agents": ["J1"],
                "seller_agents": ["ST1"],
                "pid": {"kp": 0.8, "ki": 0.05},
                "constraints": {"min_cycle_seconds": 600, "max_starts_per_hour": 6},
            }
        ],
    }


class TestRoundTrip(unittest.TestCase):
    def test_from_dict_fills_defaults(self):
        cfg = MarketConfig.from_dict(_valid_dict())
        # Schema defaults are materialised.
        self.assertEqual(cfg.meta.treatment_tag, "TREATED")
        self.assertEqual(cfg.agents[0].role, "both")
        self.assertEqual(cfg.cost_curves["buyer"].floor, 0.0)
        self.assertEqual(cfg.cost_curves["buyer"].ceiling, 1.0)
        self.assertEqual(cfg.trade_routes[0].pid.kd, 0.0)
        self.assertEqual(cfg.trade_routes[0].aggregate, "max")

    def test_canonical_round_trip_is_identity(self):
        cfg = MarketConfig.from_dict(_valid_dict())
        cfg.validate()
        again = MarketConfig.from_dict(cfg.to_dict())
        self.assertEqual(cfg, again)

    def test_optional_pump_constraints_preserved(self):
        cfg = MarketConfig.from_dict(_valid_dict())
        rc = cfg.trade_routes[0].constraints
        self.assertEqual(rc.min_cycle_seconds, 600)
        self.assertEqual(rc.max_starts_per_hour, 6)
        # And they survive to_dict (not dropped as None).
        self.assertIn("min_cycle_seconds", cfg.to_dict()["trade_routes"][0]["constraints"])

    def test_save_load_round_trip(self):
        import tempfile
        from pathlib import Path

        cfg = MarketConfig.from_dict(_valid_dict())
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "mc.json")
            cfg.save(p)
            self.assertEqual(MarketConfig.load(p), cfg)


class TestValidation(unittest.TestCase):
    def test_valid_passes(self):
        MarketConfig.from_dict(_valid_dict()).validate()  # no raise

    def test_missing_required_key_raises(self):
        d = _valid_dict()
        del d["meta"]["model_path"]
        with self.assertRaisesRegex(ValueError, "missing required key"):
            MarketConfig.from_dict(d)

    def test_bad_curve_type_rejected(self):
        d = _valid_dict()
        d["cost_curves"]["buyer"]["type"] = "exponential"
        with self.assertRaisesRegex(ValueError, "cost_curves.*type"):
            MarketConfig.from_dict(d).validate()

    def test_agent_curve_must_exist(self):
        d = _valid_dict()
        d["agents"][0]["curve"] = "nope"
        with self.assertRaisesRegex(ValueError, "not found in cost_curves"):
            MarketConfig.from_dict(d).validate()

    def test_route_references_unknown_agent(self):
        d = _valid_dict()
        d["trade_routes"][0]["buyer_agents"] = ["ghost"]
        with self.assertRaisesRegex(ValueError, "unknown agent"):
            MarketConfig.from_dict(d).validate()

    def test_onset_out_of_unit_range_rejected(self):
        d = _valid_dict()
        d["cost_curves"]["buyer"]["onset"] = 1.5
        with self.assertRaisesRegex(ValueError, r"onset.*\[0, 1\]"):
            MarketConfig.from_dict(d).validate()

    def test_setting_min_above_max_rejected(self):
        d = _valid_dict()
        d["trade_routes"][0]["constraints"]["setting_min"] = 0.9
        d["trade_routes"][0]["constraints"]["setting_max"] = 0.1
        with self.assertRaisesRegex(ValueError, "setting_min must be <= setting_max"):
            MarketConfig.from_dict(d).validate()

    def test_non_positive_control_interval_rejected(self):
        d = _valid_dict()
        d["meta"]["control_interval_seconds"] = 0
        with self.assertRaisesRegex(ValueError, "control_interval_seconds must be > 0"):
            MarketConfig.from_dict(d).validate()

    def test_original_dict_not_mutated(self):
        d = _valid_dict()
        snapshot = copy.deepcopy(d)
        MarketConfig.from_dict(d)
        self.assertEqual(d, snapshot)


if __name__ == "__main__":
    unittest.main()
