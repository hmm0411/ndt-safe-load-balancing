from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.experiments.run_forecast_hot_switch_2c20s import (
    HOT_SHARE,
    TARGET_UTILIZATION,
    achieved_utilization,
    build_distribution,
    check_achieved_utilization,
)


class HotSwitchRateTests(unittest.TestCase):
    def test_total_rate_matches_target_processed_load(self) -> None:
        rates = build_distribution(830.0, "c1", 1.0)

        # One Packet-In per flow: flows/s equals processed Packet-In/s.
        self.assertAlmostEqual(
            sum(rates.values()),
            TARGET_UTILIZATION * 830.0,
        )
        self.assertAlmostEqual(
            rates[1] / sum(rates.values()),
            HOT_SHARE,
        )

    def test_more_packet_in_per_flow_means_fewer_flows(self) -> None:
        one = sum(build_distribution(830.0, "c2", 1.0).values())
        two = sum(build_distribution(830.0, "c2", 2.0).values())

        self.assertAlmostEqual(two, one / 2.0)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_distribution(0.0, "c1", 1.0)

        with self.assertRaises(ValueError):
            build_distribution(830.0, "c1", 0.0)

    def test_achieved_utilization_skips_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forecast_samples.jsonl"
            rows = [
                {"controller_id": "c1", "utilization": 0.0}
                for _ in range(10)
            ] + [
                {"controller_id": "c1", "utilization": 0.8}
                for _ in range(20)
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            self.assertAlmostEqual(
                achieved_utilization(path, 1.0)["c1"],
                0.8,
            )

    def test_low_load_run_is_reported(self) -> None:
        # The week-8 defect: target 0.80, achieved ~0.09.
        problems = check_achieved_utilization(
            {"c1": 0.09, "c2": 0.79}
        )

        self.assertEqual(len(problems), 1)
        self.assertIn("c1", problems[0])
        self.assertEqual(
            check_achieved_utilization({"c1": 0.78, "c2": 0.84}),
            [],
        )


if __name__ == "__main__":
    unittest.main()
