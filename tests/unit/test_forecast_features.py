import unittest

from src.twin.forecasting.features import (
    build_features,
    feature_names,
)


def row(index: int):
    utilization = float(index)

    return {
        "run_id": "run-a",
        "controller_id": "c1",
        "sequence_id": "run-a:c1:0",
        "observed_at":
            f"2026-01-01T00:00:{index:02d}+00:00",
        "utilization": utilization,
        "processed_packet_in_rate":
            utilization * 100.0,
        "flow_mod_rate": utilization * 2.0,
        "process_cpu_percent":
            utilization * 3.0,
        "process_memory_rss_mb": 50.0,
        "response_p95_ms":
            utilization * 4.0,
        "managed_switch_count": 10,
        "max_switch_control_load_share": 0.5,
    }


class ForecastFeatureTests(unittest.TestCase):
    def test_feature_builder_preserves_all_rows(self):
        rows = [
            row(index)
            for index in range(20)
        ]

        result = build_features(rows)

        self.assertEqual(
            len(result),
            20,
        )

        self.assertEqual(
            result[0]["utilization"],
            0.0,
        )

        self.assertIsNone(
            result[0]["utilization_lag_1"]
        )

        self.assertIsNone(
            result[9]["utilization_lag_10"]
        )

        self.assertEqual(
            result[10]["utilization_lag_10"],
            0.0,
        )

    def test_lags_use_past_only(self):
        result = build_features([
            row(index)
            for index in range(20)
        ])

        current = result[10]

        self.assertEqual(
            current["utilization"],
            10.0,
        )

        self.assertEqual(
            current["utilization_lag_1"],
            9.0,
        )

        self.assertEqual(
            current["utilization_lag_10"],
            0.0,
        )

    def test_rolling_window_includes_current(self):
        result = build_features([
            row(index)
            for index in range(20)
        ])

        current = result[10]

        self.assertAlmostEqual(
            current["util_mean_3"],
            (8.0 + 9.0 + 10.0) / 3.0,
        )

        self.assertAlmostEqual(
            current["util_mean_5"],
            (
                6.0
                + 7.0
                + 8.0
                + 9.0
                + 10.0
            )
            / 5.0,
        )

    def test_rolling_does_not_use_future(self):
        rows_a = [
            row(index)
            for index in range(20)
        ]

        rows_b = [
            row(index)
            for index in range(20)
        ]

        rows_b[11]["utilization"] = 9999.0

        result_a = build_features(rows_a)
        result_b = build_features(rows_b)

        self.assertEqual(
            result_a[10]["util_mean_5"],
            result_b[10]["util_mean_5"],
        )

        self.assertEqual(
            result_a[10]["util_slope_5"],
            result_b[10]["util_slope_5"],
        )

    def test_delta_uses_previous_observation(self):
        result = build_features([
            row(index)
            for index in range(20)
        ])

        self.assertIsNone(
            result[0]["util_delta_1"]
        )

        self.assertEqual(
            result[10]["util_delta_1"],
            1.0,
        )

    def test_missing_numeric_field_is_rejected(self):
        rows = [
            row(index)
            for index in range(20)
        ]

        del rows[10]["flow_mod_rate"]

        with self.assertRaisesRegex(
            ValueError,
            "missing required numeric field",
        ):
            build_features(rows)

    def test_null_numeric_field_is_rejected(self):
        rows = [
            row(index)
            for index in range(20)
        ]

        rows[10]["flow_mod_rate"] = None

        with self.assertRaisesRegex(
            ValueError,
            "must not be null",
        ):
            build_features(rows)

    def test_feature_schema_is_unique(self):
        names = feature_names()

        self.assertEqual(
            len(names),
            len(set(names)),
        )

        self.assertEqual(
            len(names),
            29,
        )

        self.assertIn(
            "utilization_lag_10",
            names,
        )

        self.assertIn(
            "util_slope_5",
            names,
        )


if __name__ == "__main__":
    unittest.main()
