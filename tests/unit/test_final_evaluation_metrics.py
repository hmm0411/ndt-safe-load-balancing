from __future__ import annotations

import unittest
from typing import Any

from src.experiments.final_evaluation import metrics as fm


def series(values: list[dict[str, float]]) -> list[tuple[float, dict[str, float]]]:
    return [(float(i), u) for i, u in enumerate(values)]


def mig(switch: str, src: str, dst: str, at: float, status: str = "COMMITTED",
        **post: Any) -> dict[str, Any]:
    return {"switch_id": switch, "source": src, "target": dst, "started_at": at,
            "status": status, **post}


class OverloadTest(unittest.TestCase):
    def test_plan_case_oe_is_two(self) -> None:
        s = series([{"c1": 1.2, "c2": 0.5}] * 10)
        self.assertAlmostEqual(fm.overload_exposure(s, 1.0), 2.0)
        self.assertAlmostEqual(fm.normalized_overload_exposure(s, 1.0), 2.0 / (2 * 10))
        self.assertEqual(fm.overload_time(s, 1.0), 10.0)

    def test_boundary_is_overload_time_but_no_exposure(self) -> None:
        s = series([{"c1": 1.0, "c2": 0.2}, {"c1": 0.9, "c2": 0.2}])
        self.assertEqual(fm.overload_time(s, 1.0), 1.0)
        self.assertEqual(fm.overload_exposure(s, 1.0), 0.0)

    def test_u_is_not_clamped_and_imbalance_is_population_std(self) -> None:
        s = series([{"c1": 1.5, "c2": 0.5}])
        self.assertEqual(fm.max_u(s), 1.5)
        self.assertAlmostEqual(fm.mean_imbalance(s), 0.5)  # 2C: |U1 - U2| / 2

    def test_bad_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fm.overload_time([], 1.0)
        with self.assertRaises(ValueError):
            fm.overload_time([(1.0, {"c1": 1.0}), (1.0, {"c1": 1.0})], 1.0)
        with self.assertRaises(ValueError):
            fm.overload_time(series([{"c1": float("nan")}]), 1.0)


class MigrationTest(unittest.TestCase):
    def test_counts_rates_and_unsafe(self) -> None:
        migrations = [
            mig("s1", "c1", "c2", 10, u_dst_post=0.8),
            mig("s2", "c1", "c3", 70, u_dst_post=1.03),          # unsafe by U
            mig("s3", "c1", "c4", 130, u_dst_post=0.7, r95_dst_post_ms=90.0),  # unsafe by R95
            mig("s4", "c1", "c2", 190, status="RESTORED"),
            mig("s5", "c1", "c2", 250),                         # no post state observed
        ]
        out = fm.episode_metrics(series([{"c1": 0.9, "c2": 0.5}]), migrations, 1.0,
                                 r95_max_ms=50.0)
        self.assertEqual((out.migrations, out.committed, out.rollbacks), (5, 4, 1))
        self.assertAlmostEqual(out.success_rate or 0, 0.8)
        self.assertEqual((out.unsafe_migrations, out.unsafe_unknown), (2, 1))
        self.assertAlmostEqual(out.unsafe_rate or 0, 2 / 3)

    def test_without_migration_rates_are_none_not_zero(self) -> None:
        out = fm.episode_metrics(series([{"c1": 0.5, "c2": 0.5}]), [], 1.0)
        self.assertIsNone(out.success_rate)
        self.assertIsNone(out.unsafe_rate)

    def test_ping_pong_inside_the_window_only(self) -> None:
        migrations = [mig("s1", "c1", "c2", 0), mig("s1", "c2", "c1", 30),
                      mig("s1", "c1", "c2", 200)]
        self.assertEqual(fm.ping_pongs(migrations, 60.0), 1)


if __name__ == "__main__":
    unittest.main()
