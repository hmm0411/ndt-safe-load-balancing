from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from src.experiments.workloads.capacity_mapper import CapacityWorkloadMapper
from src.experiments.workloads.scale import ENV, parse_scales, scale_for

RUNS = Path("data/benchmarks/capacity_runs.csv")
CAP = Path("data/benchmarks/controller_capacity.json")


class ScaleTest(unittest.TestCase):
    def test_parse(self) -> None:
        self.assertEqual(parse_scales("c1=1.2, c3=1.1"), {"c1": 1.2, "c3": 1.1})
        self.assertEqual(parse_scales(None), {})
        for bad in ("c1", "c1=x", "c1=0", "c1=1.2,c1=1.1", "=1.2", "c1=nan"):
            with self.assertRaises(ValueError):
                parse_scales(bad)
        self.assertEqual(scale_for("c2", {ENV: "c1=1.2"}), 1.0)

    def test_mapper_scales_only_the_named_controller(self) -> None:
        mapper = CapacityWorkloadMapper(RUNS, CAP)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV, None)
            base_c1 = mapper.offered_rate_for_utilization("c1", 0.6)
            base_c2 = mapper.offered_rate_for_utilization("c2", 0.6)
            via_target = mapper.offered_rate_for_utilization("c1", 0.72)
        with mock.patch.dict(os.environ, {ENV: "c1=1.2"}):
            self.assertAlmostEqual(mapper.offered_rate_for_utilization("c1", 0.6), via_target)
            self.assertAlmostEqual(mapper.offered_rate_for_utilization("c2", 0.6), base_c2)
        self.assertNotAlmostEqual(base_c1, via_target)


if __name__ == "__main__":
    unittest.main()
