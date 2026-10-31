"""Benchmark script of the runtime what-if predictor (Week 12, Day 5)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_outcome_predictor import _write_artifact

from scripts.experiments import benchmark_outcome_inference as bench
from src.orchestrator.outcome_service import capacity_version

CAPACITY = bench.ROOT / "data" / "benchmarks" / "controller_capacity.json"


class BenchmarkOutcomeInferenceTest(unittest.TestCase):
    def _artifact(self, tmp: str) -> Path:
        return _write_artifact(Path(tmp), capacity_version=capacity_version(CAPACITY))

    def test_measure_one_and_ten_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            predictor, state = bench.build_predictor(self._artifact(tmp), CAPACITY)
            one = bench.measure(predictor, state, 1, warmup=2, repeats=5)
            ten = bench.measure(predictor, state, 10, warmup=2, repeats=5)
        for stats in (one, ten):
            self.assertEqual(stats["repeats"], 5.0)
            self.assertGreater(stats["p95_ms"], 0.0)
            self.assertGreaterEqual(stats["p95_ms"], stats["p50_ms"])
        self.assertAlmostEqual(ten["ms_per_candidate_mean"], ten["mean_ms"] / 10)

    def test_candidates_outside_1_to_10_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            predictor, state = bench.build_predictor(self._artifact(tmp), CAPACITY)
            for bad in (0, 11):
                with self.assertRaises(ValueError):
                    bench.measure(predictor, state, bad, warmup=0, repeats=1)

    def test_no_artifact_means_no_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as caught:
                bench.build_predictor(Path(tmp) / "missing", CAPACITY)
            self.assertIn("ARTIFACT_MISSING", str(caught.exception))

    def test_capacity_version_mismatch_means_no_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_artifact(Path(tmp), capacity_version="controller_capacity.json@sha256:0")
            with self.assertRaises(SystemExit) as caught:
                bench.build_predictor(root, CAPACITY)
            self.assertIn("CAPACITY_VERSION_MISMATCH", str(caught.exception))

    def test_fewer_than_1000_repeats_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            bench.main(["--repeats", "10"])

    def test_main_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact"
            artifact.mkdir()
            _write_artifact(artifact, capacity_version=capacity_version(CAPACITY))
            out = Path(tmp) / "out" / "latency.json"
            with mock.patch.object(bench, "MIN_REPEATS", 5):  # the real minimum is 1000
                bench.main([
                    "--artifact-dir", str(artifact), "--output", str(out),
                    "--warmup", "1", "--repeats", "5", "--candidates", "1", "10",
                ])
            body = json.loads(out.read_text(encoding="utf-8"))
        self.assertTrue(body["model_preloaded"])
        self.assertEqual(sorted(body["candidates"]), ["1", "10"])
        self.assertEqual(body["repeats"], 5)
        self.assertEqual(body["model_version"], "post-migration-v1")
        self.assertIn("p95_ms", body["candidates"]["10"])

    def test_minimum_is_1000(self) -> None:
        self.assertEqual(bench.MIN_REPEATS, 1000)


if __name__ == "__main__":
    unittest.main()
