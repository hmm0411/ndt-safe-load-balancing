"""scripts/analysis/week12_evidence.py: Week 12 Definition of Done is graded from files only."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _load() -> Any:
    spec = importlib.util.spec_from_file_location(
        "week12_evidence_cli", ROOT / "scripts" / "analysis" / "week12_evidence.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _no_tags(tag: str) -> tuple[bool, bool]:
    return False, False


class Week12EvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "evidence"
        self.out.mkdir()
        self.mod = _load()

    def grade(self) -> dict[str, bool | None]:
        items = self.mod.evaluate(
            self.out, self.root / "no-artifact", self.root / "no-capacity.json", _no_tags,
        )
        return {name: status for name, status, _ in items}

    def put(self, name: str, data: Any) -> None:
        (self.out / name).write_text(json.dumps(data), encoding="utf-8")

    def test_empty_evidence_is_missing_not_fail(self) -> None:
        grades = self.grade()
        self.assertEqual(grades["Inference benchmark"], None)
        self.assertEqual(grades["Runtime what-if predictor"], None)
        self.assertEqual(grades["make ci"], None)
        self.assertEqual(grades["Artifact, model card đủ §46"], None)
        self.assertNotIn(False, [v for k, v in grades.items() if k != "Không tự migrate"])

    def test_prediction_has_no_safe_field(self) -> None:
        self.assertIs(self.grade()["Không tự migrate"], True)

    def test_inference_benchmark(self) -> None:
        cell = {"mean_ms": 2.5, "p95_ms": 2.7}
        self.put("inference_latency.json", {"repeats": 1000, "candidates": {"1": cell, "10": cell}})
        self.assertIs(self.grade()["Inference benchmark"], True)
        self.put("inference_latency.json", {"repeats": 100, "candidates": {"1": cell, "10": cell}})
        self.assertIs(self.grade()["Inference benchmark"], False)
        self.put("inference_latency.json", {"repeats": 1000, "candidates": {"1": cell}})
        self.assertIs(self.grade()["Inference benchmark"], False)

    def test_runtime_smoke_needs_six_ready(self) -> None:
        self.put("runtime_smoke.json", {"summary": {"attempts": 6, "ready": 6}})
        self.assertIs(self.grade()["Runtime what-if predictor"], True)
        self.put("runtime_smoke.json", {"summary": {"attempts": 6, "ready": 5}})
        self.assertIs(self.grade()["Runtime what-if predictor"], False)

    def test_margins_one_sided_from_validation(self) -> None:
        self.put("coverage_metrics.json", {})
        self.put("uncertainty_margin.json", {"m_U": 0.04, "m_R": 0.01, "split": "validation"})
        self.assertIs(self.grade()["m_U, m_R một phía, test coverage"], True)
        self.put("uncertainty_margin.json", {"m_U": 0.04, "m_R": -1, "split": "validation"})
        self.assertIs(self.grade()["m_U, m_R một phía, test coverage"], False)
        self.put("uncertainty_margin.json", {"m_U": 0.04, "m_R": 0.01, "split": "test"})
        self.assertIs(self.grade()["m_U, m_R một phía, test coverage"], False)

    def test_ci_needs_two_exit_zero(self) -> None:
        (self.out / "ci.txt").write_text("exit_code: 0\nexit_code: 0\n", encoding="utf-8")
        self.assertIs(self.grade()["make ci"], True)
        (self.out / "ci.txt").write_text("exit_code: 0\nexit_code: 2\n", encoding="utf-8")
        self.assertIs(self.grade()["make ci"], False)

    def test_disjoint_runs_and_processed_rows(self) -> None:
        ok, _ = self.mod.disjoint_runs({"train_runs": ["a"], "validation_runs": ["b"], "test_runs": ["c"]})
        self.assertIs(ok, True)
        bad, _ = self.mod.disjoint_runs({"train_runs": ["a"], "validation_runs": ["a"], "test_runs": ["c"]})
        self.assertIs(bad, False)
        unknown, _ = self.mod.disjoint_runs({"train_runs": 180})
        self.assertIsNone(unknown)
        self.assertIs(self.mod.processed_rows({"rows": 300, "one_row_per_attempt": True})[0], True)
        self.assertIs(self.mod.processed_rows({"rows": 299, "one_row_per_attempt": True})[0], False)
        self.assertIs(self.mod.processed_rows({"rows": 300, "one_row_per_attempt": False})[0], False)
        self.assertIsNone(self.mod.processed_rows({})[0])

    def test_selection_and_split(self) -> None:
        self.put("model_selection.json", {"frozen_sha256": "ab"})
        self.put("test_lock.json", {"status": "completed"})
        self.assertIs(self.grade()["Chọn trên validation, test một lần"], True)
        self.put("test_lock.json", {"status": "open"})
        self.assertIs(self.grade()["Chọn trên validation, test một lần"], False)

    def test_write_dod_and_file_check(self) -> None:
        text = self.mod.write_dod(self.out, [("a", True, "x"), ("b", None, "y"), ("c", False, "z")])
        self.assertIn("1/3 PASS", text)
        self.assertIn("MISSING", text)
        self.assertTrue((self.out / "week12_dod.txt").is_file())
        problems = self.mod.check_files(self.out)
        self.assertIn("figures/fig_w12_*.png: 0 of 8", problems)
        self.assertNotIn("missing file: week12_dod.txt", problems)


if __name__ == "__main__":
    unittest.main()
