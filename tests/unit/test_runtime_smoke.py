"""scripts/analysis/runtime_smoke.py: prediction vs measured value per smoke attempt."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from outcome_testdata import write_raw_attempt

ROOT = Path(__file__).resolve().parents[2]


def _load() -> Any:
    spec = importlib.util.spec_from_file_location(
        "runtime_smoke_cli", ROOT / "scripts" / "analysis" / "runtime_smoke.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ready(dst_u: float, margin: float = 0.05) -> dict[str, Any]:
    return {
        "status": "READY", "reason": None, "inference_time_ms": 2.5,
        "predicted_src_utilization": 0.64, "predicted_dst_utilization": dst_u,
        "predicted_worst_response_p95_ms": 0.80,
        "dst_utilization_margin": margin, "response_margin_ms": 0.05,
    }


class RuntimeSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.mod = _load()

    def attempt(self, name: str, record: dict[str, Any] | None) -> Path:
        write_raw_attempt(self.root, name, "r01")  # measured after window: src 0.64, dst 0.61
        path = self.root / name
        if record is not None:
            path.chmod(0o755)  # the writer seals every attempt directory read-only
            (path / "prediction.json").write_text(json.dumps(record), encoding="utf-8")
        return path

    def test_inside_and_outside_margin(self) -> None:
        self.attempt("a-in", {"http_status": 200, "http_ms": 3.0, "response": _ready(0.63)})
        self.attempt("a-out", {"http_status": 200, "http_ms": 3.0, "response": _ready(0.80)})
        rows = {r["attempt_id"]: r for r in self.mod.build(self.root)["attempts"]}
        self.assertTrue(rows["a-in"]["dst_utilization"]["within_margin"])
        self.assertFalse(rows["a-out"]["dst_utilization"]["within_margin"])
        self.assertAlmostEqual(rows["a-in"]["dst_utilization"]["measured"], 0.61)
        self.assertAlmostEqual(rows["a-in"]["dst_utilization"]["error"], 0.02)

    def test_not_ready_has_no_numbers(self) -> None:
        response = {"status": "NOT_READY", "reason": "INSUFFICIENT_HISTORY"}
        self.attempt("a-nr", {"http_status": 200, "response": response})
        row = self.mod.build(self.root)["attempts"][0]
        self.assertEqual(row["prediction_status"], "NOT_READY")
        self.assertEqual(row["reason"], "INSUFFICIENT_HISTORY")
        self.assertNotIn("dst_utilization", row)

    def test_failed_request_and_missing_file(self) -> None:
        self.attempt("a-fail", {"error": "ConnectionRefusedError()"})
        self.attempt("a-none", None)
        rows = {r["attempt_id"]: r for r in self.mod.build(self.root)["attempts"]}
        self.assertEqual(rows["a-fail"]["prediction_status"], "REQUEST_FAILED")
        self.assertEqual(rows["a-none"]["prediction_status"], "NO_PREDICTION_FILE")

    def test_cpu_p95_and_summary(self) -> None:
        self.attempt("a-1", {"http_status": 200, "response": _ready(0.63)})
        self.attempt("a-2", {"error": "boom"})
        result = self.mod.build(self.root)
        self.assertEqual(result["summary"]["attempts"], 2)
        self.assertEqual(result["summary"]["ready"], 1)
        cpu = result["attempts"][0]["cpu_p95_percent"]
        self.assertEqual(set(cpu), {"c1", "c2"})

    def test_main_writes_file_and_refuses_empty_root(self) -> None:
        self.attempt("a-1", {"http_status": 200, "response": _ready(0.63)})
        out = self.root / "out" / "runtime_smoke.json"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.mod.main(["--root", str(self.root), "--out", str(out)]), 0)
        self.assertEqual(json.loads(out.read_text())["summary"]["ready"], 1)
        with tempfile.TemporaryDirectory() as empty, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.mod.main(["--root", empty, "--out", str(out)]), 2)


if __name__ == "__main__":
    unittest.main()
