"""Runner option --predict-url: a what-if before each MIGRATE, prediction.json beside the attempt."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from outcome_testdata import write_raw_attempt
from test_migration_runner import FakeWorld, make_plan

from src.experiments.migration import runner
from src.experiments.migration.writer import read_checksums

ROOT = Path(__file__).resolve().parents[2]
URL = "http://127.0.0.1:9000/api/v1/twin/outcome/predict"


def _load_cli() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_migration_outcome_cli", ROOT / "scripts" / "experiments" / "run_migration_outcome.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _counter(called: list[int]) -> Any:
    def predict(switch: str, source: str, target: str) -> dict[str, Any]:
        called.append(1)
        return {}

    return predict


class CollectWithPredictionTest(unittest.TestCase):
    def collect(self, world: FakeWorld, plan: Any = None, predict: Any = None) -> Any:
        ctx = world.context()
        ctx.predict = predict
        return runner.collect_attempt(
            plan or world.plan, ctx, failure="none", duration=60, window=5, post_timeout=60
        )

    def test_prediction_is_asked_once_before_the_migration(self) -> None:
        world = FakeWorld(make_plan())
        calls: list[tuple[Any, ...]] = []

        def predict(switch: str, source: str, target: str) -> dict[str, Any]:
            calls.append(("predict", switch, source, target, len(world.sent)))
            return {"response": {"status": "READY"}}

        got = self.collect(world, predict=predict)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2:], ("c1", "c2", 0))  # nothing had been sent yet
        self.assertEqual(calls[0][1], got.action.switch_id)
        self.assertEqual(got.prediction, {"response": {"status": "READY"}})
        self.assertEqual(got.transaction.status, "COMMITTED")

    def test_without_predict_nothing_changes(self) -> None:
        got = self.collect(FakeWorld(make_plan()))
        self.assertIsNone(got.prediction)

    def test_no_op_is_never_predicted(self) -> None:
        plan = make_plan(action_type="NO_OP", attempt_id="noop-stable-early-dst45-c1-c2-r01")
        called: list[int] = []
        got = self.collect(FakeWorld(plan), predict=_counter(called))
        self.assertEqual(called, [])
        self.assertIsNone(got.prediction)

    def test_skipped_migration_is_not_predicted(self) -> None:
        called: list[int] = []
        got = self.collect(FakeWorld(make_plan(), phase="late"), predict=_counter(called))
        self.assertEqual(got.transaction.status, "SKIPPED")
        self.assertEqual(called, [])


class RequestPredictionTest(unittest.TestCase):
    def test_success_is_recorded(self) -> None:
        with mock.patch.object(runner, "http_post_json", return_value=(200, {"status": "READY"})) as post:
            record = runner.request_prediction(URL, "s3", "c1", "c2")
        post.assert_called_once()
        self.assertEqual(post.call_args.args[1],
                         {"switch_id": "s3", "source_controller": "c1", "target_controller": "c2"})
        self.assertEqual((record["http_status"], record["response"]), (200, {"status": "READY"}))
        self.assertEqual(record["url"], URL)
        self.assertGreaterEqual(record["http_ms"], 0.0)

    def test_failure_is_recorded_not_raised(self) -> None:
        with mock.patch.object(runner, "http_post_json", side_effect=OSError("refused")):
            record = runner.request_prediction(URL, "s3", "c1", "c2")
        self.assertIn("refused", record["error"])
        self.assertNotIn("response", record)

    def test_http_error_status_is_recorded(self) -> None:
        with mock.patch.object(runner, "http_post_json", return_value=(400, {"detail": "bad"})):
            record = runner.request_prediction(URL, "s3", "c1", "c1")
        self.assertEqual(record["http_status"], 400)


class WriteAttemptPredictionTest(unittest.TestCase):
    def test_prediction_file_is_written_checksummed_and_valid(self) -> None:
        from src.experiments.migration import writer

        original = writer.write_attempt
        seen: dict[str, Any] = {}

        def with_prediction(*args: Any, **kwargs: Any) -> Any:
            kwargs["prediction"] = {"response": {"status": "READY"}}
            seen["result"] = original(*args, **kwargs)
            return seen["result"]

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("outcome_testdata.write_attempt", with_prediction):
                write_raw_attempt(Path(tmp), "att-1", "r01")
            attempt = Path(tmp) / "att-1"
            body = json.loads((attempt / "prediction.json").read_text(encoding="utf-8"))
            self.assertEqual(body, {"response": {"status": "READY"}})
            self.assertIn("prediction.json", read_checksums(attempt))
            self.assertTrue(seen["result"].validation["valid"])

    def test_no_prediction_means_no_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_raw_attempt(Path(tmp), "att-2", "r01")
            attempt = Path(tmp) / "att-2"
            self.assertFalse((attempt / "prediction.json").exists())
            self.assertNotIn("prediction.json", read_checksums(attempt))


class PredictUrlGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cli = _load_cli()

    def problem(self, url: str, output: str | None) -> str | None:
        return self.cli.predict_url_problem(
            url, output, "data/experiment_runs/migration-outcome-v2"
        )

    def test_good_smoke_setup(self) -> None:
        out = "data/experiment_runs/migration-outcome-smoke-w12"
        self.assertIsNone(self.problem(URL, out))

    def test_needs_explicit_output_root(self) -> None:
        self.assertIn("--output-root", self.problem(URL, None) or "")

    def test_never_the_raw_collection(self) -> None:
        for out in ("data/experiment_runs/migration-outcome-v2",
                    "/abs/path/migration-outcome-v2", "data/experiment_runs/raw-v2"):
            self.assertIn("raw collection", self.problem(URL, out) or "", out)

    def test_url_must_be_the_predict_endpoint(self) -> None:
        out = "data/experiment_runs/migration-outcome-smoke-w12"
        for url in ("http://127.0.0.1:9000/api/v1/migrations", "ftp://x/twin/outcome/predict",
                    "not a url", "http:///api/v1/twin/outcome/predict"):
            self.assertIn("--predict-url must be", self.problem(url, out) or "", url)

    def test_main_refuses_before_doing_anything(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()) as err:
            code = self.cli.main(["--dry-run", "--predict-url", URL])
        self.assertEqual(code, 2)
        self.assertIn("--output-root", err.getvalue())


if __name__ == "__main__":
    unittest.main()
