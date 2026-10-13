from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from src.twin.forecasting.tracking import Tracker, _finite, _flat

HAS_MLFLOW = importlib.util.find_spec("mlflow") is not None


class TrackerHelpersTests(unittest.TestCase):
    def test_flat_stringifies_none(self) -> None:
        self.assertEqual(
            _flat({"max_depth": None, "lr": 0.1}, prefix="rf."),
            {"rf.max_depth": "None", "rf.lr": "0.1"},
        )

    def test_finite_drops_nan_inf_bool_and_text(self) -> None:
        self.assertEqual(
            _finite({"a": 1, "b": float("nan"), "c": float("inf"), "d": True, "e": "x"}),
            {"a": 1.0},
        )

    def test_disabled_tracker_is_noop(self) -> None:
        tracker = Tracker(enabled=False)
        tracker.start({"k": "v"})
        tracker.log_params({"a": 1})
        tracker.log_metrics({"rmse": 0.1})
        tracker.log_candidate("random_forest", {"max_depth": 3}, {"rmse": 0.1}, {})
        tracker.log_model("sklearn", object(), "random_forest", None)
        tracker.end()
        self.assertIsNone(tracker.run_id)


@unittest.skipUnless(HAS_MLFLOW, "mlflow not installed")
class TrackerMlflowTests(unittest.TestCase):
    def test_parent_run_with_candidate_children(self) -> None:
        import mlflow

        with tempfile.TemporaryDirectory() as tmp:
            uri = f"sqlite:///{Path(tmp) / 'mlflow.db'}"
            tracker = Tracker(
                enabled=True,
                tracking_uri=uri,
                experiment="unit-test",
                run_name="parent",
            )
            tracker.start({"dataset_version": "test"})
            tracker.log_candidate("xgboost", {"max_depth": 3}, {"rmse": 0.2}, {"best_iteration": 7})
            tracker.log_candidate("xgboost", {"max_depth": 5}, {"rmse": 0.1}, {"best_iteration": 9})
            tracker.log_metrics({"rmse": 0.1, "bad": float("nan")}, prefix="test_xgboost_")
            tracker.end()

            client = mlflow.MlflowClient(tracking_uri=uri)
            parent = client.get_run(tracker.run_id or "")
            self.assertEqual(parent.info.status, "FINISHED")
            self.assertEqual(parent.data.tags["dataset_version"], "test")
            self.assertIn("git_commit", parent.data.tags)
            self.assertEqual(parent.data.metrics, {"test_xgboost_rmse": 0.1})

            children = client.search_runs(
                [parent.info.experiment_id],
                filter_string=f"tags.mlflow.parentRunId = '{tracker.run_id}'",
                order_by=["metrics.validation_rmse ASC"],
            )
            self.assertEqual(
                [c.data.tags["mlflow.runName"] for c in children],
                ["xgboost #2", "xgboost #1"],
            )
            self.assertEqual(children[0].data.metrics["best_iteration"], 9.0)
            mlflow.set_tracking_uri(None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
