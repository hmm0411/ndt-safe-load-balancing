from __future__ import annotations

import importlib.util
import os
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
            tracker.log_candidate(
                "xgboost", {"max_depth": 3}, {"rmse": 0.2}, {"best_iteration": 7}, [0.5, 0.3, 0.2]
            )
            tracker.log_candidate("xgboost", {"max_depth": 5}, {"rmse": 0.1}, {"best_iteration": 9})
            tracker.log_metrics({"rmse": 0.1, "bad": float("nan")}, prefix="test_xgboost_")
            tracker.end()

            client = mlflow.MlflowClient(tracking_uri=uri)
            parent = client.get_run(tracker.run_id or "")
            self.assertEqual(parent.info.status, "FINISHED")
            self.assertEqual(parent.data.tags["dataset_version"], "test")
            self.assertIn("git_commit", parent.data.tags)
            self.assertEqual(parent.data.metrics["test_xgboost_rmse"], 0.1)
            self.assertNotIn("test_xgboost_bad", parent.data.metrics)
            best = client.get_metric_history(
                tracker.run_id or "", "grid_xgboost_best_validation_rmse"
            )
            self.assertEqual(
                [(m.step, m.value) for m in sorted(best, key=lambda m: m.step)],
                [(1, 0.2), (2, 0.1)],
            )

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
            curve = client.get_metric_history(
                children[1].info.run_id, "learning_curve_validation_rmse"
            )
            self.assertEqual(sorted(m.step for m in curve), [0, 1, 2])
            mlflow.set_tracking_uri(None)  # type: ignore[arg-type]

    def test_dataset_registry_alias_and_pointer(self) -> None:
        import mlflow
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LinearRegression

        with tempfile.TemporaryDirectory() as tmp:
            # Logged model files go to ./mlruns of the working directory;
            # keep them inside the temporary directory.
            previous = os.getcwd()
            os.chdir(tmp)
            self.addCleanup(os.chdir, previous)
            uri = f"sqlite:///{Path(tmp) / 'mlflow.db'}"
            tracker = Tracker(
                enabled=True,
                tracking_uri=uri,
                experiment="unit-test-registry",
                register_models=True,
                registered_names={"linear": "unit-linear"},
            )
            tracker.start({})
            frame = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
            tracker.log_dataset(frame, "train", "training", source=str(Path(tmp) / "d.csv"), targets="y")
            x = np.array([[0.0], [1.0], [2.0]])
            tracker.log_model(
                "sklearn",
                LinearRegression().fit(x, [0.0, 1.0, 2.0]),
                "linear",
                x[:2],
                version_tags={"dataset_version": "unit"},
            )
            pointer = Path(tmp) / "mlflow_run.json"
            tracker.write_pointer(pointer)
            tracker.end()

            client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
            version = client.get_model_version_by_alias("unit-linear", "challenger")
            self.assertEqual(str(version.version), "1")
            self.assertEqual(version.tags["dataset_version"], "unit")
            run = client.get_run(tracker.run_id or "")
            assert run.inputs is not None
            self.assertEqual([d.dataset.name for d in run.inputs.dataset_inputs], ["train"])
            self.assertIn(tracker.run_id or "", pointer.read_text())
            self.assertEqual(tracker.registered_versions, {"linear": "1"})
            mlflow.set_tracking_uri(None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
