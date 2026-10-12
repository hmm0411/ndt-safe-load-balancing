import math
import unittest

import importlib.util

ML_STACK = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "sklearn", "xgboost")
)

FEATURES = ["utilization", "util_delta_1"]


def _records(count: int, offset: float = 0.0, slope: float = 0.01):
    """A ramp: U(t+h) = U(t) + 3 * slope, so the true change is constant."""
    rows = []
    for index in range(count):
        utilization = 0.4 + offset + slope * index
        rows.append(
            {
                "utilization": utilization,
                "util_delta_1": slope,
                "future_utilization": utilization + 3 * slope,
            }
        )
    return rows


@unittest.skipIf(not ML_STACK, "numpy/scikit-learn/xgboost not installed")
class TreeModelTests(unittest.TestCase):
    def setUp(self):
        self.train = _records(120)
        self.validation = _records(40, offset=0.05)

    def test_random_forest_selects_on_validation(self):
        from src.twin.forecasting.tree_models import fit_random_forest

        model, table = fit_random_forest(
            self.train,
            self.validation,
            FEATURES,
            grid={"max_depth": (2, 4), "min_samples_leaf": (5,), "max_features": (1.0,)},
        )

        self.assertEqual(len(table), 2)
        best = min(table, key=lambda row: row["rmse"])
        self.assertEqual(model.params, best["params"])

    def test_forecast_is_anchored_at_current_load(self):
        # Trees cannot extrapolate a level, but a change target follows the
        # current load to levels never seen in training (the hot-switch case).
        from src.twin.forecasting.tree_models import fit_random_forest

        model, _ = fit_random_forest(
            self.train,
            self.validation,
            FEATURES,
            grid={"max_depth": (4,), "min_samples_leaf": (5,), "max_features": (1.0,)},
        )

        far_away = _records(10, offset=-0.35)  # U ~ 0.05, below anything trained
        predictions = model.predict(far_away)
        errors = [
            abs(prediction - row["future_utilization"])
            for prediction, row in zip(predictions, far_away)
        ]

        self.assertLess(max(errors), 0.02)

    def test_xgboost_records_best_iteration(self):
        from src.twin.forecasting.tree_models import fit_xgboost

        model, table = fit_xgboost(
            self.train,
            self.validation,
            FEATURES,
            grid={"max_depth": (2,), "learning_rate": (0.1,), "min_child_weight": (1,)},
        )

        self.assertEqual(len(table), 1)
        self.assertGreaterEqual(model.extra["best_iteration"], 0)
        self.assertTrue(all(math.isfinite(v) for v in model.predict(self.validation)))

    def test_feature_importance_names_every_feature(self):
        from src.twin.forecasting.tree_models import fit_random_forest

        model, _ = fit_random_forest(
            self.train,
            self.validation,
            FEATURES,
            grid={"max_depth": (3,), "min_samples_leaf": (5,), "max_features": (1.0,)},
        )

        self.assertEqual(set(model.feature_importance()), set(FEATURES))

    def test_empty_validation_is_rejected(self):
        from src.twin.forecasting.tree_models import fit_random_forest

        with self.assertRaises(ValueError):
            fit_random_forest(self.train, [], FEATURES)


if __name__ == "__main__":
    unittest.main()
