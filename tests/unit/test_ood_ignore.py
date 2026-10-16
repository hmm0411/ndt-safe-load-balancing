"""OOD gate ignore list (process_memory_rss_mb finding, 16 Oct)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

from src.orchestrator.current_state import CurrentStateStore
from src.orchestrator.forecast_service import ForecastService
from src.twin.forecasting.fixtures import twin_state, write_persistence_artifact
from src.twin.forecasting.ood import FeatureRanges, OODGate
from src.twin.forecasting.predictor import RuntimeForecaster

FORECAST_V1 = Path("data/models/forecasting/forecast-v1")
RUNTIME_YAML = Path("configs/forecasting/runtime.yaml")


def ranges() -> FeatureRanges:
    return FeatureRanges(("a", "b"), (0.0, 0.0), (1.0, 1.0), 0.0, 3, ())


class OODGateIgnoreTest(unittest.TestCase):
    def test_ignored_feature_never_turns_the_gate_on(self) -> None:
        gate = OODGate(ranges(), ignore=("a",))
        decisions = [gate.update("c1", [5.0, 0.5]) for _ in range(5)]
        self.assertFalse(any(d.active for d in decisions))
        self.assertEqual(decisions[-1].features, ())

    def test_other_features_still_count(self) -> None:
        gate = OODGate(ranges(), ignore=("a",))
        decisions = [gate.update("c1", [5.0, 5.0]) for _ in range(3)]
        self.assertTrue(decisions[-1].active)
        self.assertEqual(decisions[-1].features, ("b",))

    def test_unknown_name_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            OODGate(ranges(), ignore=("process_memory_rss_mb_typo",))


class ForecasterIgnoreTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name) / "forecast-test"
        self.dir.mkdir()
        write_persistence_artifact(self.dir)

    def last_c2(self, fc: RuntimeForecaster, **kwargs: Any) -> Any:
        predictions: list[Any] = []
        for s in range(11):
            fc.update(twin_state(float(s)))
        for s in range(11, 14):
            predictions = fc.update(twin_state(float(s), **kwargs))
        return {p.controller_id: p for p in predictions}["c2"]

    def test_ignore_reaches_the_gate(self) -> None:
        ignored = self.last_c2(
            RuntimeForecaster(self.dir, ood_ignore=("managed_switch_count",)), switch_count=9
        )
        watched = self.last_c2(RuntimeForecaster(self.dir), switch_count=9)
        self.assertFalse(ignored.ood_fallback)
        self.assertTrue(watched.ood_fallback)

    def test_service_reads_the_list_and_reports_it(self) -> None:
        store = CurrentStateStore()
        config = {"enabled": True, "model_dir": str(self.dir),
                  "ood_ignore_features": ["managed_switch_count"]}
        service = ForecastService.from_config(config, Path("."), store)
        for s in range(11):
            service.on_twin_state(twin_state(float(s)))
        self.assertEqual(service.payload()["ood_ignored_features"], ["managed_switch_count"])


@unittest.skipUnless((FORECAST_V1 / "model_card.json").is_file(), "forecast-v1 not present")
class ForecastV1MemoryTest(unittest.TestCase):
    """The fixture's 120 MB is outside forecast-v1's 56.6-57.1 MB, like a
    restarted controller (~47 MB) on the testbed."""

    def run_14(self, ignore: tuple[str, ...]) -> list[Any]:
        fc = RuntimeForecaster(FORECAST_V1, ood_ignore=ignore)
        predictions: list[Any] = []
        for s in range(14):
            predictions = fc.update(twin_state(float(s), u=(0.58, 0.58)))
        return predictions

    def test_runtime_yaml_ignores_memory(self) -> None:
        config = yaml.safe_load(RUNTIME_YAML.read_text(encoding="utf-8"))
        self.assertIn("process_memory_rss_mb", config["ood_ignore_features"])
        watched = self.run_14(())
        ignored = self.run_14(tuple(config["ood_ignore_features"]))
        # Other fixture values may be out of range too; only memory is checked.
        self.assertTrue(all(p.ood_fallback for p in watched))
        self.assertIn("process_memory_rss_mb", watched[0].ood_features)
        for p in ignored:
            self.assertNotIn("process_memory_rss_mb", p.ood_features)


if __name__ == "__main__":
    unittest.main()
