import json
import tempfile
import unittest
from pathlib import Path

from src.twin.forecasting.loader import (
    ForecastDatasetLoadError,
    load_jsonl,
    load_official_forecast_runs,
)


class ForecastLoaderTests(unittest.TestCase):
    def test_load_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.jsonl"

            path.write_text(
                '{"a": 1}\n{"a": 2}\n',
                encoding="utf-8",
            )

            self.assertEqual(
                load_jsonl(path),
                [{"a": 1}, {"a": 2}],
            )

    def test_filters_invalid_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "stable-r01-s101"
            run_dir = root / run_id
            run_dir.mkdir()

            rows = [
                {
                    "run_id": run_id,
                    "snapshot_valid": True,
                },
                {
                    "run_id": run_id,
                    "snapshot_valid": False,
                },
            ]

            with (
                run_dir / "forecast_samples.jsonl"
            ).open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            loaded = load_official_forecast_runs(root)

            self.assertEqual(len(loaded), 1)
            self.assertTrue(loaded[0]["snapshot_valid"])

    def test_can_retain_invalid_snapshot_for_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "stable-r01-s101"
            run_dir = root / run_id
            run_dir.mkdir()

            rows = [
                {
                    "run_id": run_id,
                    "snapshot_valid": True,
                },
                {
                    "run_id": run_id,
                    "snapshot_valid": False,
                },
            ]

            with (
                run_dir / "forecast_samples.jsonl"
            ).open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            loaded = load_official_forecast_runs(
                root,
                require_valid_snapshot=False,
            )

            self.assertEqual(len(loaded), 2)

    def test_rejects_run_id_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = (
                root / "stable-r01-s101"
            )
            run_dir.mkdir()

            (
                run_dir / "forecast_samples.jsonl"
            ).write_text(
                json.dumps(
                    {
                        "run_id":
                            "stable-r02-s102",
                        "snapshot_valid": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(
                ForecastDatasetLoadError
            ):
                load_official_forecast_runs(root)


    def test_ignores_non_forecast_folders(self):
        # Capacity runs, archives and smoke runs share data/experiment_runs.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (
                "capacity-2c20s-fine-c1-30-r01",
                "_archive",
                "_smoke3",
                "stable-r01-s101",
            ):
                (root / name).mkdir()
            (root / "stable-r01-s101" / "forecast_samples.jsonl").write_text(
                json.dumps({"run_id": "stable-r01-s101", "snapshot_valid": True}) + "\n",
                encoding="utf-8",
            )

            loaded = load_official_forecast_runs(root)

            self.assertEqual([row["run_id"] for row in loaded], ["stable-r01-s101"])

    def test_expected_run_ids_limits_the_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for run_id in ("stable-r01-s101", "stable-r02-s102"):
                run_dir = root / run_id
                run_dir.mkdir()
                (run_dir / "forecast_samples.jsonl").write_text(
                    json.dumps({"run_id": run_id, "snapshot_valid": True}) + "\n",
                    encoding="utf-8",
                )

            loaded = load_official_forecast_runs(
                root, expected_run_ids={"stable-r02-s102"}
            )

            self.assertEqual([row["run_id"] for row in loaded], ["stable-r02-s102"])


if __name__ == "__main__":
    unittest.main()
