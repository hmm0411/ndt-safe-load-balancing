"""Processed outcome dataset: one row per eligible attempt, NO_OP aside, nothing imputed."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from outcome_testdata import standard_raw, write_capacity, write_split
from src.experiments.migration import fixtures as fx
from src.experiments.migration.saturation import write_outputs
from src.twin.outcome import dataset as ds
from src.twin.outcome.dataset import DATASET_COLUMNS, NOOP_COLUMNS

CONFIG = ds.load_dataset_config(Path("configs/outcome/dataset.yaml"))


class DatasetBuilderTest(unittest.TestCase):
    tmp: Path
    raw: Path
    split_path: Path
    capacity_path: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp())
        cls.raw = cls.tmp / "raw"
        cls.split_path = write_split(cls.tmp / "split.json", standard_raw(cls.raw))
        cls.capacity_path = write_capacity(cls.tmp / "cap.json")

    @classmethod
    def tearDownClass(cls) -> None:
        fx.unlock_tree(cls.tmp)
        shutil.rmtree(cls.tmp)

    def build(self) -> tuple[list[dict], list[dict], dict[str, str]]:
        return ds.build_rows(self.raw, ds.load_split_map(self.split_path),
                             ds.capacity_info(self.capacity_path)["c_safe_pps"], CONFIG,
                             ds.saturation_flags(self.raw))

    def test_rows(self) -> None:
        rows, noop, excluded = self.build()
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({r["attempt_id"] for r in rows}), 4)
        self.assertEqual([r["attempt_id"] for r in noop], ["noop-stable-middle-dst45-c1-c2-r01"])
        self.assertEqual(list(excluded), ["mig-stable-early-dst45-c1-c2-r02"])
        self.assertIn("transaction_restored", excluded["mig-stable-early-dst45-c1-c2-r02"])
        hot = next(r for r in rows if r["workload"] == "hot_switch")
        self.assertTrue(hot["overload_after"])
        self.assertAlmostEqual(hot["u_dst_post"], 1.08)
        self.assertEqual(hot["switch_share_group"], "high")
        self.assertEqual(noop[0]["analytical_u_src_post"], noop[0]["src_u_pre_mean"])

    def test_capacity_mismatch_is_an_error(self) -> None:
        wrong = write_capacity(self.tmp / "cap-wrong.json", c1=933.0, c2=933.0)
        with self.assertRaises(ValueError):
            ds.build_rows(self.raw, ds.load_split_map(self.split_path),
                          ds.capacity_info(wrong)["c_safe_pps"], CONFIG,
                          ds.saturation_flags(self.raw))

    def test_write_and_read_back(self) -> None:
        rows, noop, excluded = self.build()
        out = self.tmp / "out-write"
        result = ds.write_dataset(out, rows, noop, excluded, CONFIG, {"capacity_version": "x"})
        self.assertEqual(result["manifest"]["rows_by_split"],
                         {"train": 2, "validation": 1, "test": 1})
        self.assertTrue(result["report"]["one_row_per_attempt"])
        with (out / "outcome_dataset.csv").open(newline="") as handle:
            self.assertEqual(next(csv.reader(handle)), list(DATASET_COLUMNS))
        with (out / "noop_reference.csv").open(newline="") as handle:
            self.assertEqual(next(csv.reader(handle)), list(NOOP_COLUMNS))
        back = ds.load_dataset(out / "outcome_dataset.csv")
        self.assertEqual([r["split"] for r in back], ["train", "train", "validation", "test"])
        self.assertIs(back[-1]["overload_after"], True)
        self.assertIsInstance(back[0]["src_u_mean"], float)
        with self.assertRaises(FileExistsError):
            ds.write_dataset(out, rows, noop, excluded, CONFIG, {})

    def test_cli_build_and_analytical_evaluation(self) -> None:
        evidence = self.tmp / "evidence"
        write_outputs(self.raw, evidence)
        out = self.tmp / "out-cli"
        build = subprocess.run(
            [sys.executable, "scripts/experiments/build_outcome_dataset.py",
             "--raw-root", str(self.raw), "--split", str(self.split_path),
             "--capacity", str(self.capacity_path),
             "--saturation", str(evidence / "attempt_saturation.csv"), "--out", str(out)],
            capture_output=True, text=True, check=False)
        self.assertEqual(build.returncode, 0, build.stderr)
        self.assertIn("4 rows (train 2, validation 1, test 1), NO_OP reference 1", build.stdout)
        manifest = json.loads((out / "dataset_manifest.json").read_text())
        self.assertTrue(manifest["capacity_version"].startswith("cap.json@sha256:"))
        report = self.tmp / "analytical.csv"
        evaluate = subprocess.run(
            [sys.executable, "scripts/experiments/evaluate_outcome_analytical.py",
             "--dataset", str(out), "--split", "train", "--out", str(report)],
            capture_output=True, text=True, check=False)
        self.assertEqual(evaluate.returncode, 0, evaluate.stderr)
        self.assertIn("selection score S", evaluate.stdout)
        refused = subprocess.run(
            [sys.executable, "scripts/experiments/evaluate_outcome_analytical.py",
             "--dataset", str(out), "--split", "test"], capture_output=True, text=True,
            check=False)
        self.assertNotEqual(refused.returncode, 0)


if __name__ == "__main__":
    unittest.main()
