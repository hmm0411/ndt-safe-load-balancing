"""The outcome dataset keeps the raw collection's split by repeat, unchanged."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from outcome_testdata import standard_raw, write_capacity, write_split
from src.experiments.migration import fixtures as fx
from src.twin.outcome import dataset as ds

CONFIG = ds.load_dataset_config(Path("configs/outcome/dataset.yaml"))


class SplitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.raw = self.tmp / "raw"
        self.splits = standard_raw(self.raw)
        self.capacity = ds.capacity_info(write_capacity(self.tmp / "cap.json"))["c_safe_pps"]

    def tearDown(self) -> None:
        fx.unlock_tree(self.tmp)
        shutil.rmtree(self.tmp)

    def test_split_is_taken_unchanged(self) -> None:
        split = ds.load_split_map(write_split(self.tmp / "split.json", self.splits))
        rows, noop, excluded = ds.build_rows(self.raw, split, self.capacity, CONFIG,
                                             ds.saturation_flags(self.raw))
        for row in rows + noop:
            self.assertEqual(row["split"], split[row["attempt_id"]])
        ds.write_dataset(self.tmp / "out", rows, noop, excluded, CONFIG, {})
        written = json.loads((self.tmp / "out" / "split_manifest.json").read_text())["splits"]
        self.assertEqual(written["validation"], ["mig-burst-burst_1-dst45-c1-c2-r04"])
        self.assertEqual(written["test"], ["mig-hot_switch-peak-dst75-c1-c2-r05"])
        self.assertEqual(len(written["train"]), 2)
        everywhere = [a for v in written.values() for a in v]
        self.assertEqual(len(everywhere), len(set(everywhere)))

    def test_attempt_without_split_is_an_error(self) -> None:
        splits = {k: [a for a in v if "r04" not in a] for k, v in self.splits.items()}
        split = ds.load_split_map(write_split(self.tmp / "split.json", splits))
        with self.assertRaises(ValueError):
            ds.build_rows(self.raw, split, self.capacity, CONFIG, ds.saturation_flags(self.raw))

    def test_attempt_in_two_splits_is_an_error(self) -> None:
        splits = dict(self.splits)
        splits["test"] = splits["test"] + ["mig-burst-burst_1-dst45-c1-c2-r04"]
        with self.assertRaises(ValueError):
            ds.load_split_map(write_split(self.tmp / "split.json", splits))


if __name__ == "__main__":
    unittest.main()
