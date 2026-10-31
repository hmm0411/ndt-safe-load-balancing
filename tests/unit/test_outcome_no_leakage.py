"""No post-action information, id or label can become a model feature."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from outcome_testdata import standard_raw, write_capacity, write_raw_attempt, write_split
from src.experiments.migration import fixtures as fx
from src.twin.outcome import dataset as ds
from src.twin.outcome.features import CANDIDATE_FEATURES, FORBIDDEN_FEATURES, is_forbidden
from src.twin.outcome.targets import ALL_TARGETS

CONFIG = ds.load_dataset_config(Path("configs/outcome/dataset.yaml"))


class NoLeakageTest(unittest.TestCase):
    def test_candidates_are_allowed(self) -> None:
        self.assertEqual([n for n in CANDIDATE_FEATURES if is_forbidden(n)], [])
        for name in ALL_TARGETS + ("settle_s", "transaction_status", "workload", "switch_id",
                                   "saturated_after", "quality_score"):
            self.assertTrue(is_forbidden(name), name)
        self.assertIn("transition_hard_gate_blocked", FORBIDDEN_FEATURES)

    def test_features_do_not_depend_on_the_post_window(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        try:
            rows = {}
            for after in ((0.64, 0.61), (0.30, 1.20)):
                raw = tmp / f"raw-{after[1]}"
                write_raw_attempt(raw, "mig-stable-middle-dst45-c1-c2-r01", "r01", after=after)
                split = {"mig-stable-middle-dst45-c1-c2-r01": "train"}
                built, _, _ = ds.build_rows(raw, split, {"c1": fx.SAFE_CAPACITY,
                                                         "c2": fx.SAFE_CAPACITY},
                                            CONFIG, ds.saturation_flags(raw))
                rows[after] = built[0]
            a, b = rows.values()
            self.assertEqual({k: a[k] for k in CANDIDATE_FEATURES},
                             {k: b[k] for k in CANDIDATE_FEATURES})
            self.assertNotEqual(a["u_dst_post"], b["u_dst_post"])
        finally:
            fx.unlock_tree(tmp)
            shutil.rmtree(tmp)

    def test_written_schema_has_no_forbidden_model_feature(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        try:
            raw = tmp / "raw"
            split = ds.load_split_map(write_split(tmp / "split.json", standard_raw(raw)))
            capacity = ds.capacity_info(write_capacity(tmp / "cap.json"))["c_safe_pps"]
            rows, noop, excluded = ds.build_rows(raw, split, capacity, CONFIG,
                                                 ds.saturation_flags(raw))
            ds.write_dataset(tmp / "out", rows, noop, excluded, CONFIG, {})
            schema = json.loads((tmp / "out" / "feature_schema.json").read_text())
            self.assertTrue(schema["model_features"])
            self.assertEqual([n for n in schema["model_features"] if is_forbidden(n)], [])
            self.assertTrue(set(schema["model_features"]) <= set(CANDIDATE_FEATURES))
        finally:
            fx.unlock_tree(tmp)
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
