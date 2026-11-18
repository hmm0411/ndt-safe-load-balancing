from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.analysis import week15_evidence as w15


def touch(root: Path, rel: str, text: str = "x") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class Week15EvidenceTest(unittest.TestCase):
    def status(self, items: list[w15.Item], prefix: str) -> bool | None:
        return next(i.status for i in items if i.name.startswith(prefix))

    def test_empty_repo_is_all_missing_never_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            items = w15.evaluate(Path(tmp), lambda tag: False)
        self.assertTrue(items)
        self.assertTrue(all(i.status is None for i in items))

    def test_probe_scored_huong_2_still_passes_day3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            touch(root, "evidence/week15/probe_validation.json",
                  json.dumps({"decision": "Huong 2", "all_pass": False}))
            items = w15.evaluate(root, lambda tag: False)
        self.assertIs(self.status(items, "Day 3"), True)

    def test_capacity_needs_all_four_controllers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            touch(root, "configs/experiments/capacity_v5_rules.yaml")
            body = {"controllers": {c: {"c_safe_pps": 900.0} for c in ("c1", "c2", "c3")}}
            touch(root, "data/benchmarks/controller_capacity_v5.json", json.dumps(body))
            self.assertIs(self.status(w15.evaluate(root, lambda t: False), "Day 4: capacity-v5"),
                          False)
            body["controllers"]["c4"] = {"c_safe_pps": 910.0}
            touch(root, "data/benchmarks/controller_capacity_v5.json", json.dumps(body))
            self.assertIs(self.status(w15.evaluate(root, lambda t: False), "Day 4: capacity-v5"),
                          True)

    def test_tag_and_dod_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            items = w15.evaluate(root, lambda tag: tag == "capacity-v5")
            self.assertIs(self.status(items, "Day 4: tag"), True)
            text = w15.write_dod(root / "dod.txt", items)
        self.assertIn(f"1/{len(items)} PASS", text)


if __name__ == "__main__":
    unittest.main()
