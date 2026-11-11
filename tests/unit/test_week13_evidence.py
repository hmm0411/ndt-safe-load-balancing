from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.analysis import week13_evidence as w13


class Week13EvidenceTest(unittest.TestCase):
    def test_empty_tree_is_all_missing_never_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            items = w13.evaluate(Path(tmp), lambda tag: False)
            self.assertTrue(items)
            self.assertTrue(all(i.status is None for i in items))
            text = w13.write_dod(Path(tmp) / "dod.txt", items)
            self.assertIn(f"0/{len(items)} PASS", text)

    def test_repo_items_that_already_exist_pass(self) -> None:
        items = {i.name: i for i in w13.evaluate(w13.ROOT, lambda tag: True)}
        for name in ("Candidate chỉ sinh khi có risk, không trùng",
                     "Safety Gate 11 check, boundary inclusive",
                     "Ranking: policy có version, tie-break, NO_SAFE_ACTION",
                     "Tag ranking-policy-v1"):
            self.assertIs(items[name].status, True, name)


if __name__ == "__main__":
    unittest.main()
