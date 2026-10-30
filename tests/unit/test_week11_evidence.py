"""Week 11 evidence script: DoD grading reads only files, manual items need --confirm."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.analysis import week11_evidence as ev

SRC = Path("evidence/week11")


class DodTest(unittest.TestCase):
    def test_manual_items_missing_until_confirmed(self) -> None:
        items = dict((n, s) for n, s, _ in ev.evaluate(SRC, {}, None))
        for name in ev.MANUAL_ITEMS:
            self.assertIsNone(items[name], name)
        confirmed = dict((n, s) for n, s, _ in ev.evaluate(SRC, {ev.MANUAL_ITEMS[0]: "x"}, None))
        self.assertTrue(confirmed[ev.MANUAL_ITEMS[0]])

    def test_tag_item_follows_tag_state(self) -> None:
        name = f"Tag {ev.TAG} trên main, sau PR dev -> main"
        for state in (None, False, True):
            got = dict((n, s) for n, s, _ in ev.evaluate(SRC, {}, state))
            self.assertEqual(got[name], state)

    def test_empty_directory_has_no_pass_for_data_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            items = ev.evaluate(Path(tmp), {}, None)
            self.assertFalse(any(s for n, s, _ in items if n == "Thu chính thức"))

    def test_split_by_repeat_rejects_overlap(self) -> None:
        good = {"splits": {"train": {"repeats": ["r01"], "attempts": ["a-r01"]},
                           "test": {"repeats": ["r05"], "attempts": ["a-r05"]}}}
        bad = {"splits": {"train": {"repeats": ["r01"], "attempts": ["a-r05"]}}}
        self.assertTrue(ev.split_is_by_repeat(good))
        self.assertFalse(ev.split_is_by_repeat(bad))

    def test_check_files_flags_crlf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "x.csv").write_bytes(b"a,b\r\n1,2\r\n")
            self.assertTrue(any("CRLF" in p for p in ev.check_files(Path(tmp))))


if __name__ == "__main__":
    unittest.main()
