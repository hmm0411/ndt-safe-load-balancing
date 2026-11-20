from __future__ import annotations

import unittest
from unittest import mock

from src.experiments.capacity import capacity_runner as cr


class ReclearTest(unittest.TestCase):
    def test_flows_that_reappear_are_deleted_again(self) -> None:
        counts = iter([2, 2, 0, 0])  # s1, s2 still hold 2 flows, then none
        with mock.patch.object(cr, "count_cookie_flows", side_effect=lambda s, c: next(counts)), \
                mock.patch.object(cr, "clear_cookie_flows") as clear, \
                mock.patch.object(cr.time, "sleep"):
            cr.wait_for_cookie_absent(["s1", "s2"], "0x10", 5.0, reclear=True)
        clear.assert_called_once_with(["s1", "s2"], "0x10")

    def test_without_reclear_nothing_is_deleted_and_it_times_out(self) -> None:
        with mock.patch.object(cr, "count_cookie_flows", return_value=2), \
                mock.patch.object(cr, "clear_cookie_flows") as clear, \
                mock.patch.object(cr.time, "sleep"), \
                mock.patch.object(cr.time, "monotonic", side_effect=[0.0, 0.0, 10.0]):
            with self.assertRaisesRegex(RuntimeError, "still present"):
                cr.wait_for_cookie_absent(["s1"], "0x10", 5.0)
        clear.assert_not_called()


if __name__ == "__main__":
    unittest.main()
