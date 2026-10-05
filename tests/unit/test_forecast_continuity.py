import math
import unittest
from datetime import datetime, timedelta, timezone

from src.twin.forecasting.continuity import (
    assign_segments,
    count_windows,
    segment_lengths,
)


class ForecastContinuityTests(unittest.TestCase):
    def test_continuous_sequence(self):
        self.assertEqual(
            assign_segments([1, 2, 3], 2.5),
            [0, 0, 0],
        )

    def test_gap_starts_new_segment(self):
        self.assertEqual(
            assign_segments([1, 2, 7], 2.5),
            [0, 0, 1],
        )

    def test_multiple_gaps(self):
        self.assertEqual(
            assign_segments([1, 2, 7, 8, 20], 2.5),
            [0, 0, 1, 1, 2],
        )

    def test_empty_input(self):
        self.assertEqual(assign_segments([], 2.5), [])

    def test_duplicate_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            assign_segments([1, 2, 2], 2.5)

    def test_backwards_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            assign_segments([1, 3, 2], 2.5)

    def test_invalid_gap_threshold(self):
        with self.assertRaises(ValueError):
            assign_segments([1, 2], 0)

    def test_iso_timestamps(self):
        timestamps = [
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:01+00:00",
            "2026-01-01T00:00:05+00:00",
        ]

        self.assertEqual(
            assign_segments(timestamps, 2.5),
            [0, 0, 1],
        )

    def test_t1_t2_t3_t8_t9_is_not_five_consecutive_observations(self):
        # The task contract: a 5 s hole must break the sequence.
        self.assertEqual(
            assign_segments([1, 2, 3, 8, 9], 2.5),
            [0, 0, 0, 1, 1],
        )

    def test_gap_equal_to_limit_does_not_break(self):
        # Only a gap strictly greater than max_gap_seconds starts a segment.
        self.assertEqual(
            assign_segments([0, 2.5, 5.0], 2.5),
            [0, 0, 0],
        )

    def test_gap_just_above_limit_breaks(self):
        self.assertEqual(
            assign_segments([0, 2.6], 2.5),
            [0, 1],
        )

    def test_single_timestamp(self):
        self.assertEqual(assign_segments([42], 2.5), [0])

    def test_accepts_generator_input(self):
        self.assertEqual(
            assign_segments((value for value in (1, 2, 7)), 2.5),
            [0, 0, 1],
        )

    def test_datetime_timestamps(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        timestamps = [
            start,
            start + timedelta(seconds=1),
            start + timedelta(seconds=10),
        ]

        self.assertEqual(assign_segments(timestamps, 2.5), [0, 0, 1])

    def test_mixed_timestamp_types_are_comparable(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        timestamps = [
            start,
            "2026-01-01T00:00:01Z",
            start.timestamp() + 2,
        ]

        self.assertEqual(assign_segments(timestamps, 2.5), [0, 0, 0])

    def test_invalid_iso_string_is_rejected(self):
        with self.assertRaises(ValueError):
            assign_segments(["2026-01-01T00:00:00+00:00", "nope"], 2.5)

    def test_unsupported_type_is_rejected(self):
        with self.assertRaises(TypeError):
            assign_segments([1, None], 2.5)  # type: ignore[list-item]

    def test_non_finite_timestamps_are_rejected(self):
        # NaN/inf must not slip through the "delta <= 0" check unnoticed.
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    assign_segments([1, bad, 3], 2.5)

    def test_non_finite_gap_threshold_is_rejected(self):
        for bad in (math.nan, math.inf, -1.0):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    assign_segments([1, 2], bad)

    def test_duplicate_after_gap_is_still_rejected(self):
        with self.assertRaises(ValueError):
            assign_segments([1, 2, 9, 9], 2.5)

    def test_segment_lengths(self):
        segments = assign_segments([1, 2, 3, 8, 9], 2.5)

        self.assertEqual(segment_lengths(segments), {0: 3, 1: 2})

    def test_windows_never_cross_a_gap(self):
        segments = assign_segments([1, 2, 3, 8, 9], 2.5)

        # Window of 3: only (1,2,3) fits; 5 "continuous" points would give 3.
        self.assertEqual(count_windows(segments, 3), 1)
        self.assertEqual(count_windows(segments, 2), 3)
        self.assertEqual(count_windows(segments, 4), 0)

    def test_count_windows_validates_size(self):
        with self.assertRaises(ValueError):
            count_windows([0, 0], 0)

    def test_count_windows_empty(self):
        self.assertEqual(count_windows([], 3), 0)


if __name__ == "__main__":
    unittest.main()
