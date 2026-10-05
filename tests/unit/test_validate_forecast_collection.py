import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from scripts.experiments.validate_forecast_collection import (
    validate_collection,
    validate_official_collection,
    validate_run_directory,
)

RUN_ID = "stable-r01-s101"
CAPACITY = {"c1": 830.912131857993, "c2": 832.9337985006983}
START = datetime(2026, 10, 1, tzinfo=timezone.utc)
SAMPLES_PER_CONTROLLER = 20  # small run keeps the tests fast
RAW_VALIDATOR = (
    "scripts.experiments.validate_forecast_collection.validate_raw_samples"
)
REQUIRED_METADATA = (
    "run_id",
    "workload_type",
    "repeat",
    "seed",
    "controller_ids",
    "git_commit",
    "git_tag",
    "safe_capacity_c1",
    "safe_capacity_c2",
    "started_at",
    "finished_at",
    "topology",
)


def make_metadata(**overrides: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "run_id": RUN_ID,
        "workload_type": "stable",
        "repeat": 1,
        "seed": 101,
        "duration_seconds": float(SAMPLES_PER_CONTROLLER),
        "sampling_interval_seconds": 1.0,
        "controller_ids": ["c1", "c2"],
        "controllers": ["c1", "c2"],
        "migration_enabled": False,
        "git_commit": "abc1234",
        "git_tag": "forecast-dataset-v1",
        "safe_capacity_c1": CAPACITY["c1"],
        "safe_capacity_c2": CAPACITY["c2"],
        "started_at": START.isoformat(),
        "finished_at": (START + timedelta(seconds=20)).isoformat(),
        "topology": "capacity_2c20s",
    }
    metadata.update(overrides)
    return metadata


def make_samples(run_id: str = RUN_ID) -> list[dict[str, Any]]:
    rows = []
    for controller_id in ("c1", "c2"):
        for i in range(SAMPLES_PER_CONTROLLER):
            rate = 100.0 + i
            rows.append(
                {
                    "run_id": run_id,
                    "controller_id": controller_id,
                    "observed_at": (START + timedelta(seconds=i)).isoformat(),
                    "workload_type": "stable",
                    "workload_phase": "stable",
                    "processed_packet_in_rate": rate,
                    "flow_mod_rate": rate,
                    "process_cpu_percent": 10.0,
                    "process_memory_rss_mb": 48.0,
                    "response_p95_ms": 2.0,
                    "managed_switch_count": 10,
                    "safe_capacity_pps": CAPACITY[controller_id],
                    "utilization": rate / CAPACITY[controller_id],
                    "max_switch_control_load_share": 0.2,
                    "age_of_twin_ms": 500.0,
                    "twinning_rate": 1.0,
                    "completeness_ratio": 1.0,
                    "synchronization_jitter_ms": 100.0,
                    "snapshot_valid": True,
                    "snapshot_id": f"snap-{controller_id}-{i}",
                }
            )
    return rows


def codes(
    samples: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    **options: Any,
) -> set[str]:
    # The Week 7 raw validator is covered by its own tests; isolate it here.
    with patch(RAW_VALIDATOR, return_value=SimpleNamespace(issues=())):
        result = validate_collection(
            metadata=metadata or make_metadata(),
            samples=samples,
            **options,
        )
    return {issue.code for issue in result.issues}


def first(samples: list[dict[str, Any]], controller_id: str = "c1") -> list[dict[str, Any]]:
    return [row for row in samples if row["controller_id"] == controller_id]


class ForecastCollectionValidatorTest(unittest.TestCase):
    def test_valid_collection(self) -> None:
        self.assertEqual(codes(make_samples()), set())

    def test_duplicate_timestamp(self) -> None:
        samples = make_samples()
        rows = first(samples)
        rows[5]["observed_at"] = rows[4]["observed_at"]
        self.assertIn("timestamp_duplicate", codes(samples))

    def test_out_of_order_timestamp(self) -> None:
        samples = make_samples()
        rows = first(samples)
        rows[5]["observed_at"], rows[6]["observed_at"] = (
            rows[6]["observed_at"],
            rows[5]["observed_at"],
        )
        self.assertIn("timestamp_out_of_order", codes(samples))

    def test_sampling_gap(self) -> None:
        samples = make_samples()
        for index, row in enumerate(first(samples)):
            if index >= 10:  # 5 s hole in the middle of the run
                row["observed_at"] = (
                    START + timedelta(seconds=index + 5)
                ).isoformat()
        self.assertIn("max_gap_exceeded", codes(samples))

    def test_gap_within_limit_is_accepted(self) -> None:
        samples = make_samples()
        for index, row in enumerate(first(samples)):
            if index >= 10:  # 2 s step <= max_gap_seconds (2.5)
                row["observed_at"] = (
                    START + timedelta(seconds=index + 1)
                ).isoformat()
        self.assertNotIn("max_gap_exceeded", codes(samples))

    def test_missing_field(self) -> None:
        samples = make_samples()
        del samples[3]["snapshot_id"]
        self.assertIn("missing_fields", codes(samples))

    def test_null_field_counts_as_missing(self) -> None:
        samples = make_samples()
        samples[3]["utilization"] = None
        self.assertIn("missing_fields", codes(samples))

    def test_invalid_snapshot(self) -> None:
        samples = make_samples()
        samples[2]["snapshot_valid"] = False
        self.assertIn("snapshot_invalid", codes(samples))

    def test_wrong_controller(self) -> None:
        samples = make_samples()
        for row in first(samples, "c2"):
            row["controller_id"] = "c9"
        found = codes(samples)
        self.assertIn("controller_unexpected", found)
        self.assertIn("coverage_below_threshold", found)  # c2 has no samples

    def test_safe_capacity_not_positive(self) -> None:
        samples = make_samples()
        samples[1]["safe_capacity_pps"] = 0.0
        self.assertIn("safe_capacity_non_positive", codes(samples))
        samples[1]["safe_capacity_pps"] = -5.0
        self.assertIn("safe_capacity_non_positive", codes(samples))

    def test_safe_capacity_changes_inside_run(self) -> None:
        samples = make_samples()
        samples[1]["safe_capacity_pps"] = 900.0
        samples[1]["utilization"] = samples[1]["processed_packet_in_rate"] / 900.0
        self.assertIn("safe_capacity_inconsistent", codes(samples))

    def test_safe_capacity_differs_from_metadata(self) -> None:
        metadata = make_metadata(safe_capacity_c1=999.0)
        self.assertIn(
            "safe_capacity_metadata_mismatch", codes(make_samples(), metadata)
        )

    def test_utilization_mismatch(self) -> None:
        samples = make_samples()
        samples[4]["utilization"] = 0.9  # processed/safe is ~0.12
        self.assertIn("utilization_mismatch", codes(samples))

    def test_utilization_above_one_is_valid(self) -> None:
        samples = make_samples()
        samples[4]["processed_packet_in_rate"] = 997.095
        samples[4]["utilization"] = 997.095 / CAPACITY["c1"]  # ~1.2, no clamp
        self.assertNotIn("utilization_mismatch", codes(samples))

    def test_missing_workload_phase(self) -> None:
        samples = make_samples()
        samples[0]["workload_phase"] = ""
        del samples[1]["workload_phase"]
        self.assertIn("workload_phase_missing", codes(samples))

    def test_workload_type_mismatch(self) -> None:
        samples = make_samples()
        samples[0]["workload_type"] = "burst"
        self.assertIn("workload_type_mismatch", codes(samples))

    def test_low_coverage(self) -> None:
        samples = first(make_samples(), "c2") + first(make_samples(), "c1")[:10]
        self.assertIn("coverage_below_threshold", codes(samples))

    def test_sample_count_excess(self) -> None:
        samples = make_samples() + [dict(r) for r in first(make_samples())]
        self.assertIn("sample_count_excess", codes(samples))

    def test_migration_must_be_disabled(self) -> None:
        self.assertIn(
            "migration_enabled",
            codes(make_samples(), make_metadata(migration_enabled=True)),
        )

    def test_ownership_must_stay_fixed(self) -> None:
        samples = make_samples()
        first(samples)[7]["managed_switch_count"] = 9
        self.assertIn("ownership_changed", codes(samples))
        self.assertNotIn(
            "ownership_changed", codes(samples, require_fixed_ownership=False)
        )

    def test_run_id_contract(self) -> None:
        samples = make_samples("bad-run")
        self.assertIn(
            "run_id_format_invalid", codes(samples, make_metadata(run_id="bad-run"))
        )
        self.assertIn("run_id_seed_mismatch", codes(make_samples(), make_metadata(seed=999)))
        self.assertIn(
            "run_id_workload_mismatch",
            codes(make_samples(), make_metadata(workload_type="burst")),
        )

    def test_hot_switch_run_id_is_valid(self) -> None:
        run_id = "hot-switch-r05-s105"
        samples = make_samples(run_id)
        for row in samples:
            row["workload_type"] = "hot-switch"
        metadata = make_metadata(
            run_id=run_id, workload_type="hot-switch", repeat=5, seed=105
        )
        self.assertEqual(codes(samples, metadata), set())

    def test_metadata_contract_fields(self) -> None:
        metadata = make_metadata()
        del metadata["git_tag"]
        metadata["finished_at"] = None
        found = codes(
            make_samples(), metadata, metadata_required_fields=REQUIRED_METADATA
        )
        self.assertIn("metadata_field_missing", found)
        self.assertEqual(
            codes(make_samples(), make_metadata(), metadata_required_fields=REQUIRED_METADATA),
            set(),
        )


    def test_compatibility_validate_forecast_collection_api(self) -> None:
        config = {
            "controllers": ["c1", "c2"],
            "workloads": ["stable", "gradual", "burst", "oscillating", "hot-switch"],
            "validation": {
                "require_fixed_ownership": True,
                "min_sample_coverage_ratio": 0.95,
                "max_gap_seconds": 2.5,
                "utilization_tolerance": 1.0e-6,
            },
            "metadata": {
                "required_fields": list(REQUIRED_METADATA)
            },
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            RAW_VALIDATOR, return_value=SimpleNamespace(issues=())
        ):
            run_dir = Path(tmp) / RUN_ID
            run_dir.mkdir()
            (run_dir / "metadata.json").write_text(json.dumps(make_metadata()))
            (run_dir / "forecast_samples.jsonl").write_text(
                "\n".join(json.dumps(row) for row in make_samples()) + "\n"
            )

            from scripts.experiments.validate_forecast_collection import (
                validate_forecast_collection,
            )

            result = validate_forecast_collection(run_dir, config)

        self.assertTrue(result["valid"])
        self.assertEqual(result["controllers"]["c1"]["sample_count"], SAMPLES_PER_CONTROLLER)
        self.assertEqual(result["controllers"]["c1"]["coverage_ratio"], 1.0)

    def test_metadata_alias_accepted(self) -> None:
        metadata = make_metadata(repeat_index=1)
        del metadata["repeat"]
        del metadata["controller_ids"]  # runner still writes "controllers"
        self.assertEqual(
            codes(make_samples(), metadata, metadata_required_fields=REQUIRED_METADATA),
            set(),
        )


class OfficialCollectionTest(unittest.TestCase):
    def write_run(self, root: Path, run_id: str, samples: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
        run_dir = root / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(json.dumps(metadata))
        (run_dir / "forecast_samples.jsonl").write_text(
            "\n".join(json.dumps(row) for row in samples) + "\n"
        )

    def test_run_directory_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch(
            RAW_VALIDATOR, return_value=SimpleNamespace(issues=())
        ):
            self.write_run(Path(tmp), RUN_ID, make_samples(), make_metadata())
            result = validate_run_directory(Path(tmp) / RUN_ID)
        self.assertTrue(result.valid)
        self.assertEqual(result.total_samples, 2 * SAMPLES_PER_CONTROLLER)

    def test_official_requires_every_planned_run(self) -> None:
        plan = [SimpleNamespace(run_id=RUN_ID), SimpleNamespace(run_id="burst-r01-s101")]
        with tempfile.TemporaryDirectory() as tmp, patch(
            RAW_VALIDATOR, return_value=SimpleNamespace(issues=())
        ):
            self.write_run(Path(tmp), RUN_ID, make_samples(), make_metadata())
            result = validate_official_collection(Path(tmp), plan, expected_samples=80)
        self.assertFalse(result.valid)
        self.assertEqual(result.found_runs, 1)
        self.assertIn("official_run_missing", {i.code for i in result.issues})

    def test_official_valid_when_all_runs_present_and_valid(self) -> None:
        plan = [SimpleNamespace(run_id=RUN_ID)]
        with tempfile.TemporaryDirectory() as tmp, patch(
            RAW_VALIDATOR, return_value=SimpleNamespace(issues=())
        ):
            self.write_run(Path(tmp), RUN_ID, make_samples(), make_metadata())
            result = validate_official_collection(Path(tmp), plan, expected_samples=40)
        self.assertTrue(result.valid)
        self.assertEqual(result.total_samples, 40)

    def test_official_flags_invalid_run(self) -> None:
        plan = [SimpleNamespace(run_id=RUN_ID)]
        samples = make_samples()
        samples[0]["snapshot_valid"] = False
        with tempfile.TemporaryDirectory() as tmp, patch(
            RAW_VALIDATOR, return_value=SimpleNamespace(issues=())
        ):
            self.write_run(Path(tmp), RUN_ID, samples, make_metadata())
            result = validate_official_collection(Path(tmp), plan)
        self.assertIn("official_run_invalid", {i.code for i in result.issues})


if __name__ == "__main__":
    unittest.main()
