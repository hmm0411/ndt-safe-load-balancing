"""Day 5-7 end to end on synthetic data: frozen choice -> margins -> artifact -> figures
-> evidence; the runtime predictor reads the artifact."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import test_direct_outcome_models as direct_data
import yaml
from outcome_testdata import write_capacity
from test_outcome_selection import varied_rows

from src.twin.outcome import artifact, frozen, predictor


class ArtifactPipelineTest(unittest.TestCase):
    tmp: Path
    capacity: Path
    args: dict[str, Path]
    export: subprocess.CompletedProcess[str]

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp())
        tmp = cls.tmp
        models = tmp / "models.yaml"
        models.write_text(yaml.safe_dump(direct_data.SMALL))
        evaluation = yaml.safe_load(Path("configs/outcome/evaluation.yaml").read_text())
        evaluation["bootstrap"]["resamples"] = 200
        for spec in evaluation["ablation"].values():
            spec["drop"] = ["switch_control_load_share"]
        eval_path = tmp / "evaluation.yaml"
        eval_path.write_text(yaml.safe_dump(evaluation))
        rows = (varied_rows(60, "train", 1) + varied_rows(20, "validation", 2)
                + varied_rows(15, "test", 3))
        dataset = direct_data.write_dataset(tmp / "dataset", rows)
        cls.capacity = write_capacity(tmp / "controller_capacity.json")
        info = {"capacity_version": artifact.capacity_version(cls.capacity),
                "c_safe_pps": {"c1": 933.0, "c2": 933.0}}
        (dataset / "dataset_manifest.json").write_text(json.dumps({
            "dataset_version": "migration-outcome-processed-v1",
            "source_dataset": "migration-outcome-raw-v2", "files": {}, **info}))
        (dataset / "split_manifest.json").write_text(json.dumps({"splits": {
            s: [r["attempt_id"] for r in rows if r["split"] == s]
            for s in ("train", "validation", "test")}}))
        cls.args = {"dataset": dataset, "models": models, "eval": eval_path,
                    "direct": tmp / "direct", "hybrid": tmp / "hybrid", "final": tmp / "final",
                    "artifact": tmp / "post-migration-v1", "figures": tmp / "figures"}
        a = cls.args
        common = ["--dataset", str(dataset), "--models-config", str(models)]
        for stage, out, extra in (
                ("direct", a["direct"], []),
                ("hybrid", a["hybrid"], ["--direct-dir", str(a["direct"])]),
                ("select-and-test", a["final"],
                 ["--direct-dir", str(a["direct"]), "--hybrid-dir", str(a["hybrid"]),
                  "--eval-config", str(eval_path)])):
            result = cls.run_script("scripts/experiments/train_outcome_models.py",
                                    "--stage", stage, *common, "--out", str(out), *extra)
            assert result.returncode == 0, result.stderr
        cls.export = cls.run_export()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp)

    @staticmethod
    def run_script(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, *args], capture_output=True, text=True,
                              check=False)

    @classmethod
    def paths_args(cls) -> list[str]:
        a = cls.args
        return ["--dataset", str(a["dataset"]), "--models-config", str(a["models"]),
                "--eval-config", str(a["eval"]), "--direct-dir", str(a["direct"]),
                "--hybrid-dir", str(a["hybrid"]), "--final-dir", str(a["final"])]

    @classmethod
    def run_export(cls, *extra: str) -> subprocess.CompletedProcess[str]:
        return cls.run_script("scripts/experiments/export_outcome_artifact.py",
                              *cls.paths_args(), "--capacity-file", str(cls.capacity),
                              "--artifact-dir", str(cls.args["artifact"]), *extra)

    def chosen(self) -> dict[str, Any]:
        return dict(json.loads((self.args["final"] / "model_selection.json").read_text())
                    ["frozen"])

    def test_margins_and_coverage(self) -> None:
        self.assertEqual(self.export.returncode, 0, self.export.stderr + self.export.stdout)
        margin = json.loads((self.args["final"] / "uncertainty_margin.json").read_text())
        chosen = self.chosen()
        names = {"analytical", chosen["best_direct"], chosen["best_hybrid"]}
        self.assertEqual(set(margin["by_model"]), names)
        deployed = min(names, key=lambda n: chosen["validation_scores"][n]["S"])
        self.assertEqual(margin["deployed_model"], deployed)
        self.assertEqual(margin["m_U"], margin["by_model"][deployed]["m_U"]["value"])
        self.assertEqual(margin["by_model"][deployed]["m_U"]["rank"], 20)  # n = 20
        cov = json.loads((self.args["final"] / "coverage_metrics.json").read_text())
        entry = cov["by_model"][deployed]
        self.assertGreaterEqual(entry["m_U"]["validation"]["coverage"], 0.95)
        groups = {(r["group"], r["value"]) for r in entry["m_U"]["test"]}
        self.assertIn(("all", "all"), groups)
        self.assertIn(("workload", "burst"), groups)
        self.assertIn("missed_overload", entry["gate_u_dst"]["test"])

    def test_artifact_layout_and_runtime(self) -> None:
        self.assertEqual(self.export.returncode, 0, self.export.stderr + self.export.stdout)
        root = self.args["artifact"]
        missing = [e for e in predictor.ARTIFACT_ENTRIES if not (root / e).exists()]
        self.assertEqual(missing, [artifact.LATENCY_FILE])  # Vy adds it on the VM
        running = artifact.capacity_version(self.capacity)
        self.assertEqual(predictor.artifact_problem(root, running),
                         predictor.ARTIFACT_SCHEMA_MISMATCH)
        self.assertIn("only inference_latency.json is missing", self.export.stdout)
        manifest = json.loads((root / "training_manifest.json").read_text())
        self.assertEqual(manifest["model_version"], predictor.MODEL_VERSION)
        self.assertEqual(manifest["capacity_version"], running)
        self.assertEqual(len(manifest["test_runs"]), 15)
        short, suffix = (("xgb", ".json") if manifest["hybrid_family"] == "xgboost"
                         else ("rf", ".joblib"))
        self.assertIn(f"residual_models/residual_{short}_u_dst_post{suffix}",
                      manifest["files_sha256"])
        card = json.loads((root / "model_card.json").read_text())
        self.assertIn(card["hybrid_family"], ("random_forest", "xgboost"))
        self.assertTrue(any("does not authorize migration" in x for x in card["limitations"]))

        (root / artifact.LATENCY_FILE).write_text(json.dumps({"note": "test"}))
        try:
            self.assertIsNone(predictor.artifact_problem(root, running))
            self.assertEqual(predictor.artifact_problem(root, "other@sha256:000000000000"),
                             predictor.CAPACITY_VERSION_MISMATCH)
            runtime = predictor._load_runtime(root)
            margin = json.loads((root / "uncertainty_margin.json").read_text())
            self.assertEqual(runtime.m_u, margin["m_U"])
            check = self.run_export("--check")
            self.assertEqual(check.returncode, 0, check.stdout)
            self.assertIn("READY", check.stdout)
        finally:
            (root / artifact.LATENCY_FILE).unlink()

    def test_artifact_is_not_overwritten(self) -> None:
        again = self.run_export()
        self.assertNotEqual(again.returncode, 0)
        self.assertIn("frozen once written", again.stderr)

    def test_changed_model_file_is_refused(self) -> None:
        paths = frozen.FinalPaths(self.args["dataset"], Path("configs/outcome/dataset.yaml"),
                                  self.args["models"], self.args["eval"], self.args["direct"],
                                  self.args["hybrid"], self.args["final"])
        frozen.load_frozen(paths)  # unchanged: accepted
        model = sorted((self.args["hybrid"] / "models").iterdir())[0]
        original = model.read_bytes()
        try:
            model.write_bytes(original + b" ")
            with self.assertRaises(ValueError):
                frozen.load_frozen(paths)
        finally:
            model.write_bytes(original)

    def test_figures(self) -> None:
        result = self.run_script("scripts/analysis/plot_outcome_figures.py", *self.paths_args(),
                                 "--out", str(self.args["figures"]))
        self.assertEqual(result.returncode, 0, result.stderr)
        names = sorted(p.name for p in self.args["figures"].glob("fig_w12_*.png"))
        self.assertEqual(len(names), 8)
        self.assertEqual([n[:10] for n in names],
                         [f"fig_w12_0{i}" for i in range(1, 9)])


class CollectEvidenceTest(unittest.TestCase):
    def test_copy_then_refuse_conflicts(self) -> None:
        sys.path.insert(0, "scripts/experiments")
        try:
            import collect_outcome_evidence as collect
        finally:
            sys.path.remove("scripts/experiments")
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            for src in collect.FILES.values():
                (tmp / src).parent.mkdir(parents=True, exist_ok=True)
                (tmp / src).write_text(src)
            out = tmp / "evidence"
            script = "scripts/experiments/collect_outcome_evidence.py"
            first = subprocess.run([sys.executable, script, "--root", str(tmp), "--out",
                                    str(out)], capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertIn(f"{len(collect.FILES)} copied", first.stdout)
            again = subprocess.run([sys.executable, script, "--root", str(tmp), "--out",
                                    str(out)], capture_output=True, text=True, check=False)
            self.assertEqual(again.returncode, 0, again.stderr)
            (out / "test_lock.json").write_text("other")
            conflict = subprocess.run([sys.executable, script, "--root", str(tmp), "--out",
                                       str(out)], capture_output=True, text=True, check=False)
            self.assertNotEqual(conflict.returncode, 0)
            self.assertIn("test_lock.json", conflict.stderr)


if __name__ == "__main__":
    unittest.main()
