"""Selection score decision (2026-10-30): models are chosen with S without r95_post."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from src.twin.outcome.dataset import load_dataset_config

CONFIG_PATH = Path("configs/outcome/dataset.yaml")


class SelectionConfigTest(unittest.TestCase):
    def test_repo_decision(self) -> None:
        config = load_dataset_config(CONFIG_PATH)
        self.assertEqual(config.selection_exclude, ("r95_post",))
        self.assertEqual(config.selection_weights["r95_post"], 0.25)  # plan weights kept
        weights = config.effective_selection_weights
        self.assertNotIn("r95_post", weights)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertAlmostEqual(weights["u_dst_post"], 0.35 / 0.75)
        self.assertAlmostEqual(weights["u_src_post"], 0.25 / 0.75)
        self.assertAlmostEqual(weights["i_post"], 0.15 / 0.75)

    def test_exclusion_is_checked(self) -> None:
        data = yaml.safe_load(CONFIG_PATH.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.yaml"
            for exclude in (["latency"], list(data["selection_weights"])):
                path.write_text(yaml.safe_dump({**data, "selection_exclude": exclude}))
                with self.assertRaises(ValueError):
                    load_dataset_config(path)
            path.write_text(yaml.safe_dump({k: v for k, v in data.items()
                                            if k != "selection_exclude"}))
            config = load_dataset_config(path)
            self.assertEqual(config.effective_selection_weights, config.selection_weights)


if __name__ == "__main__":
    unittest.main()
