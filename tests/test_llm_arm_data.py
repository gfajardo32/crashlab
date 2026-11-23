import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class LLMArmDataTests(unittest.TestCase):
    def test_prompt_bins_only_allowed_causal_features(self):
        from llm_arm.prompt import make_prompt

        features = {
            "memory": [
                [5, 1024 * 2**20, 40, 100, 98765],
                [30, 2048 * 2**20, 60, 100, None],
                [35, 1024 * 2**20, 80, 100, None],
                [300, 512 * 2**20, 90, 100, None],
                [305, 12345678 * 2**20, 99999, 100, None],
            ],
            "events": [["Warning", 2004]],
            "message": "PRIVATE MESSAGE",
            "workload": "SECRET WORKLOAD",
        }
        prompt = make_prompt(SimpleNamespace(features=features, label="PRIVATE LABEL"))
        self.assertIn("0-30s: 50.00%, 1536.00 MiB", prompt)
        self.assertIn("30-60s: 80.00%, 1024.00 MiB", prompt)
        self.assertIn("270-300s: 90.00%, 512.00 MiB", prompt)
        self.assertIn("2004 Warning", prompt)
        for forbidden in ("98765", "12345678", "99999", "PRIVATE", "SECRET"):
            self.assertNotIn(forbidden, prompt)

    def test_builder_uses_18_development_groups_and_does_not_open_heldout(self):
        from llm_arm.build_dataset import build_dataset

        dev = [
            ({"slot_id": f"dev-{condition}-{j}", "condition": condition}, f"run-{condition}-{j}")
            for condition in range(6)
            for j in range(3)
        ]
        expected = {run_id for _, run_id in dev}
        cutoff = datetime(2026, 11, 23, tzinfo=timezone.utc)
        example = SimpleNamespace(cutoff=cutoff, label="no", features={"memory": [], "events": []})
        opened = []

        def load_run(slot, run_id):
            self.assertIn(run_id, expected)
            opened.append(run_id)
            return {}, SimpleNamespace(examples=[example]), Path(run_id)

        ctx = SimpleNamespace(
            freeze_commit="freeze", dev_assignments=dev, load_run=load_run,
            provenance={"freeze_commit": "freeze", "dev_assignments": [[slot["slot_id"], run_id] for slot, run_id in dev]},
        )
        with tempfile.TemporaryDirectory() as tmp, patch("llm_arm.build_dataset.load_context", return_value=ctx):
            output = Path(tmp) / "dataset"
            build_dataset(Path("protocol"), Path("runs"), output)
            rows = [json.loads(line) for line in (output / "dev.jsonl").read_text().splitlines()]
            manifest = json.loads((output / "manifest.json").read_text())
        self.assertCountEqual(opened, expected)
        self.assertEqual(len(rows), 18)
        self.assertEqual([sum(r["fold"] == fold for r in rows) for fold in range(3)], [6, 6, 6])
        for condition in range(6):
            self.assertEqual({row["fold"] for row in rows if f"run-{condition}-" in row["run_id"]}, {0, 1, 2})
        self.assertEqual(manifest["freeze_commit"], "freeze")
        self.assertTrue(manifest["dataset_sha256"])

    def test_model_settings_keep_reference_qlora_with_one_epoch(self):
        from llm_arm.train import TRAIN_OPTIONS

        self.assertEqual(TRAIN_OPTIONS["num_train_epochs"], 1)
        self.assertEqual(TRAIN_OPTIONS["per_device_train_batch_size"], 8)
        self.assertEqual(TRAIN_OPTIONS["gradient_accumulation_steps"], 2)
        self.assertEqual(TRAIN_OPTIONS["learning_rate"], 2e-4)
        self.assertEqual(TRAIN_OPTIONS["optim"], "adamw_8bit")

    def test_cross_validation_predicts_each_row_with_its_run_held_back(self):
        from llm_arm.train import write_oof_predictions

        rows = [
            {"run_id": f"run-{fold}-{run}", "cutoff_utc": f"cutoff-{cutoff}", "label": "no",
             "prompt": "prompt", "fold": fold}
            for fold in range(3) for run in range(6) for cutoff in range(2)
        ]
        trained = []
        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(empty_cache=lambda: None)

        def check_groups(training, validation, output_dir):
            training_runs = {row["run_id"] for row in training}
            validation_runs = {row["run_id"] for row in validation}
            self.assertEqual(len(training_runs), 12)
            self.assertEqual(len(validation_runs), 6)
            self.assertFalse(training_runs & validation_runs)
            fold, = {row["fold"] for row in validation}
            trained.append(fold)
            return None, None

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("llm_arm.train.train_model", side_effect=check_groups),
            patch("llm_arm.train.make_predictor", return_value=lambda prompt: 0.25),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            scratch = Path(tmp)
            path = scratch / "oof_predictions.jsonl"
            write_oof_predictions(rows, path, scratch)
            predictions = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(trained, [0, 1, 2])
        self.assertCountEqual(
            predictions,
            [{key: row[key] for key in ("run_id", "cutoff_utc", "fold")} | {"probability": 0.25} for row in rows],
        )

    def test_training_rejects_dataset_from_a_different_freeze_before_work(self):
        from llm_arm.train import train

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("llm_arm.train.read_dataset", return_value=({"freeze_commit": "older"}, [])),
            patch("llm_arm.protocol_data.verify_v32_freeze", return_value="current"),
            patch("llm_arm.train.write_oof_predictions",
                  side_effect=lambda rows, path, scratch: path.write_text("")) as train_cv,
        ):
            output = Path(tmp) / "artifacts" / "cv"
            with self.assertRaisesRegex(ValueError, "freeze"):
                train("cv", Path(tmp) / "dataset", output)
            train_cv.assert_not_called()
            self.assertFalse(output.parent.exists())

if __name__ == "__main__":
    unittest.main()
