import json
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from math import nextafter
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_arm.metrics import score_alarms, score_slopes
from llm_arm.select_threshold import (
    THRESHOLD_REPORT_VERSION,
    best_threshold,
    load_selection,
    match_oof_predictions,
    select_development,
)
from score_slope import REPORT_VERSION, write_report


class LlmArmSelectionTests(unittest.TestCase):
    def setUp(self):
        self.base = datetime(2026, 11, 23, tzinfo=timezone.utc)

    def test_scoring_matches_protocol_band_and_budget_phases(self):
        cutoffs = [self.base + timedelta(seconds=s) for s in (0, 10, 20, 30, 40)]
        analysis = SimpleNamespace(
            onset=self.base + timedelta(seconds=330),
            diagnostic=False,
            examples=[
                SimpleNamespace(cutoff=t, label=label)
                for t, label in zip(cutoffs, ("no", "no", "no", "yes", "yes"))
            ],
        )
        slot = {"slot_id": "rampA", "condition": "ramp10"}
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workload.csv").write_text(
                f"event,timestamp_utc\nstarted,{cutoffs[1].isoformat()}\n"
            )
            record = score_alarms(
                slot, "run-1", {}, analysis, run_dir, dict.fromkeys(cutoffs, True),
            )
        self.assertTrue(record["caught_on_time"])
        self.assertEqual(record["first_alarm_lead_seconds"], 300)
        self.assertEqual(record["overall"]["episodes"], 1)
        self.assertEqual(record["budget"]["eligible_negative_seconds"], 10)
        self.assertEqual(record["budget"]["episodes"], 1)
        self.assertEqual(record["phases"]["initial_rise"]["eligible_negative_seconds"], 20)

    def test_threshold_selection_uses_budget_and_fewer_false_alarms(self):
        def row(threshold, catches, false_alarm_rate):
            return dict(threshold=threshold, caught_on_time_runs=catches,
                        budget={"episodes_per_hour": false_alarm_rate})

        self.assertEqual(best_threshold([
            row(0.4, 3, 1.01), row(0.5, 2, 1.0), row(0.6, 2, 0.25), row(0.7, 1, 0.0),
        ]), 0.6)
        self.assertIsNone(best_threshold([row(0.4, 3, 1.01)]))

    def test_out_of_fold_predictions_require_exact_rows_and_assigned_fold(self):
        examples = {
            ("r1", self.base): {"label": "yes", "fold": 0},
            ("r2", self.base): {"label": "no", "fold": 1},
            ("r3", self.base): {"label": "no", "fold": 2},
        }

        def row(run_id, fold):
            return dict(run_id=run_id, cutoff_utc=self.base.isoformat(), fold=fold,
                        probability=0.6)

        rows = [row("r1", 0), row("r2", 1), row("r3", 2)]
        self.assertEqual(match_oof_predictions(examples, rows), dict.fromkeys(examples, 0.6))
        with self.assertRaises(ValueError):
            match_oof_predictions(examples, [row("r1", 1), *rows[1:]])
        with self.assertRaises(ValueError):
            match_oof_predictions(examples, rows[:-1])
        with self.assertRaises(ValueError):
            match_oof_predictions(examples, [*rows, row("held-out", 0)])
        with self.assertRaises(ValueError):
            match_oof_predictions(examples, [*rows, rows[0]])

    def test_selection_refuses_unrecorded_or_mismatched_threshold(self):
        provenance = {"freeze_commit": "fixture"}
        context = SimpleNamespace(provenance=provenance)
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            slope = directory / "slope_dev.json"
            write_report(slope, {**provenance, "split": "dev", "report_version": REPORT_VERSION,
                                 "selected_lookback_seconds": None, "table": []})
            with self.assertRaises(FileNotFoundError):
                load_selection(context, directory)
            llm = directory / "threshold.json"
            write_report(llm, {**provenance, "split": "dev", "report_version": THRESHOLD_REPORT_VERSION,
                               "selected_threshold": None, "table": [],
                               "slope_dev_sha256": sha256(slope.read_bytes()).hexdigest()})
            with self.assertRaisesRegex(ValueError, "threshold"):
                load_selection(context, directory)

    def test_development_selection_with_both_freezes_and_three_fold_predictions(self):
        provenance = {"freeze_commit": "new", "protocol_sha256": "fixture"}
        assignments = [({"slot_id": run, "condition": "healthy"}, run)
                       for run in ("r1", "r2", "r3")]
        cutoff = self.base + timedelta(minutes=10)
        with TemporaryDirectory() as tmp:
            old_dir = Path(tmp)
            folds = [[run] for _, run in assignments]
            expected = {}
            rows = []
            probabilities = []
            for fold, (slot, run) in enumerate(assignments):
                folder = old_dir / run
                folder.mkdir()
                label = "yes" if fold == 0 else "no"
                onset = cutoff + timedelta(seconds=300) if fold == 0 else None
                metadata = {"freeze_commit": "old" if fold == 0 else "new"}
                analysis = SimpleNamespace(onset=onset, diagnostic=False, examples=[
                    SimpleNamespace(cutoff=cutoff, label=label),
                ])
                expected[run] = metadata, analysis, folder
                (folder / "workload.csv").write_text(
                    f"event,timestamp_utc\nstarted,{self.base.isoformat()}\n"
                )
                rows.append({"run_id": run, "cutoff_utc": cutoff.isoformat(),
                             "label": label, "prompt": "fixture", "fold": fold})
                probabilities.append({"run_id": run, "cutoff_utc": cutoff.isoformat(),
                                      "fold": fold, "probability": (0.8, 0.5, 0.2)[fold]})

            def fixture_rows(directory, filename, values):
                directory.mkdir()
                path = directory / filename
                path.write_text("".join(json.dumps(row) + "\n" for row in values))
                return sha256(path.read_bytes()).hexdigest()

            dataset, cv, selection = (old_dir / name for name in ("dataset", "cv", "selection"))
            dataset_hash = fixture_rows(dataset, "dev.jsonl", rows)
            predictions_hash = fixture_rows(cv, "oof_predictions.jsonl", probabilities)
            manifest = {**provenance, "folds": folds, "dataset_sha256": dataset_hash}
            write_report(dataset / "manifest.json", manifest)
            write_report(cv / "manifest.json", {**manifest, "mode": "cv", "predictions_sha256": predictions_hash})
            context = SimpleNamespace(
                provenance=provenance,
                dev_assignments=assignments,
                runs_dir=old_dir,
                read_metadata=lambda slot, run: expected[run][0],
                load_run=lambda slot, run: expected[run],
            )

            def frozen_slope(slot, run, folder, lookbacks, commit):
                metadata, analysis, _ = expected[run]
                self.assertEqual(commit, metadata["freeze_commit"])
                record = score_alarms(slot, run, metadata, analysis, folder, {cutoff: False})
                return {lookback: record for lookback in lookbacks}

            with patch("llm_arm.metrics.score_run", side_effect=frozen_slope) as scoring:
                slope, llm = select_development(context, old_dir, dataset, cv, selection)
            self.assertEqual(scoring.call_count, 3)
            self.assertEqual(slope["selected_lookback_seconds"], 300)
            self.assertEqual(llm["selected_threshold"], 0.8)
            self.assertEqual(llm["selected_metrics"]["caught_on_time_runs"], 1)
            self.assertEqual(load_selection(context, selection), (slope, llm))
            (old_dir / "slope_held_out_started.json").touch()
            with self.assertRaisesRegex(ValueError, "locked"):
                select_development(context, old_dir, dataset, cv, selection)

    def test_saturated_probability_still_has_no_alarm_candidate(self):
        context = SimpleNamespace(provenance={"freeze_commit": "fixture"}, dev_assignments=[])
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            probabilities = {("negative", self.base): 1.0}

            def score(_runs, probabilities, threshold):
                return {"caught_on_time_runs": 0,
                        "budget": {"episodes_per_hour": 3 if 1.0 >= threshold else 0}}

            with (patch("llm_arm.select_threshold.load_development",
                        return_value=([], probabilities, "dataset", "predictions")),
                  patch("llm_arm.select_threshold.threshold_scores", side_effect=score),
                  patch("llm_arm.select_threshold.score_slopes", return_value=[])):
                slope, llm = select_development(context, directory, directory, directory, directory)
            self.assertEqual(llm["selected_threshold"], nextafter(1.0, 2.0))
            self.assertEqual(load_selection(context, directory), (slope, llm))

    def test_existing_combined_report_locks_development_even_without_marker(self):
        context = SimpleNamespace()
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "heldout_v32.json").touch()
            with self.assertRaisesRegex(ValueError, "locked"), patch(
                "llm_arm.select_threshold.load_development",
                side_effect=AssertionError("development data was opened"),
            ):
                select_development(context, directory, directory, directory, directory)


if __name__ == "__main__":
    unittest.main()
