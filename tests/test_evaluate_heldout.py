import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate_heldout as heldout


class EventBaselineTests(unittest.TestCase):
    def test_strict_lower_bound_and_causal_upper_bound(self):
        cutoff = datetime(2026, 11, 23, tzinfo=timezone.utc)
        self.assertFalse(heldout.event_alarm_at([cutoff - timedelta(seconds=300)], cutoff))
        self.assertTrue(heldout.event_alarm_at([cutoff - timedelta(seconds=299)], cutoff))
        self.assertTrue(heldout.event_alarm_at([cutoff], cutoff))
        self.assertFalse(heldout.event_alarm_at([cutoff + timedelta(microseconds=1)], cutoff))


class CombinedHeldoutTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.protocol = Path(temp.name)
        self.runs = self.protocol / "runs"
        self.selection = self.protocol / "selection"
        self.adapter = self.protocol / "adapter"
        self.marker = self.protocol / "slope_held_out_started.json"
        self.report = self.protocol / "heldout_v32.json"
        self.provenance = {"freeze_commit": "new", "dev_assignments": [["slot-1", "dev-run"]]}
        self.slot = {"slot_id": "slot-2", "condition": "healthy"}
        self.metadata = {"freeze_commit": "old"}
        self.context = SimpleNamespace(
            provenance=self.provenance,
            assignments=Mock(return_value=[(self.slot, "held-run")]),
            read_metadata=Mock(return_value=self.metadata),
            load_run=Mock(),
        )
        self.selection_records = (
            {"selected_lookback_seconds": 60}, {"selected_threshold": 0.75, "dataset_sha256": "dataset"},
        )
        self.adapter_manifest = {**self.provenance, "mode": "final", "dataset_sha256": "dataset"}

    def evaluate(self):
        return heldout.evaluate(self.protocol, self.runs, self.selection, self.adapter)

    def test_prior_slope_report_refuses_before_any_data_access(self):
        (self.protocol / "slope_held_out.json").touch()
        with patch.object(heldout, "load_context") as load_context:
            with self.assertRaisesRegex(ValueError, "already"):
                self.evaluate()
        load_context.assert_not_called()

    def test_missing_selection_or_mismatched_adapter_does_not_burn_attempt(self):
        with patch.object(heldout, "load_context", return_value=self.context):
            with patch.object(heldout, "load_selection", side_effect=FileNotFoundError):
                with self.assertRaises(FileNotFoundError):
                    self.evaluate()
            with patch.object(heldout, "load_selection", return_value=self.selection_records):
                with patch.object(heldout, "validate_adapter", return_value={
                    **self.adapter_manifest, "dataset_sha256": "unrelated",
                }):
                    with self.assertRaisesRegex(ValueError, "adapter"):
                        self.evaluate()
        self.assertFalse(self.marker.exists())
        self.context.load_run.assert_not_called()

    def test_marker_exists_before_features_and_blocks_retries(self):
        def compute(*_args):
            self.assertTrue(self.marker.is_file())
            raise RuntimeError("simulated failure after held-out access")

        with (
            patch.object(heldout, "load_context", return_value=self.context) as load_context,
            patch.object(heldout, "load_selection", return_value=self.selection_records),
            patch.object(heldout, "validate_adapter", return_value=self.adapter_manifest),
            patch.object(heldout, "load_predictor", return_value=Mock()),
            patch.object(heldout, "compute_arms", side_effect=compute) as compute_arms,
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                self.evaluate()
            with self.assertRaisesRegex(ValueError, "already"):
                self.evaluate()
        load_context.assert_called_once()
        compute_arms.assert_called_once()
        self.assertFalse(self.report.exists())

    def test_all_three_arms_publish_in_single_report(self):
        arms = {"slope": [{"lookback_seconds": 60}], "event": {"episodes": 0}, "llm": {"episodes": 0}}
        with (
            patch.object(heldout, "load_context", return_value=self.context),
            patch.object(heldout, "load_selection", return_value=self.selection_records),
            patch.object(heldout, "validate_adapter", return_value=self.adapter_manifest),
            patch.object(heldout, "load_predictor", return_value=Mock()),
            patch.object(heldout, "compute_arms", return_value=arms),
        ):
            path, report = self.evaluate()
        self.assertEqual(path, self.report)
        self.assertEqual(report["arms"], arms)
        self.assertEqual(report["held_out_assignments"], [["slot-2", "held-run"]])
        self.assertTrue(self.report.is_file())

    def test_fixed_event_and_threshold_use_the_same_cutoffs(self):
        first = datetime(2026, 11, 23, 1, tzinfo=timezone.utc)
        second = first + timedelta(seconds=10)
        analysis = SimpleNamespace(examples=[SimpleNamespace(cutoff=first), SimpleNamespace(cutoff=second)])
        metadata = {
            **self.metadata,
            "observation_start_utc": (first - timedelta(minutes=10)).isoformat(),
            "observation_end_utc": (second + timedelta(minutes=5)).isoformat(),
            "boot_time_local": first.isoformat(),
        }
        self.context.load_run.return_value = metadata, analysis, self.runs / "held-run"
        assignments = [(self.slot, "held-run")]
        with (
            patch.object(heldout, "score_slopes", return_value=["slope results"]) as score_slopes,
            patch.object(heldout, "read_events", return_value=[
                (first, "Warning", 7), (second, "Warning", 2004),
            ]),
            patch.object(heldout, "make_prompt", side_effect=["prompt 1", "prompt 2"]),
            patch.object(heldout, "score_alarms", side_effect=lambda *_args: _args[-1]) as score_alarms,
            patch.object(heldout, "summarize_records", side_effect=lambda records: records),
        ):
            arms = heldout.compute_arms(self.context, assignments, 120, 0.75, Mock(side_effect=[0.75, 0.74]))
        score_slopes.assert_called_once_with(self.context, assignments, (120,))
        self.assertEqual(score_alarms.call_count, 2)
        self.assertEqual(arms, {
            "slope": ["slope results"],
            "event": [{first: False, second: True}],
            "llm": [{first: True, second: False}],
        })


if __name__ == "__main__":
    unittest.main()
