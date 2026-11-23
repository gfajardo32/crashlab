"""Synthetic checks for the frozen responsiveness score."""
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import score_slope


class ScoreSlopeTests(unittest.TestCase):
    origin = datetime(2026, 11, 22, tzinfo=timezone.utc)
    freeze_commit = "a" * 40

    def at(self, seconds):
        return self.origin + timedelta(seconds=seconds)

    def test_episodes_break_on_gaps_or_nonalarms(self):
        windows = [(self.at(t), alarm) for t, alarm in
                   [(0, True), (10, True), (20, False), (30, True), (50, True), (60, True)]]
        self.assertEqual(score_slope.episode_count(windows), 3)

    def test_on_time_band_is_inclusive(self):
        onset = self.at(500)
        for lead in (289, 290, 295, 300, 301):
            with self.subTest(lead=lead):
                result = score_slope.positive_result(onset, [(self.at(500-lead), "yes", True)])
                self.assertEqual(result["caught_on_time"], 290 <= lead <= 300)
                self.assertEqual(result["on_time_evaluable"], 290 <= lead <= 300)

    def test_missing_band_is_not_a_miss(self):
        onset = self.at(500)
        result = score_slope.positive_result(onset, [(self.at(220), "yes", True)])
        self.assertFalse(result["on_time_evaluable"])
        self.assertTrue(result["caught_at_any_lead"])
        self.assertEqual(result["first_alarm_lead_seconds"], 280)
        eligible = score_slope.positive_result(onset, [(self.at(200), "yes", False)])
        self.assertTrue(eligible["on_time_evaluable"])
        self.assertFalse(eligible["caught_on_time"])

    def test_phase_boundaries(self):
        stepped = {"started": self.at(100), "pause_start": self.at(200), "resume": self.at(380)}
        plateau = {"started": self.at(100), "plateau_settle": self.at(300),
                   "topup_started": self.at(400), "topup_allocation_end": self.at(425)}
        for condition, events, cases in (
            ("stepped", stepped, [(99, "baseline"), (100, "initial_rise"), (200, "pause"),
                                   (439, "pause"), (440, "post_pause_rise")]),
            ("plateau96", plateau, [(299, "initial_rise"), (300, "settled_hold"),
                                    (400, "top_up"), (484, "top_up"), (485, "settled_hold")]),
            ("healthy", plateau, [(99, "baseline"), (100, "healthy")]),
        ):
            for time, expected in cases:
                with self.subTest(condition=condition, time=time):
                    self.assertEqual(score_slope.phase_at(self.at(time), condition, events), expected)

    def test_fit_uses_strict_bounds_and_last_sample_anchor(self):
        samples = [(self.at(t), value) for t, value in
                   [(0, 0.1), (5, .7), (10, .7045), (15, .709), (20, .7135), (25, .732), (30, .7155), (31, 1.0)]]
        self.assertFalse(score_slope.alarm_at(samples, self.at(30), 30))
        self.assertFalse(score_slope.alarm_at(samples[:4], self.at(30), 30))
        linear = [(self.at(t), .7 + t/1000) for t in range(5, 31, 5)]
        self.assertTrue(score_slope.alarm_at(linear, self.at(30), 30))

    def test_selection_budget_and_ties(self):
        entries = [(30, 20, 1.01), (60, 5, 0.5), (120, 5, 0.4), (300, 5, 0.4)]
        table = [
            {"lookback_seconds": lookback, "caught_on_time_runs": hits,
             "budget": {"episodes_per_hour": rate}}
            for lookback, hits, rate in entries
        ]
        self.assertEqual(score_slope.select_lookback(table), 300)
        self.assertIsNone(score_slope.select_lookback(table[:1]))
        table[1]["budget"]["episodes_per_hour"] = 1.0
        table[1]["caught_on_time_runs"] = 6
        self.assertEqual(score_slope.select_lookback(table), 60)

    def test_dev_only_and_held_out_one_shot(self):
        for rate, selected in ((0, 300), (1.01, None)):
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as directory:
                self.check_dev_and_holdout(Path(directory), rate, selected)

    def test_legacy_pilot_cannot_be_scored_even_if_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "run.json").write_text('{"run_id":"pilot"}')
            with patch.object(score_slope, "analyze_run") as analyze:
                with self.assertRaisesRegex(ValueError, "pilot runs are excluded"):
                    score_slope.score_run({"slot_id": "test"}, "pilot", run, (30,), self.freeze_commit)
            analyze.assert_not_called()

    def test_pilot_run_id_rejected_even_for_invalid_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "run_log.csv"
            log.write_text("slot_id,run_id,status,reason\nslot-dev,trial_pilot,invalid,test\n")
            slots = [{"slot_id": "slot-dev", "split": "dev"}]
            with self.assertRaisesRegex(ValueError, "Pilot run"):
                score_slope.assignments_for(slots, log, "dev")

    def test_run_metadata_rejects_pilots_and_wrong_or_missing_freeze(self):
        for extra, message in (({"pilot": True, "freeze_commit": self.freeze_commit}, "pilot"),
                               ({"freeze_commit": "b" * 40}, "freeze"),
                               ({}, "freeze")):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                self.write_metadata(run, **extra)
                with patch.object(score_slope, "analyze_run") as analyze:
                    with self.assertRaisesRegex(ValueError, message):
                        score_slope.score_run({"slot_id": "test"}, "trace", run, (30,), self.freeze_commit)
                    analyze.assert_not_called()

    def write_metadata(self, run, **extra):
        (run / "run.json").write_text(json.dumps({"protocol_version": 3, **extra}))

    def check_dev_and_holdout(self, protocol, rate, selected):
        (protocol / "PROTOCOL_v3.md").write_text("frozen protocol")
        (protocol / "protocol_v3.json").write_text(json.dumps({"runs": [
            {"slot_id": "slot-dev", "condition": "ramp5", "split": "dev"},
            {"slot_id": "slot-held", "condition": "ramp5", "split": "held_out"},
        ]}))
        (protocol / "run_log.csv").write_text(
            "slot_id,run_id,status,reason\n"
            "slot-dev,old-dev,invalid,reboot\nslot-dev,dev-trace,valid,\n"
            "slot-held,held-trace,valid,\n"
        )
        runs = protocol / "runs"
        (runs / "dev-trace").mkdir(parents=True)
        self.write_metadata(runs / "dev-trace", freeze_commit=self.freeze_commit)
        table = [
            {"lookback_seconds": lookback, "caught_on_time_runs": 1,
             "budget": {"episodes_per_hour": rate}}
            for lookback in score_slope.LOOKBACKS
        ]
        with patch.object(score_slope, "verify_protocol_freeze", return_value=self.freeze_commit), \
             patch.object(score_slope, "score_split", return_value=table) as score:
            _, dev = score_slope.evaluate("dev", protocol, runs)
            self.assertEqual(dev["dev_assignments"], [["slot-dev", "dev-trace"]])
            self.assertEqual(dev["freeze_commit"], self.freeze_commit)
            self.assertEqual(score.call_args.args[0][0][1], "dev-trace")
            self.assertEqual(dev["selected_lookback_seconds"], selected)
            self.assertFalse((protocol / "slope_held_out_started.json").exists())
        (runs / "held-trace").mkdir()
        self.write_metadata(runs / "held-trace", freeze_commit="b" * 40)
        with patch.object(score_slope, "verify_protocol_freeze", return_value=self.freeze_commit), \
             patch.object(score_slope, "score_split", return_value=table if selected is None else table[-1:]) as held_score:
            with self.assertRaisesRegex(ValueError, "freeze"):
                score_slope.evaluate("held-out", protocol, runs)
            self.assertFalse((protocol / "slope_held_out_started.json").exists())
            held_score.assert_not_called()
            self.write_metadata(runs / "held-trace", freeze_commit=self.freeze_commit)
            _, held = score_slope.evaluate("held-out", protocol, runs)
            self.assertEqual(held_score.call_args.args[0][0][1], "held-trace")
            self.assertEqual(held_score.call_args.args[2], score_slope.LOOKBACKS if selected is None else (300,))
            self.assertEqual(held["held_out_assignments"], [["slot-held", "held-trace"]])
            self.assertEqual(held["freeze_commit"], self.freeze_commit)
            with self.assertRaisesRegex(ValueError, "already started"):
                score_slope.evaluate("held-out", protocol, runs)
            held_score.assert_called_once()


if __name__ == "__main__":
    unittest.main()
