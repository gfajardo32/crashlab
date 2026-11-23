"""Synthetic tests for protocol collection and label-blind quality checks."""

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_protocol


class RunProtocolTests(unittest.TestCase):
    start = datetime(2026, 11, 22, tzinfo=timezone.utc)
    slot = {
        "order": 1,
        "slot_id": "ramp15-01",
        "condition": "ramp15",
        "collector_args": ["1", "--surface-data", "--responsiveness", "--protocol-v3", "--ramp", "15"],
    }

    def stamp(self, seconds):
        return (self.start + timedelta(seconds=seconds)).isoformat()

    def setup_project(self, project):
        protocol = project / "protocol"
        protocol.mkdir()
        later = {**self.slot, "order": 2, "slot_id": "ramp15-02"}
        (protocol / "protocol_v3.json").write_text(json.dumps({"runs": [later, self.slot]}))
        log = protocol / "run_log.csv"
        log.write_text("slot_id,run_id,status,reason\n")
        return log

    def fixture(self, directory, *, missing=0, reboot=False, stopped=True):
        meta = {
            "run_id": directory.name,
            "protocol_version": 3,
            "experiment": "surface-responsiveness-v3",
            "fault": "responsiveness-ramp",
            "observation_start_utc": self.stamp(0),
            "observation_end_utc": self.stamp(1260),
            "reboot_times_utc": [self.stamp(800)] if reboot else [],
            "inject_time_utc": self.stamp(480),
            "workload_time_cap_minutes": 12,
            "allocation_chunk_mib": 15,
            "allocation_chunk_interval_seconds": 1,
            "workload_stop_reason": "time_cap",
            "workload_stop_verified": stopped,
        }
        (directory / "run.json").write_text(json.dumps(meta))
        with (directory / "workload.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("timestamp_utc", "event", "reason", "onset_utc"))
            writer.writerow((self.stamp(480), "started", "supervised_ramp", ""))
            writer.writerow((self.stamp(1200), "intervention", "time_cap", ""))
            if stopped:
                writer.writerow((self.stamp(1201), "stopped", "time_cap", ""))
        with (directory / "probe_powershell.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("probe_timestamp_utc", "probe_status", "probe_latency_ms"))
            for slot in range(missing, 252):
                writer.writerow((self.stamp(slot * 5), "timeout" if slot % 2 else "ok", 50))
        return meta

    def test_invalid_attempt_leaves_lowest_order_slot_pending(self):
        schedule = [{"order": 2, "slot_id": "second"}, self.slot]
        rows = [{"slot_id": self.slot["slot_id"], "status": "invalid"}]
        self.assertEqual(run_protocol.next_slot(schedule, rows), self.slot)
        rows.append({"slot_id": self.slot["slot_id"], "status": "valid"})
        self.assertEqual(run_protocol.next_slot(schedule, rows), schedule[0])

    def test_validity_ignores_onset_and_does_not_run_labeler(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            meta = self.fixture(directory)
            self.assertEqual(run_protocol.validate_run(self.slot, directory), "")
            meta["local_probe_onset_utc"] = "not a timestamp"
            meta["onset_utc"] = self.stamp(900)
            (directory / "run.json").write_text(json.dumps(meta))
            with (directory / "workload.csv").open(newline="") as stream:
                rows = list(csv.reader(stream))
            for row in rows[1:]:
                row[-1] = "not a timestamp"
            with (directory / "workload.csv").open("w", newline="") as stream:
                csv.writer(stream).writerows(rows)
            self.assertEqual(run_protocol.validate_run(self.slot, directory), "")

    def test_missing_slots_reboots_and_unverified_stops_are_invalid(self):
        for options, reason in (({"missing": 13}, "coverage"), ({"reboot": True}, "reboot"),
                                ({"stopped": False}, "stop")):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                self.fixture(directory, **options)
                self.assertIn(reason, run_protocol.validate_run(self.slot, directory))

    def test_dry_run_prints_exact_args_and_leaves_log_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            log = self.setup_project(project)
            before = log.read_bytes()
            with patch.object(run_protocol, "launch_collector") as launch, redirect_stdout(io.StringIO()) as text:
                self.assertEqual(run_protocol.run_next(project, project / "runs", dry_run=True), 0)
            launch.assert_not_called()
            self.assertIn(self.slot["slot_id"], text.getvalue())
            self.assertIn(" ".join(self.slot["collector_args"]), text.getvalue())
            self.assertEqual(log.read_bytes(), before)

    def test_failed_collector_appends_invalid_and_retries_same_slot(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            log = self.setup_project(project)
            with patch.object(run_protocol, "launch_collector", return_value=(1, "")) as launch:
                for _ in range(2):
                    self.assertEqual(run_protocol.run_next(project, project / "runs"), 1)
            self.assertEqual(launch.call_count, 2)
            for call in launch.call_args_list:
                self.assertEqual(call.args[0][2:], self.slot["collector_args"])
            with log.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([(r["slot_id"], r["status"]) for r in rows],
                             [(self.slot["slot_id"], "invalid")] * 2)

    def test_successfully_saved_run_advances_to_next_slot(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            log = self.setup_project(project)
            runs = project / "runs"
            runs.mkdir()
            completed = runs / "laptop_fixture"
            def save_fixture(*args):
                completed.mkdir()
                self.fixture(completed)
                return 0, completed
            with patch.object(run_protocol, "launch_collector", side_effect=save_fixture):
                self.assertEqual(run_protocol.run_next(project, runs), 0)
            with log.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows, [{"slot_id": self.slot["slot_id"], "run_id": completed.name,
                                     "status": "valid", "reason": ""}])
            with redirect_stdout(io.StringIO()) as text:
                self.assertEqual(run_protocol.run_next(project, runs, dry_run=True), 0)
            self.assertIn("Slot 2: ramp15-02", text.getvalue())

    def test_append_preserves_last_row_when_log_has_no_trailing_newline(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            log = self.setup_project(project)
            log.write_text(log.read_text() + "ramp15-01,,invalid,old failure")
            with patch.object(run_protocol, "launch_collector", return_value=(1, "")):
                self.assertEqual(run_protocol.run_next(project, project / "runs"), 1)
            with log.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["reason"], "old failure")

    def test_collector_stdout_pairs_run_banner_with_saved_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            saved = project / "laptop_fixture"
            code = (f'print("=== {saved.name} ==="); '
                    f'print("  saved responsiveness telemetry -> " + {str(saved)!r})')
            with redirect_stdout(io.StringIO()) as text:
                result = run_protocol.launch_collector([sys.executable, "-c", code], project)
            self.assertEqual(result, (0, saved))
            self.assertIn(str(saved), text.getvalue())

    def plateau_fixture(self, directory, *, reason, cap=22, with_hold=False, with_topup=False):
        self.fixture(directory)
        meta_path = directory / "run.json"
        meta = json.loads(meta_path.read_text())
        intervention = 480 + (cap * 60 if reason == "time_cap" else 600)
        end = intervention + 60
        meta.update(workload_time_cap_minutes=cap, target_commit_percent=96,
                    workload_stop_reason=reason, observation_end_utc=self.stamp(end))
        meta_path.write_text(json.dumps(meta))
        with (directory / "workload.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("timestamp_utc", "event", "reason", "onset_utc"))
            writer.writerow((self.stamp(480), "started", "supervised_plateau", ""))
            if with_hold:
                writer.writerow((self.stamp(intervention - 100), "plateau_settle", "", ""))
                if with_topup:
                    writer.writerow((self.stamp(intervention - 75), "topup_started", "", ""))
                    writer.writerow((self.stamp(intervention - 50), "topup_allocation_end", "", ""))
            writer.writerow((self.stamp(intervention), "intervention", reason, "ignored"))
            writer.writerow((self.stamp(intervention + 1), "stopped", reason, "ignored"))
        with (directory / "probe_powershell.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("probe_timestamp_utc", "probe_status"))
            writer.writerows((self.stamp(stamp), "ok") for stamp in range(0, end, 5))
        return intervention

    def test_only_verified_early_probe_stop_waives_missing_hold(self):
        slot = {**self.slot, "condition": "plateau96"}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.plateau_fixture(directory, reason="probe_onset")
            self.assertEqual(run_protocol.validate_run(slot, directory), "")
            self.plateau_fixture(directory, reason="time_cap")
            self.assertIn("plateau hold marker", run_protocol.validate_run(slot, directory))

    def test_hold_uses_only_final_interval_and_counts_missing_against_full_hold(self):
        slot = {**self.slot, "condition": "plateau96"}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            stop = self.plateau_fixture(directory, reason="time_cap", with_hold=True, with_topup=True)
            def write_memory(in_band):
                with (directory / "memory.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(("timestamp_utc", "committed_bytes", "commit_limit_bytes"))
                    for stamp in range(stop - 100, stop + 60, 5):
                        if stop - 50 + in_band * 5 <= stamp < stop:
                            continue
                        percent = 96 if stop - 50 <= stamp < stop else 20
                        writer.writerow((self.stamp(stamp), percent, 100))
            write_memory(8)
            self.assertEqual(run_protocol.validate_run(slot, directory), "")
            write_memory(7)
            self.assertIn("70.0%", run_protocol.validate_run(slot, directory))

    def test_stepped_hold_uses_pause_start_to_resume(self):
        slot = {**self.slot, "condition": "stepped", "pattern": "A"}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            meta = self.fixture(directory)
            meta.update(workload_time_cap_minutes=25, stepped_pattern="A", pause_commit_percent=75,
                        pause_duration_seconds=60, workload_stop_reason="probe_onset")
            (directory / "run.json").write_text(json.dumps(meta))
            with (directory / "workload.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("timestamp_utc", "event", "reason"))
                writer.writerow((self.stamp(480), "started", "stepped_A"))
                writer.writerow((self.stamp(720), "pause_start", "target_reached"))
                writer.writerow((self.stamp(780), "resume", "pause_complete"))
                writer.writerow((self.stamp(1200), "intervention", "probe_onset"))
                writer.writerow((self.stamp(1201), "stopped", "probe_onset"))
            path = directory / "memory.csv"
            path.write_text("timestamp_utc,committed_bytes,commit_limit_bytes\n" + "".join(
                f"{self.stamp(stamp)},{76 if 720 <= stamp < 780 else 90},100\n"
                for stamp in range(700, 801, 5)))
            self.assertEqual(run_protocol.validate_run(slot, directory), "")
            path.write_text(path.read_text().replace(",76,100", ",77,100"))
            self.assertIn("below 80%", run_protocol.validate_run(slot, directory))


if __name__ == "__main__":
    unittest.main()
