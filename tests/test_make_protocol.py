import csv
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "make_protocol.py"
SCHEDULE = "protocol_v3.json"
RUN_LOG = "run_log.csv"


class MakeProtocolTests(unittest.TestCase):
    def generate(self, directory, *, check=True, runs_dir=None):
        command = [sys.executable, str(SCRIPT), "--protocol-dir", str(directory)]
        if runs_dir is not None:
            command.extend(["--runs-dir", str(runs_dir)])
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=check,
        )

    def test_seeded_schedule_has_all_conditions_and_prespecified_splits(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            self.generate(first)
            self.generate(second)
            schedule = (Path(first) / SCHEDULE).read_bytes()
            self.assertEqual(schedule, (Path(second) / SCHEDULE).read_bytes())
            with (Path(first) / RUN_LOG).open(newline="") as f:
                self.assertEqual(list(csv.reader(f)), [["slot_id", "run_id", "status", "reason"]])
        protocol = json.loads(schedule)
        self.assertEqual(protocol["seed"], 20261122)
        runs = protocol["runs"]
        self.assertEqual(len({run["slot_id"] for run in runs}), 30)
        self.assertEqual([run["order"] for run in runs], list(range(1, 31)))
        self.assertEqual(
            Counter((run["condition"], run["split"]) for run in runs),
            Counter(
                {
                    (condition, split): count
                    for condition in ("healthy", "plateau96", "ramp5", "ramp10", "ramp15", "stepped")
                    for split, count in (("dev", 3), ("held_out", 2))
                }
            ),
        )
        self.assertNotEqual(
            [run["condition"] for run in runs],
            sorted(run["condition"] for run in runs),
        )
        patterns = {
            split: [run["pattern"] for run in runs if run["condition"] == "stepped" and run["split"] == split]
            for split in ("dev", "held_out")
        }
        self.assertEqual(sorted(patterns["held_out"]), ["A", "B"])
        self.assertEqual(set(patterns["dev"]), {"A", "B"})
        self.assertEqual(len(patterns["dev"]), 3)

    def test_commands_match_schedule_and_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            self.generate(directory)
            runs = json.loads((Path(directory) / SCHEDULE).read_text())["runs"]
        healthy_seeds = set()
        for run in runs:
            with self.subTest(slot=run["slot_id"]):
                condition = run["condition"]
                if condition == "healthy":
                    seed = run["burst_seed"]
                    self.assertIs(type(seed), int)
                    healthy_seeds.add(seed)
                    tail = ["--healthy-control", "--burst-seed", str(seed)]
                else:
                    self.assertIsNone(run["burst_seed"])
                    if condition == "stepped":
                        tail = ["--stepped", run["pattern"]]
                    elif condition == "plateau96":
                        tail = ["--plateau-percent", "96"]
                    else:
                        tail = ["--ramp", condition.removeprefix("ramp")]
                if condition != "stepped":
                    self.assertIsNone(run["pattern"])
                self.assertEqual(
                    run["collector_args"],
                    ["1", "--surface-data", "--responsiveness", "--protocol-v3", *tail],
                )
        self.assertEqual(len(healthy_seeds), 5)

    def test_existing_run_log_survives_and_frozen_schedule_conflicts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.generate(directory)
            schedule = Path(directory) / SCHEDULE
            log = Path(directory) / RUN_LOG
            original_schedule = schedule.read_bytes()
            log_bytes = log.read_bytes() + b"healthy-01,attempt_01,invalid,timeout\n"
            log.write_bytes(log_bytes)
            self.generate(directory)
            self.assertEqual(schedule.read_bytes(), original_schedule)
            self.assertEqual(log.read_bytes(), log_bytes)
            different_schedule = original_schedule.replace(b"20261122", b"20261123", 1)
            schedule.write_bytes(different_schedule)
            result = self.generate(directory, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(schedule.read_bytes(), different_schedule)
            self.assertEqual(log.read_bytes(), log_bytes)

    def test_pilot_folder_id_rejected_even_for_invalid_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            self.generate(directory)
            schedule = Path(directory) / SCHEDULE
            log = Path(directory) / RUN_LOG
            original_schedule = schedule.read_bytes()
            log_bytes = log.read_bytes() + b"stepped-01,trial_pilot,invalid,timeout\n"
            log.write_bytes(log_bytes)

            result = self.generate(directory, check=False)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Pilot run", result.stderr)
            self.assertEqual(schedule.read_bytes(), original_schedule)
            self.assertEqual(log.read_bytes(), log_bytes)

    def test_pilot_metadata_rejected_even_without_pilot_suffix(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as runs_dir:
            self.generate(directory)
            log = Path(directory) / RUN_LOG
            log_bytes = log.read_bytes() + b"healthy-01,renamed_run,valid,\n"
            log.write_bytes(log_bytes)
            run = Path(runs_dir) / "renamed_run"
            run.mkdir()
            (run / "run.json").write_text('{"pilot": true}', encoding="utf-8")

            result = self.generate(directory, check=False, runs_dir=runs_dir)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Pilot run", result.stderr)
            self.assertEqual(log.read_bytes(), log_bytes)


if __name__ == "__main__":
    unittest.main()
