import datetime as dt
import hashlib
import io
import json
import subprocess
import tempfile
import runpy
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import Mock, patch


MODULE = Path(__file__).resolve().parents[1] / "crashloop.py"
PINNED_SOURCES = (
    "crashloop.py", "surface_responsiveness.py", "make_surface_dataset.py",
    "score_slope.py", "make_protocol.py",
)


class ProtocolCollectorTests(unittest.TestCase):
    def setUp(self):
        self.main = runpy.run_path(str(MODULE))["main"]
        self.env = self.main.__globals__

    def test_ramp15_uses_requested_rate_and_cap(self):
        script = self.env["ramp_workload_script"]("workload.csv", "probe.csv", ramp_rate=15)
        self.assertIn("-d 15 -e 1", script)
        self.assertIn("720000", script)

    def test_stepped_target_and_duration_are_local_and_pattern_specific(self):
        for pattern, target, hold in (("A", 75, 60), ("B", 60, 360)):
            script = self.env["stepped_workload_script"]("workload.csv", "probe.csv", "memory.csv",
                                                         dt.datetime.now(dt.timezone.utc), pattern)
            self.assertIn(f"$pauseTarget = {target}", script)
            self.assertIn(f"$pauseSeconds = {hold}", script)
            self.assertIn("'pause_start'", script)
            self.assertIn("'resume'", script)

    def test_bursts_seeded_and_within_protocol_bounds(self):
        build = self.env["healthy_burst_schedule"]
        schedule = build(20261122)
        self.assertEqual(schedule, build(20261122))
        self.assertNotEqual(schedule, build(20261123))
        start = 0
        for offset, size, hold in schedule:
            self.assertTrue(60 <= offset - start <= 180)
            self.assertTrue(200 <= size <= 600)
            self.assertTrue(15 <= hold <= 60)
            self.assertLessEqual(offset + hold, 30 * 60)
            start = offset

    def test_freeze_pins_protocol_and_sources_and_returns_revision(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol = root / "protocol"
            protocol.mkdir()
            files = (protocol / "PROTOCOL_v3.md", protocol / "protocol_v3.json")

            def git(*args):
                return subprocess.check_output(["git", "-C", folder, *args], text=True).strip()

            def commit(message):
                git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", message)

            git("init", "-q")
            for name in PINNED_SOURCES:
                (root / name).write_text(name, encoding="utf-8")
            for file in files:
                file.write_text(file.name, encoding="utf-8")
            git("add", "-A")
            commit("Freeze\n\n" + "\n".join(hashlib.sha256(file.read_bytes()).hexdigest() for file in files))
            freeze = git("rev-parse", "HEAD")
            self.assertEqual(self.env["verify_protocol_freeze"](root), freeze)

            amended_source = root / "crashloop.py"
            files[0].write_text("Protocol v3.1: forced restart", encoding="utf-8")
            amended_source.write_text("Restart-Computer -Force", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                self.env["verify_protocol_freeze"](root)
            git("add", "--", str(files[0]), str(amended_source))
            commit("Freeze v3.1\n\n" + "\n".join(hashlib.sha256(file.read_bytes()).hexdigest() for file in files))
            amended_freeze = git("rev-parse", "HEAD")
            self.assertNotEqual(amended_freeze, freeze)
            self.assertEqual(self.env["verify_protocol_freeze"](root), amended_freeze)

            for file in (*files, *(root / name for name in PINNED_SOURCES)):
                with self.subTest(file=file):
                    original = file.read_bytes()
                    file.write_text("edited", encoding="utf-8")
                    with self.assertRaises(RuntimeError):
                        self.env["verify_protocol_freeze"](root)
                    git("add", "--", str(file))
                    file.write_bytes(original)
                    with self.assertRaises(RuntimeError):
                        self.env["verify_protocol_freeze"](root)
                    git("reset", "-q", "HEAD", "--", str(file))
            edited_source = amended_source
            frozen_source = edited_source.read_bytes()
            edited_source.write_text("changed in a later commit", encoding="utf-8")
            git("add", "--", str(edited_source))
            commit("Change source after freeze")
            with self.assertRaises(RuntimeError):
                self.env["verify_protocol_freeze"](root)
            git("rm", "-q", "--", str(edited_source))
            commit("Remove a pinned source")
            edited_source.write_bytes(frozen_source)
            with self.assertRaises(RuntimeError):
                self.env["verify_protocol_freeze"](root)

    def test_protocol_restart_forces_console_session_to_close(self):
        old_boot = "2026-11-22T07:30:00+00:00"
        new_boot = "2026-11-22T07:32:00+00:00"
        freeze = "a" * 40
        restart = Mock(return_value=(True, ""))
        overrides = {
            "verify_protocol_freeze": Mock(return_value=freeze),
            "boot_time": Mock(return_value=old_boot),
            "verified_boot_time": Mock(return_value=new_boot),
            "surface_utc_now": Mock(return_value="2026-11-22T07:37:00+00:00"),
            "wait_until": lambda predicate, *args: predicate(),
            "ps": restart,
        }
        with patch.dict(self.env, overrides), patch.object(self.env["time"], "sleep"):
            self.assertEqual(self.env["restart_for_protocol"](), (new_boot, freeze))
        restart.assert_called_once()
        self.assertTrue(restart.call_args.args[0].endswith("; Restart-Computer -Force"))

    def test_protocol_cli_routes_the_stepped_pattern(self):
        argv = ["crashloop.py", "1", "--surface-data", "--responsiveness",
                "--protocol-v3", "--stepped", "A"]
        collector = Mock()
        with patch.object(sys, "argv", argv), patch.dict(self.env, {"collect_responsiveness_run": collector}):
            self.main()
        self.assertEqual(collector.call_args.kwargs["stepped"], "A")
        self.assertTrue(collector.call_args.kwargs["protocol_v3"])

    def test_pilot_cli_routes_stepped_and_seeded_burst_without_protocol(self):
        for options, expected in (
            (("--stepped", "B"), {"stepped": "B", "burst_seed": None}),
            (("--healthy-control", "--burst-seed", "43"), {"stepped": None, "burst_seed": 43}),
        ):
            with self.subTest(options=options):
                argv = ["crashloop.py", "1", "--surface-data", "--responsiveness", "--pilot", *options]
                collector = Mock()
                with patch.object(sys, "argv", argv), patch.dict(self.env, {"collect_responsiveness_run": collector}):
                    self.main()
                kwargs = collector.call_args.kwargs
                self.assertTrue(kwargs["pilot"])
                self.assertFalse(kwargs["protocol_v3"])
                for key, value in expected.items():
                    self.assertEqual(kwargs[key], value)

    def test_pilot_and_protocol_flags_are_exclusive(self):
        argv = ["crashloop.py", "--surface-data", "--responsiveness", "--stepped", "A",
                "--protocol-v3", "--pilot"]
        collector = Mock()
        with patch.object(sys, "argv", argv), patch.dict(self.env, {"collect_responsiveness_run": collector}), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.main()
        collector.assert_not_called()

    def test_collected_pilot_and_protocol_metadata(self):
        boot = "2026-11-22T07:30:00+00:00"
        freeze = "a" * 40
        for pilot in (False, True):
            with self.subTest(pilot=pilot), tempfile.TemporaryDirectory() as folder:
                runs = Path(folder) / "runs"
                runs.mkdir()
                mock_restart = Mock(return_value=(boot, freeze))
                overrides = {
                    "RUNS_DIR": str(runs),
                    "restart_for_protocol": mock_restart,
                    "boot_time": Mock(return_value=boot),
                    "verified_boot_time": Mock(return_value=boot),
                    "launch_responsiveness_sampler": Mock(),
                    "launch_protocol_workload": Mock(return_value={
                        "inject_time_utc": "2026-11-22T07:38:05+00:00", "workload_time_cap_minutes": 30,
                    }),
                    "ps": Mock(return_value=(True, boot)),
                    "wait_until": lambda predicate, *args: predicate(),
                    "record_ssh_probe": Mock(),
                    "remove_workload_task": Mock(return_value=True),
                    "copy_remote": Mock(),
                    "collect_event_exports": Mock(),
                    "summarize_ramp": Mock(),
                }
                with patch.dict(self.env, overrides), patch.object(self.env["time"], "sleep"):
                    self.env["collect_responsiveness_run"](
                        timeout_ms=4000, burst_seed=43, protocol_v3=not pilot, pilot=pilot,
                    )
                (run,) = runs.iterdir()
                metadata = json.loads((run / "run.json").read_text())
                if pilot:
                    mock_restart.assert_not_called()
                    self.assertTrue(run.name.endswith("_pilot"))
                    self.assertIs(metadata["pilot"], True)
                    self.assertNotIn("freeze_commit", metadata)
                else:
                    mock_restart.assert_called_once()
                    self.assertEqual(metadata["freeze_commit"], freeze)
                    self.assertEqual(metadata["protocol_version"], 3)


if __name__ == "__main__":
    unittest.main()
