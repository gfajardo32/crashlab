"""Run the next frozen protocol slot and check collection quality without labels."""

import argparse
import csv
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

from crashloop import HIGH_PLATEAU_PERCENT, PLATEAU_HOLD_LIMIT_MIN, PROTOCOL_HEALTHY_MIN
from crashloop import RAMP_LIMIT_MINUTES, STEPPED_LIMIT_MIN, STEPPED_PATTERNS
from surface_responsiveness import FAILURES, STEP, read_samples, slotted, timestamp


PROJECT = Path(__file__).resolve().parent
RUN_LOG_FIELDS = ("slot_id", "run_id", "status", "reason")
RUN_BANNER = re.compile(r"=== (laptop_[A-Za-z0-9_-]+) ===")
SAVED_PREFIX = "  saved responsiveness telemetry -> "


def next_slot(slots, attempts):
    completed = {row["slot_id"] for row in attempts if row["status"] == "valid"}
    return min((slot for slot in slots if slot["slot_id"] not in completed),
               key=lambda slot: slot["order"], default=None)


def read_attempts(stream, slots):
    reader = csv.DictReader(stream)
    if reader.fieldnames != list(RUN_LOG_FIELDS):
        raise ValueError("Unexpected protocol run-log header")
    known = {slot["slot_id"] for slot in slots}
    rows = list(reader)
    if any(row["slot_id"] not in known or row["status"] not in ("valid", "invalid")
           or any(row[key] is None for key in RUN_LOG_FIELDS) for row in rows):
        raise ValueError("Invalid entry in protocol/run_log.csv")
    return rows


def workload_plan(slot, metadata):
    condition = slot["condition"]
    expected = {}
    if condition == "healthy":
        marker, minutes = "healthy_bursts", PROTOCOL_HEALTHY_MIN
        expected["burst_seed"] = slot["burst_seed"]
    elif condition == "plateau96":
        marker, minutes = "supervised_plateau", PLATEAU_HOLD_LIMIT_MIN
        expected["target_commit_percent"] = HIGH_PLATEAU_PERCENT
    elif condition == "stepped":
        pattern = slot["pattern"]
        target, duration = STEPPED_PATTERNS[pattern]
        marker, minutes = f"stepped_{pattern}", STEPPED_LIMIT_MIN
        expected.update(stepped_pattern=pattern, pause_commit_percent=target, pause_duration_seconds=duration)
    else:
        rate = int(condition.removeprefix("ramp"))
        marker, minutes = "supervised_ramp", RAMP_LIMIT_MINUTES[rate]
        expected.update(allocation_chunk_mib=rate, allocation_chunk_interval_seconds=1)
    expected["workload_time_cap_minutes"] = minutes
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("workload metadata does not match the slot configuration")
    return marker, timedelta(minutes=minutes)


def collection_events(path):
    names = {"started", "intervention", "stopped", "pause_start", "resume", "plateau_settle",
             "topup_started", "topup_allocation_end"}
    events = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if not {"event", "timestamp_utc", "reason"}.issubset(rows.fieldnames or ()):
            raise ValueError("workload.csv is missing required event columns")
        for row in rows:
            name = row["event"]
            if name in names:
                if name in events:
                    raise ValueError(f"duplicate workload event: {name}")
                events[name] = timestamp(row["timestamp_utc"]), row["reason"]
    return events


def verify_workload(events, marker, duration, start, end, metadata):
    if "started" not in events or events["started"][1] != marker:
        raise ValueError("workload did not record the configured start")
    if "intervention" not in events or "stopped" not in events or metadata.get("workload_stop_verified") is not True:
        raise ValueError("workload stop was not verified")
    started = events["started"][0]
    intervention, reason = events["intervention"]
    stopped, confirmation = events["stopped"]
    if not start <= started < intervention <= stopped <= end:
        raise ValueError("workload start and stop are outside the observation interval or out of order")
    if reason not in ("time_cap", "probe_onset") or confirmation != reason or metadata.get("workload_stop_reason") != reason:
        raise ValueError("workload stopped for a reason outside the configured protocol")
    elapsed = intervention - started
    if elapsed > duration + STEP or (reason == "time_cap" and elapsed < duration - STEP):
        raise ValueError("workload stop time does not match its configured cap")
    return intervention, reason, elapsed


def read_probe_presence(row):
    status = row["probe_status"]
    return True if status == "ok" or status in FAILURES else None


def verify_probe_coverage(path, start, end):
    whole_slots, tail = divmod(end - start, STEP)
    expected = whole_slots + bool(tail)
    samples = read_samples(path, "probe_timestamp_utc", ("probe_status",), read_probe_presence)
    observed = sum(start <= slot < end for slot in slotted(samples, start, end))
    if 100 * observed < 95 * expected:
        raise ValueError(f"PowerShell slot coverage {observed}/{expected} is below 95%")


def commit_reading(row):
    committed, limit = int(row["committed_bytes"]), int(row["commit_limit_bytes"])
    return (committed, limit) if committed >= 0 and limit > 0 else None


def verify_hold(path, observation_start, observation_end, start, stop, target):
    # Only in-hold readings can cover any of the hold. Missing slots still
    # contribute to its full duration; endpoint slots get their clipped length.
    samples = read_samples(path, "timestamp_utc", ("committed_bytes", "commit_limit_bytes"), commit_reading)
    counters = slotted(((when, value) for when, value in samples if start <= when < stop),
                       observation_start, observation_end)
    in_band = timedelta()
    for when, (committed, limit) in counters.items():
        if (target - 1) * limit <= 100 * committed <= (target + 1) * limit:
            in_band += max(timedelta(), min(when + STEP, stop) - max(when, start))
    duration = stop - start
    if 5 * in_band < 4 * duration:
        raise ValueError(f"commit hold in band for {in_band / duration:.1%}, below 80%")


def validate_hold(slot, events, run_dir, start, end, stop, early_probe_stop):
    condition = slot["condition"]
    if condition == "stepped":
        target, planned_seconds = STEPPED_PATTERNS[slot["pattern"]]
        pause = events.get("pause_start")
        resume = events.get("resume")
        if resume is None and early_probe_stop and (
                pause is None or stop < pause[0] + timedelta(seconds=planned_seconds)):
            return
        if pause is None or resume is None or not events["started"][0] <= pause[0] < resume[0] <= stop:
            raise ValueError("missing or invalid stepped hold markers")
        if abs(resume[0] - pause[0] - timedelta(seconds=planned_seconds)) > STEP:
            raise ValueError("stepped hold duration does not match its pattern")
        hold_start, hold_end = pause[0], resume[0]
    elif condition == "plateau96":
        if early_probe_stop:
            return
        target = HIGH_PLATEAU_PERCENT
        marker = "topup_allocation_end" if "topup_started" in events else "plateau_settle"
        hold = events.get(marker)
        minimum = events.get("topup_started", events["started"])[0]
        if hold is None or not minimum <= hold[0] < stop:
            raise ValueError("missing or invalid plateau hold marker")
        hold_start, hold_end = hold[0], stop
    else:
        return
    verify_hold(run_dir / "memory.csv", start, end, hold_start, hold_end, target)


def validate_run(slot, run_dir):
    """Return an empty reason for a valid run, or a collection failure reason."""
    try:
        metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8-sig"))
        if (metadata.get("protocol_version") != 3 or metadata.get("pilot") or
                metadata.get("run_id") != run_dir.name):
            raise ValueError("run metadata does not identify this protocol run")
        reboots = metadata["reboot_times_utc"]
        if not isinstance(reboots, list) or reboots:
            raise ValueError("unexpected reboot or invalid reboot records")
        start = timestamp(metadata["observation_start_utc"])
        end = timestamp(metadata["observation_end_utc"])
        if not start < end:
            raise ValueError("invalid observation interval")
        marker, cap = workload_plan(slot, metadata)
        events = collection_events(run_dir / "workload.csv")
        stop, reason, elapsed = verify_workload(events, marker, cap, start, end, metadata)
        verify_probe_coverage(run_dir / "probe_powershell.csv", start, end)
        validate_hold(slot, events, run_dir, start, end, stop, reason == "probe_onset" and elapsed < cap)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError, csv.Error) as error:
        return str(error)
    return ""


def launch_collector(command, project):
    run_id = ""
    saved = None
    with subprocess.Popen(command, cwd=project, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace") as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            banner = RUN_BANNER.fullmatch(line.strip())
            if banner:
                run_id = banner[1]
            if line.startswith(SAVED_PREFIX):
                saved = Path(line[len(SAVED_PREFIX):].strip())
        code = process.wait()
    # Return a path only for an identified and explicitly saved run. An
    # announced ID still identifies a failed attempt with no saved folder.
    return code, saved if saved is not None and saved.name == run_id else run_id


def run_next(project, runs_dir, *, dry_run=False):
    protocol = project / "protocol"
    slots = json.loads((protocol / "protocol_v3.json").read_text(encoding="utf-8-sig"))["runs"]
    with (protocol / "run_log.csv").open("r" if dry_run else "r+", encoding="utf-8-sig", newline="") as log:
        if not dry_run:
            fcntl.flock(log, fcntl.LOCK_EX | fcntl.LOCK_NB)
        slot = next_slot(slots, read_attempts(log, slots))
        if slot is None:
            print("All protocol slots already have a valid run.")
            return 0
        command = [sys.executable, str(project / "crashloop.py"), *slot["collector_args"]]
        print(f"Slot {slot['order']}: {slot['slot_id']}", flush=True)
        print(shlex.join(command), flush=True)
        if dry_run:
            return 0
        before = {p.name for p in runs_dir.iterdir()} if runs_dir.is_dir() else set()
        run_id = ""
        try:
            code, result = launch_collector(command, project)
            run_id = result.name if isinstance(result, Path) else result
            if code != 0:
                reason = f"collector exited with status {code}"
            elif (not isinstance(result, Path) or result.parent.resolve() != runs_dir.resolve() or
                  not result.is_dir() or result.is_symlink() or run_id in before):
                reason = "collector did not save an identifiable new run folder"
            else:
                reason = validate_run(slot, result)
        except KeyboardInterrupt:
            reason = "collector interrupted"
        except OSError as error:
            reason = f"collector could not run: {error}"
        status = "invalid" if reason else "valid"
        log.seek(0)
        needs_newline = not log.read().endswith(("\n", "\r"))
        log.seek(0, os.SEEK_END)
        if needs_newline:
            log.write("\n")
        csv.writer(log).writerow((slot["slot_id"], run_id, status, reason))
        log.flush()
        os.fsync(log.fileno())
        print(f"{status}: {run_id or 'no run folder'}" + (f" ({reason})" if reason else ""))
        return int(bool(reason))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show the next slot and command without collecting")
    args = parser.parse_args()
    try:
        return run_next(PROJECT, PROJECT.parent / "crashlab-data", dry_run=args.dry_run)
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        print(f"run_protocol: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
