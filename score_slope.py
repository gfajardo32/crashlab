"""Evaluate the frozen CrashLab protocol v3 commit-slope alarm."""

import argparse
from bisect import bisect_right
from collections import Counter
import csv
from datetime import timedelta
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import median
import tempfile

from crashloop import verify_protocol_freeze
from surface_responsiveness import analyze_run, read_samples, timestamp

LOOKBACKS = (30, 60, 120, 300)
STEP = timedelta(seconds=10)
BUDGET = 1.0
BUDGET_PHASES = frozenset(("baseline", "healthy", "pause", "top_up", "settled_hold"))
PHASES = ("baseline", "initial_rise", "pause", "post_pause_rise", "top_up", "settled_hold", "healthy")
REPORT_VERSION = "crashlab-slope-v3"
HERE = Path(__file__).resolve().parent


def alarm_at(samples, cutoff, lookback, times=None):
    times = [stamp for stamp, _ in samples] if times is None else times
    lo = bisect_right(times, cutoff - timedelta(seconds=lookback))
    hi = bisect_right(times, cutoff)
    window = samples[lo:hi]
    if len(window) < lookback // 5 - 1:
        return False
    x = [(stamp - cutoff).total_seconds() for stamp, _ in window]
    y = [ratio for _, ratio in window]
    x_mean, y_mean = sum(x) / len(x), sum(y) / len(y)
    variance = sum((point - x_mean) ** 2 for point in x)
    if variance == 0:
        return False
    slope = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) / variance
    return y[-1] + 300 * slope >= 1.0


def episode_count(negative_windows):
    """Count starts in one ordered stream; a gap or non-alarm ends a streak."""
    episodes = 0
    previous = None
    for cutoff, alarm in negative_windows:
        if alarm:
            if previous is None or cutoff - previous != STEP:
                episodes += 1
            previous = cutoff
        else:
            previous = None
    return episodes


def phase_at(cutoff, condition, events):
    if cutoff < events["started"]:
        return "baseline"
    if condition == "healthy":
        return "healthy"
    if condition == "stepped" and "pause_start" in events:
        if cutoff < events["pause_start"]:
            return "initial_rise"
        resume = events.get("resume")
        return "pause" if resume is None or cutoff < resume + timedelta(seconds=60) else "post_pause_rise"
    if condition == "plateau96":
        topup = events.get("topup_started")
        if topup is not None and cutoff >= topup:
            return ("top_up" if cutoff < events["topup_allocation_end"] + timedelta(seconds=60)
                    else "settled_hold")
        if "plateau_settle" in events and cutoff >= events["plateau_settle"]:
            return "settled_hold"
    return "initial_rise"


def positive_result(onset, windows):
    leads = [(onset - cutoff).total_seconds() for cutoff, label, _ in windows if label == "yes"] if onset else []
    alarms = [(onset - cutoff).total_seconds()
              for cutoff, label, alarm in windows if label == "yes" and alarm] if onset else []
    return {
        "positive": onset is not None,
        "on_time_evaluable": any(290 <= lead <= 300 for lead in leads),
        "caught_on_time": any(290 <= lead <= 300 for lead in alarms),
        "caught_at_any_lead": bool(alarms),
        "first_alarm_lead_seconds": max(alarms, default=None),
    }


def commit_ratio(row):
    committed, limit = float(row["committed_bytes"]), float(row["commit_limit_bytes"])
    if isfinite(committed) and isfinite(limit) and committed >= 0 and limit > 0:
        return committed / limit
    return None


def workload_events(path, metadata, condition):
    events = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            events.setdefault(row["event"], timestamp(row["timestamp_utc"]))
    if "started" not in events:
        raise ValueError(f"{path}: missing workload start")
    if condition == "plateau96" and "topup_started" in events and "topup_allocation_end" not in events:
        duration = metadata["topup_allocation_chunk_count"] * metadata["allocation_chunk_interval_seconds"]
        events["topup_allocation_end"] = events["topup_started"] + timedelta(seconds=duration)
    return events


def window_summary(windows):
    seconds = len(windows) * int(STEP.total_seconds())
    episodes = episode_count(windows)
    return {
        "eligible_negative_windows": len(windows),
        "eligible_negative_seconds": seconds,
        "episodes": episodes,
        "episodes_per_hour": 3600 * episodes / seconds if seconds else None,
    }


def read_protocol_metadata(run_id, run_dir, freeze_commit):
    if run_id.endswith("_pilot"):
        raise ValueError(f"Pilot run {run_id!r} cannot be scored")
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8-sig"))
    if metadata.get("pilot") is True:
        raise ValueError(f"{run_id}: pilot runs are excluded")
    if metadata.get("protocol_version") != 3:
        raise ValueError(f"{run_id}: expected protocol v3 telemetry; pilot runs are excluded")
    if metadata.get("freeze_commit") != freeze_commit:
        raise ValueError(f"{run_id}: run freeze commit does not match the current freeze")
    return metadata


def score_run(slot, run_id, run_dir, lookbacks, freeze_commit):
    metadata = read_protocol_metadata(run_id, run_dir, freeze_commit)
    labeled = analyze_run(run_dir, metadata)
    workload = workload_events(run_dir / "workload.csv", metadata, slot["condition"])
    samples = read_samples(run_dir / "memory.csv", "timestamp_utc",
                           ("committed_bytes", "commit_limit_bytes"), commit_ratio)
    times = [stamp for stamp, _ in samples]
    phases = [phase_at(example.cutoff, slot["condition"], workload) for example in labeled.examples]
    results = {}
    for lookback in lookbacks:
        windows = [(example.cutoff, example.label, alarm_at(samples, example.cutoff, lookback, times))
                   for example in labeled.examples]
        negative = [(cutoff, alarm) for cutoff, label, alarm in windows if label == "no"]
        phased = [(phase, cutoff, alarm) for phase, (cutoff, label, alarm) in zip(phases, windows) if label == "no"]
        # The overall streak is continuous across phase boundaries. Scoped
        # phase/budget streams break a streak at any excluded or missing cutoff.
        results[lookback] = {
            "slot_id": slot["slot_id"],
            "run_id": run_id,
            "condition": slot["condition"],
            "pattern": slot.get("pattern"),
            "onset_utc": labeled.onset.isoformat() if labeled.onset else None,
            "diagnostic": labeled.diagnostic,
            **positive_result(labeled.onset, windows),
            "overall": window_summary(negative),
            "budget": window_summary([(cutoff, alarm) for phase, cutoff, alarm in phased if phase in BUDGET_PHASES]),
            "phases": {
                phase: window_summary([(cutoff, alarm) for current, cutoff, alarm in phased if current == phase])
                for phase in PHASES
            },
        }
    return results


def pooled(summaries):
    totals = Counter()
    for summary in summaries:
        totals.update({key: summary[key] for key in ("eligible_negative_windows", "eligible_negative_seconds", "episodes")})
    time = totals["eligible_negative_seconds"]
    return {**totals, "episodes_per_hour": 3600 * totals["episodes"] / time if time else None}


def score_split(assignments, runs_dir, lookbacks, freeze_commit):
    by_lookback = {lookback: [] for lookback in lookbacks}
    for slot, run_id in assignments:
        results = score_run(slot, run_id, runs_dir / run_id, lookbacks, freeze_commit)
        for lookback, result in results.items():
            by_lookback[lookback].append(result)
    table = []
    for lookback, records in by_lookback.items():
        leads = [r["first_alarm_lead_seconds"] for r in records if r["caught_at_any_lead"]]
        table.append({
            "lookback_seconds": lookback,
            "positive_runs": sum(r["positive"] for r in records),
            "on_time_evaluable_positive_runs": sum(r["on_time_evaluable"] for r in records),
            "on_time_not_evaluable_positive_runs": sum(r["positive"] and not r["on_time_evaluable"] for r in records),
            "caught_on_time_runs": sum(r["caught_on_time"] for r in records),
            "caught_at_any_lead_runs": len(leads),
            "median_first_alarm_lead_seconds": median(leads) if leads else None,
            "overall": pooled(r["overall"] for r in records),
            "budget": pooled(r["budget"] for r in records),
            "phases": {phase: pooled(r["phases"][phase] for r in records) for phase in PHASES},
            "runs": records,
        })
    return table


def select_lookback(table):
    within = [row for row in table if row["budget"]["episodes_per_hour"] is not None
              and row["budget"]["episodes_per_hour"] <= BUDGET]
    best = min(within, key=lambda row: (-row["caught_on_time_runs"], row["budget"]["episodes_per_hour"],
                                       -row["lookback_seconds"]), default=None)
    return best["lookback_seconds"] if best else None


def assignments_for(slots, log, split):
    known = {slot["slot_id"] for slot in slots}
    valid = {}
    used_runs = set()
    with log.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            slot, run, status = row["slot_id"], row["run_id"], row["status"]
            if run and run.endswith("_pilot"):
                raise ValueError(f"Pilot run {run!r} cannot be referenced in the protocol run log")
            if slot not in known or status not in ("valid", "invalid"):
                raise ValueError(f"Unrecognized run-log entry for slot {slot!r}: {status}")
            if status == "invalid":
                continue
            if not run or Path(run).name != run or run in (".", "..") or slot in valid or run in used_runs:
                raise ValueError(f"Duplicate or invalid run-log mapping for slot {slot!r}: {run!r}")
            valid[slot] = run
            used_runs.add(run)
    selected = [slot for slot in slots if slot["split"] == split]
    if not selected:
        raise ValueError(f"No {split} slots in schedule")
    missing = [slot["slot_id"] for slot in selected if slot["slot_id"] not in valid]
    if missing:
        raise ValueError(f"Incomplete {split} run log: {', '.join(missing)}")
    return [(slot, valid[slot["slot_id"]]) for slot in selected]


def assignment_ids(assignments):
    return [[slot["slot_id"], run_id] for slot, run_id in assignments]


def write_report(path, report):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.close()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def evaluate(command, protocol_dir, runs_dir):
    marker = protocol_dir / "slope_held_out_started.json"
    held_out = protocol_dir / "slope_held_out.json"
    if marker.exists() or held_out.exists():
        raise ValueError("Held-out scoring has already started; the protocol can only be evaluated once")

    schedule_path = protocol_dir / "protocol_v3.json"
    schedule = schedule_path.read_bytes()
    slots = json.loads(schedule)["runs"]
    log = protocol_dir / "run_log.csv"
    dev_path = protocol_dir / "slope_dev.json"
    dev_assignments = assignments_for(slots, log, "dev")
    freeze_commit = verify_protocol_freeze(protocol_dir.parent)
    provenance = {
        "report_version": REPORT_VERSION,
        "freeze_commit": freeze_commit,
        "protocol_sha256": sha256((protocol_dir / "PROTOCOL_v3.md").read_bytes()).hexdigest(),
        "schedule_sha256": sha256(schedule).hexdigest(),
        "dev_assignments": assignment_ids(dev_assignments),
    }
    if command == "dev":
        table = score_split(dev_assignments, runs_dir, LOOKBACKS, freeze_commit)
        result = {**provenance, "split": "dev", "selected_lookback_seconds": select_lookback(table), "table": table}
        write_report(dev_path, result)
        return dev_path, result

    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    if any(dev.get(key) != value for key, value in provenance.items()) or dev.get("split") != "dev":
        raise ValueError("Development result does not match the current protocol, schedule and run log")
    selected = dev["selected_lookback_seconds"]
    if selected not in (*LOOKBACKS, None) or selected != select_lookback(dev["table"]):
        raise ValueError("Development selection does not match the frozen selection rule")
    holdout = assignments_for(slots, log, "held_out")
    missing = [run_id for _, run_id in holdout if not (runs_dir / run_id).is_dir()]
    if missing:
        raise ValueError(f"Held-out run folders missing: {', '.join(missing)}")
    for _, run_id in holdout:
        read_protocol_metadata(run_id, runs_dir / run_id, freeze_commit)
    result = {
        **provenance,
        "split": "held_out",
        "selected_lookback_seconds": selected,
        "held_out_assignments": assignment_ids(holdout),
        "dev_report_sha256": sha256(dev_path.read_bytes()).hexdigest(),
    }
    # Exclusive creation precedes held-out scoring. Preserve the marker
    # after any failure so partially observed scores cannot be retried.
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    result["table"] = score_split(
        holdout, runs_dir, LOOKBACKS if selected is None else (selected,), freeze_commit,
    )
    write_report(held_out, result)
    return held_out, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("dev", "held-out"))
    parser.add_argument("--protocol-dir", type=Path, default=HERE / "protocol")
    parser.add_argument("--runs-dir", type=Path, default=HERE.parent / "crashlab-data")
    args = parser.parse_args()
    try:
        path, result = evaluate(args.command, args.protocol_dir, args.runs_dir)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, csv.Error) as error:
        parser.exit(1, f"score_slope: {error}\n")
    for row in result["table"]:
        rate = row["budget"]["episodes_per_hour"]
        rate_display = "n/a" if rate is None else f"{rate:.3f}"
        print(f"{row['lookback_seconds']:>3} s: {row['caught_on_time_runs']}/"
              f"{row['on_time_evaluable_positive_runs']} on time; "
              f"{row['on_time_not_evaluable_positive_runs']} not evaluable; {rate_display} budget episodes/hour")
    print(f"Selected lookback: {result['selected_lookback_seconds']}; report: {path}")


if __name__ == "__main__":
    main()
