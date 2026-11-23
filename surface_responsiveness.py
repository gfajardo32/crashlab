"""Build five-minute Windows PowerShell responsiveness examples."""

import argparse
from bisect import bisect_right
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from math import isfinite
from pathlib import Path
import re

from make_surface_dataset import read_events

STEP = timedelta(seconds=5)
SLOTS = 60
SPAN = STEP * SLOTS
MIN_COUNTERS = 54
MIN_HEALTHY = 54
SLOW_PROBE_MS = 1000
FAILURES = frozenset(("spawn_error", "timeout", "exit_nonzero", "probe_error"))
COUNTERS = ("available_bytes", "committed_bytes", "commit_limit_bytes", "pages_per_sec")
FRACTION = re.compile(r"([.,]\d{6})\d+")
MAX_EVENTS = 30


def timestamp(value):
    """Read a timezone-aware PowerShell/.NET timestamp on Python 3.8+."""
    normalized = FRACTION.sub(r"\1", value)
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a UTC offset")
    return parsed


def read_samples(path, time_field, fields, convert):
    samples = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {time_field, *fields}.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path.name} is missing required columns")
        for row in reader:
            try:
                when = timestamp(row[time_field])
                sample = convert(row)
            except (AttributeError, TypeError, ValueError):
                # An interrupted write, invalid timestamp or invalid reading
                # creates a missing observation and never a failed probe.
                continue
            if sample is not None:
                samples.append((when, sample))
    return sorted(samples, key=lambda sample: sample[0])


def counter_values(row):
    available, committed, limit = (float(row[field]) for field in COUNTERS[:3])
    pages = float(row[COUNTERS[3]]) if row[COUNTERS[3]] else None
    values = (available, committed, limit, pages)
    if limit > 0 and all(value is None or (isfinite(value) and value >= 0) for value in values):
        return values
    return None


def probe_observation(row):
    status = row["probe_status"]
    if status != "ok" and status not in FAILURES:
        return None
    try:
        duration = float(row.get("probe_latency_ms", ""))
    except (TypeError, ValueError):
        duration = None
    if duration is not None and (not isfinite(duration) or duration < 0):
        duration = None
    return status, duration


def probe_limits(metadata):
    cap = metadata.get("probe_observation_cap_ms", metadata.get("probe_timeout_ms"))
    threshold = metadata.get("probe_failure_threshold_ms")
    for value in (cap, threshold):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError("probe limits must be positive milliseconds")
    if threshold is not None and (cap is None or threshold > cap):
        raise ValueError("probe failure threshold exceeds its observation cap")
    return cap, threshold


def slotted(samples, start, end):
    """Assign each sample to its closest five-second slot, once."""
    nearest = {}
    for when, value in samples:
        if not start <= when <= end:
            continue
        slot = start + STEP * int(((when - start) + STEP / 2) // STEP)
        if slot > end:
            continue
        distance = abs(when - slot)
        if slot not in nearest or distance < nearest[slot][0]:
            nearest[slot] = (distance, value)
    return {slot: value for slot, (_, value) in nearest.items()}


def missing_spans(slots, statuses):
    start = None
    count = 0
    for slot, status in zip(slots, statuses):
        if status == "missing":
            start = slot if count == 0 else start
            count += 1
        elif count:
            yield start, start + (count - 1) * STEP, count
            count = 0
    if count:
        yield start, start + (count - 1) * STEP, count


def first_onset(slots, statuses, reboots):
    count = 0
    for slot, status in zip(slots, statuses):
        if any(slot - STEP < reboot <= slot for reboot in reboots):
            count = 0
        count = count + 1 if status in FAILURES else 0
        if count == 3:
            return slot - 2 * STEP
    return None


@dataclass(frozen=True)
class Example:
    cutoff: datetime
    label: str
    features: dict


@dataclass
class RunAnalysis:
    onset: datetime
    sensitivity_onsets: dict
    observation_cap_ms: int
    failure_threshold_ms: int
    slow_sensitivity_eligible: bool
    diagnostic: bool
    probe_slots: dict
    counter_slots: dict
    examples: list
    candidates: dict
    counter_drops: dict


def window_features(counters, events, start):
    points = []
    # Use elapsed seconds, never run identity, absolute timestamps or settings.
    for offset in range(5, 301, 5):
        slot = start + timedelta(seconds=offset)
        if slot in counters:
            available, committed, limit, pages = counters[slot]
            points.append([offset, available, committed, limit, pages])
    event_features = [[level, event_id] for _, level, event_id in events[-MAX_EVENTS:]]
    return {
        "memory_columns": ["elapsed_seconds", *COUNTERS],
        "memory": points,
        "event_columns": ["level", "id"],
        "events": event_features,
    }


def analyze_run(run_dir, metadata):
    start = timestamp(metadata["observation_start_utc"])
    end = timestamp(metadata["observation_end_utc"])
    if end <= start:
        raise ValueError("observation end must follow its start")
    slots = [start + i * STEP for i in range((end - start) // STEP + 1)]
    probes = read_samples(run_dir / "probe_powershell.csv", "probe_timestamp_utc", ("probe_status",), probe_observation)
    probe_slots = slotted(probes, start, end)
    observations = [probe_slots.get(slot, ("missing", None)) for slot in slots]
    cap, threshold = probe_limits(metadata)
    statuses = [
        ("missing" if duration is None else "timeout" if duration > threshold else "ok")
        if threshold is not None and status == "ok" else status
        for status, duration in observations
    ]
    reboots = [timestamp(value) for value in metadata.get("reboot_times_utc", [])]
    onset = first_onset(slots, statuses, reboots)
    # With older unlabeled cap metadata, completed measurements are still
    # usable if there are no potentially right-censored timeouts.
    slow_eligible = (cap >= SLOW_PROBE_MS if cap is not None else
                     all(status != "timeout" for status, _ in observations))
    sensitivity_onsets = {
        "missing_as_fail": first_onset(
            slots, ("timeout" if status == "missing" else status for status in statuses), reboots
        ),
        "three_launches_over_1s": first_onset(
            slots, ("timeout" if duration is not None and duration > SLOW_PROBE_MS else "ok"
                    for _, duration in observations), reboots
        ) if slow_eligible else None,
    }
    silence = [(a, b) for a, b, count in missing_spans(slots, statuses) if count >= 6]
    diagnostic = bool(onset is None and silence and silence[-1][1] == slots[-1])

    samples = read_samples(run_dir / "memory.csv", "timestamp_utc", COUNTERS, counter_values)
    counter_slots = slotted(samples, start, end)
    result = RunAnalysis(
        onset, sensitivity_onsets, cap, threshold, slow_eligible, diagnostic,
        dict(zip(slots, statuses)), counter_slots, [], Counter(), Counter()
    )
    if diagnostic:
        return result

    interventions = [timestamp(value) for value in metadata.get("intervention_times_utc", [])]
    # Older captures also listed the top-up as an intervention. Remove only
    # that entry before adding reboots, so censoring still applies to exits.
    if metadata.get("topup_time_utc") is not None:
        topup = timestamp(metadata["topup_time_utc"])
        if topup in interventions:
            interventions.remove(topup)
    interventions = sorted(reboots + interventions)
    zone = timestamp(metadata["boot_time_local"]).tzinfo
    events = read_events(run_dir, zone, start, end)
    sample_times = [when for when, _ in samples]
    event_times = [when for when, _, _ in events]
    # Five-second slots define observations; emit windows on the existing ten-second stride.
    for index in range(SLOTS, len(slots), 2):
        cutoff = slots[index]
        horizon = cutoff + SPAN
        if onset is not None and cutoff >= onset:
            break
        lookback = cutoff - SPAN
        if any(lookback < reboot <= cutoff for reboot in reboots):
            continue
        stop = next((event for event in interventions if cutoff <= event <= horizon), None)
        positive = onset is not None and cutoff < onset <= horizon and (stop is None or onset < stop)
        if not positive and stop is not None:
            continue
        observed_through = onset if positive else horizon
        if any(cutoff < gap_end and gap_start <= observed_through for gap_start, gap_end in silence):
            continue
        if positive:
            label = "yes"
        else:
            future = statuses[index + 1:index + SLOTS + 1]
            if len(future) != SLOTS or future.count("ok") < MIN_HEALTHY or any(
                count > 2 for _, _, count in missing_spans(slots[index + 1:index + SLOTS + 1], future)
            ):
                continue
            label = "no"
        result.candidates[label] += 1
        # Filter raw timestamps before snapping: a sample collected just after
        # the cutoff cannot fill even an earlier nominal slot.
        causal = samples[bisect_right(sample_times, lookback):bisect_right(sample_times, cutoff)]
        window_counters = slotted(causal, lookback, cutoff)
        window_counters.pop(lookback, None)
        if len(window_counters) < MIN_COUNTERS:
            result.counter_drops[label] += 1
            continue
        window_events = events[bisect_right(event_times, lookback):bisect_right(event_times, cutoff)]
        features = window_features(window_counters, window_events, lookback)
        result.examples.append(Example(cutoff, label, features))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    example_offset = 0
    with (args.out_dir / "examples.jsonl").open("w", encoding="utf-8") as stream:
        for run in sorted(path for path in args.runs_dir.iterdir() if (path / "probe_powershell.csv").is_file()):
            try:
                metadata = json.loads((run / "run.json").read_text(encoding="utf-8-sig"))
                analysis = analyze_run(run, metadata)
            except (OSError, TypeError, ValueError, KeyError, csv.Error) as error:
                manifest[run.name] = {"error": str(error)}
                continue
            counts = Counter(example.label for example in analysis.examples)
            manifest[run.name] = {
                "diagnostic": analysis.diagnostic,
                "onset_utc": analysis.onset.astimezone(timezone.utc).isoformat() if analysis.onset else None,
                "probe_observation_cap_ms": analysis.observation_cap_ms,
                "probe_failure_threshold_ms": analysis.failure_threshold_ms,
                "sensitivity_max_threshold_ms": analysis.observation_cap_ms,
                "three_launches_over_1s_eligible": analysis.slow_sensitivity_eligible,
                "sensitivity_onsets_utc": {
                    rule: onset.astimezone(timezone.utc).isoformat() if onset else None
                    for rule, onset in analysis.sensitivity_onsets.items()
                },
                "sensitivity_assumptions": {
                    "missing_as_fail": "Count each missing slot as a failed launch.",
                    "three_launches_over_1s": "Three consecutive observed attempts with latency strictly above 1000 ms.",
                },
                "labels": counts,
                "example_offset": example_offset,
                "example_count": len(analysis.examples),
                "lookback_candidates": analysis.candidates,
                "lookback_drops": analysis.counter_drops,
                "cutoffs_utc": [example.cutoff.astimezone(timezone.utc).isoformat() for example in analysis.examples],
            }
            for example in analysis.examples:
                stream.write(json.dumps({"features": example.features, "label": example.label}) + "\n")
            example_offset += len(analysis.examples)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
