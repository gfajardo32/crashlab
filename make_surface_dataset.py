"""Build matched event-only and event-plus-memory Surface datasets."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import json
from math import isfinite
from pathlib import Path
import random

LOOKBACK = timedelta(minutes=5)
HORIZON = timedelta(minutes=5)
STRIDE = timedelta(seconds=10)
MAX_SAMPLE_GAP = timedelta(seconds=12)
MAX_EDGE_GAP = timedelta(seconds=8)
MEMORY_BIN = timedelta(seconds=30)
MEMORY_COLUMNS = ("available_bytes", "committed_bytes", "commit_limit_bytes", "pages_per_sec")
MAX_EVENTS = 30
MIB = 1024 ** 2
SEED = 42


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a UTC offset")
    return parsed


def read_memory(path, start, end):
    samples = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = csv.DictReader(source)
        if not {"timestamp_utc", *MEMORY_COLUMNS}.issubset(rows.fieldnames or ()):
            raise ValueError("memory.csv is missing required columns")
        for row in rows:
            try:
                time = timestamp(row["timestamp_utc"]).astimezone(timezone.utc)
                available, committed, limit, pages = (float(row[name]) for name in MEMORY_COLUMNS)
            except (TypeError, ValueError):
                # A crash may interrupt the final CSV row. Coverage gates below
                # exclude any windows affected by missing samples.
                continue
            values = (available, committed, limit, pages)
            if start <= time <= end and limit > 0 and all(isfinite(value) and value >= 0 for value in values):
                samples[time] = values
    return sorted(samples.items())


def read_events(run_dir, zone, start, end):
    paths = [path for path in run_dir.glob("*System*.csv") if "GOLD" not in path.name.upper()]
    if not paths:
        raise ValueError("missing Windows System CSV")

    events = []
    with max(paths).open(encoding="utf-8-sig", newline="") as source:
        rows = csv.DictReader(source, strict=True)
        if not {"TimeCreated", "LevelDisplayName", "Id"}.issubset(rows.fieldnames or ()):
            raise ValueError("System CSV is missing required columns")
        for row in rows:
            try:
                time = datetime.strptime(row["TimeCreated"], "%m/%d/%Y %I:%M:%S %p").replace(tzinfo=zone)
            except ValueError:
                time = datetime.fromisoformat(row["TimeCreated"])
                if time.tzinfo is None:
                    time = time.replace(tzinfo=zone)
            if start <= time <= end:
                level = row["LevelDisplayName"]
                if not level or not level.strip() or "\n" in level or "\r" in level:
                    raise ValueError("invalid System event level")
                events.append((time.astimezone(timezone.utc), level, int(row["Id"])))
    return sorted(events)


def memory_lines(samples, start, cutoff):
    lines = []
    bin_start = start
    while bin_start < cutoff:
        bin_end = bin_start + MEMORY_BIN
        values = [
            (a, c, limit, p) for time, (a, c, limit, p) in samples
            if bin_start <= time < bin_end or time == cutoff == bin_end
        ]
        # The coverage check guarantees a sample in each bin.
        available = min(item[0] for item in values) / MIB
        commit_percent = 100 * max(item[1] / item[2] for item in values)
        pages = max(item[3] for item in values)
        elapsed = int((bin_start - start).total_seconds())
        lines.append(f"  t={elapsed:03d} {available:.0f} {commit_percent:.1f} {pages:.1f}")
        bin_start = bin_end
    return "\n".join(lines)


def examples_for_run(run_dir, metadata):
    status = metadata.get("status")
    died = metadata.get("died")
    if status == "died" and died is True:
        death = timestamp(metadata["death_time_utc"]).astimezone(timezone.utc)
    elif status == "healthy" and died is False:
        if metadata.get("inject_time_utc") is not None or metadata.get("death_time_utc") is not None:
            raise ValueError("healthy control has an injection or crash timestamp")
        death = None
    else:
        raise ValueError(f"run has no verified crash or healthy-control label (status: {status!r})")

    start = timestamp(metadata["observation_start_utc"]).astimezone(timezone.utc)
    end = timestamp(metadata["observation_end_utc"]).astimezone(timezone.utc)
    zone = timestamp(metadata["boot_time_local"]).tzinfo
    if end <= start or (death is not None and not start < death <= end):
        raise ValueError("invalid observation or crash bounds")

    data_end = death or end
    samples = read_memory(run_dir / "memory.csv", start, data_end)
    events = read_events(run_dir, zone, start, data_end)
    sample_times = [time for time, _ in samples]
    event_times = [entry[0] for entry in events]
    if not sample_times:
        return

    question = "Question: will this machine crash within the next 5 minutes? Answer yes or no."
    cutoff = sample_times[0] + LOOKBACK
    last_cutoff = death if death is not None else end - HORIZON
    while cutoff <= last_cutoff and cutoff != death:
        window_start = cutoff - LOOKBACK
        left = bisect_left(sample_times, window_start)
        right = bisect_right(sample_times, cutoff)
        times = sample_times[left:right]
        if (
            times
            and times[0] - window_start <= MAX_EDGE_GAP
            and cutoff - times[-1] <= MAX_EDGE_GAP
            and all(later - earlier <= MAX_SAMPLE_GAP for earlier, later in zip(times, times[1:]))
        ):
            event_end = bisect_right(event_times, cutoff)
            event_start = max(bisect_left(event_times, window_start), event_end - MAX_EVENTS)
            event_lines = [
                f"  t={int((time - window_start).total_seconds()):03d} {level} {event_id}"
                for time, level, event_id in events[event_start:event_end]
            ]
            body = "\n".join(event_lines or ["  (no events logged)"])
            header = f"Windows System events in the last 5 minutes (t = seconds since window start):\n{body}"
            memory = memory_lines(samples[left:right], window_start, cutoff)
            completion = " yes" if death is not None and death <= cutoff + HORIZON else " no"
            yield (
                {"prompt": f"{header}\n{question}", "completion": completion},
                {
                    "prompt": f"{header}\nMemory by 30-second interval (t, min available MiB, "
                              f"peak commit %, peak pages/sec):\n{memory}\n{question}",
                    "completion": completion,
                },
            )
        cutoff += STRIDE


def split_runs(runs):
    splits = {"train": [], "eval": [], "test": []}
    rng = random.Random(SEED)
    for has_crash in (True, False):
        group = [run for run in runs if run[1] == has_crash]
        rng.shuffle(group)
        count = len(group)
        n_test = max(1, round(count * 0.2)) if count >= 2 else 0
        n_eval = max(1, round(count * 0.15)) if count >= 3 else 0
        splits["test"].extend(group[:n_test])
        splits["eval"].extend(group[n_test:n_test + n_eval])
        splits["train"].extend(group[n_test + n_eval:])
    return splits


def build_dataset(runs_dir, out_dir, *, excluded_run_ids=()):
    runs = []
    skipped = {}
    seen = set()
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        state_path = run_dir / "run.json"
        if not state_path.is_file():
            continue
        try:
            metadata = json.loads(state_path.read_text(encoding="utf-8-sig"))
            if "observation_start_utc" not in metadata and not (run_dir / "memory.csv").exists():
                continue
            run_id = metadata["run_id"]
            if not isinstance(run_id, str) or not run_id or run_id in seen:
                raise ValueError("missing or duplicate run ID")
            if run_id in excluded_run_ids:
                skipped[run_dir.name] = "excluded by --exclude-run"
                continue
            pairs = list(examples_for_run(run_dir, metadata))
            if not pairs:
                raise ValueError("no eligible five-minute windows with sufficient memory coverage")
        except (OSError, TypeError, ValueError, KeyError, csv.Error) as exc:
            skipped[run_dir.name] = str(exc)
            continue
        seen.add(run_id)
        runs.append((run_id, metadata["died"], pairs))

    modes = ("event_only", "combined")
    for mode in modes:
        (out_dir / mode).mkdir(parents=True, exist_ok=True)
    splits = {}
    for split, members in split_runs(runs).items():
        labels = Counter(pair[0]["completion"].strip() for _, _, pairs in members for pair in pairs)
        splits[split] = {
            "runs": [run_id for run_id, _, _ in members],
            "examples": sum(labels.values()),
            "yes": labels["yes"],
            "no": labels["no"],
        }
        for index, mode in enumerate(modes):
            with (out_dir / mode / f"{split}.jsonl").open("w", encoding="utf-8") as output:
                for _, _, pairs in members:
                    for pair in pairs:
                        output.write(json.dumps(pair[index]) + "\n")

    manifest = {"splits": splits, "skipped_runs": skipped}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=project.parent / "crashlab-data")
    parser.add_argument("--out-dir", type=Path, default=project / "surface_llm_data")
    parser.add_argument("--exclude-run", action="append", default=[], metavar="RUN_ID",
                        help="Skip a known contaminated run; repeat for each run ID")
    args = parser.parse_args()
    if not args.runs_dir.is_dir():
        parser.error(f"missing run directory: {args.runs_dir}")
    manifest = build_dataset(args.runs_dir, args.out_dir, excluded_run_ids=set(args.exclude_run))
    for name, split in manifest["splits"].items():
        print(f"{name}: {len(split['runs'])} runs, {split['yes']} yes, {split['no']} no")
    if any(not split["yes"] or not split["no"] for split in manifest["splits"].values()):
        print("Some splits lack a label class. Collect more independent crash and healthy runs before fine-tuning.")
    if manifest["skipped_runs"]:
        print(f"Skipped {len(manifest['skipped_runs'])} runs. See manifest.json for reasons.")


if __name__ == "__main__":
    main()
