"""Generate the frozen candidates for the Surface responsiveness v3 schedule."""

import argparse
import csv
import json
import os
import random
import tempfile
from pathlib import Path


SEED = 20261122
CONDITIONS = ("healthy", "plateau96", "ramp5", "ramp10", "ramp15", "stepped")
PROTOCOL_DIR = Path(__file__).resolve().parent / "protocol"
RUNS_DIR = PROTOCOL_DIR.parent.parent / "crashlab-data"
RUN_LOG_HEADER = "slot_id,run_id,status,reason\n"
COLLECTOR_ARGS = ("1", "--surface-data", "--responsiveness", "--protocol-v3")


def build_protocol():
    rng = random.Random(SEED)
    healthy_seeds = rng.sample(range(2**31), 5)
    runs = []
    for condition in CONDITIONS:
        patterns = ("A", "B", "A", "B", rng.choice(("A", "B"))) if condition == "stepped" else (None,) * 5
        slots = []
        for index, pattern in enumerate(patterns):
            burst_seed = healthy_seeds[index] if condition == "healthy" else None
            if condition == "healthy":
                args = ["--healthy-control", "--burst-seed", str(burst_seed)]
            elif condition == "plateau96":
                args = ["--plateau-percent", "96"]
            elif condition == "stepped":
                args = ["--stepped", pattern]
            else:
                args = ["--ramp", condition.removeprefix("ramp")]
            slots.append(
                {
                    "slot_id": f"{condition}-{index + 1:02d}",
                    "condition": condition,
                    "pattern": pattern,
                    "burst_seed": burst_seed,
                    "collector_args": [*COLLECTOR_ARGS, *args],
                }
            )

        rng.shuffle(slots)
        if condition == "stepped":
            # Hold out one randomly chosen replicate of each pattern.
            held_out = {
                next(slot["slot_id"] for slot in slots if slot["pattern"] == pattern)
                for pattern in ("A", "B")
            }
        else:
            held_out = {slot["slot_id"] for slot in slots[3:]}
        for slot in slots:
            slot["split"] = "held_out" if slot["slot_id"] in held_out else "dev"
        runs.extend(slots)

    rng.shuffle(runs)
    for index, run in enumerate(runs, start=1):
        run["order"] = index
    return {"protocol_version": 3, "seed": SEED, "runs": runs}


def publish_if_absent(path, content):
    """Publish a complete file atomically without replacing an existing file."""
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as staging:
        temporary_path = Path(staging.name)
        try:
            staging.write(content)
            staging.close()
            os.link(temporary_path, path)
        except FileExistsError:
            pass
        finally:
            temporary_path.unlink()


def reject_pilot_runs(log_path, runs_dir):
    if not log_path.exists():
        return
    runs_root = runs_dir.resolve()
    with log_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            run_id = (row.get("run_id") or "").strip()
            if not run_id:
                continue
            if run_id.endswith("_pilot"):
                raise ValueError(f"Pilot run {run_id!r} cannot appear in the protocol run log")
            if Path(run_id).name != run_id or run_id in (".", ".."):
                raise ValueError(f"Invalid run ID in the protocol run log: {run_id!r}")
            metadata_path = (runs_root / run_id / "run.json").resolve()
            if not metadata_path.is_relative_to(runs_root):
                raise ValueError(f"Protocol run metadata is outside the runs directory: {run_id!r}")
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
            except FileNotFoundError:
                # Invalid attempts can end before any trace was written.
                continue
            if not isinstance(metadata, dict):
                raise ValueError(f"Invalid protocol run metadata for {run_id!r}")
            if metadata.get("pilot") is True:
                raise ValueError(f"Pilot run {run_id!r} cannot appear in the protocol run log")


def write_protocol(directory, runs_dir=RUNS_DIR):
    protocol = build_protocol()
    log_path = directory / "run_log.csv"
    reject_pilot_runs(log_path, runs_dir)
    directory.mkdir(parents=True, exist_ok=True)
    schedule_path = directory / "protocol_v3.json"
    if not schedule_path.exists():
        publish_if_absent(schedule_path, json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    if json.loads(schedule_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError(f"Existing schedule differs from the seeded v3 schedule: {schedule_path}")

    if not log_path.exists():
        publish_if_absent(log_path, RUN_LOG_HEADER)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, default=PROTOCOL_DIR)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args()
    try:
        write_protocol(args.protocol_dir, args.runs_dir)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))
    print(f"Protocol v3 schedule and run log ready in {args.protocol_dir}")


if __name__ == "__main__":
    main()
