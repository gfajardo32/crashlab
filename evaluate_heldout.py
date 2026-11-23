"""Run the frozen Protocol v3.2 held-out evaluation exactly once."""

import argparse
from bisect import bisect_right
import csv
from datetime import timedelta
import json
from math import isfinite, nextafter
from pathlib import Path

from llm_arm.metrics import score_alarms, score_slopes, summarize_records
from llm_arm.prompt import make_prompt
from llm_arm.protocol_data import load_context
from llm_arm.select_threshold import load_selection
from llm_arm.train import load_predictor, validate_adapter
from make_surface_dataset import read_events
from score_slope import LOOKBACKS, assignment_ids, write_report
from surface_responsiveness import timestamp


HERE = Path(__file__).resolve().parent
EVENT_LOOKBACK = timedelta(seconds=300)
REPORT_NAME = "heldout_v32.json"
MARKER_NAME = "slope_held_out_started.json"
PRIOR_SLOPE_REPORT = "slope_held_out.json"


def event_alarm_at(event_times, cutoff):
    """Test whether a sorted list of Event 2004 timestamps hits the window."""
    return bisect_right(event_times, cutoff) > bisect_right(event_times, cutoff - EVENT_LOOKBACK)


def compute_arms(context, assignments, lookback, threshold, predict):
    slope = score_slopes(context, assignments, LOOKBACKS if lookback is None else (lookback,))
    event_records, llm_records = [], []
    for slot, run_id in assignments:
        metadata, analysis, run_dir = context.load_run(slot, run_id)
        start = timestamp(metadata["observation_start_utc"])
        end = timestamp(metadata["observation_end_utc"])
        zone = timestamp(metadata["boot_time_local"]).tzinfo
        # Use the raw event list, because a feature window may cap its events.
        event_times = [when for when, _, event_id in read_events(run_dir, zone, start, end)
                       if event_id == 2004]
        event_alarms, llm_alarms = {}, {}
        for example in analysis.examples:
            cutoff = example.cutoff
            event_alarms[cutoff] = event_alarm_at(event_times, cutoff)
            probability = predict(make_prompt(example))
            if not isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Final adapter returned an invalid alarm probability")
            llm_alarms[cutoff] = probability >= threshold
        event_records.append(score_alarms(slot, run_id, metadata, analysis, run_dir, event_alarms))
        llm_records.append(score_alarms(slot, run_id, metadata, analysis, run_dir, llm_alarms))
    return {
        "slope": slope,
        "event": summarize_records(event_records),
        "llm": summarize_records(llm_records),
    }


def evaluate(protocol_dir, runs_dir, selection_dir, adapter_dir):
    marker = protocol_dir / MARKER_NAME
    output = protocol_dir / REPORT_NAME
    if any(path.exists() or path.is_symlink()
           for path in (marker, output, protocol_dir / PRIOR_SLOPE_REPORT)):
        raise ValueError("Held-out scoring has already started; the protocol can only be evaluated once")

    context = load_context(protocol_dir, runs_dir)
    slope, llm = load_selection(context, selection_dir)
    lookback = slope["selected_lookback_seconds"]
    threshold = llm["selected_threshold"]
    if lookback is not None and lookback not in LOOKBACKS:
        raise ValueError("Invalid recorded slope selection")
    if type(threshold) not in (int, float) or not isfinite(threshold) or not 0 <= threshold <= nextafter(1.0, 2.0):
        raise ValueError("A valid LLM alarm threshold must be recorded before scoring held-out runs")

    adapter = validate_adapter(adapter_dir)
    if (any(adapter.get(key) != value for key, value in context.provenance.items())
            or adapter.get("mode") != "final" or adapter.get("dataset_sha256") != llm["dataset_sha256"]):
        raise ValueError("The final adapter must match the current freeze and selected development dataset")
    assignments = context.assignments("held_out")
    for slot, run_id in assignments:
        context.read_metadata(slot, run_id)

    predict = load_predictor(adapter_dir)
    report = {
        **context.provenance,
        "split": "held_out",
        "held_out_assignments": assignment_ids(assignments),
        "selected_lookback_seconds": lookback,
        "selected_threshold": threshold,
        "dataset_sha256": llm["dataset_sha256"],
        "adapter_sha256": adapter.get("adapter_sha256"),
        "event_rule": {"system_event_id": 2004, "lookback_seconds": 300},
    }
    # Use the existing slope lock so neither command can evaluate after the
    # other. Keep the claim if scoring fails after the first held-out read.
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    report["arms"] = compute_arms(context, assignments, lookback, threshold, predict)
    write_report(output, report)
    return output, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, default=HERE / "protocol")
    parser.add_argument("--runs-dir", type=Path, default=HERE.parent / "crashlab-data")
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        path, _ = evaluate(args.protocol_dir, args.runs_dir, args.selection_dir, args.adapter_dir)
    except (OSError, ImportError, RuntimeError, ValueError, KeyError, TypeError, csv.Error) as error:
        parser.exit(1, f"evaluate_heldout: {error}\n")
    print(f"Held-out slope, event, and LLM scores: {path}")


if __name__ == "__main__":
    main()
