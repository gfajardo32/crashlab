"""Select v3.2 slope and LLM alarm settings using development runs only."""

import argparse
import json
import math
from hashlib import sha256
from pathlib import Path

from score_slope import BUDGET, LOOKBACKS, REPORT_VERSION, select_lookback, timestamp, workload_events, write_report

from .metrics import score_alarms, score_slopes, summarize_records

HERE = Path(__file__).resolve().parent.parent
THRESHOLD_REPORT_VERSION = "crashlab-llm-threshold-v3.2"
NO_ALARM_THRESHOLD = math.nextafter(1.0, 2.0)


def best_threshold(table):
    eligible = [row for row in table if row["budget"]["episodes_per_hour"] is not None
                and row["budget"]["episodes_per_hour"] <= BUDGET]
    # Use the higher, more conservative threshold only if both frozen
    # selection metrics are identical.
    best = min(
        eligible,
        key=lambda row: (
            -row["caught_on_time_runs"], row["budget"]["episodes_per_hour"], -row["threshold"],
        ),
        default=None,
    )
    return best["threshold"] if best is not None else None


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checked_rows(path, checksum):
    data = path.read_bytes()
    if sha256(data).hexdigest() != checksum:
        raise ValueError(f"{path}: content does not match the recorded SHA-256")
    return [json.loads(line) for line in data.splitlines()]


def assert_provenance(context, manifest, name):
    if any(manifest.get(key) != value for key, value in context.provenance.items()):
        raise ValueError(f"{name} does not match the current development protocol or runs")


def validate_folds(folds, dev_ids):
    if not isinstance(folds, list) or len(folds) != 3 or any(not isinstance(fold, list) or not fold for fold in folds):
        raise ValueError("Expected exactly three nonempty development folds")
    run_fold = {}
    for index, fold in enumerate(folds):
        for run_id in fold:
            if run_id in run_fold:
                raise ValueError(f"{run_id}: development run appears more than once in the folds")
            run_fold[run_id] = index
    if run_fold.keys() != dev_ids:
        raise ValueError("Fold membership must partition exactly the scheduled development runs")
    return run_fold


def example_key(row):
    cutoff = timestamp(row["cutoff_utc"])
    return row["run_id"], cutoff


def match_oof_predictions(examples, rows):
    predictions = {}
    for row in rows:
        key = example_key(row)
        if key not in examples or key in predictions:
            raise ValueError(f"Unexpected or duplicate out-of-fold prediction: {key}")
        if type(row["fold"]) is not int or row["fold"] != examples[key]["fold"]:
            raise ValueError(f"{key}: prediction was not produced for this example's held-back fold")
        probability = row["probability"]
        if (type(probability) not in (int, float) or not math.isfinite(probability)
                or not 0 <= probability <= 1):
            raise ValueError(f"{key}: expected a finite probability between zero and one")
        predictions[key] = probability
    if predictions.keys() != examples.keys():
        raise ValueError("Out-of-fold predictions do not cover every eligible development cutoff")
    return predictions


def load_development(context, dataset_dir, cv_dir):
    dataset = read_json(dataset_dir / "manifest.json")
    crossval = read_json(cv_dir / "manifest.json")
    assert_provenance(context, dataset, "Development dataset")
    assert_provenance(context, crossval, "Cross-validation")
    if (crossval.get("mode") != "cv" or crossval.get("dataset_sha256") != dataset["dataset_sha256"]
            or crossval.get("folds") != dataset["folds"]):
        raise ValueError("Cross-validation must use the recorded development dataset and folds")

    assignments = context.dev_assignments
    run_fold = validate_folds(dataset["folds"], {run_id for _, run_id in assignments})
    expected = {}
    runs = []
    for slot, run_id in assignments:
        metadata, analysis, run_dir = context.load_run(slot, run_id)
        for example in analysis.examples:
            expected[run_id, example.cutoff] = {"label": example.label, "fold": run_fold[run_id]}
        events = workload_events(run_dir / "workload.csv", metadata, slot["condition"])
        runs.append((slot, run_id, metadata, analysis, run_dir, events))

    seen = set()
    rows = checked_rows(dataset_dir / "dev.jsonl", dataset["dataset_sha256"])
    for row in rows:
        key = example_key(row)
        if (key not in expected or key in seen or type(row["fold"]) is not int
                or any(row[field] != value for field, value in expected[key].items())):
            raise ValueError(f"Unexpected, duplicate or inconsistent development example: {key}")
        seen.add(key)
    if seen != expected.keys():
        raise ValueError("Development dataset does not cover the current eligible development cutoffs")
    if not seen:
        raise ValueError("The development runs have no eligible cutoffs for threshold selection")

    probabilities = match_oof_predictions(
        expected, checked_rows(cv_dir / "oof_predictions.jsonl", crossval["predictions_sha256"]),
    )
    return runs, probabilities, dataset["dataset_sha256"], crossval["predictions_sha256"]


def threshold_scores(runs, probabilities, threshold):
    return summarize_records([
        score_alarms(
            slot, run_id, metadata, analysis, run_dir,
            {example.cutoff: probabilities[run_id, example.cutoff] >= threshold
             for example in analysis.examples},
            events=events,
        )
        for slot, run_id, metadata, analysis, run_dir, events in runs
    ])


def select_development(context, protocol_dir, dataset_dir, cv_dir, output_dir):
    reports = ("slope_held_out_started.json", "slope_held_out.json", "heldout_v32.json")
    if any((protocol_dir / path).exists() for path in reports):
        raise ValueError("Held-out scoring has already started; development settings are locked")

    runs, probabilities, dataset_sha, predictions_sha = load_development(context, dataset_dir, cv_dir)
    table = []
    # Inclusive thresholds group equal predictions; the extra candidate
    # represents no alarms even when a probability is exactly one.
    for threshold in sorted({NO_ALARM_THRESHOLD, *probabilities.values()}, reverse=True):
        metrics = threshold_scores(runs, probabilities, threshold)
        table.append({
            "threshold": threshold,
            "caught_on_time_runs": metrics["caught_on_time_runs"],
            "budget": metrics["budget"],
        })
    threshold = best_threshold(table)
    slope_table = score_slopes(context, context.dev_assignments, LOOKBACKS)
    slope = {
        **context.provenance,
        "report_version": REPORT_VERSION,
        "split": "dev",
        "selected_lookback_seconds": select_lookback(slope_table),
        "table": slope_table,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    slope_path = output_dir / "slope_dev.json"
    write_report(slope_path, slope)
    threshold_report = {
        **context.provenance,
        "report_version": THRESHOLD_REPORT_VERSION,
        "split": "dev",
        "selected_threshold": threshold,
        "dataset_sha256": dataset_sha,
        "predictions_sha256": predictions_sha,
        "slope_dev_sha256": sha256(slope_path.read_bytes()).hexdigest(),
        "table": table,
        "selected_metrics": threshold_scores(runs, probabilities, threshold) if threshold is not None else None,
    }
    write_report(output_dir / "threshold.json", threshold_report)
    return slope, threshold_report


def load_selection(context, selection_dir):
    slope_path = selection_dir / "slope_dev.json"
    slope = read_json(slope_path)
    llm = read_json(selection_dir / "threshold.json")
    assert_provenance(context, slope, "Development slope selection")
    assert_provenance(context, llm, "Development LLM selection")
    if (slope.get("split") != "dev" or slope.get("report_version") != REPORT_VERSION
            or slope["selected_lookback_seconds"] not in (*LOOKBACKS, None)
            or slope["selected_lookback_seconds"] != select_lookback(slope["table"])):
        raise ValueError("Development slope selection is missing or inconsistent")
    threshold = llm.get("selected_threshold")
    if (llm.get("split") != "dev" or llm.get("report_version") != THRESHOLD_REPORT_VERSION
            or type(threshold) not in (int, float) or not 0 <= threshold <= NO_ALARM_THRESHOLD
            or threshold != best_threshold(llm["table"])
            or llm.get("slope_dev_sha256") != sha256(slope_path.read_bytes()).hexdigest()):
        raise ValueError("Development LLM threshold is missing, stale or inconsistent")
    return slope, llm


def main():
    from .protocol_data import load_context

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, default=HERE / "protocol")
    parser.add_argument("--runs-dir", type=Path, default=HERE.parent / "crashlab-data")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--cv-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        context = load_context(args.protocol_dir, args.runs_dir)
        slope, llm = select_development(
            context, args.protocol_dir, args.dataset_dir, args.cv_dir, args.output_dir,
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"select_threshold: {error}\n")
    print(f"Development slope lookback: {slope['selected_lookback_seconds']}")
    print(f"Development LLM threshold: {llm['selected_threshold']}")
    print(f"Selection records: {args.output_dir}")


if __name__ == "__main__":
    main()
