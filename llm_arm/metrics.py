"""Use the frozen slope scorer's metrics for all protocol v3.2 arms."""

from statistics import median

from score_slope import (
    BUDGET_PHASES, PHASES, phase_at, pooled, positive_result, score_run,
    window_summary, workload_events,
)


def score_alarms(slot, run_id, metadata, analysis, run_dir, alarms, *, events=None):
    """Score one run's eligible cutoffs with the same rules as score_slope."""
    expected = {example.cutoff for example in analysis.examples}
    if alarms.keys() != expected:
        raise ValueError(f"{run_id}: alarms do not match the eligible cutoffs")
    if events is None:
        events = workload_events(run_dir / "workload.csv", metadata, slot["condition"])
    windows = [(example.cutoff, example.label, alarms[example.cutoff]) for example in analysis.examples]
    negatives = [(cutoff, alarm) for cutoff, label, alarm in windows if label == "no"]
    phased = [(phase_at(cutoff, slot["condition"], events), cutoff, alarm)
              for cutoff, alarm in negatives]
    return {
        "slot_id": slot["slot_id"],
        "run_id": run_id,
        "condition": slot["condition"],
        "pattern": slot.get("pattern"),
        "onset_utc": analysis.onset.isoformat() if analysis.onset else None,
        "diagnostic": analysis.diagnostic,
        **positive_result(analysis.onset, windows),
        "overall": window_summary(negatives),
        "budget": window_summary([(cutoff, alarm) for phase, cutoff, alarm in phased if phase in BUDGET_PHASES]),
        "phases": {
            phase: window_summary([(cutoff, alarm) for current, cutoff, alarm in phased if current == phase])
            for phase in PHASES
        },
    }


def summarize_records(records):
    """Aggregate run-level catches and eligible negative exposure."""
    leads = [r["first_alarm_lead_seconds"] for r in records if r["caught_at_any_lead"]]
    return {
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
    }


def score_slopes(context, assignments, lookbacks):
    """Run each untouched slope rule against its mechanically verified run freeze."""
    by_lookback = {lookback: [] for lookback in lookbacks}
    for slot, run_id in assignments:
        metadata = context.read_metadata(slot, run_id)
        run_dir = context.runs_dir / run_id
        for lookback, result in score_run(slot, run_id, run_dir, lookbacks, metadata["freeze_commit"]).items():
            by_lookback[lookback].append(result)
    return [
        {"lookback_seconds": lookback, **summarize_records(records)}
        for lookback, records in by_lookback.items()
    ]
