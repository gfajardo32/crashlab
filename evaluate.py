"""
evaluate.py
Train the non-LLM baseline and score it the way this project actually cares
about: detection rate, lead time, and false-alarm rate, NOT accuracy.

Model: logistic regression over the windowed event-count vectors (the
conventional baseline). It outputs, per window, P(crash within H seconds).

Evaluation (per test run, at a given probability threshold):
  - in-horizon windows  = label 1 (c <= H): where an alarm is a real warning
  - safe-zone windows    = label 0 (c > H): where an alarm is a false alarm
  - detected      : any in-horizon window crosses the threshold
  - lead time     : c of the EARLIEST in-horizon alarm (max obs_time among them)
  - false alarms  : safe-zone windows that cross the threshold
Aggregated across the test runs, then swept over thresholds so the
lead-time vs false-alarm tradeoff is visible.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # silence the date-parse noise

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

from make_dataset import (
    list_runs, split_runs, build_vocab, rows_from_runs, META_COLS, S,
)


def build_frames():
    """Same leak-free pipeline as make_dataset, but keep the full frames so we
    still have obs_time and run for each window (needed for the evaluation)."""
    run_dirs = list_runs()
    train_runs, test_runs = split_runs(run_dirs)      # same seed -> same split
    vocab = build_vocab(train_runs)                   # train-only vocabulary
    train_df = rows_from_runs(train_runs, vocab)
    test_df = rows_from_runs(test_runs, vocab)
    feature_cols = [c for c in train_df.columns if c not in META_COLS]
    return train_df, test_df, feature_cols


def evaluate_threshold(test_df, threshold, stride=S):
    """The metric suite at one probability threshold."""
    n_runs = test_df["run"].nunique()
    detected = 0
    lead_times = []
    fp = 0        # safe-zone alarms (false positives)
    n_safe = 0    # safe-zone windows (for the per-hour rate)

    for _, g in test_df.groupby("run"):
        in_horizon = g[g["label"] == 1]
        safe_zone = g[g["label"] == 0]

        alarms = in_horizon[in_horizon["prob"] >= threshold]
        if len(alarms) > 0:
            detected += 1
            # earliest in-horizon alarm = largest c = most lead time
            lead_times.append(alarms["obs_time"].max())

        fp += int((safe_zone["prob"] >= threshold).sum())
        n_safe += len(safe_zone)

    detection_rate = detected / n_runs
    median_lead = float(np.median(lead_times)) if lead_times else 0.0
    safe_hours = (n_safe * stride) / 3600.0
    far_per_hour = (fp / safe_hours) if safe_hours > 0 else float("nan")
    return {
        "detected": detected, "n_runs": n_runs,
        "detection_rate": detection_rate,
        "median_lead": median_lead,
        "far_per_hour": far_per_hour,
    }


def main():
    train_df, test_df, feature_cols = build_frames()
    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_test = test_df.reindex(columns=feature_cols, fill_value=0)
    y_test = test_df["label"]

    # --- train the baseline ---
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    clf.fit(X_train, y_train)

    test_df = test_df.copy()
    test_df["prob"] = clf.predict_proba(X_test)[:, 1]

    # --- threshold-independent health check (accuracy is deliberately NOT used) ---
    auc = roc_auc_score(y_test, test_df["prob"])
    ap = average_precision_score(y_test, test_df["prob"])
    print(f"test runs: {test_df['run'].nunique()}   test windows: {len(test_df)}")
    print(f"window-level ROC-AUC: {auc:.3f}   PR-AUC: {ap:.3f}")
    print()

    # --- the operating curve: sweep the threshold ---
    print(f"{'thresh':>6} {'detected':>9} {'lead(med)':>10} {'FAR/hr':>8}")
    print("-" * 38)
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        m = evaluate_threshold(test_df, t)
        det = f"{m['detected']}/{m['n_runs']}"
        print(f"{t:>6.2f} {det:>9} {m['median_lead']:>8.0f}s {m['far_per_hour']:>8.2f}")


if __name__ == "__main__":
    main()
