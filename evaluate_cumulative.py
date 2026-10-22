"""
evaluate_cumulative.py
Same baseline + evaluation as evaluate.py, but with CUMULATIVE features:
each window counts every event from boot up to the observation point, instead
of only the last L seconds. This kills the "empty window" problem and lets a
slow, accumulating signal (resource-exhaustion events piling up) actually show.

Compare the operating curve this prints against the snapshot version.
"""

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

from build_dataset import load_run
from make_dataset import list_runs, split_runs, build_vocab, L, S, H, META_COLS
from windowing import generate_windows_for_run, max_countdown_from_run
from evaluate import evaluate_threshold


def extract_cumulative(run_df, c, event_vocab):
    """Every event from boot up to the observation point: secs_before_crash >= c."""
    window_df = run_df[run_df["secs_before_crash"] >= c]
    features = {"total_volume": len(window_df), "oov_events": 0}
    for eid in event_vocab:
        features[f"event_{eid}"] = 0
    if window_df.empty:
        return features
    for eid, cnt in window_df["Id"].value_counts().to_dict().items():
        if eid in event_vocab:
            features[f"event_{eid}"] = cnt
        else:
            features["oov_events"] += cnt
    return features


def rows_cumulative(run_dirs, event_vocab):
    rows = []
    for run_dir in run_dirs:
        try:
            run_df = load_run(run_dir)
        except (ValueError, FileNotFoundError):
            continue
        max_c = max_countdown_from_run(run_df)
        for c, label in generate_windows_for_run(max_c, L, S, H):
            feats = extract_cumulative(run_df, c, event_vocab)
            feats["label"] = label
            feats["obs_time"] = c
            feats["run"] = os.path.basename(run_dir)
            rows.append(feats)
    return pd.DataFrame(rows)


def main():
    run_dirs = list_runs()
    train_runs, test_runs = split_runs(run_dirs)      # same seed/split as snapshot
    vocab = build_vocab(train_runs)                   # same train-only vocabulary

    train_df = rows_cumulative(train_runs, vocab)
    test_df = rows_cumulative(test_runs, vocab)
    feature_cols = [c for c in train_df.columns if c not in META_COLS]

    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_test = test_df.reindex(columns=feature_cols, fill_value=0)
    y_test = test_df["label"]

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    clf.fit(X_train, y_train)
    test_df = test_df.copy()
    test_df["prob"] = clf.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, test_df["prob"])
    ap = average_precision_score(y_test, test_df["prob"])
    print("=== CUMULATIVE (since-boot) features ===")
    print(f"test runs: {test_df['run'].nunique()}   test windows: {len(test_df)}")
    print(f"window-level ROC-AUC: {auc:.3f}   PR-AUC: {ap:.3f}")
    print()
    print(f"{'thresh':>6} {'detected':>9} {'lead(med)':>10} {'FAR/hr':>8}")
    print("-" * 38)
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        m = evaluate_threshold(test_df, t)
        det = f"{m['detected']}/{m['n_runs']}"
        print(f"{t:>6.2f} {det:>9} {m['median_lead']:>8.0f}s {m['far_per_hour']:>8.2f}")


if __name__ == "__main__":
    main()
