"""
fair_baseline.py
Give the conventional baseline its best fair shot before concluding it fails:
  - richer features: cumulative event-ID counts PLUS cumulative severity counts
    (Critical/Error/Warning/Information/Verbose from LevelDisplayName)
  - two models: logistic regression (linear) and random forest (nonlinear,
    ignores noise features, captures interactions)
Same leak-free split, same evaluation. Compare against the plain cumulative LR
(ROC-AUC 0.74, FAR ~48/hr even at 0.90).
"""

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

from build_dataset import load_run
from make_dataset import list_runs, split_runs, build_vocab, L, S, H, META_COLS
from windowing import generate_windows_for_run, max_countdown_from_run
from evaluate import evaluate_threshold

LEVELS = ["Critical", "Error", "Warning", "Information", "Verbose"]


def extract(run_df, c, vocab):
    """Cumulative (since-boot) event-ID counts + cumulative severity counts."""
    w = run_df[run_df["secs_before_crash"] >= c]
    feats = {"total_volume": len(w), "oov_events": 0}
    for eid in vocab:
        feats[f"event_{eid}"] = 0
    for lv in LEVELS:
        feats[f"sev_{lv}"] = 0
    if w.empty:
        return feats
    for eid, cnt in w["Id"].value_counts().to_dict().items():
        if eid in vocab:
            feats[f"event_{eid}"] = cnt
        else:
            feats["oov_events"] += cnt
    for lv, cnt in w["LevelDisplayName"].value_counts().to_dict().items():
        if lv in LEVELS:
            feats[f"sev_{lv}"] = cnt
    return feats


def rows(run_dirs, vocab):
    out = []
    for rd in run_dirs:
        try:
            run_df = load_run(rd)
        except (ValueError, FileNotFoundError):
            continue
        mc = max_countdown_from_run(run_df)
        for c, label in generate_windows_for_run(mc, L, S, H):
            f = extract(run_df, c, vocab)
            f["label"] = label
            f["obs_time"] = c
            f["run"] = os.path.basename(rd)
            out.append(f)
    return pd.DataFrame(out)


def report(name, clf, train_df, test_df, feature_cols):
    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_test = test_df.reindex(columns=feature_cols, fill_value=0)
    y_test = test_df["label"]
    clf.fit(X_train, y_train)
    td = test_df.copy()
    td["prob"] = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, td["prob"])
    ap = average_precision_score(y_test, td["prob"])
    print(f"\n=== {name} (cumulative + severity) ===")
    print(f"ROC-AUC: {auc:.3f}   PR-AUC: {ap:.3f}")
    print(f"{'thresh':>6} {'detected':>9} {'lead(med)':>10} {'FAR/hr':>8}")
    print("-" * 38)
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        m = evaluate_threshold(td, t)
        det = f"{m['detected']}/{m['n_runs']}"
        print(f"{t:>6.2f} {det:>9} {m['median_lead']:>8.0f}s {m['far_per_hour']:>8.2f}")


def main():
    run_dirs = list_runs()
    train_runs, test_runs = split_runs(run_dirs)
    vocab = build_vocab(train_runs)
    train_df = rows(train_runs, vocab)
    test_df = rows(test_runs, vocab)
    feature_cols = [c for c in train_df.columns if c not in META_COLS]
    print(f"features: {len(feature_cols)}  (event IDs + severity + volume)")

    report(
        "logistic regression",
        make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=1000, class_weight="balanced")),
        train_df, test_df, feature_cols,
    )
    report(
        "random forest",
        RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               random_state=42, n_jobs=-1),
        train_df, test_df, feature_cols,
    )


if __name__ == "__main__":
    main()
