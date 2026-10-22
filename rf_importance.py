"""
rf_importance.py
Which features did the random forest actually rely on? Importances are >= 0 and
sum to 1 (they measure how much each feature reduced impurity across the trees).
They don't show direction, only reliance.

Reading it:
  - total_volume at the top            -> leaning on "how long the run has gone" (stopwatch)
  - sev_Error / sev_Critical high      -> reacting to error/critical bursts (real)
  - event_2004 / event_26 high         -> resource-exhaustion precursor (real)
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.ensemble import RandomForestClassifier

from make_dataset import list_runs, split_runs, build_vocab, META_COLS
from fair_baseline import rows


def main():
    run_dirs = list_runs()
    train_runs, _ = split_runs(run_dirs)
    vocab = build_vocab(train_runs)

    train_df = rows(train_runs, vocab)
    feature_cols = [c for c in train_df.columns if c not in META_COLS]
    X, y = train_df[feature_cols], train_df["label"]

    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    clf.fit(X, y)

    imp = sorted(zip(feature_cols, clf.feature_importances_),
                 key=lambda kv: kv[1], reverse=True)

    print("=== random forest feature importances (desc, sum to 1.0) ===")
    print(f"{'feature':<16} {'importance':>10}")
    print("-" * 28)
    for name, v in imp:
        print(f"{name:<16} {v:>10.3f}")


if __name__ == "__main__":
    main()
