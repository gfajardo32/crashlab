"""
coefficients.py
Show which features the cumulative logistic-regression model actually leaned on.
Because the features are standardized first, the coefficients are directly
comparable: bigger |weight| = more influence. Positive weight pushes a window
toward CRASH (label 1), negative pushes it toward SAFE.

Reading it:
  - total_volume near the top  -> "stopwatch": it's counting how long the run went
  - event_2004 / event_26 high & positive -> a real resource-exhaustion precursor
  - death-cascade IDs (7031, 7009, 7038) negative -> explains prob dropping at the end
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from make_dataset import list_runs, split_runs, build_vocab, META_COLS
from evaluate_cumulative import rows_cumulative


def main():
    run_dirs = list_runs()
    train_runs, _ = split_runs(run_dirs)
    vocab = build_vocab(train_runs)

    train_df = rows_cumulative(train_runs, vocab)
    feature_cols = [c for c in train_df.columns if c not in META_COLS]
    X_train, y_train = train_df[feature_cols], train_df["label"]

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    clf.fit(X_train, y_train)

    coefs = clf.named_steps["logisticregression"].coef_[0]
    ranked = sorted(zip(feature_cols, coefs), key=lambda kv: abs(kv[1]), reverse=True)

    print("=== cumulative model: standardized coefficients (|weight| desc) ===")
    print("positive -> pushes toward CRASH ; negative -> toward SAFE")
    print(f"{'feature':<16} {'weight':>8}")
    print("-" * 26)
    for name, w in ranked:
        print(f"{name:<16} {w:>8.3f}")


if __name__ == "__main__":
    main()
