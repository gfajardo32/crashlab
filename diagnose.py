"""
diagnose.py
Look inside the baseline's failure: for a few test runs, walk the countdown and
show, per window, the model's probability against the true label and the events
that were actually present. The point is to see WHERE the model loses the signal
(e.g. positive windows that are silent, or events that arrive after the horizon).
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from evaluate import build_frames


def train_and_predict():
    train_df, test_df, feature_cols = build_frames()
    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_test = test_df.reindex(columns=feature_cols, fill_value=0)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    clf.fit(X_train, y_train)
    test_df = test_df.copy()
    test_df["prob"] = clf.predict_proba(X_test)[:, 1]
    return test_df, feature_cols


def show_run(g, event_cols, max_rows=45):
    g = g.sort_values("obs_time", ascending=False)   # early (high c) -> crash (c=0)
    n = len(g)
    step = max(1, n // max_rows)

    have_events = g[g["total_volume"] > 0]
    onset = have_events["obs_time"].max() if len(have_events) else None
    onset_str = f"{onset:.0f}s" if onset is not None else "never"

    print(f"\n=== {g['run'].iloc[0]} ===")
    print(f"    run length {g['obs_time'].max():.0f}s, {n} windows")
    print(f"    first events enter a window at c={onset_str}   (label flips to 1 at c=300)")
    print(f"    {'c(s)':>6} {'lbl':>3} {'prob':>5}  {'evts':>4}  {'bar':<20} event IDs present")
    for i, (_, row) in enumerate(g.iterrows()):
        if i % step and i != n - 1:
            continue
        bar = "#" * int(row["prob"] * 20)
        present = [c.replace("event_", "") for c in event_cols if row[c] > 0]
        ids = ",".join(present[:6]) + (" ..." if len(present) > 6 else "")
        print(f"    {row['obs_time']:>6.0f} {int(row['label']):>3} {row['prob']:>5.2f}  "
              f"{int(row['total_volume']):>4}  {bar:<20} {ids}")


def main():
    test_df, feature_cols = train_and_predict()
    event_cols = [c for c in feature_cols if c.startswith("event_")]
    runs = list(test_df["run"].unique())[:2]
    for r in runs:
        show_run(test_df[test_df["run"] == r], event_cols)


if __name__ == "__main__":
    main()
