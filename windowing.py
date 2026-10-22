"""
windowing.py

The feature pipeline that turns runs into labeled training examples for
crash prediction.

Everything slides along the `secs_before_crash` axis (a countdown: large at
boot, 0 at the crash). Anchoring on the crash instead of boot makes the label
trivial and needs no absolute-time / datetime math:

    a point sitting `c` seconds before the crash is labeled 1 if c <= H, else 0.

Two building blocks live here:
  - generate_windows_for_run : the observation points and their labels
  - extract_features_for_window : one fixed-length feature vector per window

Still to come (they enforce the leak-free rule): split runs into train/test,
build the event vocabulary from the training runs only, then loop everything
into a single dataset.
"""

import pandas as pd


def generate_windows_for_run(max_countdown, L, S, H):
    """
    Generate (c, label) pairs for one run by walking the countdown axis.

    Parameters
    ----------
    max_countdown : float
        Largest secs_before_crash in the run (~ how long before the crash the
        first logged event fired, i.e. the usable run length in seconds).
    L : float
        Lookback window length in seconds. The first observation point needs a
        full L seconds of history behind it (the warm-up).
    S : float
        Stride between observation points, in seconds.
    H : float
        Prediction horizon in seconds. Crash within H  =>  label 1.

    Returns
    -------
    list[tuple[float, int]]
        (c, label) pairs, where c = seconds before the crash at the observation
        point and label = 1 if c <= H else 0. Points run from (max_countdown - L)
        down to 0, stepping down by S.
    """
    windows = []
    c = max_countdown - L          # first point that has a full L of history behind it
    while c >= 0:
        label = 1 if c <= H else 0
        windows.append((round(c, 3), label))
        c -= S
    return windows


def max_countdown_from_run(run_df):
    """
    Pull max_countdown out of a real run DataFrame (from build_dataset.load_run).

    This is the one-liner that bridges the toy inputs to real data: the largest
    secs_before_crash value is the run's start on the countdown axis.
    """
    return float(run_df["secs_before_crash"].max())


def extract_features_for_window(run_df, c, L, event_vocab):
    """
    Build one fixed-length feature vector for the window ending at observation
    point c.

    Parameters
    ----------
    run_df : pd.DataFrame
        Raw events for a single run. Must contain 'secs_before_crash' and 'Id'
        (the Windows Event ID column, as harvested).
    c : float
        Observation point, in seconds before the crash.
    L : float
        Lookback window length in seconds.
    event_vocab : set
        Allowed event IDs, built strictly from the TRAINING runs. Anything not
        in here is routed to the OOV bucket so the vector length stays fixed.

    Returns
    -------
    dict
        A single fixed-length row of features: total volume, one count per
        known event ID (zero if it did not fire), and an OOV count.
    """
    # 1. Slice the window. On a countdown axis the L seconds of history behind
    #    the observation point are the events with c <= secs_before_crash <= c+L.
    #    (This never includes events AFTER the observation point, so no leakage.)
    mask = (run_df["secs_before_crash"] <= c + L) & (run_df["secs_before_crash"] >= c)
    window_df = run_df[mask]

    # 2. Initialize the fixed-length vector (zeros for every known event ID).
    features = {"total_volume": len(window_df), "oov_events": 0}
    for event_id in event_vocab:
        features[f"event_{event_id}"] = 0

    # If the machine was silent in this window, the zero vector is the answer.
    if window_df.empty:
        return features

    # 3. Aggregate. Known IDs fill their slot; unknown IDs go to the OOV bucket.
    event_counts = window_df["Id"].value_counts().to_dict()
    for event_id, count in event_counts.items():
        if event_id in event_vocab:
            features[f"event_{event_id}"] = count
        else:
            features["oov_events"] += count

    # Note: severity rollups (LevelDisplayName == "Error"/"Warning") slot in here
    # the same way when we add them.
    return features


if __name__ == "__main__":
    # --- windowing self-check ---
    # A run crashing 600s after its first event, L=60, S=10, H=300 should give
    # 55 windows, 31 positive, first positive at c = 300.
    w = generate_windows_for_run(max_countdown=600, L=60, S=10, H=300)
    print("=== windowing self-check ===")
    print(f"total windows     : {len(w)}")
    print(f"positive (label 1): {sum(lbl for _, lbl in w)}")
    print(f"first positive at : c = {next(c for c, lbl in w if lbl == 1)}")
    print(f"first 3           : {w[:3]}")
    print(f"last 3            : {w[-3:]}")

    # --- feature self-check ---
    # Window at c=90, L=60 keeps events with 90 <= secs_before_crash <= 150:
    # that's the events at 100 (Id 41) and 90 (Id 999). Vocab knows {41, 7031},
    # so Id 999 is routed to OOV. Expect total_volume=2, event_41=1,
    # event_7031=0, oov_events=1.
    print("\n=== feature self-check ===")
    df = pd.DataFrame({
        "secs_before_crash": [500, 100, 90, 10],
        "Id":                [7031, 41, 999, 41],
    })
    feat = extract_features_for_window(df, c=90, L=60, event_vocab={41, 7031})
    print(feat)
    ok = (feat["total_volume"] == 2 and feat["event_41"] == 1
          and feat["event_7031"] == 0 and feat["oov_events"] == 1)
    print("PASS" if ok else "FAIL")
