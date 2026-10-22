"""
make_dataset.py
Assemble all clean crash runs into a leak-free train/test dataset.

The order of operations IS the leak protection:
  1. list_runs   : find every clean, usable crash run on disk
  2. split_runs  : split the RUN list into train/test (whole runs), fixed seed
  3. build_vocab : event vocabulary from TRAINING runs only
  4/5. rows_from_runs : turn each run into feature+label rows (test uses train vocab)
  6. build_dataset : split each side into X (features) and y (labels)
"""

import os
import glob
import json
import random
import pandas as pd

from build_dataset import load_run
from windowing import (
    generate_windows_for_run,
    max_countdown_from_run,
    extract_features_for_window,
)

# ---- config ----
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "crashlab-data")
L = 60           # lookback window length (seconds)
S = 10           # stride between observation points (seconds)
H = 300          # prediction horizon (seconds): crash within 5 min => label 1
TEST_FRAC = 0.2  # fraction of RUNS held out for test
SEED = 42        # fixes the split so the dataset is reproducible

META_COLS = ["label", "obs_time", "run"]  # bookkeeping columns, never features


def list_runs(data_dir=DATA_DIR):
    """
    Step 1. Every run folder that is a clean, usable crash:
    died == True, ttf_seconds > 0 (no clock-corrupted runs), and a harvested
    System CSV present. Returns a sorted list of run directory paths.
    """
    runs = []
    for run_dir in sorted(glob.glob(os.path.join(data_dir, "*"))):
        meta_path = os.path.join(run_dir, "run.json")
        if not os.path.isdir(run_dir) or not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, encoding="utf-8-sig") as f:
                meta = json.load(f)
        except Exception:
            continue
        if not meta.get("died"):
            continue
        ttf = meta.get("ttf_seconds")
        if ttf is None or ttf <= 0:            # excludes the clock-corrupted runs
            continue
        csvs = [f for f in glob.glob(os.path.join(run_dir, "*System*.csv"))
                if "GOLD" not in os.path.basename(f)]
        if not csvs:                           # excludes runs with no harvested log
            continue
        runs.append(run_dir)
    return runs


def split_runs(run_dirs, test_frac=TEST_FRAC, seed=SEED):
    """
    Step 2. Split the RUN list (not the windows) into train/test, reproducibly.
    This happens before any log is opened for features, so no window from a
    test run can ever influence training.
    """
    rng = random.Random(seed)
    shuffled = run_dirs[:]
    rng.shuffle(shuffled)
    n_test = int(round(len(shuffled) * test_frac))
    test_runs = shuffled[:n_test]
    train_runs = shuffled[n_test:]
    return train_runs, test_runs


def build_vocab(train_runs):
    """
    Step 3. Union of every event Id seen in the TRAINING runs only. Test runs
    are excluded, so anything they show that training never saw becomes OOV.
    """
    vocab = set()
    for run_dir in train_runs:
        try:
            run_df = load_run(run_dir)
        except (ValueError, FileNotFoundError):
            continue
        vocab.update(int(i) for i in run_df["Id"].unique())
    return vocab


def rows_from_runs(run_dirs, event_vocab):
    """
    Steps 4 & 5. Turn a list of runs into labeled feature rows using a fixed
    vocab. Same code path for train and test; the only difference is which
    runs come in and that both use the TRAIN vocab.
    """
    rows = []
    for run_dir in run_dirs:
        try:
            run_df = load_run(run_dir)
        except (ValueError, FileNotFoundError):
            continue
        max_c = max_countdown_from_run(run_df)
        for c, label in generate_windows_for_run(max_c, L, S, H):
            feats = extract_features_for_window(run_df, c, L, event_vocab)
            feats["label"] = label
            feats["obs_time"] = c
            feats["run"] = os.path.basename(run_dir)
            rows.append(feats)
    return pd.DataFrame(rows)


def build_dataset():
    """Run the whole pipeline and return X_train, y_train, X_test, y_test (+ extras)."""
    run_dirs = list_runs()                              # 1
    train_runs, test_runs = split_runs(run_dirs)        # 2
    vocab = build_vocab(train_runs)                     # 3
    train_df = rows_from_runs(train_runs, vocab)        # 4
    test_df = rows_from_runs(test_runs, vocab)          # 5

    # 6. features vs labels. Lock the feature columns from TRAIN, then force
    #    TEST into that exact same set/order (fill_value=0 is belt-and-suspenders;
    #    the columns already match because both used the train vocab).
    feature_cols = [c for c in train_df.columns if c not in META_COLS]
    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_test = test_df.reindex(columns=feature_cols, fill_value=0)
    y_test = test_df["label"]
    return X_train, y_train, X_test, y_test, vocab, train_runs, test_runs


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, vocab, train_runs, test_runs = build_dataset()
    print(f"runs: {len(train_runs)} train / {len(test_runs)} test")
    print(f"vocab size (train event IDs): {len(vocab)}")
    print(f"feature vector length: {X_train.shape[1]}")
    print(f"X_train: {X_train.shape}   X_test: {X_test.shape}")
    print(f"train positives: {int(y_train.sum())} / {len(y_train)} "
          f"({y_train.mean():.1%})")
    print(f"test  positives: {int(y_test.sum())} / {len(y_test)} "
          f"({y_test.mean():.1%})")
