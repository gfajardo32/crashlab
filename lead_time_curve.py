"""
lead_time_curve.py
Characterize the honest baseline (bounded 5-min window + severity, random forest)
across many random train/test splits, and answer: how far ahead can we warn?

For each of N_SPLITS seeds we split runs, train, and score each test run. Pooling
every test-run evaluation across splits gives a robust picture (each run lands in
the test set several times) and settles the "only 11 test runs, one seed" caveat.

Speed: every run's CSV is loaded and parsed ONCE (cached), so the 10 splits reuse
the loaded data instead of re-reading 53 files ten times over.
"""

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import random
from functools import lru_cache

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from build_dataset import load_run
from make_dataset import list_runs, META_COLS, L, S, H
from windowing import generate_windows_for_run, max_countdown_from_run
from bounded_baseline import extract   # honest bounded-window + severity features

N_SPLITS = 10
THRESHOLD = 0.80
R_GRID = [60, 120, 180, 240, 300]


@lru_cache(maxsize=None)
def cached_load(run_dir):
    return load_run(run_dir)


def split_seed(run_dirs, seed, test_frac=0.2):
    rng = random.Random(seed)
    sh = run_dirs[:]
    rng.shuffle(sh)
    n_test = round(len(sh) * test_frac)
    return sh[n_test:], sh[:n_test]      # train, test


def build_vocab(train_runs):
    v = set()
    for rd in train_runs:
        try:
            run_df = cached_load(rd)
        except (ValueError, FileNotFoundError):
            continue
        v.update(int(i) for i in run_df["Id"].unique())
    return v


def rows(run_dirs, vocab):
    out = []
    for rd in run_dirs:
        try:
            run_df = cached_load(rd)
        except (ValueError, FileNotFoundError):
            continue
        mc = max_countdown_from_run(run_df)
        for c, label in generate_windows_for_run(mc, L, S, H):
            f = extract(run_df, c, vocab)
            f["label"] = label
            f["obs_time"] = c
            f["run"] = os.path.basename(rd)
            out.append(f)
    import pandas as pd
    return pd.DataFrame(out)


def run_one(train_runs, test_runs):
    vocab = build_vocab(train_runs)
    train_df = rows(train_runs, vocab)
    test_df = rows(test_runs, vocab)
    fcols = [c for c in train_df.columns if c not in META_COLS]

    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    clf.fit(train_df[fcols], train_df["label"])
    test_df = test_df.copy()
    test_df["prob"] = clf.predict_proba(test_df.reindex(columns=fcols, fill_value=0))[:, 1]

    leads, fp, nsafe = [], 0, 0
    for _, g in test_df.groupby("run"):
        pos = g[g["label"] == 1]
        neg = g[g["label"] == 0]
        alarms = pos[pos["prob"] >= THRESHOLD]
        leads.append(alarms["obs_time"].max() if len(alarms) else 0.0)  # 0 = missed
        fp += int((neg["prob"] >= THRESHOLD).sum())
        nsafe += len(neg)
    return leads, fp, nsafe


def main():
    run_dirs = list_runs()
    all_leads, tot_fp, tot_safe, fars = [], 0, 0, []

    print(f"training {N_SPLITS} random-split forests "
          f"(sklearn takes a few seconds to load first)...", flush=True)
    for seed in range(N_SPLITS):
        tr, te = split_seed(run_dirs, seed)
        leads, fp, nsafe = run_one(tr, te)
        all_leads += leads
        tot_fp += fp
        tot_safe += nsafe
        fars.append(fp / ((nsafe * S) / 3600) if nsafe else float("nan"))
        print(f"  split {seed + 1}/{N_SPLITS} done", flush=True)
    print()

    all_leads = np.array(all_leads)
    detected = all_leads > 0
    pooled_far = tot_fp / ((tot_safe * S) / 3600)

    print(f"honest RF baseline | threshold {THRESHOLD} | {N_SPLITS} splits "
          f"| {len(all_leads)} test-run evaluations")
    print(f"overall detection : {detected.mean():.0%}")
    print(f"median lead (of detected): {np.median(all_leads[detected]):.0f}s")
    print(f"pooled false-alarm rate  : {pooled_far:.1f}/hr "
          f"(per-split range {np.nanmin(fars):.1f}-{np.nanmax(fars):.1f})")
    print()
    print("lead-time curve: fraction of crashes warned at least this far ahead")
    print(f"{'lead >=':>9}  {'caught':>7}")
    print("-" * 20)
    for R in R_GRID:
        print(f"{R:>7}s   {(all_leads >= R).mean():>6.0%}")


if __name__ == "__main__":
    main()
