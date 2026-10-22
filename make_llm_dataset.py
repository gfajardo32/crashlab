"""
make_llm_dataset.py
Turn the windowed crash data into text examples for fine-tuning a small LM.

Each window becomes one (prompt, completion) pair:
  prompt     = the window's events in time order (relative seconds, level, event id)
  completion = " yes" / " no"  (crash within H seconds)

Leak-free THREE-way split by whole run: train / eval (validation) / test.
Writes llm_data/{train,eval,test}.jsonl.
"""

import os
import json
import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from build_dataset import load_run
from make_dataset import list_runs, L, S, H
from windowing import generate_windows_for_run, max_countdown_from_run

WINDOW = 300     # 5-minute lookback, same as the honest baseline
CAP = 80         # max events shown per window (keep the most recent if more)
SEED = 42
TEST_FRAC = 0.20
EVAL_FRAC = 0.15
OUT_DIR = "llm_data"


def render_window(run_df, c):
    """The window's events as text: t = seconds since window start, level, id."""
    w = run_df[(run_df["secs_before_crash"] >= c) &
               (run_df["secs_before_crash"] <= c + WINDOW)]
    if len(w) > CAP:
        w = w.nsmallest(CAP, "secs_before_crash")     # keep the most recent CAP
    w = w.sort_values("secs_before_crash", ascending=False)  # oldest first

    if len(w) == 0:
        body = "  (no events logged)"
    else:
        lines = []
        for _, r in w.iterrows():
            t = int(round((c + WINDOW) - r["secs_before_crash"]))
            lines.append(f"  t={t:<4} {str(r['LevelDisplayName']):<11} {int(r['Id'])}")
        body = "\n".join(lines)

    return ("You are monitoring a Windows machine's system event log.\n"
            "Events in the last 5 minutes (t = seconds since window start):\n"
            f"{body}\n"
            "Question: will this machine crash within the next 5 minutes? "
            "Answer yes or no.")


def three_way_split(run_dirs, seed=SEED):
    rng = random.Random(seed)
    sh = run_dirs[:]
    rng.shuffle(sh)
    n = len(sh)
    n_test = round(n * TEST_FRAC)
    n_eval = round(n * EVAL_FRAC)
    test = sh[:n_test]
    eval_ = sh[n_test:n_test + n_eval]
    train = sh[n_test + n_eval:]
    return train, eval_, test


def write_jsonl(path, run_dirs):
    n, n_pos = 0, 0
    with open(path, "w") as f:
        for rd in run_dirs:
            try:
                run_df = load_run(rd)
            except (ValueError, FileNotFoundError):
                continue
            mc = max_countdown_from_run(run_df)
            for c, label in generate_windows_for_run(mc, L, S, H):
                rec = {"prompt": render_window(run_df, c),
                       "completion": " yes" if label == 1 else " no"}
                f.write(json.dumps(rec) + "\n")
                n += 1
                n_pos += label
    return n, n_pos


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    run_dirs = list_runs()
    train, eval_, test = three_way_split(run_dirs)
    print(f"runs: {len(train)} train / {len(eval_)} eval / {len(test)} test\n")

    for name, runs in [("train", train), ("eval", eval_), ("test", test)]:
        n, n_pos = write_jsonl(os.path.join(OUT_DIR, f"{name}.jsonl"), runs)
        print(f"{name:>5}: {n:>5} examples, {n_pos:>4} positive ({n_pos/n:.0%})")

    # show one positive and one negative example from the train file
    print("\n--- sample examples ---")
    shown_pos = shown_neg = False
    with open(os.path.join(OUT_DIR, "train.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            if rec["completion"] == " yes" and not shown_pos:
                print("\n[POSITIVE]\n" + rec["prompt"] + rec["completion"])
                shown_pos = True
            elif rec["completion"] == " no" and not shown_neg:
                print("\n[NEGATIVE]\n" + rec["prompt"] + rec["completion"])
                shown_neg = True
            if shown_pos and shown_neg:
                break


if __name__ == "__main__":
    main()
