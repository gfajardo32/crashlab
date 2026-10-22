# CrashLab — crash prediction pipeline

Predict Windows crashes from event logs. Each run folder in `../crashlab-data/`
is one induced run-to-failure (mem-slow fault): the System event log plus a
`run.json` with the boot and crash timestamps. The goal is to warn ~5 minutes
before a crash, and to be judged on lead time and false-alarm rate, not accuracy.

Data lives in `../crashlab-data/` (a separate git repo). Code lives here.

## Pipeline order

`build_dataset.load_run` (read one run) -> `windowing` (turn a run into labeled
feature windows) -> `make_dataset` (assemble all runs, leak-free split) ->
`evaluate` (train a baseline, score it). The other scripts are variants and
diagnostics built on those.

## Config knobs

Set in `make_dataset.py`: `L`=60 (lookback, s), `S`=10 (stride, s), `H`=300
(horizon: crash within 5 min => label 1), `TEST_FRAC`=0.2, `SEED`=42.

## Core pipeline

- **build_dataset.py** — `load_run(run_dir)`: reads one run folder, aligns the
  crash time to the guest's local clock, and returns the System-log events
  between boot and crash with a `secs_before_crash` countdown column.

- **windowing.py** — turns one run into training examples. `generate_windows_for_run`
  makes `(observation point, label)` pairs on the countdown axis;
  `extract_features_for_window` turns each window's events into a fixed-length
  bag-of-event-counts vector with an out-of-vocabulary bucket.

- **make_dataset.py** — the leak-free assembler. Lists clean runs, splits them
  into train/test **by whole run** (seeded), builds the event vocabulary from
  **training runs only**, and produces `X_train, y_train, X_test, y_test`. Run it
  directly to print the dataset summary.

## Baseline + evaluation

- **evaluate.py** — trains the logistic-regression baseline on 60s-snapshot
  features and runs the real evaluation: per test run it computes detection,
  lead time, and false-alarm rate, swept across probability thresholds
  (`evaluate_threshold` is the shared metric function).

- **evaluate_cumulative.py** — same baseline, but features are cumulative counts
  since boot instead of a 60s snapshot. Fixes the empty-window problem; raised
  AUC but partly leaks uptime.

- **fair_baseline.py** — the baseline's fair shot: cumulative event counts plus
  severity counts (Error/Warning/etc.), trained with both logistic regression
  and a random forest.

- **bounded_baseline.py** — the honest baseline: a bounded 5-minute lookback
  window (no since-boot uptime leak) plus severity, LR vs random forest. The
  version to trust for the baseline comparison.

## Diagnostics

- **diagnose.py** — walks two test runs window by window, printing the model's
  probability against the true label and the events actually present. Shows
  *where* the model loses the signal.

- **coefficients.py** — ranks the cumulative logistic-regression model's
  standardized weights. Used to check whether it learned a real precursor or a
  stopwatch.

- **rf_importance.py** — ranks the random forest's feature importances, the same
  stopwatch-vs-real-signal check for the forest.

## Running

From this folder, with the virtualenv active (`pandas`, `numpy`, `scikit-learn`):

```bash
python3 make_dataset.py        # build + summarize the dataset
python3 evaluate.py            # snapshot baseline + evaluation
python3 bounded_baseline.py    # the honest baseline
python3 diagnose.py            # look inside a couple of runs
```
