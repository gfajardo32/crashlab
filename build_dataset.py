"""
build_dataset.py
Turn CrashLab run folders into labeled training windows for crash prediction.

Stage 1 (this file so far): load_run() cleans one run folder into the events
that actually belong to that run's boot -> crash window, in a single timezone.
"""

import json
import glob
import os
import pandas as pd


def load_run(run_dir):
    """
    Load one run folder and return the System-log events that happened
    between this run's boot and its crash, as a DataFrame with:
        TimeCreated, Id, LevelDisplayName, ProviderName, MachineName, Message,
        ts                 (parsed datetime, guest-local naive)
        secs_before_crash  (seconds before the crash; 0 = the crash instant)

    Only valid for runs that actually died (death_time_utc present).
    """
    # --- anchors from run.json --------------------------------------------
    with open(os.path.join(run_dir, "run.json"), encoding="utf-8-sig") as f:
        meta = json.load(f)

    if not meta.get("died") or not meta.get("death_time_utc"):
        raise ValueError(f"{run_dir}: run did not die, no crash anchor")

    # boot_time_local carries the guest's UTC offset (e.g. -07:00).
    # Keep the wall-clock part; that offset is also how we localize the UTC crash.
    boot_ts_aware = pd.to_datetime(meta["boot_time_local"])
    guest_tz = boot_ts_aware.tzinfo
    boot = boot_ts_aware.tz_localize(None)
    crash = pd.to_datetime(meta["death_time_utc"]).tz_convert(guest_tz).tz_localize(None)

    # --- newest System CSV in this folder, excluding the GOLD baseline ----
    csvs = [
        f for f in glob.glob(os.path.join(run_dir, "*System*.csv"))
        if "GOLD" not in os.path.basename(f)
    ]
    if not csvs:
        raise FileNotFoundError(f"No System CSV in {run_dir}")
    # sort by filename (…_YYYYMMDD_HHMMSS.csv); mtime is meaningless after git clone
    newest = max(csvs, key=os.path.basename)

    # --- load, parse, filter to this run's real window --------------------
    df = pd.read_csv(newest, encoding="utf-8-sig")
    df["ts"] = pd.to_datetime(df["TimeCreated"])
    df["secs_before_crash"] = (crash - df["ts"]).dt.total_seconds()

    run_df = df[(df["ts"] >= boot) & (df["ts"] <= crash)].copy()
    run_df = run_df.sort_values("ts").reset_index(drop=True)
    return run_df


if __name__ == "__main__":
    test_run = os.path.join(
        os.path.dirname(__file__), "..", "crashlab-data",
        "crashlab-01_20260909_015940_mem-slow-50mb",
    )
    df = load_run(test_run)
    print("shape:", df.shape)
    print(df[["secs_before_crash", "Id", "ProviderName"]].to_string())
