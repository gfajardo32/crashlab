"""
check_pilot.py - verify ONE crash run + ONE control before the overnight batch.

Reuses make_surface_dataset's exact parsing/windowing so the result matches the
real builder: if the builder would skip a run, this says so and why.

Usage (from the crashlab/ folder):
    python3 check_pilot.py                       # auto-pick newest crash + control
    python3 check_pilot.py <crash_run> <control_run>
"""
import sys, json
from pathlib import Path
from datetime import timezone

import make_surface_dataset as M

DATA = (Path(__file__).resolve().parent.parent / "crashlab-data")


def has_memory(d):
    return (d / "memory.csv").exists()


def meta_of(d):
    try:
        return json.loads((d / "run.json").read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def pick(died_value):
    cands = [d for d in DATA.iterdir() if d.is_dir() and has_memory(d)
             and meta_of(d).get("died") is died_value]
    return max(cands, key=lambda d: d.name) if cands else None


def trajectory(run_dir, meta):
    """Print the memory trajectory over the run so pressure build-up is visible."""
    start = M.timestamp(meta["observation_start_utc"]).astimezone(timezone.utc)
    death = M.timestamp(meta["death_time_utc"]).astimezone(timezone.utc) if meta["died"] else None
    end = M.timestamp(meta["observation_end_utc"]).astimezone(timezone.utc)
    samples = M.read_memory(run_dir / "memory.csv", start, death or end)
    if not samples:
        print("    no memory samples in bounds"); return
    gaps = [(b[0] - a[0]).total_seconds() for a, b in zip(samples, samples[1:])]
    print(f"    samples: {len(samples)}   max gap: {max(gaps):.0f}s   "
          f"span: {(samples[-1][0]-samples[0][0]).total_seconds()/60:.1f} min")
    anchor = death or end
    print("    t_to_end(s)  availMiB  commit%  pages/s   (every ~30s)")
    last = None
    for t, (a, c, lim, p) in samples:
        sec = int((anchor - t).total_seconds())
        if last is None or last - sec >= 30:
            print(f"      {sec:6d}      {a/M.MIB:7.0f}   {100*c/lim:6.1f}   {p:6.1f}")
            last = sec


def check(run_dir, kind):
    print(f"\n=== {kind}: {run_dir.name} ===")
    meta = meta_of(run_dir)
    for f in ("observation_start_utc", "observation_end_utc", "boot_time_local", "died", "death_time_utc"):
        print(f"    run.json {f}: {meta.get(f, '*** MISSING ***')}")
    try:
        pairs = list(M.examples_for_run(run_dir, meta))
    except Exception as e:
        print(f"    >>> WOULD BE SKIPPED: {e}")
        return False, 0, 0
    yes = sum(1 for e, _ in pairs if e["completion"].strip() == "yes")
    no = len(pairs) - yes
    print(f"    windows: {len(pairs)}   yes={yes}  no={no}")
    if meta.get("died"):
        trajectory(run_dir, meta)
        if yes == 0:
            print("    >>> WARNING: crash run produced ZERO positive windows "
                  "(coverage gate likely dropped the near-death samples)")
    else:
        if no == 0:
            print("    >>> WARNING: control produced ZERO healthy windows "
                  "(too short, or sampling gaps)")
    return True, yes, no


def main():
    if len(sys.argv) == 3:
        crash, control = DATA / sys.argv[1], DATA / sys.argv[2]
    else:
        crash, control = pick(True), pick(False)
    if not crash:
        print("no crash run with memory.csv found"); return
    if not control:
        print("no control run with memory.csv found"); return

    ok_c, cy, cn = check(crash, "CRASH")
    ok_h, hy, hn = check(control, "CONTROL")

    print("\n=== VERDICT ===")
    go = ok_c and ok_h and cy > 0 and hn > 0
    if go:
        print("GO: both runs parse, crash has positive windows, control has healthy windows.")
    else:
        print("NO-GO: fix the warnings above before the overnight batch.")


if __name__ == "__main__":
    main()
