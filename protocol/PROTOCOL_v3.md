# CrashLab Surface responsiveness protocol v3 (DRAFT until frozen)

Frozen = this file and protocol_v3.json committed to git, with their SHA-256 recorded in the commit message, BEFORE replicate run 1. Any later change becomes v3.1 with a logged reason. Scoring rules may not change after any held-out run is scored.

## 1. Question
Can a commit-slope rule warn of loss of local PowerShell launch responsiveness 5 minutes ahead, and which lookback (30, 60, 120, 300 s) works best? The chosen rule is evaluated once on held-out runs, which then decides whether fine-tuning is worth doing.

## 2. Schedule (30 fresh-boot runs)
| Condition | Dev | Held-out |
|---|---|---|
| healthy (realistic load, >=30 min after baseline) | 3 | 2 |
| plateau96 (closed-loop top-up, 98% guard, 22 min cap) | 3 | 2 |
| ramp5 (5 MiB/s, 25 min cap) | 3 | 2 |
| ramp10 (10 MiB/s, 15 min cap) | 3 | 2 |
| ramp15 (15 MiB/s, 12 min cap) | 3 | 2 |
| stepped (pattern A or B, 25 min cap) | 3 (A, B, + one seeded) | 2 (A, B) |

- Splits and run order are generated once by a seeded shuffle (seed 20261122) and written to protocol_v3.json. The order is followed exactly.
- Every run: Restart-Computer, wait for SSH, wait 5 min after boot, then 8 min baseline, then the workload.
- Existing pilot runs are kept for design documentation only, never used for selection or evaluation.
- Pattern A pause changed from 180 s to 60 s after pilot laptop_20261122_204250_responsiveness-stepped_pilot showed no lookback could catch it on time.

## 3. Workloads
- Stepped pattern A (short high pause): ramp 10 MiB/s to 75% commit (closed-loop), hold 60 s, resume 10 MiB/s until onset or cap. The on-time cutoff should fall early in the pause, so long lookbacks that still see the ramp can catch on time while short lookbacks go flat.
- Stepped pattern B (long low pause): ramp 10 MiB/s to 60% commit, hold 360 s, resume 10 MiB/s until onset or cap. Onset is ~6-7 min after resume, so pause windows are negatives and the on-time cutoff falls after resume. Long lookbacks that span the flat pause catch late.
- Healthy realistic load: seeded transient commit bursts, a new burst every 60-180 s, size 200-600 MiB, held 15-60 s then released, plus a light CPU duty loop. Bursts stay below 85% commit. Burst schedule is seeded per run and logged to workload.csv. (Healthy-only. Known limitation: background bursts appear only in healthy runs, which matters for future learned models, not for slope rules.)

## 4. Labels
- Every run is labeled from the PowerShell probe only, regardless of workload config. A plateau that fails is a positive run; a ramp that hits its cap without onset contributes only negatives.
- Onset, slots, censoring, 4 s observation cap, 1,365 ms failure threshold: unchanged from the current labeler. Top-ups and pauses are workload events, not interventions.

## 5. Alarm rule (frozen)
At each cutoff (10 s stride), fit least squares to commit/limit samples in (cutoff - L, cutoff]. Alarm if last sample + slope x 300 s >= 1.0. Require at least floor(L/5) - 1 samples, otherwise no alarm. L in {30, 60, 120, 300} s.

## 6. Scoring (frozen)
- Caught on time: positive run has an alarm at a cutoff with lead time in [290, 300] s. If those windows are ineligible, the run is reported as not evaluable, not as a miss.
- Episode: maximal set of alarming eligible negative windows with consecutive cutoffs (10 s apart). Any gap breaks the episode.
- False-alarm rate: episodes per hour of eligible negative time (eligible negative windows x 10 s).
- Phases by cutoff time, from workload.csv events: baseline; initial rise (injection to first pause, plateau settle, or onset); pause (pause start to resume + 60 s); post-pause rise; top-up (top-up start to allocation end + 60 s); settled hold; healthy.
- Budget (set now): at most 1.0 false-alarm episode per hour, over baseline + healthy + pause + top-up + settled-hold time pooled across dev runs. Initial and post-pause rise are reported separately and excluded from the budget, because they can be indistinguishable from a failing ramp.
- Selection on dev runs only: among lookbacks within budget, pick the most on-time catches; ties go to the lower false-alarm rate, then to the longer lookback. If none is within budget, report the full trade-off table, select nothing, and do not move the budget.
- Held-out: evaluated once with the chosen L (or, if none was chosen, the trade-off table is reported once for all four). Coverage QA on held-out runs is allowed before then; scoring is not.
- Also reported (not used for selection): catches at any lead, median first-alarm lead, per-run results, per-phase false-alarm rates including rise.

## 7. Run validity (decided before data, never based on labels)
A run is invalid and re-collected in the same slot (same condition, split, pattern) if: workload failed to start or stop as configured; PowerShell slot coverage < 95% outside the onset sequence; unexpected reboot; plateau or stepped hold not within +/-1 point of target for >= 80% of the hold; Mac-side loss of the trace. Every invalid run and reason is logged in protocol/run_log.csv.

## Amendments

v3.1: Surface restart is forced. The unforced restart failed before slot 1 (logged invalid in run_log.csv) because autologon keeps a console session. No protocol data existed.
