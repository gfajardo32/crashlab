"""Format only eligible, cutoff-bounded responsiveness features for the LLM."""

from statistics import median


MIB = 2**20
LOOKBACK_SECONDS = 300
BIN_SECONDS = 30
LEVELS = {level.casefold(): level for level in ("Critical", "Error", "Warning", "Information", "Verbose", "LogAlways")}


def make_prompt(example):
    bins = [[] for _ in range(LOOKBACK_SECONDS // BIN_SECONDS)]
    for elapsed, available, committed, limit, _ in example.features["memory"]:
        if 0 < elapsed <= LOOKBACK_SECONDS:
            slot = (elapsed - 1) // BIN_SECONDS
            bins[slot].append((100 * committed / limit, available / MIB))

    lines = ["Memory in the last five minutes (commit percent, available MiB)"]
    for index, points in enumerate(bins):
        lo, hi = index * BIN_SECONDS, (index + 1) * BIN_SECONDS
        values = "missing"
        if points:
            commit, available = zip(*points)
            values = f"{median(commit):.2f}%, {median(available):.2f} MiB"
        lines.append(f"{lo}-{hi}s: {values}")

    lines.extend(("", "System events (Id and level)"))
    for level, event_id in example.features["events"]:
        lines.append(f"{int(event_id)} {LEVELS.get(level.casefold(), 'Unknown')}")
    if not example.features["events"]:
        lines.append("None")
    lines.extend(("", "Will local PowerShell launch responsiveness fail within the next five minutes? Answer yes or no."))
    return "\n".join(lines)
