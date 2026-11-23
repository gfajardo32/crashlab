"""Build Protocol v3.2 LLM examples from the development runs only."""

import argparse
from collections import defaultdict
from datetime import timezone
import hashlib
import json
from pathlib import Path
from random import Random
import tempfile

from llm_arm.prompt import make_prompt
from llm_arm.protocol_data import load_context


PROJECT = Path(__file__).resolve().parents[1]
FOLD_COUNT = 3
DEVELOPMENT_COUNT = 18
FOLD_SEED = 20261122


def assign_folds(assignments):
    if len(assignments) != DEVELOPMENT_COUNT or len(set(run_id for _, run_id in assignments)) != DEVELOPMENT_COUNT:
        raise ValueError("The LLM dataset requires 18 distinct development runs")
    by_condition = defaultdict(list)
    for slot, run_id in assignments:
        by_condition[slot["condition"]].append(run_id)
    if len(by_condition) != 6 or any(len(runs) != FOLD_COUNT for runs in by_condition.values()):
        raise ValueError("The schedule must contain three development runs for each of six conditions")

    folds = [[] for _ in range(FOLD_COUNT)]
    rng = Random(FOLD_SEED)
    for condition in sorted(by_condition):
        group = sorted(by_condition[condition])
        rng.shuffle(group)
        for fold, run_id in zip(folds, group):
            fold.append(run_id)
    return folds


def build_dataset(protocol_dir, runs_dir, output_dir):
    ctx = load_context(protocol_dir, runs_dir)
    folds = assign_folds(ctx.dev_assignments)
    fold_by_run = {run_id: index for index, fold in enumerate(folds) for run_id in fold}
    if output_dir.exists():
        raise ValueError(f"Dataset output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    counts = [0] * FOLD_COUNT
    with tempfile.TemporaryDirectory(prefix=".llm-dataset-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        path = staging / "dev.jsonl"
        with path.open("w", encoding="utf-8") as out:
            for slot, run_id in ctx.dev_assignments:
                _, analysis, _ = ctx.load_run(slot, run_id)
                fold = fold_by_run[run_id]
                for example in analysis.examples:
                    row = {
                        "run_id": run_id,
                        "cutoff_utc": example.cutoff.astimezone(timezone.utc).isoformat(),
                        "label": example.label,
                        "prompt": make_prompt(example),
                        "fold": fold,
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts[fold] += 1
        if not all(counts):
            raise ValueError("Every development fold must contain eligible examples")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()

        manifest = {**ctx.provenance, "folds": folds, "dataset_sha256": digest, "fold_row_counts": counts}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(output_dir)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, default=PROJECT / "protocol")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT.parent / "crashlab-data")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_dataset(args.protocol_dir, args.runs_dir, args.output_dir)
    print(f"Wrote {sum(manifest['fold_row_counts'])} LLM windows from {DEVELOPMENT_COUNT} development runs")


if __name__ == "__main__":
    main()
