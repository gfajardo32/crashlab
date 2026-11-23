"""Verify the append-only v3.2 freeze and provide split-scoped trace access."""

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from crashloop import verify_protocol_freeze
from score_slope import assignment_ids, assignments_for, read_protocol_metadata
from surface_responsiveness import analyze_run


LEGACY_FREEZE = "fe5bc6719b5c808a11f86498c53eed7974f50b64"
LEGACY_PINS = (
    "crashloop.py", "surface_responsiveness.py", "make_surface_dataset.py",
    "score_slope.py", "make_protocol.py", "protocol/protocol_v3.json",
)
REQUIRED_V32_PINS = (
    "evaluate_heldout.py", "llm_arm/protocol_data.py", "llm_arm/build_dataset.py",
    "llm_arm/prompt.py", "llm_arm/train.py", "llm_arm/train.slurm",
    "llm_arm/select_threshold.py", "llm_arm/metrics.py",
)
PROTOCOL_DOC = "protocol/PROTOCOL_v3.md"
AMENDMENT = (
    "\nv3.2: Made before any held-out scoring. Runs collected under v3.1 (`fe5bc67`) and v3.2 "
    "are pooled because all collection, labeling, and slope-scoring files are byte-identical between the two freezes.\n\n"
    "(a) Event baseline: alarm at a cutoff if any System event Id 2004 occurred in the previous 300 s. "
    "Fixed rule, no tuning.\n\n"
    "(b) LLM arm: prompts built only from data at or before the cutoff: the 5-min lookback of commit percent "
    "and available MiB in 30 s bins, plus System event Id and level only (no message text, no run metadata). "
    "Same base model and QLoRA settings as train_llm.py, 1 epoch. Trained on the 18 development runs only. "
    "The alarm threshold is chosen with 3-fold cross-validation grouped by run on the development runs, "
    "using the same scoring and the same 1.0 episodes/hour budget as the slope rule (most on-time catches "
    "within budget; ties go to the lower false-alarm rate). The final model is trained on all 18 development runs.\n\n"
    "(c) Held-out: one command scores the chosen slope rule, the event baseline, and the final LLM together, "
    "once, and refuses to run unless the slope selection and the LLM threshold are already recorded.\n"
)


def git(project, *args):
    result = subprocess.run(["git", "-C", str(project), *args], capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Cannot verify v3.2 freeze: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def verify_v32_freeze(protocol_dir):
    project = Path(protocol_dir).resolve().parent
    if Path(protocol_dir).resolve() != project / "protocol":
        raise ValueError("The protocol directory must be the frozen repository's protocol directory")
    current = verify_protocol_freeze(project)
    if current == LEGACY_FREEZE:
        raise ValueError("The v3.2 amendment must be frozen in Git")
    legacy_doc = git(project, "show", f"{LEGACY_FREEZE}:{PROTOCOL_DOC}")
    current_doc = git(project, "show", f"{current}:{PROTOCOL_DOC}")
    if current_doc != legacy_doc + AMENDMENT.encode("utf-8"):
        raise ValueError("Protocol v3.2 must append exactly the amendment to the v3.1 text")

    for name in LEGACY_PINS:
        legacy = git(project, "ls-tree", LEGACY_FREEZE, "--", name)
        if not legacy or legacy != git(project, "ls-tree", current, "--", name):
            raise ValueError(f"v3.1-pinned file differs at the v3.2 freeze: {name}")

    # Pin every committed script in the arm, including its shared helpers.
    arm_files = git(project, "ls-tree", "-r", "--name-only", current, "--", "llm_arm").decode().splitlines()
    pinned = {*arm_files, "evaluate_heldout.py"}
    local_scripts = {
        p.relative_to(project).as_posix()
        for p in (project / "llm_arm").rglob("*")
        if p.suffix in (".py", ".slurm") and not any(part.startswith(".") for part in p.relative_to(project).parts)
    }
    if unpinned := sorted(local_scripts - pinned):
        raise ValueError(f"v3.2 contains unpinned arm files: {', '.join(unpinned)}")
    for name in REQUIRED_V32_PINS:
        if not git(project, "ls-tree", current, "--", name):
            raise ValueError(f"Missing v3.2 pinned file: {name}")
    for name in pinned:
        path = project / name
        if (git(project, "ls-tree", current, "--", name) != git(project, "ls-tree", "HEAD", "--", name)
                or path.is_symlink() or not path.is_file()
                or git(project, "show", f"{current}:{name}") != path.read_bytes()):
            raise ValueError(f"v3.2 pinned file differs from its freeze: {name}")
    if git(project, "diff", "--cached", "--name-only", "HEAD", "--", *sorted(pinned)):
        raise ValueError("v3.2 pinned files have staged changes")
    return current


@dataclass(frozen=True)
class ProtocolContext:
    protocol_dir: Path
    runs_dir: Path
    freeze_commit: str
    slots: list
    dev_assignments: list

    @property
    def provenance(self):
        return {
            "freeze_commit": self.freeze_commit,
            "protocol_sha256": hashlib.sha256((self.protocol_dir / "PROTOCOL_v3.md").read_bytes()).hexdigest(),
            "schedule_sha256": hashlib.sha256((self.protocol_dir / "protocol_v3.json").read_bytes()).hexdigest(),
            "dev_assignments": assignment_ids(self.dev_assignments),
        }

    def assignments(self, split):
        if split == "dev":
            return self.dev_assignments
        if split != "held_out":
            raise ValueError(f"Unknown protocol split: {split}")
        return assignments_for(self.slots, self.protocol_dir / "run_log.csv", split)

    def run_path(self, run_id):
        path = self.runs_dir / run_id
        if Path(run_id).name != run_id or path.is_symlink() or not path.resolve().is_relative_to(self.runs_dir.resolve()):
            raise ValueError(f"Invalid protocol run path: {run_id!r}")
        return path

    def read_metadata(self, slot, run_id):
        path = self.run_path(run_id)
        metadata = json.loads((path / "run.json").read_text(encoding="utf-8-sig"))
        freeze = metadata.get("freeze_commit")
        if freeze not in (LEGACY_FREEZE, self.freeze_commit):
            raise ValueError(f"{run_id}: run freeze is neither the v3.1 nor the current v3.2 freeze")
        if metadata.get("run_id") != run_id:
            raise ValueError(f"{run_id}: run metadata has a different ID")
        return read_protocol_metadata(run_id, path, freeze)

    def load_run(self, slot, run_id):
        path = self.run_path(run_id)
        metadata = self.read_metadata(slot, run_id)
        return metadata, analyze_run(path, metadata), path


def load_context(protocol_dir, runs_dir):
    protocol_dir, runs_dir = Path(protocol_dir), Path(runs_dir)
    freeze_commit = verify_v32_freeze(protocol_dir)
    slots = json.loads((protocol_dir / "protocol_v3.json").read_bytes())["runs"]
    dev = assignments_for(slots, protocol_dir / "run_log.csv", "dev")
    if len(dev) != 18:
        raise ValueError("The frozen schedule must supply exactly 18 development runs")
    return ProtocolContext(protocol_dir, runs_dir, freeze_commit, slots, dev)
