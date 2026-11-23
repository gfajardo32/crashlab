import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

class ProvenanceTests(unittest.TestCase):
    def git(self, *args):
        return subprocess.check_output(
            ["git", "-C", str(self.root), *args], text=True, stderr=subprocess.DEVNULL,
        ).strip()

    def setUp(self):
        from llm_arm import protocol_data
        self.module = protocol_data
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = self.root / "protocol"
        self.protocol.mkdir()
        self.git("init")
        self.git("config", "user.name", "Synthetic Test")
        self.git("config", "user.email", "fixture@example.invalid")
        self.doc = self.protocol / "PROTOCOL_v3.md"
        self.doc.write_text("# Fixture v3.1\n")
        self.old_doc = self.doc.read_bytes()
        for name in self.module.LEGACY_PINS:
            (self.root / name).write_text("original\n")
        self.old = self.freeze("v3.1")
        patcher = patch.object(self.module, "LEGACY_FREEZE", self.old)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.doc.write_bytes(self.old_doc + self.module.AMENDMENT.encode("utf-8"))
        for name in self.module.REQUIRED_V32_PINS:
            p = self.root / name
            p.parent.mkdir(exist_ok=True)
            p.write_text("new implementation\n")
        self.new = self.freeze("v3.2")

    def freeze(self, name):
        hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            self.doc, self.protocol / "protocol_v3.json",
        )]
        self.git("add", ".")
        self.git("commit", "-m", name, "-m", "\n".join(hashes))
        return self.git("rev-parse", "HEAD")

    def test_accepts_append_only_freeze_and_rejects_later_new_file_change(self):
        self.assertEqual(self.module.verify_v32_freeze(self.protocol), self.new)
        (self.root / "llm_arm/train.py").write_text("changed\n")
        with self.assertRaisesRegex((RuntimeError, ValueError), "pinned"):
            self.module.verify_v32_freeze(self.protocol)

    def test_rejects_amendment_with_old_pin_change(self):
        self.git("reset", "--soft", self.old)
        (self.root / "crashloop.py").write_text("changed\n")
        self.freeze("bad v3.2")
        with self.assertRaisesRegex((RuntimeError, ValueError), "v3.1-pinned"):
            self.module.verify_v32_freeze(self.protocol)

    def test_rejects_an_unpinned_llm_source_file(self):
        (self.root / "llm_arm/extra.py").write_text("unfrozen\n")
        with self.assertRaisesRegex(ValueError, "unpinned"):
            self.module.verify_v32_freeze(self.protocol)

    def test_rejects_rewrite_of_old_protocol_text_even_with_current_hashes(self):
        self.doc.write_bytes(b"# Altered prior text\n" + self.module.AMENDMENT.encode())
        self.freeze("bad v3.2")
        with self.assertRaisesRegex((RuntimeError, ValueError), "append"):
            self.module.verify_v32_freeze(self.protocol)

    def test_only_two_freezes_are_accepted_in_metadata(self):
        runs = self.root / "raw"
        d = runs / "laptop_example"
        d.mkdir(parents=True)
        ctx = self.module.ProtocolContext(self.protocol, runs, self.new, [], [])
        path = d / "run.json"
        for commit in (self.old, self.new, "unknown"):
            path.write_text(json.dumps({"run_id": d.name, "freeze_commit": commit, "protocol_version": 3}))
            if commit == "unknown":
                with self.assertRaisesRegex(ValueError, "freeze"):
                    ctx.read_metadata({}, d.name)
            else:
                self.assertEqual(ctx.read_metadata({}, d.name)["freeze_commit"], commit)

if __name__ == "__main__":
    unittest.main()
