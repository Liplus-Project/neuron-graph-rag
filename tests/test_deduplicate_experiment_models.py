from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path

from tools.deduplicate_experiment_models import (
    BACKUP_SUFFIX,
    apply,
    finalize,
    make_plan,
    rollback,
    verify,
)


class ExperimentModelDedupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.sources = [self.workspace / "one" / "model-cache", self.workspace / "two" / "ce-cache"]
        self.first = self.sources[0] / "models--sample" / "snapshots" / "revision" / "model.safetensors"
        self.second = self.sources[1] / "models--sample" / "snapshots" / "revision" / "model.safetensors"
        self.unique = self.sources[1] / "models--other" / "snapshots" / "revision" / "tokenizer.json"
        for path, content in ((self.first, b"same model bytes"), (self.second, b"same model bytes"), (self.unique, b"different")):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        self.store = self.workspace / "shared-store"
        self.plan_path = self.workspace / "plan.json"
        self.receipt_path = self.workspace / "receipt.json"

    def _plan(self):
        return make_plan(self.workspace, self.store, self.sources, self.plan_path)

    def test_plan_apply_verify_and_finalize_preserve_paths_and_bytes(self):
        plan = self._plan()
        self.assertEqual(plan["file_count"], 3)
        self.assertEqual(plan["unique_content_count"], 2)
        self.assertEqual(plan["maximum_reclaimable_bytes"], len(b"same model bytes"))
        self.assertNotEqual(os.stat(self.first).st_ino, os.stat(self.second).st_ino)

        applied = apply(self.plan_path)
        self.assertEqual(applied["replaced_files"], 1)
        self.assertEqual(verify(self.plan_path)["backups_retained"], 1)
        self.assertTrue(os.path.samefile(self.first, self.second))
        self.assertEqual(self.first.read_bytes(), b"same model bytes")
        self.assertEqual(self.unique.read_bytes(), b"different")

        receipt = finalize(self.plan_path, self.receipt_path)
        self.assertEqual(receipt["backups_removed"], 1)
        self.assertEqual(verify(self.plan_path)["backups_retained"], 0)
        self.assertFalse(self.second.with_name(self.second.name + BACKUP_SUFFIX).exists())
        self.assertTrue(self.receipt_path.is_file())

    def test_rollback_restores_original_inode_and_content(self):
        self._plan()
        apply(self.plan_path)
        restored = rollback(self.plan_path)
        self.assertEqual(restored["restored_files"], 1)
        self.assertFalse(os.path.samefile(self.first, self.second))
        self.assertEqual(self.first.read_bytes(), b"same model bytes")
        self.assertEqual(self.second.read_bytes(), b"same model bytes")
        self.assertFalse(self.second.with_name(self.second.name + BACKUP_SUFFIX).exists())

    def test_changed_source_fails_before_mutation(self):
        self._plan()
        self.second.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "content changed"):
            apply(self.plan_path)
        self.assertFalse(self.store.exists())
        self.assertEqual(self.first.read_bytes(), b"same model bytes")
        self.assertEqual(self.second.read_bytes(), b"modified")

    def test_tampered_store_path_is_rejected_before_mutation(self):
        self._plan()
        plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        plan["files"][0]["store_path"] = "workspace/experiments/other-file"
        self.plan_path.write_text(json.dumps(plan), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "store path does not match digest"):
            apply(self.plan_path)
        self.assertFalse(self.store.exists())


if __name__ == "__main__":
    unittest.main()
