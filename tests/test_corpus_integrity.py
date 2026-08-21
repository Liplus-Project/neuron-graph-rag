from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag.corpus_integrity import (
    registered_manifest_commit,
    verify_historical_source_hashes,
    verify_manifest_source_hashes,
    verify_source_sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


class SourceHashVerificationTest(unittest.TestCase):
    def verify(self, raw: bytes, expected: bytes):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.txt"
            path.write_bytes(raw)
            return verify_source_sha256(path, _sha256(expected))

    def test_raw_match_wins_before_newline_validation(self) -> None:
        raw = b"alpha\r\nbeta\ngamma\rdelta"

        result = self.verify(raw, raw)

        self.assertTrue(result.accepted)
        self.assertEqual(result.decision, "raw_match")
        self.assertEqual(result.alternate_sha256, None)

    def test_repository_declares_canonical_lf_for_text(self) -> None:
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")

        self.assertIn("* text=auto eol=lf", attributes)

    def test_lf_source_matches_crlf_hash_only_by_whole_file_conversion(self) -> None:
        result = self.verify(b"alpha\nbeta\n", b"alpha\r\nbeta\r\n")

        self.assertTrue(result.accepted)
        self.assertEqual(result.decision, "newline_equivalent_lf_to_crlf")
        self.assertEqual(result.alternate_sha256, result.expected_sha256)

    def test_crlf_source_matches_lf_hash_only_by_whole_file_conversion(self) -> None:
        result = self.verify(b"alpha\r\nbeta\r\n", b"alpha\nbeta\n")

        self.assertTrue(result.accepted)
        self.assertEqual(result.decision, "newline_equivalent_crlf_to_lf")
        self.assertEqual(result.alternate_sha256, result.expected_sha256)

    def test_mixed_newlines_are_rejected_even_when_partial_rewrite_would_match(self) -> None:
        result = self.verify(b"alpha\r\nbeta\n", b"alpha\nbeta\n")

        self.assertFalse(result.accepted)
        self.assertEqual(result.decision, "rejected_mixed_newlines")

    def test_bare_cr_is_rejected(self) -> None:
        result = self.verify(b"alpha\rbeta\r\n", b"alpha\nbeta\n")

        self.assertFalse(result.accepted)
        self.assertEqual(result.decision, "rejected_bare_cr")

    def test_body_difference_and_json_reordering_are_not_canonicalized(self) -> None:
        body_result = self.verify(b"alpha changed\n", b"alpha\r\n")
        json_result = self.verify(b'{"b":2,"a":1}\n', b'{"a":1,"b":2}\n')

        self.assertFalse(body_result.accepted)
        self.assertEqual(body_result.decision, "rejected_hash_mismatch")
        self.assertFalse(json_result.accepted)
        self.assertEqual(json_result.decision, "rejected_hash_mismatch")

    def test_raw_mismatch_without_newlines_is_rejected(self) -> None:
        result = self.verify(b"alpha", b"beta")

        self.assertFalse(result.accepted)
        self.assertEqual(result.decision, "rejected_no_newline_conversion")


class HistoricalSourceVerificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self._git("init", "--initial-branch=main")
        self._git("config", "user.name", "Historical Source Test")
        self._git("config", "user.email", "historical-source@example.invalid")
        self.artifact = self.root / "src" / "artifact.txt"
        self.manifest = self.root / "tests" / "frozen.manifest.json"
        self.artifact.parent.mkdir(parents=True)
        self.manifest.parent.mkdir(parents=True)
        self.registered_bytes = b"registered\nsource\n"
        self.artifact.write_bytes(self.registered_bytes)
        self.expected = hashlib.sha256(self.registered_bytes).hexdigest()
        self.manifest.write_text(
            json.dumps(
                {"artifact_sha256": {"src/artifact.txt": self.expected}},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        self._git("add", ".")
        self._git("commit", "-m", "test: register historical source")
        self.commit = self._git("rev-parse", "HEAD").stdout.strip()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _git(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_registered_commit_bytes_survive_current_working_tree_evolution(self) -> None:
        self.artifact.write_bytes(b"later repository evolution\n")
        self._git("add", "src/artifact.txt")
        self._git("commit", "-m", "test: evolve current source")

        snapshot = verify_manifest_source_hashes(
            self.root,
            self.manifest,
            {"src/artifact.txt": self.expected},
        )

        self.assertEqual(snapshot.source_commit, self.commit)
        self.assertEqual(snapshot.artifact_bytes["src/artifact.txt"], self.registered_bytes)
        self.assertEqual(registered_manifest_commit(self.root, self.manifest), self.commit)

    def test_manifest_and_registered_blob_tampering_fail_closed(self) -> None:
        original_manifest = self.manifest.read_bytes()
        self.manifest.write_bytes(original_manifest + b" ")
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            verify_manifest_source_hashes(
                self.root,
                self.manifest,
                {"src/artifact.txt": self.expected},
            )

        self.manifest.write_bytes(original_manifest)
        self.artifact.write_bytes(b"tampered registered bytes\n")
        self.manifest.write_text(
            json.dumps(
                {
                    "artifact_sha256": {"src/artifact.txt": self.expected},
                    "revision": 2,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        self._git("add", ".")
        self._git("commit", "-m", "test: forge registered source")
        with self.assertRaisesRegex(ValueError, "historical source hash mismatch"):
            verify_manifest_source_hashes(
                self.root,
                self.manifest,
                {"src/artifact.txt": self.expected},
            )

    def test_unknown_commit_and_missing_path_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "commit is unavailable"):
            verify_historical_source_hashes(
                self.root,
                "0" * 40,
                {"src/artifact.txt": self.expected},
            )
        with self.assertRaisesRegex(ValueError, "path is missing"):
            verify_historical_source_hashes(
                self.root,
                self.commit,
                {"src/missing.txt": self.expected},
            )

    def test_nonancestor_commit_fails_closed(self) -> None:
        tree = self._git("show", "-s", "--format=%T", self.commit).stdout.strip()
        nonancestor = self._git(
            "commit-tree", tree, "-m", "test: detached historical source"
        ).stdout.strip()

        with self.assertRaisesRegex(ValueError, "not an ancestor"):
            verify_historical_source_hashes(
                self.root,
                nonancestor,
                {"src/artifact.txt": self.expected},
            )

    def test_newline_alternate_is_explicit(self) -> None:
        expected_crlf = hashlib.sha256(b"registered\r\nsource\r\n").hexdigest()
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_historical_source_hashes(
                self.root,
                self.commit,
                {"src/artifact.txt": expected_crlf},
            )
        accepted = verify_historical_source_hashes(
            self.root,
            self.commit,
            {"src/artifact.txt": expected_crlf},
            allow_text_newline_alternate=True,
        )
        self.assertEqual(accepted.artifact_bytes["src/artifact.txt"], self.registered_bytes)


if __name__ == "__main__":
    unittest.main()
