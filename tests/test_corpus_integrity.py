from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag.corpus_integrity import verify_source_sha256


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


if __name__ == "__main__":
    unittest.main()
