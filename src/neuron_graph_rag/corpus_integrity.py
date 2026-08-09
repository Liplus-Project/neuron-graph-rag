"""Portable, raw-first integrity checks for future frozen source corpora."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceHashVerification:
    """The integrity decision and hashes used to reach it."""

    accepted: bool
    decision: str
    reason: str
    expected_sha256: str
    raw_sha256: str
    alternate_sha256: str | None = None


def verify_source_sha256(
    path: str | Path, expected_sha256: str
) -> SourceHashVerification:
    """Verify bytes before allowing one whole-file LF/CRLF conversion.

    A raw hash match always wins. On a raw mismatch, only an input containing
    exclusively LF or exclusively CRLF line endings is eligible for conversion.
    This deliberately does not parse or canonicalize the source content.
    """

    raw = Path(path).read_bytes()
    raw_sha256 = _sha256(raw)
    if raw_sha256 == expected_sha256:
        return SourceHashVerification(
            accepted=True,
            decision="raw_match",
            reason="raw SHA-256 matched",
            expected_sha256=expected_sha256,
            raw_sha256=raw_sha256,
        )

    if b"\r" not in raw:
        if b"\n" not in raw:
            return _rejected(
                "rejected_no_newline_conversion",
                "raw mismatch and input has no newline to convert",
                expected_sha256,
                raw_sha256,
            )
        alternate_sha256 = _sha256(raw.replace(b"\n", b"\r\n"))
        if alternate_sha256 == expected_sha256:
            return SourceHashVerification(
                accepted=True,
                decision="newline_equivalent_lf_to_crlf",
                reason="whole-file LF to CRLF conversion matched",
                expected_sha256=expected_sha256,
                raw_sha256=raw_sha256,
                alternate_sha256=alternate_sha256,
            )
        return _rejected(
            "rejected_hash_mismatch",
            "raw and whole-file LF to CRLF hashes differ from expected",
            expected_sha256,
            raw_sha256,
            alternate_sha256,
        )

    newline_remainder = raw.replace(b"\r\n", b"")
    if b"\r" in newline_remainder:
        return _rejected(
            "rejected_bare_cr",
            "raw mismatch and input contains a bare CR",
            expected_sha256,
            raw_sha256,
        )
    if b"\n" in newline_remainder:
        return _rejected(
            "rejected_mixed_newlines",
            "raw mismatch and input mixes CRLF with LF",
            expected_sha256,
            raw_sha256,
        )

    alternate_sha256 = _sha256(raw.replace(b"\r\n", b"\n"))
    if alternate_sha256 == expected_sha256:
        return SourceHashVerification(
            accepted=True,
            decision="newline_equivalent_crlf_to_lf",
            reason="whole-file CRLF to LF conversion matched",
            expected_sha256=expected_sha256,
            raw_sha256=raw_sha256,
            alternate_sha256=alternate_sha256,
        )
    return _rejected(
        "rejected_hash_mismatch",
        "raw and whole-file CRLF to LF hashes differ from expected",
        expected_sha256,
        raw_sha256,
        alternate_sha256,
    )


def _rejected(
    decision: str,
    reason: str,
    expected_sha256: str,
    raw_sha256: str,
    alternate_sha256: str | None = None,
) -> SourceHashVerification:
    return SourceHashVerification(
        accepted=False,
        decision=decision,
        reason=reason,
        expected_sha256=expected_sha256,
        raw_sha256=raw_sha256,
        alternate_sha256=alternate_sha256,
    )


def _sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()
