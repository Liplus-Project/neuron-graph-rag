"""Raw and repository-historical integrity checks for frozen sources."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Mapping
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


@dataclass(frozen=True)
class HistoricalSourceSnapshot:
    """Verified bytes read from one registered repository commit."""

    source_commit: str
    artifact_bytes: dict[str, bytes]
    artifact_sha256: dict[str, str]


def registered_manifest_commit(
    repository_root: str | Path, manifest_path: str | Path
) -> str:
    """Resolve the commit that registered the current frozen manifest bytes.

    The manifest itself remains the immutable registry.  Its most recent commit
    is therefore the source boundary for the hashes it contains; later working
    tree changes must not move that boundary.
    """

    root = Path(repository_root).resolve()
    relative = _repository_relative(root, manifest_path)
    completed = _git(root, "log", "-1", "--format=%H", "--", relative)
    commit = completed.stdout.decode("ascii", errors="strict").strip()
    if not commit:
        raise ValueError(f"frozen manifest is not registered in git: {relative}")
    _verify_commit_boundary(root, commit)
    registered = _git_bytes(root, commit, relative)
    current = (root / relative).read_bytes()
    if current != registered:
        raise ValueError(f"frozen manifest differs from registered commit: {relative}")
    return commit


def verify_historical_source_hashes(
    repository_root: str | Path,
    source_commit: str,
    expected_hashes: Mapping[str, str],
    *,
    allow_text_newline_alternate: bool = False,
) -> HistoricalSourceSnapshot:
    """Verify registered blobs without consulting same-path working-tree bytes."""

    root = Path(repository_root).resolve()
    commit = str(source_commit)
    _verify_commit_boundary(root, commit)
    if not isinstance(expected_hashes, Mapping) or not expected_hashes:
        raise ValueError("historical source hash registry must be non-empty")

    artifact_bytes: dict[str, bytes] = {}
    artifact_sha256: dict[str, str] = {}
    for raw_relative, raw_expected in expected_hashes.items():
        relative = _repository_relative(root, str(raw_relative))
        expected = str(raw_expected)
        registered = _git_bytes(root, commit, relative)
        verification = verify_source_bytes(
            registered,
            expected,
            allow_text_newline_alternate=allow_text_newline_alternate,
        )
        if not verification.accepted:
            raise ValueError(
                f"historical source hash mismatch at {commit}: {relative}"
            )
        artifact_bytes[relative] = registered
        artifact_sha256[relative] = expected
    return HistoricalSourceSnapshot(commit, artifact_bytes, artifact_sha256)


def verify_manifest_source_hashes(
    repository_root: str | Path,
    manifest_path: str | Path,
    expected_hashes: Mapping[str, str],
    *,
    allow_text_newline_alternate: bool = False,
) -> HistoricalSourceSnapshot:
    """Verify a manifest registry against the commit that registered it."""

    root = Path(repository_root).resolve()
    commit = registered_manifest_commit(root, manifest_path)
    return verify_historical_source_hashes(
        root,
        commit,
        expected_hashes,
        allow_text_newline_alternate=allow_text_newline_alternate,
    )


def verify_source_sha256(
    path: str | Path, expected_sha256: str
) -> SourceHashVerification:
    """Verify bytes before allowing one whole-file LF/CRLF conversion.

    A raw hash match always wins. On a raw mismatch, only an input containing
    exclusively LF or exclusively CRLF line endings is eligible for conversion.
    This deliberately does not parse or canonicalize the source content.
    """

    return verify_source_bytes(
        Path(path).read_bytes(),
        expected_sha256,
        allow_text_newline_alternate=True,
    )


def verify_source_bytes(
    raw: bytes,
    expected_sha256: str,
    *,
    allow_text_newline_alternate: bool,
) -> SourceHashVerification:
    """Verify bytes, optionally allowing one whole-file newline conversion."""

    raw_sha256 = _sha256(raw)
    expected = _prefixed_sha256(expected_sha256)
    if raw_sha256 == expected:
        return SourceHashVerification(
            accepted=True,
            decision="raw_match",
            reason="raw SHA-256 matched",
            expected_sha256=expected_sha256,
            raw_sha256=raw_sha256,
        )

    if not allow_text_newline_alternate:
        return _rejected(
            "rejected_hash_mismatch",
            "raw bytes differ from expected",
            expected_sha256,
            raw_sha256,
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
        if alternate_sha256 == expected:
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
    if alternate_sha256 == expected:
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


def _prefixed_sha256(value: str) -> str:
    return value if value.startswith("sha256:") else f"sha256:{value}"


def _repository_relative(root: Path, path: str | Path) -> str:
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"historical source path is outside repository: {path}") from error
    if not relative.parts or ".git" in relative.parts:
        raise ValueError(f"invalid historical source path: {path}")
    return relative.as_posix()


def _verify_commit_boundary(root: Path, commit: str) -> None:
    if not commit or any(character.isspace() for character in commit):
        raise ValueError(f"invalid historical source commit: {commit!r}")
    try:
        _git(root, "cat-file", "-e", f"{commit}^{{commit}}")
    except RuntimeError as error:
        raise ValueError(f"historical source commit is unavailable: {commit}") from error
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if ancestor.returncode != 0:
        raise ValueError(f"historical source commit is not an ancestor of HEAD: {commit}")


def _git_bytes(root: Path, commit: str, relative: str) -> bytes:
    try:
        return _git(root, "show", f"{commit}:{relative}").stdout
    except RuntimeError as error:
        raise ValueError(
            f"historical source path is missing at {commit}: {relative}"
        ) from error


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or f"git {' '.join(arguments)} failed")
    return completed
