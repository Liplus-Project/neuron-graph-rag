from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import stat
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

ALGORITHM_VERSION = "ngr.wslc-runtime-content/v1"
BASE_DIGEST = "sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3"
DEPENDENCY_ARTIFACT_REGISTRY = Path("/opt/ngr-v6/dependency-artifacts.json")
ROOTS = (
    (Path("/usr/local/bin/python3.11"), PurePosixPath("python-runtime/python3.11")),
    (
        Path("/usr/local/lib/python3.11/site-packages"),
        PurePosixPath("site-packages"),
    ),
    (Path("/opt/ngr-v6/requirements.lock"), PurePosixPath("protocol/requirements.lock")),
    (
        Path("/opt/ngr-v6/dependency-report.json"),
        PurePosixPath("protocol/dependency-report.json"),
    ),
    (
        Path("/opt/ngr-v6/validate_runtime.py"),
        PurePosixPath("protocol/validate_runtime.py"),
    ),
    (
        Path("/opt/ngr-v6/runtime_content.py"),
        PurePosixPath("protocol/runtime_content.py"),
    ),
)
EXCLUSION_REGISTRY = (
    {"kind": "component", "value": "__pycache__"},
    {"kind": "suffix", "value": ".pyc"},
    {"kind": "component", "value": ".cache"},
    {"kind": "component", "value": "cache"},
    {"kind": "suffix", "value": ".tmp"},
    {"kind": "suffix", "value": ".log"},
    {"kind": "absolute-root", "value": "/opt/ngr-v6/runtime"},
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def exclusion_registry_sha256() -> str:
    return sha256_bytes(canonical_json_bytes(EXCLUSION_REGISTRY))


def _validate_relative_path(value: str) -> PurePosixPath:
    if not value or "\\" in value or value.startswith("/"):
        raise ValueError("normalized path must be a relative POSIX path")
    path = PurePosixPath(value)
    if any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("normalized path traversal is forbidden")
    return path


def _excluded(path: PurePosixPath) -> bool:
    for rule in EXCLUSION_REGISTRY:
        kind = rule["kind"]
        value = rule["value"]
        if kind == "component" and value in path.parts:
            return True
        if kind == "suffix" and path.name.endswith(value):
            return True
    return False


def _entry(path: Path, relative: PurePosixPath) -> dict[str, Any]:
    normalized = _validate_relative_path(relative.as_posix()).as_posix()
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        target = os.readlink(path)
        return {
            "path": normalized,
            "type": "symlink",
            "symlink_target": target,
            "size": len(os.fsencode(target)),
        }
    if stat.S_ISREG(mode):
        raw = path.read_bytes()
        return {
            "path": normalized,
            "type": "file",
            "content_sha256": sha256_bytes(raw),
            "size": len(raw),
        }
    raise ValueError(f"unsupported runtime entry type: {normalized}")


def collect_entries(
    roots: Iterable[tuple[Path, PurePosixPath]] = ROOTS,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for source, prefix in roots:
        _validate_relative_path(prefix.as_posix())
        if not source.exists() and not source.is_symlink():
            raise ValueError(f"required fingerprint root is missing: {prefix}")
        if source.is_symlink() or source.is_file():
            if not _excluded(prefix):
                entries.append(_entry(source, prefix))
            continue
        if not source.is_dir():
            raise ValueError(f"unsupported fingerprint root: {prefix}")
        for directory, names, filenames in os.walk(source, followlinks=False):
            names.sort(key=lambda item: item.encode("utf-8"))
            filenames.sort(key=lambda item: item.encode("utf-8"))
            directory_path = Path(directory)
            relative_directory = directory_path.relative_to(source)
            kept_names: list[str] = []
            for name in names:
                item = directory_path / name
                relative = prefix / PurePosixPath(relative_directory.as_posix()) / name
                if _excluded(relative):
                    continue
                if item.is_symlink():
                    entries.append(_entry(item, relative))
                else:
                    kept_names.append(name)
            names[:] = kept_names
            for name in filenames:
                item = directory_path / name
                relative = prefix / PurePosixPath(relative_directory.as_posix()) / name
                if not _excluded(relative):
                    entries.append(_entry(item, relative))
    return normalize_entries(entries)


def normalize_entries(entries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    exact: set[bytes] = set()
    folded: set[str] = set()
    for raw in entries:
        row = dict(raw)
        path = _validate_relative_path(str(row.get("path", ""))).as_posix()
        encoded = path.encode("utf-8")
        casefolded = path.casefold()
        if encoded in exact:
            raise ValueError("duplicate normalized path")
        if casefolded in folded:
            raise ValueError("case-colliding normalized path")
        exact.add(encoded)
        folded.add(casefolded)
        kind = row.get("type")
        if kind == "file":
            if set(row) != {"path", "type", "content_sha256", "size"}:
                raise ValueError("regular file entry shape mismatch")
            if not isinstance(row["size"], int) or row["size"] < 0:
                raise ValueError("regular file size mismatch")
            digest = row["content_sha256"]
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("regular file SHA-256 mismatch")
        elif kind == "symlink":
            if set(row) != {"path", "type", "symlink_target", "size"}:
                raise ValueError("symlink entry shape mismatch")
            if not isinstance(row["symlink_target"], str):
                raise ValueError("symlink target mismatch")
            if row["size"] != len(os.fsencode(row["symlink_target"])):
                raise ValueError("symlink size mismatch")
        else:
            raise ValueError("runtime entry type mismatch")
        row["path"] = path
        normalized.append(row)
    normalized.sort(key=lambda row: row["path"].encode("utf-8"))
    return normalized


def build_report(
    entries: Iterable[Mapping[str, Any]],
    *,
    dependency_artifact_registry_sha256: str,
    python_identity: Mapping[str, str] | None = None,
    base_digest: str = BASE_DIGEST,
) -> dict[str, Any]:
    normalized = normalize_entries(entries)
    identity = dict(
        python_identity
        or {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "abi": "cp311",
        }
    )
    payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "base_digest": base_digest,
        "python": identity,
        "dependency_artifact_registry_sha256": dependency_artifact_registry_sha256,
        "normalized_entries": normalized,
        "entry_count": len(normalized),
    }
    return {**payload, "fingerprint_sha256": sha256_bytes(canonical_json_bytes(payload))}


def validate_report(report: Mapping[str, Any]) -> None:
    if set(report) != {
        "algorithm_version",
        "base_digest",
        "python",
        "dependency_artifact_registry_sha256",
        "normalized_entries",
        "entry_count",
        "fingerprint_sha256",
    }:
        raise ValueError("runtime content report fields mismatch")
    if report.get("algorithm_version") != ALGORITHM_VERSION:
        raise ValueError("runtime content algorithm mismatch")
    if report.get("base_digest") != BASE_DIGEST:
        raise ValueError("runtime content base digest mismatch")
    if report.get("python") != {
        "implementation": "CPython",
        "version": "3.11.15",
        "abi": "cp311",
    }:
        raise ValueError("runtime content Python identity mismatch")
    entries = normalize_entries(report.get("normalized_entries", []))
    if report.get("entry_count") != len(entries):
        raise ValueError("runtime content entry count mismatch")
    digest = report.get("dependency_artifact_registry_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("dependency artifact registry SHA-256 mismatch")
    payload = {key: report[key] for key in report if key != "fingerprint_sha256"}
    if report.get("fingerprint_sha256") != sha256_bytes(canonical_json_bytes(payload)):
        raise ValueError("runtime content fingerprint mismatch")


def reports_equivalent(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    validate_report(left)
    validate_report(right)
    return dict(left) == dict(right)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit normalized WSLC runtime content")
    parser.parse_args(argv)
    dependency_sha256 = sha256_bytes(DEPENDENCY_ARTIFACT_REGISTRY.read_bytes())
    report = build_report(
        collect_entries(),
        dependency_artifact_registry_sha256=dependency_sha256,
    )
    validate_report(report)
    sys.stdout.buffer.write(canonical_json_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
