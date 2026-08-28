from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import stat
import sys
from collections.abc import Iterable, Mapping
from email.parser import BytesParser
from email.policy import default
from pathlib import Path, PurePosixPath
from typing import Any

ALGORITHM_VERSION = "ngr.wslc-runtime-content/v2"
BASE_DIGEST = "sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3"
DEPENDENCY_ARTIFACT_REGISTRY = Path("/opt/ngr-v7/dependency-artifacts.json")
EXPECTED_DISTRIBUTION_REGISTRY = Path("/opt/ngr-v7/expected-distributions.json")
SITE_PACKAGES = Path("/usr/local/lib/python3.11/site-packages")
ROOTS = (
    (Path("/usr/local/bin/python3.11"), PurePosixPath("python-runtime/python3.11")),
    (
        Path("/usr/local/lib/python3.11/site-packages"),
        PurePosixPath("site-packages"),
    ),
    (Path("/opt/ngr-v7/requirements.lock"), PurePosixPath("protocol/requirements.lock")),
    (
        Path("/opt/ngr-v7/dependency-report.json"),
        PurePosixPath("protocol/dependency-report.json"),
    ),
    (
        Path("/opt/ngr-v7/validate_runtime.py"),
        PurePosixPath("protocol/validate_runtime.py"),
    ),
    (
        Path("/opt/ngr-v7/runtime_content.py"),
        PurePosixPath("protocol/runtime_content.py"),
    ),
    (
        EXPECTED_DISTRIBUTION_REGISTRY,
        PurePosixPath("protocol/expected-distributions.json"),
    ),
)
EXCLUSION_REGISTRY = (
    {"kind": "component", "value": "__pycache__"},
    {"kind": "suffix", "value": ".pyc"},
    {"kind": "component", "value": ".cache"},
    {"kind": "component", "value": "cache"},
    {"kind": "suffix", "value": ".tmp"},
    {"kind": "suffix", "value": ".log"},
    {"kind": "absolute-root", "value": "/opt/ngr-v7/runtime"},
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def exclusion_registry_sha256() -> str:
    return sha256_bytes(canonical_json_bytes(EXCLUSION_REGISTRY))


def canonicalize_name(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("distribution name must be a non-empty trimmed string")
    canonical = re.sub(r"[-_.]+", "-", value).lower()
    if not canonical or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", canonical):
        raise ValueError("distribution canonical name is malformed")
    return canonical


def normalize_distribution_inventory(
    inventory: Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    paths: set[bytes] = set()
    folded_paths: set[str] = set()
    names: set[str] = set()
    for raw in inventory:
        row = dict(raw)
        if set(row) != {
            "metadata_path",
            "name",
            "version",
            "canonical_name",
            "metadata_sha256",
        }:
            raise ValueError("filesystem distribution inventory shape mismatch")
        metadata_path = _validate_relative_path(str(row["metadata_path"])).as_posix()
        parts = PurePosixPath(metadata_path).parts
        if (
            len(parts) != 3
            or parts[0] != "site-packages"
            or not parts[1].endswith(".dist-info")
            or parts[2] != "METADATA"
        ):
            raise ValueError("filesystem distribution METADATA path mismatch")
        encoded = metadata_path.encode("utf-8")
        folded = metadata_path.casefold()
        if encoded in paths or folded in folded_paths:
            raise ValueError("duplicate or case-colliding METADATA path")
        paths.add(encoded)
        folded_paths.add(folded)
        name = row["name"]
        version = row["version"]
        if (
            not isinstance(name, str)
            or not isinstance(version, str)
            or not name
            or not version
            or name != name.strip()
            or version != version.strip()
            or "\n" in name
            or "\n" in version
        ):
            raise ValueError("filesystem distribution Name/Version is malformed")
        canonical = canonicalize_name(name)
        if row["canonical_name"] != canonical or canonical in names:
            raise ValueError("duplicate or mismatched canonical distribution name")
        names.add(canonical)
        digest = row["metadata_sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("filesystem METADATA SHA-256 mismatch")
        normalized.append(
            {
                "metadata_path": metadata_path,
                "name": name,
                "version": version,
                "canonical_name": canonical,
                "metadata_sha256": digest,
            }
        )
    normalized.sort(key=lambda row: row["metadata_path"].encode("utf-8"))
    return normalized


def collect_filesystem_distribution_inventory(
    site_packages: Path = SITE_PACKAGES,
) -> list[dict[str, str]]:
    if not site_packages.is_dir() or site_packages.is_symlink():
        raise ValueError("site-packages root is missing or not a real directory")
    rows: list[dict[str, str]] = []
    for directory in sorted(
        site_packages.glob("*.dist-info"), key=lambda path: path.name.encode("utf-8")
    ):
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("dist-info entry must be a real directory")
        metadata = directory / "METADATA"
        if not metadata.is_file() or metadata.is_symlink():
            raise ValueError(f"dist-info METADATA is missing: {directory.name}")
        raw = metadata.read_bytes()
        parsed = BytesParser(policy=default).parsebytes(raw, headersonly=True)
        name = parsed.get("Name")
        version = parsed.get("Version")
        if name is None or version is None:
            raise ValueError(f"dist-info METADATA is malformed: {directory.name}")
        rows.append(
            {
                "metadata_path": f"site-packages/{directory.name}/METADATA",
                "name": name,
                "version": version,
                "canonical_name": canonicalize_name(name),
                "metadata_sha256": sha256_bytes(raw),
            }
        )
    return normalize_distribution_inventory(rows)


def _validate_inventory_entry_correspondence(
    entries: Iterable[Mapping[str, Any]], inventory: Iterable[Mapping[str, Any]]
) -> None:
    entry_paths = {
        str(row["path"])
        for row in entries
        if str(row.get("path", "")).startswith("site-packages/")
        and str(row.get("path", "")).endswith(".dist-info/METADATA")
    }
    inventory_paths = {str(row["metadata_path"]) for row in inventory}
    if entry_paths != inventory_paths:
        raise ValueError("normalized content and filesystem METADATA inventory differ")


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
    expected_distribution_registry_sha256: str,
    filesystem_distributions: Iterable[Mapping[str, Any]],
    python_identity: Mapping[str, str] | None = None,
    base_digest: str = BASE_DIGEST,
) -> dict[str, Any]:
    normalized = normalize_entries(entries)
    distributions = normalize_distribution_inventory(filesystem_distributions)
    _validate_inventory_entry_correspondence(normalized, distributions)
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
        "expected_distribution_registry_sha256": expected_distribution_registry_sha256,
        "normalized_entries": normalized,
        "entry_count": len(normalized),
        "filesystem_distributions": distributions,
        "filesystem_distribution_count": len(distributions),
    }
    return {**payload, "fingerprint_sha256": sha256_bytes(canonical_json_bytes(payload))}


def validate_report(report: Mapping[str, Any]) -> None:
    if set(report) != {
        "algorithm_version",
        "base_digest",
        "python",
        "dependency_artifact_registry_sha256",
        "expected_distribution_registry_sha256",
        "normalized_entries",
        "entry_count",
        "filesystem_distributions",
        "filesystem_distribution_count",
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
    expected_digest = report.get("expected_distribution_registry_sha256")
    if (
        not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        or not isinstance(expected_digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_digest)
    ):
        raise ValueError("dependency artifact registry SHA-256 mismatch")
    distributions = normalize_distribution_inventory(
        report.get("filesystem_distributions", [])
    )
    if report.get("filesystem_distribution_count") != len(distributions):
        raise ValueError("filesystem distribution count mismatch")
    _validate_inventory_entry_correspondence(entries, distributions)
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
    expected_sha256 = sha256_bytes(EXPECTED_DISTRIBUTION_REGISTRY.read_bytes())
    report = build_report(
        collect_entries(),
        dependency_artifact_registry_sha256=dependency_sha256,
        expected_distribution_registry_sha256=expected_sha256,
        filesystem_distributions=collect_filesystem_distribution_inventory(),
    )
    validate_report(report)
    sys.stdout.buffer.write(canonical_json_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
