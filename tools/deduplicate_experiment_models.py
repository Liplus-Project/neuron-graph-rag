"""Preserve experiment snapshot paths while sharing identical model bytes.

This tool only considers files below explicit source roots whose path contains
``snapshots``. It never follows symlinks and never touches registered evidence.
Apply retains original bytes as per-file backups until verify/finalize succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

BACKUP_SUFFIX = ".ngr-dedup-backup"
SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_reparse_point(path: Path) -> bool:
    metadata = path.lstat()
    return bool(getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)) or path.is_symlink()


def _within(root: Path, path: Path) -> Path:
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes workspace: {path}") from exc


def _path(root: Path, relative: str) -> Path:
    item = Path(relative)
    if item.is_absolute() or ".." in item.parts or not item.parts:
        raise ValueError(f"unsafe relative path: {relative}")
    resolved = root / item
    _within(root, resolved)
    return resolved


def _read_plan(path: Path) -> tuple[dict, Path, Path]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported plan schema")
    root = Path(plan["workspace_root"]).resolve(strict=True)
    store = _path(root, plan["store_root"])
    if store.is_symlink():
        raise ValueError("store root must not be a symlink")
    sources = [_path(root, relative) for relative in plan["source_roots"]]
    if any(store == source or store.is_relative_to(source) for source in sources):
        raise ValueError("shared store overlaps a source root")
    for row in plan["files"]:
        source = _path(root, row["path"])
        _path(root, row["store_path"])
        if len(row["sha256"]) != 64 or row["size"] < 0:
            raise ValueError("invalid file identity in plan")
        if "snapshots" not in source.parts or not any(source.is_relative_to(parent) for parent in sources):
            raise ValueError(f"planned source is outside snapshot roots: {source}")
        expected_store = Path(plan["store_root"]) / "sha256" / row["sha256"][:2] / row["sha256"]
        if Path(row["store_path"]) != expected_store:
            raise ValueError(f"store path does not match digest: {row['store_path']}")
    return plan, root, store


def _write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def make_plan(workspace: Path, store: Path, sources: list[Path], output: Path) -> dict:
    root = workspace.resolve(strict=True)
    store_relative = _within(root, store)
    if store.exists():
        raise FileExistsError("new shared store path must be absent at plan time")
    if not sources:
        raise ValueError("at least one source root is required")
    files: dict[str, dict] = {}
    source_roots = []
    skipped_reparse_points = []
    for source in sources:
        base = source.resolve(strict=True)
        source_roots.append(_within(root, base).as_posix())
        if not base.is_dir() or _is_reparse_point(base):
            raise ValueError(f"source root must be a real directory: {source}")
        for item in base.rglob("*"):
            metadata = item.lstat()
            if _is_reparse_point(item):
                if stat.S_ISDIR(metadata.st_mode):
                    raise ValueError(f"source contains a linked directory: {item}")
                skipped_reparse_points.append(item.relative_to(root).as_posix())
                continue
            if not stat.S_ISREG(metadata.st_mode) or "snapshots" not in item.relative_to(base).parts:
                continue
            relative = _within(root, item).as_posix()
            size = item.stat().st_size
            digest = _sha256(item)
            store_path = store_relative / "sha256" / digest[:2] / digest
            files[relative] = {
                "path": relative,
                "size": size,
                "sha256": digest,
                "store_path": store_path.as_posix(),
            }
    if not files:
        raise ValueError("no snapshot files were found")
    rows = [files[key] for key in sorted(files)]
    unique = {(row["sha256"], row["size"]) for row in rows}
    logical_bytes = sum(row["size"] for row in rows)
    unique_bytes = sum(size for _, size in unique)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "workspace_root": str(root),
        "store_root": store_relative.as_posix(),
        "source_roots": sorted(set(source_roots)),
        "skipped_reparse_points": sorted(set(skipped_reparse_points)),
        "files": rows,
        "file_count": len(rows),
        "unique_content_count": len(unique),
        "logical_bytes": logical_bytes,
        "unique_content_bytes": unique_bytes,
        "maximum_reclaimable_bytes": logical_bytes - unique_bytes,
        "disk_free_bytes_at_plan": shutil.disk_usage(root).free,
    }
    _write_exclusive(output, plan)
    return plan


def _verify_file(path: Path, row: dict) -> None:
    if _is_reparse_point(path) or not path.is_file():
        raise ValueError(f"snapshot file missing or linked symbolically: {path}")
    if path.stat().st_size != row["size"] or _sha256(path) != row["sha256"]:
        raise ValueError(f"snapshot content changed: {path}")


def preflight(plan: dict, root: Path) -> None:
    for row in plan["files"]:
        source = _path(root, row["path"])
        backup = source.with_name(source.name + BACKUP_SUFFIX)
        if backup.exists():
            raise FileExistsError(f"unfinished backup already exists: {backup}")
        _verify_file(source, row)
        target = _path(root, row["store_path"])
        if target.exists():
            _verify_file(target, row)
        if source.stat().st_dev != root.stat().st_dev:
            raise ValueError(f"source is on another filesystem: {source}")


def apply(plan_path: Path) -> dict:
    plan, root, store = _read_plan(plan_path)
    preflight(plan, root)
    store.mkdir(parents=True, exist_ok=True)
    first_by_digest: dict[str, dict] = {}
    for row in plan["files"]:
        first_by_digest.setdefault(row["sha256"], row)
    for row in first_by_digest.values():
        source = _path(root, row["path"])
        target = _path(root, row["store_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(source, target)
        _verify_file(target, row)
    replaced = 0
    try:
        for row in plan["files"]:
            source = _path(root, row["path"])
            target = _path(root, row["store_path"])
            if os.path.samefile(source, target):
                continue
            backup = source.with_name(source.name + BACKUP_SUFFIX)
            os.replace(source, backup)
            os.link(target, source)
            _verify_file(source, row)
            if not os.path.samefile(source, target):
                raise ValueError(f"hardlink identity mismatch: {source}")
            replaced += 1
    except BaseException:
        rollback(plan_path)
        raise
    return {"replaced_files": replaced, "backups_retained": replaced}


def verify(plan_path: Path) -> dict:
    plan, root, _ = _read_plan(plan_path)
    backups = 0
    for row in plan["files"]:
        source = _path(root, row["path"])
        target = _path(root, row["store_path"])
        _verify_file(source, row)
        _verify_file(target, row)
        if not os.path.samefile(source, target):
            raise ValueError(f"source is not backed by shared store: {source}")
        backup = source.with_name(source.name + BACKUP_SUFFIX)
        if backup.exists():
            _verify_file(backup, row)
            backups += 1
    return {"verified_files": len(plan["files"]), "backups_retained": backups}


def rollback(plan_path: Path) -> dict:
    plan, root, _ = _read_plan(plan_path)
    restored = 0
    for row in plan["files"]:
        source = _path(root, row["path"])
        backup = source.with_name(source.name + BACKUP_SUFFIX)
        if not backup.exists():
            continue
        _verify_file(backup, row)
        if source.exists():
            _verify_file(source, row)
            os.unlink(source)
        os.replace(backup, source)
        restored += 1
    return {"restored_files": restored}


def finalize(plan_path: Path, receipt_path: Path) -> dict:
    plan, root, _ = _read_plan(plan_path)
    checked = verify(plan_path)
    if receipt_path.exists():
        raise FileExistsError(f"receipt already exists: {receipt_path}")
    removed = 0
    for row in plan["files"]:
        source = _path(root, row["path"])
        backup = source.with_name(source.name + BACKUP_SUFFIX)
        if backup.exists():
            _verify_file(backup, row)
            os.unlink(backup)
            removed += 1
    if removed != checked["backups_retained"]:
        raise RuntimeError("backup count changed during finalization")
    verified = verify(plan_path)
    if verified["backups_retained"]:
        raise RuntimeError("backups remain after finalization")
    for row in plan["files"]:
        target = _path(root, row["store_path"])
        os.chmod(target, stat.S_IREAD)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "plan_sha256": _sha256(plan_path),
        "verified_files": verified["verified_files"],
        "backups_removed": removed,
        "logical_bytes": plan["logical_bytes"],
        "unique_content_bytes": plan["unique_content_bytes"],
        "maximum_reclaimable_bytes": plan["maximum_reclaimable_bytes"],
        "disk_free_bytes_at_plan": plan["disk_free_bytes_at_plan"],
        "disk_free_bytes_after": shutil.disk_usage(root).free,
        "store_files_read_only": True,
    }
    _write_exclusive(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    planner = subcommands.add_parser("plan")
    planner.add_argument("--workspace", type=Path, required=True)
    planner.add_argument("--store", type=Path, required=True)
    planner.add_argument("--source", type=Path, action="append", required=True)
    planner.add_argument("--output", type=Path, required=True)
    for name in ("apply", "verify", "rollback", "finalize"):
        command = subcommands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        if name == "finalize":
            command.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        plan = make_plan(args.workspace, args.store, args.source, args.output)
        result = {
            "plan": str(args.output),
            "file_count": plan["file_count"],
            "unique_content_count": plan["unique_content_count"],
            "maximum_reclaimable_bytes": plan["maximum_reclaimable_bytes"],
            "skipped_reparse_points": len(plan["skipped_reparse_points"]),
        }
    elif args.command == "apply":
        result = apply(args.plan)
    elif args.command == "verify":
        result = verify(args.plan)
    elif args.command == "rollback":
        result = rollback(args.plan)
    else:
        result = finalize(args.plan, args.receipt)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
