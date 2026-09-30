"""Explicit, recoverable migration of the legacy default database only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path
from typing import Iterator

from .database_home import _check_database_integrity, _copy_database_without_overwrite
from .user_config import atomic_json, config_path, ngr_home, read_config, write_config


def migration_completed(home: Path) -> bool:
    try:
        record = json.loads((ngr_home(home) / "home-migration.json").read_text(encoding="utf-8"))
        return (record.get("status") == "complete"
                and record.get("source") == str((home / ".ngrdb/knowledge.db").resolve())
                and record.get("destination") == str((ngr_home(home) / "db/knowledge.db").resolve())
                and (ngr_home(home) / "db/knowledge.db").is_file())
    except (OSError, ValueError, AttributeError):
        return False


def _runtime_processes() -> bool:
    if os.name == "nt":
        # Return only a boolean. Command lines may contain secrets.
        script = ("$p=Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object { "
                  "$_.ProcessId -ne $env:_NGR_MIGRATOR_PID -and "
                  "($_.Name -eq 'NGR.exe' -or $_.CommandLine -match 'neuron_graph_rag_mcp') -and "
                  "$_.CommandLine -notmatch '--migrate-home|--version|--configure-clients' }; "
                  "if ($p) { 'running' } else { 'stopped' }")
        environment = dict(os.environ, _NGR_MIGRATOR_PID=str(os.getpid()))
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                                env=environment, capture_output=True, text=True, timeout=20)
        if result.returncode != 0 or result.stdout.strip() not in ("running", "stopped"):
            raise RuntimeError("Could not verify that NGR processes have stopped")
        return result.stdout.strip() == "running"
    proc = Path("/proc")
    if not proc.is_dir():
        raise RuntimeError("Automatic process verification is unavailable on this platform")
    for item in proc.iterdir():
        if not item.name.isdecimal() or int(item.name) == os.getpid():
            continue
        try:
            args = (item / "cmdline").read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        except PermissionError as error:
            raise RuntimeError("Could not inspect processes before migration") from error
        if (b"neuron_graph_rag_mcp" in args or b"neuron_graph_rag_mcp.tray_controller" in args
                or any(Path(os.fsdecode(arg)).name in ("NGR.exe", "neuron-graph-rag-mcp") for arg in args if arg)):
            return True
    return False


@contextmanager
def _exclusive_file(path: Path) -> Iterator[None]:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            if stream.seek(0, 2) == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        for row in connection.iterdump():
            digest.update(row.encode("utf-8") + b"\n")
    return digest.hexdigest()


def migrate_home(*, home: str | Path | None = None, confirm_stopped: bool = False) -> dict:
    root = (Path.home() if home is None else Path(home)).resolve()
    if not confirm_stopped:
        raise ValueError("Exit every NGR client and tray, then pass --confirm-stopped")
    if _runtime_processes():
        raise RuntimeError("NGR processes are running; exit all clients and the tray before migration")
    document = read_config(root)
    if "database" in document or os.environ.get("NGR_DATABASE", "").strip():
        raise ValueError("Explicit databases are user managed; clear the override before default-home migration")
    source = root / ".ngrdb/knowledge.db"
    target = ngr_home(root) / "db/knowledge.db"
    backup = ngr_home(root) / "backups/legacy-knowledge.db"
    receipt = ngr_home(root) / "home-migration.json"
    if migration_completed(root):
        (ngr_home(root) / "home-migration.pending").unlink(missing_ok=True)
        return {"status": "complete", "destination": str(target), "backup": str(backup)}
    if not source.is_file():
        raise FileNotFoundError("Legacy default database was not found")
    with ExitStack() as stack:
        stack.enter_context(_exclusive_file(ngr_home(root) / "home-migration.lock"))
        pending = ngr_home(root) / "home-migration.pending"
        pending.touch()
        # Cover both old startup locks and the new runtime lock directory.
        for directory in (root / ".ngrdb", ngr_home(root)):
            for lock in sorted(directory.glob("shared-local-mcp-*.lock")):
                stack.enter_context(_exclusive_file(lock))
        if _runtime_processes():
            raise RuntimeError("NGR restarted during migration preparation; stop clients and retry")
        writer = stack.enter_context(closing(sqlite3.connect(source, timeout=0)))
        writer.execute("BEGIN IMMEDIATE")
        try:
            signature = _logical_digest(source)
            if receipt.exists():
                record = json.loads(receipt.read_text(encoding="utf-8"))
                if (record.get("source") != str(source) or record.get("destination") != str(target)
                        or record.get("backup") != str(backup) or record.get("source_digest") != signature
                        or record.get("status") not in ("planned", "copied")
                        or record.get("config") != document
                        or (record.get("status") == "copied" and (
                            record.get("destination_digest") != _digest(target)
                            or record.get("backup_digest") != _digest(backup)))):
                    raise RuntimeError("Migration recovery conflicts with current data or configuration; preserve all files and inspect backups")
            else:
                if target.exists() or backup.exists():
                    raise FileExistsError("Unrecorded migration output already exists")
                record = {"status": "planned", "source": str(source), "destination": str(target),
                          "backup": str(backup), "source_digest": signature, "config": document}
                atomic_json(receipt, record)
            for output in (backup, target):
                output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                if not output.exists():
                    _copy_database_without_overwrite(source if output == backup else backup, output)
                _check_database_integrity(output)
                if _logical_digest(output) != signature:
                    raise RuntimeError("Migration output differs from the stopped source; preserve files and inspect backups")
            record.update(status="copied", destination_digest=_digest(target), backup_digest=_digest(backup))
            atomic_json(receipt, record)
            if _runtime_processes():
                raise RuntimeError("NGR restarted before configuration cutover; stop clients and retry")
            old_config = config_path(root)
            config_backup = ngr_home(root) / "backups/config.before-home-migration.json"
            if old_config.is_file() and not config_backup.exists():
                # Non-overwriting copy also protects against a concurrent backup publisher.
                with config_backup.open("xb") as stream:
                    stream.write(old_config.read_bytes())
            write_config(document, home=root)
            record["status"] = "complete"
            atomic_json(receipt, record)
            pending.unlink()
            return record
        finally:
            writer.rollback()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Migrate the stopped legacy default DB to ~/.ngr/db")
    parser.add_argument("--confirm-stopped", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = migrate_home(confirm_stopped=args.confirm_stopped)
    except (OSError, ValueError, RuntimeError, sqlite3.Error):
        parser.exit(2, "NGR migration stopped. Existing data and backups are retained; resolve conflicts and retry.\n")
    print(f"Migration complete: {result['destination']}; recovery backup: {result['backup']}")


if __name__ == "__main__":
    main()
