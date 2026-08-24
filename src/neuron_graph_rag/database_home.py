from __future__ import annotations

import os
import sqlite3
import tempfile
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

DATABASE_ENVIRONMENT_VARIABLE = "NGR_DATABASE"
DATABASE_HOME_DIRECTORY = ".ngrdb"
DATABASE_FILENAME = "knowledge.db"


@dataclass(frozen=True, slots=True)
class DatabaseResolution:
    path: Path
    source: str


@dataclass(frozen=True, slots=True)
class DatabaseMigrationResult:
    source: Path
    destination: Path
    backup: Path


def resolve_database(
    command_line: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    home: str | Path | None = None,
) -> DatabaseResolution:
    environment = os.environ if environ is None else environ
    home_path = Path.home() if home is None else Path(home)
    if command_line is not None:
        value = str(command_line)
        if not value.strip():
            raise ValueError("--database must not be empty")
        return DatabaseResolution(_expand_user_path(value, home_path), "command_line")
    environment_value = environment.get(DATABASE_ENVIRONMENT_VARIABLE, "")
    if environment_value.strip():
        return DatabaseResolution(
            _expand_user_path(environment_value, home_path), "environment"
        )
    return DatabaseResolution(
        home_path / DATABASE_HOME_DIRECTORY / DATABASE_FILENAME,
        "default",
    )


def resolve_database_path(
    command_line: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    home: str | Path | None = None,
) -> Path:
    return resolve_database(command_line, environ=environ, home=home).path


def prepare_database(resolution: DatabaseResolution) -> Path:
    if resolution.source == "default":
        resolution.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    return resolution.path


def migrate_database(
    source: str | Path,
    destination: str | Path,
    backup: str | Path,
) -> DatabaseMigrationResult:
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    backup_path = Path(backup).expanduser().resolve()
    _require_distinct_paths(source_path, destination_path, backup_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"source database does not exist: {source_path}")
    if destination_path.exists():
        raise FileExistsError(
            f"refusing to overwrite destination database: {destination_path}"
        )
    if backup_path.exists():
        raise FileExistsError(f"refusing to overwrite backup database: {backup_path}")

    _check_database_integrity(source_path)
    backup_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _copy_database_without_overwrite(source_path, backup_path)
    _check_database_integrity(backup_path)
    destination_published = False
    try:
        _copy_database_without_overwrite(backup_path, destination_path)
        destination_published = True
        _check_database_integrity(destination_path)
    except Exception:
        if destination_published:
            destination_path.unlink()
        raise
    return DatabaseMigrationResult(source_path, destination_path, backup_path)


def _expand_user_path(value: str, home: Path) -> Path:
    if value == "~":
        return home
    if value.startswith(("~/", "~\\")):
        return home / value[2:]
    return Path(value).expanduser()


def _require_distinct_paths(source: Path, destination: Path, backup: Path) -> None:
    normalized = {
        os.path.normcase(str(source)),
        os.path.normcase(str(destination)),
        os.path.normcase(str(backup)),
    }
    if len(normalized) != 3:
        raise ValueError("source, destination, and backup must be distinct paths")


def _read_only_uri(path: Path) -> str:
    return path.as_uri() + "?mode=ro"


def _check_database_integrity(path: Path) -> None:
    with closing(sqlite3.connect(_read_only_uri(path), uri=True)) as connection:
        results = [row[0] for row in connection.execute("PRAGMA integrity_check")]
        foreign_key_violations = list(connection.execute("PRAGMA foreign_key_check"))
    if results != ["ok"] or foreign_key_violations:
        raise RuntimeError(f"database integrity check failed: {path}")


def _copy_database_without_overwrite(source: Path, destination: Path) -> None:
    temporary_file = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    try:
        with (
            closing(sqlite3.connect(_read_only_uri(source), uri=True)) as source_connection,
            closing(sqlite3.connect(temporary_path)) as destination_connection,
        ):
            source_connection.backup(destination_connection)
        _check_database_integrity(temporary_path)
        os.link(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
