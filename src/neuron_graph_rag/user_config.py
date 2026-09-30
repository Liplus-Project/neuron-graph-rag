"""One user-owned NGR configuration, independent of the client's working directory."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PORT = 8765
PATH_FIELDS = ("database", "cuda_cache", "cuda_e5_snapshot", "cuda_v2_m3_snapshot")
FIELDS = (*PATH_FIELDS, "port", "cuda_device")


def ngr_home(home: str | Path | None = None) -> Path:
    return (Path.home() if home is None else Path(home)) / ".ngr"


def config_path(home: str | Path | None = None) -> Path:
    return ngr_home(home) / "config.json"


def require_home_available(home: str | Path | None = None) -> None:
    if (ngr_home(home) / "home-migration.pending").exists():
        raise ValueError("Home migration is pending; stop all clients and rerun --migrate-home --confirm-stopped")


def read_config(home: str | Path | None = None) -> dict[str, Any]:
    path = config_path(home)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (ValueError, UnicodeError) as error:
        raise ValueError("NGR config.json must be valid UTF-8 JSON") from error
    validate_config(value)
    return value


def validate_config(value: Any) -> None:
    if not isinstance(value, dict) or set(value) - set(FIELDS):
        raise ValueError("NGR config.json has unsupported fields or is not an object")
    for name in PATH_FIELDS:
        if name in value and (not isinstance(value[name], str) or not value[name].strip()):
            raise ValueError(f"NGR {name} must be a nonempty path")
    for name, minimum, maximum in (("port", 1, 65535), ("cuda_device", 0, None)):
        if name in value:
            number = value[name]
            if (type(number) is not int or number < minimum
                    or (maximum is not None and number > maximum)):
                raise ValueError(f"NGR {name} has an invalid integer value")


def write_config(value: dict[str, Any], *, home: str | Path | None = None) -> None:
    validate_config(value)
    atomic_json(config_path(home), value)


def ensure_config(*, home: str | Path | None = None) -> None:
    path = config_path(home)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write("{}\n")
    except FileExistsError:
        read_config(home)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def expand_path(value: str | Path, home: Path, *, base: Path | None = None) -> Path:
    text = str(value)
    if text == "~":
        path = home
    elif text.startswith(("~/", "~\\")):
        path = home / text[2:]
    else:
        path = Path(text).expanduser()
    if base is not None and not path.is_absolute():
        path = base / path
    return path


@dataclass(frozen=True)
class SharedConfiguration:
    database: Path
    port: int
    database_source: str = "default"
    cuda_cache: Path | None = None
    cuda_e5_snapshot: Path | None = None
    cuda_v2_m3_snapshot: Path | None = None
    cuda_device: int = 0

    def cuda_identity(self) -> dict[str, Any]:
        if self.cuda_cache is None:
            return {}
        return {**{name: str(getattr(self, name)) for name in PATH_FIELDS[1:]},
                "cuda_device": self.cuda_device}

    def apply(self, namespace: Any) -> None:
        for name in FIELDS:
            setattr(namespace, name, getattr(self, name))


def resolve_shared(command_line: Any = None, *, environ: Mapping[str, str] | None = None,
                   home: str | Path | None = None) -> SharedConfiguration:
    from .database_home import resolve_database

    root = Path.home() if home is None else Path(home)
    require_home_available(root)
    environment = os.environ if environ is None else environ
    document = read_config(root)
    values: dict[str, Any] = {}
    for name in FIELDS[1:]:
        cli = getattr(command_line, name, None)
        env = environment.get("NGR_" + name.upper())
        if cli is not None:
            value = cli
        elif env is not None:
            if not env.strip():
                raise ValueError(f"NGR_{name.upper()} must not be empty")
            value = env
        else:
            value = document.get(name, DEFAULT_PORT if name == "port" else 0 if name == "cuda_device" else None)
        if name in ("port", "cuda_device"):
            if isinstance(value, bool):
                raise ValueError(f"NGR {name} must be an integer")
            value = int(value)
        elif value is not None:
            value = expand_path(value, root, base=ngr_home(root) if cli is None and env is None else None).resolve()
        values[name] = value
    validate_config({name: value for name, value in values.items() if name in ("port", "cuda_device")})
    paths = [values[name] for name in PATH_FIELDS[1:]]
    if any(paths) and not all(paths):
        raise ValueError("CUDA requires all three path options")
    if not any(paths) and values["cuda_device"] != 0:
        raise ValueError("cuda_device requires CUDA path options")
    database = resolve_database(getattr(command_line, "database", None), environ=environment, home=root,
                                configuration=document)
    return SharedConfiguration(database.path.resolve(), database_source=database.source, **values)
