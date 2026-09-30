"""Shared client registration; native files contain only the stdio launch entry."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import uuid
from pathlib import Path

from neuron_graph_rag.user_config import FIELDS, atomic_json, ensure_config, ngr_home, read_config
from .windows_registration import SERVER_NAME, TOKEN_ENV, _config_path, _ensure_token, _register_claude


def desktop_candidates(*, home: Path | None = None, environ: dict | None = None) -> list[Path]:
    root = Path.home() if home is None else home
    environment = os.environ if environ is None else environ
    roaming = Path(environment.get("APPDATA", root / "AppData/Roaming"))
    local = Path(environment.get("LOCALAPPDATA", root / "AppData/Local"))
    normal = roaming / "Claude/claude_desktop_config.json"
    paths = [normal] if normal.is_file() else []
    packages = local / "Packages"
    paths.extend(sorted(path for path in packages.glob("Claude_*/LocalCache/Roaming/Claude/claude_desktop_config.json") if path.is_file()))
    return list(dict.fromkeys(paths))


def select_desktop_config() -> Path | None:
    paths = desktop_candidates()
    if len(paths) == 1:
        return paths[0]
    print("Select the configuration opened by Claude Desktop Settings > Developer > Edit Config.")
    for number, path in enumerate(paths, 1):
        print(f"{number}: {path}")
    answer = input("Candidate number or full config file path (Enter cancels): ").strip()
    if not answer:
        return None
    if answer.isdecimal() and 1 <= int(answer) <= len(paths):
        return paths[int(answer) - 1]
    path = Path(answer).expanduser()
    if not path.is_absolute() or path.name != "claude_desktop_config.json" or not path.parent.is_dir():
        raise ValueError("Select an existing Desktop config directory and its config filename")
    return path


def _read(path: Path, *, toml: bool = False) -> tuple[dict, bytes | None]:
    if not path.exists():
        return {}, None
    original = path.read_bytes()
    document = (tomllib.loads(original.decode("utf-8-sig")) if toml
                else json.loads(original.decode("utf-8-sig")))
    key = "mcp_servers" if toml else "mcpServers"
    if not isinstance(document, dict) or not isinstance(document.get(key, {}), dict):
        raise ValueError("Invalid native MCP configuration")
    return document, original


def _snapshot(path: Path, original: bytes | None) -> Path | None:
    if (path.read_bytes() if path.exists() else None) != original:
        raise RuntimeError("Client configuration changed during setup; retry after closing the client")
    if original is None:
        return None
    directory = ngr_home() / "backups/clients"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    backup = directory / f"{path.name}.{uuid.uuid4().hex}.bak"
    with backup.open("xb") as stream:
        stream.write(original)
    return backup


def _choice(client: str, path: Path, exists: bool) -> bool:
    print(f"Proposed entry: client={client}, name={SERVER_NAME}, shared local NGR executable.")
    print(f"Native configuration: {path}; NGR details are read from ~/.ngr/config.json.")
    if exists:
        print("A same-name entry exists. Other servers and private values are not shown.")
        return input("Back up and replace only this entry? [y/N] ").lower() == "y"
    return input("Back up and add this entry? [y/N] ").lower() == "y"


def register_json(client: str, path: Path) -> None:
    document, original = _read(path)
    servers = document.setdefault("mcpServers", {})
    if not _choice(client, path, SERVER_NAME in servers):
        return
    _snapshot(path, original)
    servers[SERVER_NAME] = {"command": sys.executable, "args": ["--shared"]}
    atomic_json(path, document)
    print(f"{client}: registered {SERVER_NAME}. Restart the client to load the entry and user token.")


def _codex_command() -> list[str]:
    executable = shutil.which("codex")
    if executable is None:
        raise OSError("Codex CLI was not found")
    path = Path(executable)
    if path.suffix.lower() in (".cmd", ".bat", ".ps1"):
        # Run the documented npm entry with Node rather than a shell interpreter.
        # Shell shims could expand path characters or inherited secret variables.
        node = shutil.which("node")
        script = path.parent / "node_modules/@openai/codex/bin/codex.js"
        if node is None or not script.is_file():
            raise OSError("Codex npm entry could not be resolved; install a native CLI or npm Node entry")
        return [node, str(script)]
    return [executable]


def register_codex() -> None:
    command = _codex_command()
    path = _config_path("codex")
    document, original = _read(path, toml=True)
    existing = SERVER_NAME in document.get("mcp_servers", {})
    if not _choice("codex", path, existing):
        return
    backup = _snapshot(path, original)
    try:
        operations = ([command + ["mcp", "remove", SERVER_NAME]] if existing else [])
        operations.append(command + ["mcp", "add", SERVER_NAME, "--", sys.executable, "--shared"])
        for operation in operations:
            result = subprocess.run(operation, capture_output=True, text=True, check=False)
            if result.returncode:
                raise OSError("Codex MCP registration failed; CLI output is withheld")
        # Codex filters the stdio child's environment. Forward names only;
        # no token or other environment values are serialized into TOML.
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        headers = (f"[mcp_servers.{SERVER_NAME}]", f'[mcp_servers."{SERVER_NAME}"]')
        positions = [i for i, line in enumerate(lines) if line.strip() in headers]
        if len(positions) != 1:
            raise OSError("Codex launch table could not be resolved")
        names = [TOKEN_ENV, "USERPROFILE", "HOME", "APPDATA", "LOCALAPPDATA"]
        names.extend("NGR_" + field.upper() for field in FIELDS)
        lines.insert(positions[0] + 1, "env_vars = " + json.dumps(names) + "\n")
        fd, temporary = tempfile.mkstemp(prefix=".ngr-codex-", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
                stream.write("".join(lines))
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        updated, _ = _read(path, toml=True)
        entry = updated.get("mcp_servers", {}).get(SERVER_NAME, {})
        if entry.get("command") != sys.executable or entry.get("args") != ["--shared"]:
            raise OSError("Codex registration did not match the proposed launch entry")
        before = dict(document.get("mcp_servers", {}))
        after = dict(updated.get("mcp_servers", {}))
        before.pop(SERVER_NAME, None)
        after.pop(SERVER_NAME, None)
        if before != after:
            raise OSError("Codex registration changed another server")
        if ({key: value for key, value in document.items() if key != "mcp_servers"}
                != {key: value for key, value in updated.items() if key != "mcp_servers"}):
            raise OSError("Codex registration changed another client setting")
        if entry.get("env_vars") != names:
            raise OSError("Codex registration did not forward the required environment names")
    except Exception:
        if backup is not None:
            shutil.copyfile(backup, path)
        elif path.exists():
            path.unlink()
        raise
    print("Codex: shared host entry registered. Restart Codex and ChatGPT Desktop on this host.")


def main() -> None:
    if os.name != "nt" or not getattr(sys, "frozen", False):
        raise SystemExit("Client setup is available in the installed Windows package")
    try:
        read_config()
        print("NGR common client setup. Each write requires your choice; backups are stored in ~/.ngr.")
        selected = [client for client in ("codex", "claude-user", "claude-desktop", "claude-project")
                    if input(f"Configure {client}? [y/N] ").lower() == "y"]
        if not selected:
            return
        ensure_config()
        _ensure_token()
        _notify_environment()
        for client in selected:
            if client == "codex":
                register_codex()
            elif client == "claude-user":
                register_json(client, _config_path("claude"))
            elif client == "claude-desktop":
                path = select_desktop_config()
                if path is not None:
                    register_json(client, path)
            else:
                _register_claude(backup_root=ngr_home() / "backups/clients")
    except (OSError, ValueError, RuntimeError):
        raise SystemExit("Client setup stopped; inspect the configuration and ~/.ngr/backups. Private output is withheld.") from None


def _notify_environment() -> None:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    notify = user32.SendMessageTimeoutW
    notify.argtypes = (wintypes.HWND, wintypes.UINT, wintypes.WPARAM,
                       wintypes.LPCWSTR, wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_size_t))
    notify.restype = wintypes.LPARAM
    result = ctypes.c_size_t()
    if not notify(0xFFFF, 0x001A, 0, "Environment", 0x0002, 5000, ctypes.byref(result)):
        raise OSError("Environment notification failed; sign out and restart before using new user variables")
