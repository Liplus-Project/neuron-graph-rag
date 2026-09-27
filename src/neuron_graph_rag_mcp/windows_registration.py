"""Opt-in, non-destructive Windows MCP client registration for an installed EXE."""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .http_server import TOKEN_ENV, _validate_bearer_token

SERVER_NAME = "ngr-shared"


def _config_path(client: str) -> Path:
    if client == "codex":
        return Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "config.toml"
    return Path.home() / ".claude.json"


def _show_existing(client: str, config: Path) -> None:
    # Client output is untrusted free text and may contain secrets in arbitrary
    # arguments. Never echo any of its bytes.
    print(f"Existing MCP entry: client={client}, name={SERVER_NAME}.")
    print(f"Configuration file: {config}")
    if config.is_file() and input("Open the existing configuration in your editor? [y/N] ").lower() == "y":
        os.startfile(config)
        input("Inspect it locally, then press Enter to continue.")


def _backup(path: Path) -> Path | None:
    if not path.is_file():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = path.with_name(f"{path.name}.ngr-backup-{stamp}")
    if target.exists():
        raise FileExistsError(target)
    shutil.copy2(path, target)
    return target


def _ensure_token() -> None:
    import winreg

    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, "Environment", 0,
                            winreg.KEY_READ | winreg.KEY_SET_VALUE) as key:
        try:
            current, _ = winreg.QueryValueEx(key, TOKEN_ENV)
        except FileNotFoundError:
            current = ""
        if current:
            _validate_bearer_token(current)
        else:
            current = secrets.token_urlsafe(32)
            winreg.SetValueEx(key, TOKEN_ENV, 0, winreg.REG_SZ, current)
    os.environ[TOKEN_ENV] = current
    print("Private user token is ready. Restart an already running MCP client to load it.")


def _register(client: str) -> None:
    executable = shutil.which(client)
    if not executable:
        print(f"{client}: CLI was not found; skipped.")
        return
    result = subprocess.run([executable, "mcp", "get", SERVER_NAME],
                            text=True, capture_output=True, check=False)
    config = _config_path(client)
    if result.returncode == 0:
        _show_existing(client, config)
        if input("Back up the existing configuration? [y/N] ").lower() == "y":
            backup = _backup(config)
            print(f"Backup: {backup}" if backup else "Configuration file was not found.")
        print("Existing entry was preserved. Rename or remove it yourself before registering.")
        return
    details = (result.stdout + result.stderr).lower()
    if not any(word in details for word in ("not found", "does not exist", "no mcp server")):
        print(f"{client}: could not verify whether {SERVER_NAME} exists; skipped.")
        return
    command = [executable, "mcp", "add"]
    if client == "claude":
        command += ["--scope", "user", "--transport", "stdio", SERVER_NAME, "--", sys.executable, "--shared"]
    else:
        command += [SERVER_NAME, "--", sys.executable, "--shared"]
    print(f"Proposed {client} entry: name={SERVER_NAME}, command={sys.executable} --shared")
    print(f"Configuration: {config}; token is read from {TOKEN_ENV}, not saved in the MCP entry.")
    if input(f"Register in {client}? [y/N] ").lower() != "y":
        return
    backup = _backup(config)
    if backup:
        print(f"Backup: {backup}")
    subprocess.run(command, check=True)
    print(f"{client}: registered {SERVER_NAME}.")


def main() -> None:
    if os.name != "nt" or not getattr(sys, "frozen", False):
        raise SystemExit("Client setup is available in the installed Windows package")
    print("NGR MCP client setup. No client settings change without an explicit choice.")
    print("Existing same-name entries can be inspected locally and are never replaced.")
    selected = [name for name in ("codex", "claude")
                if input(f"Configure {name}? [y/N] ").lower() == "y"]
    if not selected:
        return
    try:
        _ensure_token()
        for client in selected:
            _register(client)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Client setup stopped: {type(error).__name__}; existing settings remain available") from error


if __name__ == "__main__":
    main()
