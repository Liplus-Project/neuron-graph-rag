"""Opt-in, non-destructive Windows MCP client registration for an installed EXE."""

from __future__ import annotations

import os
import json
import hashlib
import secrets
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .http_server import TOKEN_ENV, _validate_bearer_token

SERVER_NAME = "ngr-shared"
EXE_ENV = "NGR_MCP_EXE"


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


def _backup(path: Path, backup_dir: Path | None = None) -> Path | None:
    if not path.is_file():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
    target = (backup_dir / f"{path.name}.ngr-backup-{stamp}" if backup_dir is not None
              else path.with_name(f"{path.name}.ngr-backup-{stamp}"))
    if target.exists():
        raise FileExistsError(target)
    shutil.copy2(path, target)
    return target


def _backup_root() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return base / "Neuron Graph RAG" / "mcp-backups"


def _project_backup_dir(project: Path, backup_root: Path | None = None) -> Path:
    project_id = hashlib.sha256(str(project).encode("utf-8")).hexdigest()[:16]
    return (backup_root if backup_root is not None else _backup_root()) / project_id


def _select_claude_project() -> Path | None:
    """Ask for a project root, without deriving it from the installation path."""
    # The Windows package already requires Windows PowerShell; use its native
    # folder dialog instead of adding Tcl/Tk to the frozen executable.
    script = ("[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false); "
              "Add-Type -AssemblyName System.Windows.Forms; "
              "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; "
              "$dialog.Description = 'Select Claude Code project root'; "
              "if ($dialog.ShowDialog() -eq 'OK') { [Console]::Out.Write($dialog.SelectedPath) }")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-STA", "-Command", script],
                            text=True, encoding="utf-8", capture_output=True, check=False)
    if result.returncode != 0:
        raise OSError("Claude Code project folder selection failed")
    selected = result.stdout.strip()
    return Path(selected).resolve() if selected else None


def _mcp_servers(path: Path) -> dict:
    if not path.exists():
        return {}
    if not path.is_file():
        raise ValueError("MCP configuration path is not a file")
    try:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("MCP configuration is not valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise ValueError("MCP configuration root must be an object")
    servers = document.get("mcpServers", {})
    if not isinstance(servers, dict):
        raise ValueError("MCP configuration mcpServers must be an object")
    return servers


def _ensure_executable_env() -> None:
    import winreg

    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, "Environment", 0,
                            winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, EXE_ENV, 0, winreg.REG_SZ, sys.executable)
    os.environ[EXE_ENV] = sys.executable


def _register_claude(*, backup_root: Path | None = None) -> None:
    executable = shutil.which("claude")
    if not executable:
        print("claude: CLI was not found; skipped.")
        return
    project = _select_claude_project()
    if project is None:
        print("Claude Code project selection cancelled.")
        return
    if not project.is_dir():
        raise ValueError("Selected Claude Code project directory does not exist")
    config = project / ".mcp.json"
    servers = _mcp_servers(config)
    if SERVER_NAME in servers:
        _show_existing("claude", config)
        if input("Back up the existing project configuration? [y/N] ").lower() == "y":
            print(f"Backup: {_backup(config, _project_backup_dir(project, backup_root))}")
        print("Existing project entry was preserved. Review its backup before changing it yourself.")
        return
    user_config = _config_path("claude")
    user_entry = SERVER_NAME in _mcp_servers(user_config)
    print(f"Proposed Claude Code project entry: name={SERVER_NAME}, command=${{{EXE_ENV}}} --shared")
    print(f"Project configuration: {config}; token is read from {TOKEN_ENV}.")
    if user_entry:
        print("An existing user-scope entry also applies to other projects. It will be preserved unless you choose removal after project registration.")
    if input("Register in this Claude Code project? [y/N] ").lower() != "y":
        return
    _ensure_executable_env()
    backup = _backup(config, _project_backup_dir(project, backup_root))
    if backup:
        print(f"Backup: {backup}")
    command = [executable, "mcp", "add", "--scope", "project", "--transport", "stdio",
               SERVER_NAME, "--", f"${{{EXE_ENV}}}", "--shared"]
    result = subprocess.run(command, cwd=project, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise OSError("Claude Code project registration failed")
    if SERVER_NAME not in _mcp_servers(config):
        raise OSError("Claude Code did not write the selected project configuration")
    print(f"Claude Code: registered {SERVER_NAME} in {config}.")
    if user_entry:
        print("Removing the user-scope entry disables it in other projects without their own registration.")
        if input("Back up and remove the old user-scope entry? [y/N] ").lower() == "y":
            user_backup = _backup(user_config, (backup_root if backup_root is not None else _backup_root()) / "user")
            if user_backup is None:
                raise OSError("Claude Code user configuration disappeared before backup")
            print(f"Backup: {user_backup}")
            removed = subprocess.run([executable, "mcp", "remove", SERVER_NAME, "--scope", "user"],
                                     cwd=project, text=True, capture_output=True, check=False)
            if removed.returncode != 0 or SERVER_NAME in _mcp_servers(user_config):
                raise OSError("Claude Code user-scope removal could not be verified")
            print("Claude Code: old user-scope entry removed.")


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
    if client == "claude":
        _register_claude()
        return
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
