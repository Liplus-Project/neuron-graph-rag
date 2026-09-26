"""Per-user Windows notification-area control for the shared MCP service.

The process is launched outside MCP clients' Job Objects by shared_proxy.
Win32 APIs are loaded only in this process; importing the module is portable.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
from typing import Any

from .shared_proxy import (
    TRAY_CONFIG_ENV, _ensure_service, _paused_path, _probe, _startup_lock,
    _state_dir, _stop_service,
)
from .http_server import TOKEN_ENV, _validate_bearer_token


def _pause(args: argparse.Namespace, token: str, database: Path) -> None:
    with _startup_lock(_state_dir() / f"shared-local-mcp-{args.port}.lock"):
        _paused_path(args.port).touch()
        if _probe(args.port, token, database, None) is not None:
            _stop_service(args.port, token, database)


def _resume(args: argparse.Namespace, token: str, database: Path,
            config: dict[str, Any]) -> None:
    with _startup_lock(_state_dir() / f"shared-local-mcp-{args.port}.lock"):
        _paused_path(args.port).unlink(missing_ok=True)
    try:
        _ensure_service(args, token, database, config, start_tray=False)
    except (RuntimeError, OSError):
        with _startup_lock(_state_dir() / f"shared-local-mcp-{args.port}.lock"):
            _paused_path(args.port).touch()
        raise


def _exit(args: argparse.Namespace, token: str, database: Path) -> None:
    # Exiting closes both the controller and its service. The next MCP
    # connection is allowed to start a fresh pair.
    with _startup_lock(_state_dir() / f"shared-local-mcp-{args.port}.lock"):
        _paused_path(args.port).touch()
        if _probe(args.port, token, database, None) is not None:
            _stop_service(args.port, token, database)
        _paused_path(args.port).unlink(missing_ok=True)


def _run_tray(args: argparse.Namespace, token: str, database: Path,
              config: dict[str, Any], marker: Path, fingerprint: str) -> None:
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    shell32 = ctypes.windll.shell32
    kernel32 = ctypes.windll.kernel32
    LRESULT = ctypes.c_ssize_t
    WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT,
                                 wintypes.WPARAM, wintypes.LPARAM)

    class WNDCLASSW(ctypes.Structure):
        _fields_ = [("style", wintypes.UINT), ("lpfnWndProc", WNDPROC),
                    ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
                    ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON),
                    ("hCursor", wintypes.HCURSOR), ("hbrBackground", wintypes.HBRUSH),
                    ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR)]

    class NOTIFYICONDATAW(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("hWnd", wintypes.HWND),
                    ("uID", wintypes.UINT), ("uFlags", wintypes.UINT),
                    ("uCallbackMessage", wintypes.UINT), ("hIcon", wintypes.HICON),
                    ("szTip", wintypes.WCHAR * 128), ("dwState", wintypes.DWORD),
                    ("dwStateMask", wintypes.DWORD), ("szInfo", wintypes.WCHAR * 256),
                    ("uTimeoutOrVersion", wintypes.UINT),
                    ("szInfoTitle", wintypes.WCHAR * 64), ("dwInfoFlags", wintypes.DWORD)]

    class POINT(ctypes.Structure):
        _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    user32.LoadIconW.restype = wintypes.HICON
    user32.LoadIconW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR]
    user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR,
                                      wintypes.DWORD, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_int, wintypes.HWND,
                                      wintypes.HMENU, wintypes.HINSTANCE, ctypes.c_void_p]
    user32.CreatePopupMenu.restype = wintypes.HMENU
    user32.AppendMenuW.argtypes = [wintypes.HMENU, wintypes.UINT,
                                   ctypes.c_size_t, wintypes.LPCWSTR]
    user32.TrackPopupMenu.argtypes = [wintypes.HMENU, wintypes.UINT, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_int, wintypes.HWND,
                                      ctypes.c_void_p]
    user32.GetCursorPos.argtypes = [ctypes.POINTER(POINT)]
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.DestroyMenu.argtypes = [wintypes.HMENU]
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    user32.SetTimer.argtypes = [wintypes.HWND, ctypes.c_size_t, wintypes.UINT, ctypes.c_void_p]
    user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                                   wintypes.UINT, wintypes.UINT]
    user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.MessageBoxW.argtypes = [wintypes.HWND, wintypes.LPCWSTR,
                                   wintypes.LPCWSTR, wintypes.UINT]
    user32.DefWindowProcW.restype = LRESULT
    user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT,
                                      wintypes.WPARAM, wintypes.LPARAM]
    shell32.Shell_NotifyIconW.argtypes = [wintypes.DWORD, ctypes.POINTER(NOTIFYICONDATAW)]

    WM_TRAY = 0x8001
    WM_TIMER = 0x0113
    WM_DESTROY = 0x0002
    WM_RBUTTONUP = 0x0205
    WM_LBUTTONUP = 0x0202
    WM_CONTEXTMENU = 0x007B
    NIM_ADD, NIM_MODIFY, NIM_DELETE = 0, 1, 2
    NIF_MESSAGE, NIF_ICON, NIF_TIP = 1, 2, 4
    MF_STRING, MF_GRAYED = 0, 1
    TPM_RETURNCMD, TPM_RIGHTBUTTON = 0x100, 2
    state = {"text": "Starting", "exit": False}
    icon = NOTIFYICONDATAW()
    icon.cbSize = ctypes.sizeof(icon)
    icon.uID = 1
    icon.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP
    icon.uCallbackMessage = WM_TRAY
    icon.hIcon = user32.LoadIconW(None, ctypes.cast(ctypes.c_void_p(32512), wintypes.LPCWSTR))

    def status() -> str:
        paused = _paused_path(args.port).exists()
        try:
            running = _probe(args.port, token, database, config) is not None
        except (RuntimeError, OSError):
            return "Unavailable"
        if paused:
            return "Stopping" if running else "Stopped"
        return "Running" if running else "Unavailable"

    def refresh() -> None:
        current = status()
        if current != state["text"]:
            state["text"] = current
            icon.szTip = f"NGR shared MCP - {current}"
            shell32.Shell_NotifyIconW(NIM_MODIFY, ctypes.byref(icon))

    def menu(hwnd: int) -> None:
        refresh()
        handle = user32.CreatePopupMenu()
        user32.AppendMenuW(handle, MF_STRING | MF_GRAYED, 0, f"NGR: {state['text']}")
        user32.AppendMenuW(handle, MF_STRING | (MF_GRAYED if state["text"] == "Stopped" else 0),
                           1, "Stop and release GPU")
        user32.AppendMenuW(handle, MF_STRING | (0 if state["text"] == "Stopped" else MF_GRAYED),
                           2, "Resume")
        user32.AppendMenuW(handle, MF_STRING, 3, "Exit")
        point = POINT()
        user32.GetCursorPos(ctypes.byref(point))
        user32.SetForegroundWindow(hwnd)
        choice = user32.TrackPopupMenu(handle, TPM_RETURNCMD | TPM_RIGHTBUTTON,
                                       point.x, point.y, 0, hwnd, None)
        user32.DestroyMenu(handle)
        try:
            if choice == 1:
                _pause(args, token, database)
            elif choice == 2:
                _resume(args, token, database, config)
            elif choice == 3:
                _exit(args, token, database)
                state["exit"] = True
                user32.DestroyWindow(hwnd)
        except (RuntimeError, OSError):
            user32.MessageBoxW(hwnd, "The operation failed. Check the NGR diagnostic log.",
                               "NGR shared MCP", 0x10)
        refresh()

    @WNDPROC
    def window_proc(hwnd: int, message: int, wparam: int, lparam: int) -> int:
        if message == WM_TRAY and lparam in (WM_RBUTTONUP, WM_LBUTTONUP, WM_CONTEXTMENU):
            menu(hwnd)
            return 0
        if message == WM_TIMER:
            refresh()
            return 0
        if message == WM_DESTROY:
            shell32.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(icon))
            user32.PostQuitMessage(0)
            return 0
        return user32.DefWindowProcW(hwnd, message, wparam, lparam)

    class_name = f"NGRSharedMCPTray{args.port}"
    instance = kernel32.GetModuleHandleW(None)
    window_class = WNDCLASSW()
    window_class.lpfnWndProc = window_proc
    window_class.hInstance = instance
    window_class.lpszClassName = class_name
    if not user32.RegisterClassW(ctypes.byref(window_class)):
        raise OSError("could not register NGR tray window")
    hwnd = user32.CreateWindowExW(0, class_name, "NGR shared MCP", 0,
                                  0, 0, 0, 0, None, None, instance, None)
    if not hwnd:
        raise OSError("could not create NGR tray window")
    icon.hWnd = hwnd
    icon.szTip = "NGR shared MCP - Starting"
    if not shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(icon)):
        raise OSError("could not add NGR tray icon")
    marker.write_text(json.dumps({"pid": os.getpid(),
                                  "fingerprint": fingerprint}), encoding="utf-8")
    refresh()
    user32.SetTimer(hwnd, 1, 3000, None)
    message = wintypes.MSG()
    while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
        user32.TranslateMessage(ctypes.byref(message))
        user32.DispatchMessageW(ctypes.byref(message))


def main() -> None:
    if os.name != "nt":
        raise SystemExit("NGR tray controller requires Windows")
    raw = os.environ.pop(TRAY_CONFIG_ENV, None)
    if raw is None:
        raise SystemExit("NGR tray controller must be started by shared MCP")
    payload = json.loads(raw)
    port = int(payload["port"])
    database = Path(payload["database"])
    config = payload["config"]
    token = os.environ.get(TOKEN_ENV, "")
    _validate_bearer_token(token)
    args = argparse.Namespace(port=port, cuda_device=config.get("cuda_device", 0),
                              cuda_cache=config.get("cuda_cache"),
                              cuda_e5_snapshot=config.get("cuda_e5_snapshot"),
                              cuda_v2_m3_snapshot=config.get("cuda_v2_m3_snapshot"))
    marker = _state_dir() / f"shared-local-mcp-{port}.tray.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        _run_tray(args, token, database, config, marker, payload["fingerprint"])
    finally:
        try:
            if json.loads(marker.read_text(encoding="utf-8")).get("pid") == os.getpid():
                marker.unlink(missing_ok=True)
        except (FileNotFoundError, ValueError, OSError):
            pass


if __name__ == "__main__":
    main()
