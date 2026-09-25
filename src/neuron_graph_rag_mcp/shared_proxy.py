"""Start one loopback NGR service on demand and bridge stdio MCP clients to it."""

from __future__ import annotations

import argparse
import asyncio
import errno
import http.client
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import httpx2 as httpx
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

from neuron_graph_rag.database_home import resolve_database

from .http_server import DEFAULT_PORT, TOKEN_ENV, _validate_bearer_token


SERVICE_MARKER = "neuron-graph-rag-shared-local-mcp/v1"
START_TIMEOUT = 20.0


@contextmanager
def _startup_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _probe(port: int, token: str, database: Path, config: dict[str, Any] | None) -> int | None:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request("GET", "/_ngr/identity", headers={"Authorization": f"Bearer {token}"})
        response = connection.getresponse()
        body = response.read(8192)
    except OSError as error:
        if getattr(error, "errno", None) in (errno.ECONNREFUSED, 10061):
            return None
        if isinstance(error, TimeoutError):
            with socket.socket() as candidate:
                try:
                    candidate.bind(("127.0.0.1", port))
                except OSError:
                    pass
                else:
                    return None
        raise RuntimeError(f"port {port} is occupied or unresponsive: {type(error).__name__}") from error
    finally:
        connection.close()
    if response.status == 401:
        raise RuntimeError(f"port {port} rejected the shared MCP token")
    if response.status != 200:
        raise RuntimeError(f"port {port} is occupied by an incompatible service (HTTP {response.status})")
    try:
        identity = json.loads(body)
    except (ValueError, UnicodeDecodeError) as error:
        raise RuntimeError(f"port {port} returned an invalid service identity") from error
    if not isinstance(identity, dict) or identity.get("service") != SERVICE_MARKER:
        raise RuntimeError(f"port {port} is occupied by a different service")
    if identity.get("database") != str(database) or (config is not None and identity.get("config") != config):
        raise RuntimeError(f"port {port} has a different NGR database or CUDA configuration")
    pid = identity.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise RuntimeError(f"port {port} returned an invalid service identity")
    return pid


def _wait_process_exit(pid: int, timeout: float) -> bool:
    if os.name == "nt":
        import ctypes

        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        kernel.WaitForSingleObject.restype = ctypes.c_uint32
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            if kernel.GetLastError() == 87:  # ERROR_INVALID_PARAMETER: process already exited
                return True
            raise OSError(f"cannot wait for shared MCP service process {pid}")
        try:
            return kernel.WaitForSingleObject(handle, int(timeout * 1000)) == 0
        finally:
            kernel.CloseHandle(handle)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists():
            try:
                if stat.read_text(encoding="ascii").split()[2] == "Z":
                    return True
            except (OSError, IndexError):
                pass
        time.sleep(0.1)
    return False


def _start_service(args: argparse.Namespace, database: Path, log_path: Path) -> subprocess.Popen[bytes]:
    command = [
        sys.executable, "-m", "neuron_graph_rag_mcp", "--http",
        "--database", str(database), "--port", str(args.port),
    ]
    for name in ("cuda_cache", "cuda_e5_snapshot", "cuda_v2_m3_snapshot"):
        value = getattr(args, name)
        if value:
            command.extend(("--" + name.replace("_", "-"), str(Path(value).expanduser().resolve())))
    if args.cuda_device != 0:
        command.extend(("--cuda-device", str(args.cuda_device)))
    kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
                              "close_fds": True}
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            | subprocess.CREATE_BREAKAWAY_FROM_JOB
        )
    else:
        kwargs["start_new_session"] = True
    with log_path.open("ab") as log:
        kwargs["stderr"] = log
        return subprocess.Popen(command, **kwargs)


def _ensure_service(args: argparse.Namespace, token: str, database: Path,
                    config: dict[str, Any]) -> None:
    state_dir = Path.home() / ".ngrdb"
    lock_path = state_dir / f"shared-local-mcp-{args.port}.lock"
    log_path = state_dir / f"shared-local-mcp-{args.port}.log"
    with _startup_lock(lock_path):
        if _probe(args.port, token, database, config):
            return
        process = _start_service(args, database, log_path)
        deadline = time.monotonic() + START_TIMEOUT
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"shared MCP service exited with code {process.returncode}; see {log_path}"
                )
            if _probe(args.port, token, database, config):
                return
            time.sleep(0.1)
        raise RuntimeError(f"shared MCP service did not become ready; see {log_path}")


async def _run_proxy(port: int, token: str) -> None:
    url = f"http://127.0.0.1:{port}/mcp/"
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {token}"}, timeout=None) as http_client:
        async with streamable_http_client(url, http_client=http_client,
                                          terminate_on_close=False) as (remote_read, remote_write):
            async with ClientSession(remote_read, remote_write) as remote:
                await remote.initialize()

                async def list_tools(_context: object, params: types.PaginatedRequestParams | None) -> types.ListToolsResult:
                    return await remote.list_tools(params=params)

                async def call_tool(_context: object, params: types.CallToolRequestParams) -> types.CallToolResult:
                    return await remote.call_tool(params.name, params.arguments)

                server: Server[Any] = Server(
                    "neuron-graph-rag", version="0.1.0",
                    on_list_tools=list_tools, on_call_tool=call_tool,
                )
                async with stdio_server() as (read_stream, write_stream):
                    await server.run(
                        read_stream, write_stream,
                        InitializationOptions(
                            server_name="neuron-graph-rag", server_version="0.1.0",
                            capabilities=server.get_capabilities(
                                notification_options=NotificationOptions(), experimental_capabilities={}
                            ),
                        ),
                    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Connect stdio MCP to an on-demand shared NGR service")
    parser.add_argument("--database", help="SQLite path (default: NGR_DATABASE or ~/.ngrdb/knowledge.db)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--cuda-cache")
    parser.add_argument("--cuda-e5-snapshot")
    parser.add_argument("--cuda-v2-m3-snapshot")
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args(argv)
    token = os.environ.get(TOKEN_ENV, "")
    try:
        _validate_bearer_token(token)
        if not 1 <= args.port <= 65535:
            raise ValueError("--port must be from 1 through 65535")
        if args.cuda_device < 0:
            raise ValueError("--cuda-device must be non-negative")
        cuda_paths = (args.cuda_cache, args.cuda_e5_snapshot, args.cuda_v2_m3_snapshot)
        if any(cuda_paths) and not all(cuda_paths):
            raise ValueError("CUDA requires all three --cuda path options")
        if not any(cuda_paths) and args.cuda_device != 0:
            raise ValueError("--cuda-device requires CUDA path options")
        database = resolve_database(args.database, environ=os.environ).path.expanduser().resolve()
        config: dict[str, Any] = {}
        if all(cuda_paths):
            config = {
                "cuda_cache": str(Path(args.cuda_cache).expanduser().resolve()),
                "cuda_e5_snapshot": str(Path(args.cuda_e5_snapshot).expanduser().resolve()),
                "cuda_v2_m3_snapshot": str(Path(args.cuda_v2_m3_snapshot).expanduser().resolve()),
                "cuda_device": args.cuda_device,
            }
        _ensure_service(args, token, database, config)
        asyncio.run(_run_proxy(args.port, token))
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f"NGR shared MCP: {error}\n")


def stop_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stop the shared local NGR MCP service")
    parser.add_argument("--database", help="SQLite path used by the service")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    token = os.environ.get(TOKEN_ENV, "")
    try:
        _validate_bearer_token(token)
        if not 1 <= args.port <= 65535:
            raise ValueError("--port must be from 1 through 65535")
        database = resolve_database(args.database, environ=os.environ).path.expanduser().resolve()
        with _startup_lock(Path.home() / ".ngrdb" / f"shared-local-mcp-{args.port}.lock"):
            pid = _probe(args.port, token, database, None)
            if pid is None:
                raise RuntimeError("shared MCP service is not running")
            connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=5)
            try:
                connection.request("POST", "/_ngr/stop", headers={"Authorization": f"Bearer {token}"})
                response = connection.getresponse()
                response.read()
                if response.status != 200:
                    raise RuntimeError(f"stop request failed (HTTP {response.status})")
            finally:
                connection.close()
            if not _wait_process_exit(pid, START_TIMEOUT):
                raise RuntimeError("shared MCP service did not stop within 20 seconds")
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f"NGR shared MCP: {error}\n")


if __name__ == "__main__":
    main()
