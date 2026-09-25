"""Single-process loopback Streamable HTTP entry for the optional MCP adapter."""

from __future__ import annotations

import argparse
import os
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Mount
from starlette.types import ASGIApp, Receive, Scope, Send

from neuron_graph_rag.cpu_shortlist_retrieval import LocalPinnedE5
from neuron_graph_rag.cuda_shortlist_retrieval import CudaShortlistRetriever, LocalPinnedCudaV2M3
from neuron_graph_rag.database_home import prepare_database, resolve_database

from .server import create_server


DEFAULT_PORT = 8765
TOKEN_ENV = "NGR_MCP_HTTP_BEARER_TOKEN"


def _validate_bearer_token(token: str) -> None:
    if len(token) < 32 or not token.isascii() or not all(
        character.isalnum() or character in "_-" for character in token
    ):
        raise ValueError(f"{TOKEN_ENV} must be a private base64url token of at least 32 characters")


class BearerTokenGate:
    """Reject unauthenticated requests before the MCP session manager sees them."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        _validate_bearer_token(token)
        self.app = app
        self.expected = f"Bearer {token}".encode("ascii")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            credentials = [value for name, value in scope["headers"] if name.lower() == b"authorization"]
            if len(credentials) != 1 or not secrets.compare_digest(credentials[0], self.expected):
                await Response(status_code=401, headers={"WWW-Authenticate": "Bearer"})(scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_http_app(database: str | Path, *, port: int = DEFAULT_PORT,
                    cuda_retriever: Any = None, bearer_token: str) -> Starlette:
    """Create one process-owned adapter and one session manager for all clients."""
    if not 1 <= port <= 65535:
        raise ValueError("port must be from 1 through 65535")
    # Validate before opening the database or allocating the optional retriever.
    _validate_bearer_token(bearer_token)
    server, adapter = create_server(database, cuda_retriever=cuda_retriever, expose_cuda=True)
    manager = StreamableHTTPSessionManager(
        server, json_response=True,
        security_settings=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[f"127.0.0.1:{port}"],
            allowed_origins=[f"http://127.0.0.1:{port}"],
        ),
    )

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        try:
            async with manager.run():
                yield
        finally:
            adapter.close()

    app = Starlette(routes=[Mount("/mcp", app=BearerTokenGate(manager.handle_request, bearer_token))],
                    lifespan=lifespan)
    app.state.ngr_adapter = adapter
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the shared local NGR MCP service")
    parser.add_argument("--database", help="SQLite path (default: NGR_DATABASE or ~/.ngrdb/knowledge.db)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--cuda-cache", help="Existing or new local E5 shortlist cache path")
    parser.add_argument("--cuda-e5-snapshot", help="Pinned local E5 ONNX snapshot directory")
    parser.add_argument("--cuda-v2-m3-snapshot", help="Pinned local v2-m3 snapshot directory")
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args(argv)
    bearer_token = os.environ.get(TOKEN_ENV, "")
    try:
        _validate_bearer_token(bearer_token)
    except ValueError as error:
        parser.error(str(error))
    if not 1 <= args.port <= 65535:
        parser.error("--port must be from 1 through 65535")
    if args.cuda_device < 0:
        parser.error("--cuda-device must be non-negative")
    paths = (args.cuda_cache, args.cuda_e5_snapshot, args.cuda_v2_m3_snapshot)
    if any(paths) and not all(paths):
        parser.error("CUDA requires --cuda-cache, --cuda-e5-snapshot, and --cuda-v2-m3-snapshot together")
    retriever = None
    if all(paths):
        e5 = Path(args.cuda_e5_snapshot).expanduser()
        v2 = Path(args.cuda_v2_m3_snapshot).expanduser()
        if not (e5 / "tokenizer.json").is_file() or not (e5 / "onnx" / "model.onnx").is_file():
            parser.error("pinned E5 snapshot is incomplete")
        if not all((v2 / name).is_file() for name in ("config.json", "model.safetensors", "tokenizer.json")):
            parser.error("pinned v2-m3 snapshot is incomplete")
        retriever = CudaShortlistRetriever(
            Path(args.cuda_cache).expanduser(), LocalPinnedE5(e5),
            LocalPinnedCudaV2M3(v2, device=args.cuda_device),
        )
    try:
        database = prepare_database(resolve_database(args.database, environ=os.environ))
    except ValueError as error:
        parser.error(str(error))
    app = create_http_app(database, port=args.port, cuda_retriever=retriever,
                          bearer_token=bearer_token)
    print(f"NGR MCP: http://127.0.0.1:{args.port}/mcp/", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
