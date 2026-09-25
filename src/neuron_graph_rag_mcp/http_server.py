"""Single-process loopback Streamable HTTP entry for the optional MCP adapter."""

from __future__ import annotations

import argparse
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.routing import Mount

from neuron_graph_rag.cpu_shortlist_retrieval import LocalPinnedE5
from neuron_graph_rag.cuda_shortlist_retrieval import CudaShortlistRetriever, LocalPinnedCudaV2M3
from neuron_graph_rag.database_home import prepare_database, resolve_database

from .server import create_server


DEFAULT_PORT = 8765


def create_http_app(database: str | Path, *, port: int = DEFAULT_PORT,
                    cuda_retriever: Any = None) -> Starlette:
    """Create one process-owned adapter and one session manager for all clients."""
    if not 1 <= port <= 65535:
        raise ValueError("port must be from 1 through 65535")
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

    app = Starlette(routes=[Mount("/mcp", app=manager.handle_request)], lifespan=lifespan)
    app.state.ngr_adapter = adapter
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the shared local NGR MCP service")
    parser.add_argument("--database", help="SQLite path (default: NGR_DATABASE or ~/.ngrdb/knowledge.db)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--cuda-cache", help="Existing or new local E5 shortlist cache path")
    parser.add_argument("--cuda-e5-snapshot", help="Pinned local E5 ONNX snapshot directory")
    parser.add_argument("--cuda-v2-m3-snapshot", help="Pinned local v2-m3 snapshot directory")
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()
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
    app = create_http_app(database, port=args.port, cuda_retriever=retriever)
    print(f"NGR MCP: http://127.0.0.1:{args.port}/mcp/", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
