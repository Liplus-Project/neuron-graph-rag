"""Resolve the installed NGR version for MCP initialization metadata."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def application_version() -> str:
    try:
        return version("neuron-graph-rag")
    except PackageNotFoundError:
        return "0.0.0+source"
