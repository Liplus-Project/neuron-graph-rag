from __future__ import annotations

import argparse
import asyncio
import json
import re
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from mcp import types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError

from neuron_graph_rag import FeedbackContractError, FeedbackLedger, SourceUseEvent
from neuron_graph_rag.evidence_feedback import EngineConfig, NeuronGraphRAG

CONTRACT_VERSION = "ngr.mcp.feedback/v1"

SEARCH_DESCRIPTION = (
    "Search Neuron Graph RAG and return ranked source candidates with a trace_id. "
    "Returned hits are only retrieved candidates: retrieval does not mean a source was "
    "selected, validated, or used, and search alone never reinforces graph weights. "
    "Retain trace_id until feedback is complete. After a returned source is actually used "
    "in a downstream answer, implementation decision, or review, call record_source_use "
    "with that trace_id and node_id using stage used; record selected and validated when "
    "those earlier transitions occur. In the persistent NGR core, trace handles do not "
    "expire automatically."
)
SOURCE_USE_DESCRIPTION = (
    "Record ordered source-use transitions for candidates from one Neuron Graph RAG "
    "search trace. Use selected only when a source is chosen for inspection, validated "
    "only after its exact source is checked and accepted as usable, and used only after it "
    "becomes an actual basis of a downstream answer, implementation decision, or review. "
    "Transitions must occur in order. A newly recorded used source can add one independent "
    "evidence item per credited edge; graph reinforcement occurs only when that edge's "
    "configured evidence quorum has been reached. The default quorum is one. Retrieved, "
    "selected, validated, retries, duplicate traces, and duplicate stages add no evidence "
    "and do not reinforce. If the trace handle has expired or does not exist, this tool "
    "returns unknown_trace."
)
OUTCOME_DESCRIPTION = (
    "Record a delayed outcome for sources that were already marked used, such as "
    "confirmed, corrected, rolled_back, or superseded. In v1, delayed outcomes are audit "
    "and evaluation records only: they do not add, subtract, undo, or otherwise change "
    "graph weights. Do not use this tool instead of record_source_use for immediate "
    "source-use feedback. If the trace handle has expired or does not exist, this tool "
    "returns unknown_trace."
)

_IDEMPOTENCY = re.compile(r"^[A-Za-z0-9._:-]+$")
_TRACE = re.compile(r"^[0-9a-f]{32}$")


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


SEARCH_INPUT = _object(
    {
        "contract_version": {"type": "string", "const": CONTRACT_VERSION},
        "query": {"type": "string", "minLength": 1, "maxLength": 8192},
        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 5},
    },
    ["contract_version", "query"],
)
SOURCE_USE_INPUT = _object(
    {
        "contract_version": {"type": "string", "const": CONTRACT_VERSION},
        "idempotency_key": {"type": "string", "minLength": 1, "maxLength": 128, "pattern": _IDEMPOTENCY.pattern},
        "trace_id": {"type": "string", "pattern": _TRACE.pattern},
        "events": {
            "type": "array", "minItems": 1, "maxItems": 100,
            "items": _object(
                {
                    "node_id": {"type": "string", "minLength": 1, "maxLength": 512},
                    "stage": {"type": "string", "enum": ["selected", "validated", "used"]},
                },
                ["node_id", "stage"],
            ),
        },
    },
    ["contract_version", "idempotency_key", "trace_id", "events"],
)
OUTCOME_INPUT = _object(
    {
        "contract_version": {"type": "string", "const": CONTRACT_VERSION},
        "idempotency_key": {"type": "string", "minLength": 1, "maxLength": 128, "pattern": _IDEMPOTENCY.pattern},
        "trace_id": {"type": "string", "pattern": _TRACE.pattern},
        "node_ids": {
            "type": "array", "minItems": 1, "maxItems": 100, "uniqueItems": True,
            "items": {"type": "string", "minLength": 1, "maxLength": 512},
        },
        "outcome": {"type": "string", "enum": ["confirmed", "corrected", "rolled_back", "superseded"]},
        "summary": {"type": "string", "minLength": 1, "maxLength": 2000},
        "external_ref": {"type": "string", "maxLength": 2048, "format": "uri"},
    },
    ["contract_version", "idempotency_key", "trace_id", "node_ids", "outcome", "summary"],
)

_STEP_OUTPUT = _object(
    {
        "source_id": {"type": "string"},
        "target_id": {"type": "string"},
        "edge_type": {"type": "string"},
        "edge_weight": {"type": "number"},
        "factuality": {"type": "number"},
    },
    ["source_id", "target_id", "edge_type", "edge_weight", "factuality"],
)
_EDGE_OUTPUT = _object(
    {
        "source_id": {"type": "string"},
        "target_id": {"type": "string"},
        "edge_type": {"type": "string"},
        "old_weight": {"type": "number"},
        "new_weight": {"type": "number"},
    },
    ["source_id", "target_id", "edge_type", "old_weight", "new_weight"],
)
_EVIDENCE_OUTPUT = _object(
    {
        "source_id": {"type": "string"},
        "target_id": {"type": "string"},
        "edge_type": {"type": "string"},
        "count": {"type": "integer", "minimum": 1},
        "quorum": {"type": "integer", "minimum": 1},
        "activated": {"type": "boolean"},
    },
    ["source_id", "target_id", "edge_type", "count", "quorum", "activated"],
)
SEARCH_OUTPUT = _object(
    {
        "contract_version": {"type": "string", "const": CONTRACT_VERSION},
        "trace_id": {"type": "string", "pattern": _TRACE.pattern},
        "query": {"type": "string"},
        "created_at": {"type": "number"},
        "trace_expires_at": {"type": ["number", "null"]},
        "hits": {
            "type": "array",
            "items": _object(
                {
                    "node_id": {"type": "string"},
                    "rank": {"type": "integer", "minimum": 1},
                    "text": {"type": "string"},
                    "metadata": {"type": "object"},
                    "confidence": {"type": "number"},
                    "source_use_stage": {"type": "string", "const": "retrieved"},
                    "scores": _object(
                        {
                            "sparse": {"type": "number"},
                            "dense": {"type": "number"},
                            "entry": {"type": "number"},
                            "graph_activation": {"type": "number"},
                            "final": {"type": "number"},
                        },
                        ["sparse", "dense", "entry", "graph_activation", "final"],
                    ),
                    "paths": {
                        "type": "array",
                        "items": _object(
                            {
                                "seed_id": {"type": "string"},
                                "contribution": {"type": "number"},
                                "steps": {"type": "array", "items": _STEP_OUTPUT},
                            },
                            ["seed_id", "contribution", "steps"],
                        ),
                    },
                },
                [
                    "node_id", "rank", "text", "metadata", "confidence",
                    "source_use_stage", "scores", "paths",
                ],
            ),
        },
    },
    ["contract_version", "trace_id", "query", "created_at", "trace_expires_at", "hits"],
)
SOURCE_USE_OUTPUT = _object(
    {
        "contract_version": {"type": "string", "const": CONTRACT_VERSION},
        "receipt_id": {"type": "string", "pattern": _TRACE.pattern},
        "trace_id": {"type": "string", "pattern": _TRACE.pattern},
        "events": {
            "type": "array",
            "items": _object(
                {
                    "node_id": {"type": "string"},
                    "stage": {"type": "string", "enum": ["selected", "validated", "used"]},
                    "changed": {"type": "boolean"},
                },
                ["node_id", "stage", "changed"],
            ),
        },
        "newly_used_node_ids": {"type": "array", "items": {"type": "string"}},
        "feedback": {
            "anyOf": [
                {"type": "null"},
                _object(
                    {
                        "feedback_id": {"type": "string", "pattern": _TRACE.pattern},
                        "used_node_ids": {"type": "array", "items": {"type": "string"}},
                        "reinforced_edges": {"type": "array", "items": _EDGE_OUTPUT},
                        "evidence": {"type": "array", "items": _EVIDENCE_OUTPUT},
                    },
                    ["feedback_id", "used_node_ids", "reinforced_edges", "evidence"],
                ),
            ]
        },
    },
    ["contract_version", "receipt_id", "trace_id", "events", "newly_used_node_ids", "feedback"],
)
OUTCOME_OUTPUT = _object(
    {
        "contract_version": {"type": "string", "const": CONTRACT_VERSION},
        "outcome_id": {"type": "string", "pattern": _TRACE.pattern},
        "trace_id": {"type": "string", "pattern": _TRACE.pattern},
        "node_ids": {"type": "array", "items": {"type": "string"}},
        "outcome": {"type": "string", "enum": ["confirmed", "corrected", "rolled_back", "superseded"]},
        "recorded_at": {"type": "number"},
        "reinforcement_applied": {"type": "boolean", "const": False},
    },
    ["contract_version", "outcome_id", "trace_id", "node_ids", "outcome", "recorded_at", "reinforcement_applied"],
)


def _annotations(*, idempotent: bool) -> types.ToolAnnotations:
    return types.ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=idempotent,
        open_world_hint=False,
    )


TOOLS = (
    types.Tool(
        name="search",
        description=SEARCH_DESCRIPTION,
        input_schema=SEARCH_INPUT,
        output_schema=SEARCH_OUTPUT,
        annotations=_annotations(idempotent=False),
    ),
    types.Tool(
        name="record_source_use",
        description=SOURCE_USE_DESCRIPTION,
        input_schema=SOURCE_USE_INPUT,
        output_schema=SOURCE_USE_OUTPUT,
        annotations=_annotations(idempotent=True),
    ),
    types.Tool(
        name="record_outcome",
        description=OUTCOME_DESCRIPTION,
        input_schema=OUTCOME_INPUT,
        output_schema=OUTCOME_OUTPUT,
        annotations=_annotations(idempotent=True),
    ),
)


class FeedbackMCPAdapter:
    def __init__(
        self, database: str | Path, *, config: EngineConfig | None = None
    ) -> None:
        self.engine = NeuronGraphRAG(database, config=config)
        self.feedback = FeedbackLedger(self.engine)

    def close(self) -> None:
        self.engine.close()

    async def list_tools(self, *_: object) -> types.ListToolsResult:
        return types.ListToolsResult(tools=list(TOOLS), result_type="complete")

    async def call_tool(self, _context: object, params: types.CallToolRequestParams) -> types.CallToolResult:
        if params.name not in {tool.name for tool in TOOLS}:
            raise MCPError(-32601, "Unknown tool")
        arguments = params.arguments or {}
        try:
            self._require_object(arguments)
            if params.name == "search":
                output = self._search(arguments)
            elif params.name == "record_source_use":
                output = self._record_source_use(arguments)
            else:
                output = self._record_outcome(arguments)
            return self._success(output)
        except FeedbackContractError as error:
            return self._error(error.code, str(error), error.retryable)
        except (TypeError, ValueError, KeyError) as error:
            return self._error("invalid_argument", self._safe_validation_message(error), False)
        except sqlite3.Error:
            return self._error("core_unavailable", "local NGR database is unavailable", True)
        except Exception:  # noqa: BLE001 - the MCP boundary must not leak internals
            return self._error("internal_error", "tool execution failed", False)

    def _search(self, data: dict[str, Any]) -> dict[str, Any]:
        self._keys(data, {"contract_version", "query", "limit"}, {"contract_version", "query"})
        self._version(data)
        query = self._trimmed_string(data["query"], "query", 8192)
        limit = data.get("limit", 5)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 through 100")
        try:
            trace = self.engine.search(query, limit=limit)
        except ValueError as error:
            if str(error) == "Cannot search an empty corpus":
                raise FeedbackContractError("empty_corpus", "local NGR corpus is empty") from error
            raise
        hits = []
        for rank, hit in enumerate(trace.hits, start=1):
            hits.append(
                {
                    "node_id": hit.node.node_id,
                    "rank": rank,
                    "text": hit.node.text,
                    "metadata": hit.node.metadata,
                    "confidence": hit.node.confidence,
                    "source_use_stage": "retrieved",
                    "scores": {
                        "sparse": hit.sparse_score,
                        "dense": hit.dense_score,
                        "entry": hit.entry_score,
                        "graph_activation": hit.graph_activation,
                        "final": hit.final_score,
                    },
                    "paths": [
                        {
                            "seed_id": path.seed_id,
                            "contribution": path.contribution,
                            "steps": [
                                {
                                    "source_id": step.source_id,
                                    "target_id": step.target_id,
                                    "edge_type": step.edge_type,
                                    "edge_weight": step.edge_weight,
                                    "factuality": step.factuality,
                                }
                                for step in path.steps
                            ],
                        }
                        for path in hit.paths
                    ],
                }
            )
        return {
            "contract_version": CONTRACT_VERSION,
            "trace_id": trace.trace_id,
            "query": trace.query,
            "created_at": trace.created_at,
            "trace_expires_at": None,
            "hits": hits,
        }

    def _record_source_use(self, data: dict[str, Any]) -> dict[str, Any]:
        self._keys(
            data,
            {"contract_version", "idempotency_key", "trace_id", "events"},
            {"contract_version", "idempotency_key", "trace_id", "events"},
        )
        self._version(data)
        key = self._idempotency_key(data["idempotency_key"])
        trace_id = self._trace_id(data["trace_id"])
        raw_events = data["events"]
        if not isinstance(raw_events, list) or not 1 <= len(raw_events) <= 100:
            raise ValueError("events must contain 1 through 100 items")
        events = []
        for raw in raw_events:
            self._require_object(raw)
            self._keys(raw, {"node_id", "stage"}, {"node_id", "stage"})
            node_id = self._node_id(raw["node_id"])
            stage = raw["stage"]
            if stage not in {"selected", "validated", "used"}:
                raise ValueError("stage must be selected, validated, or used")
            events.append(SourceUseEvent(node_id, stage))
        receipt = self.feedback.record_source_use(
            trace_id, events, idempotency_key=key
        )
        feedback = None
        if receipt.feedback is not None:
            feedback = {
                "feedback_id": receipt.feedback.feedback_id,
                "used_node_ids": list(receipt.feedback.used_node_ids),
                "reinforced_edges": [
                    {
                        "source_id": edge.source_id,
                        "target_id": edge.target_id,
                        "edge_type": edge.edge_type,
                        "old_weight": edge.old_weight,
                        "new_weight": edge.new_weight,
                    }
                    for edge in receipt.feedback.reinforced_edges
                ],
                "evidence": [
                    {
                        "source_id": item.source_id,
                        "target_id": item.target_id,
                        "edge_type": item.edge_type,
                        "count": item.count,
                        "quorum": item.quorum,
                        "activated": item.activated,
                    }
                    for item in receipt.feedback.evidence
                ],
            }
        return {
            "contract_version": CONTRACT_VERSION,
            "receipt_id": receipt.receipt_id,
            "trace_id": receipt.trace_id,
            "events": [
                {"node_id": event.node_id, "stage": event.stage, "changed": event.changed}
                for event in receipt.events
            ],
            "newly_used_node_ids": list(receipt.newly_used_node_ids),
            "feedback": feedback,
        }

    def _record_outcome(self, data: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "contract_version", "idempotency_key", "trace_id", "node_ids",
            "outcome", "summary", "external_ref",
        }
        self._keys(data, allowed, allowed - {"external_ref"})
        self._version(data)
        key = self._idempotency_key(data["idempotency_key"])
        trace_id = self._trace_id(data["trace_id"])
        raw_node_ids = data["node_ids"]
        if not isinstance(raw_node_ids, list) or not 1 <= len(raw_node_ids) <= 100:
            raise ValueError("node_ids must contain 1 through 100 items")
        node_ids = tuple(self._node_id(value) for value in raw_node_ids)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("node_ids must be unique")
        outcome = data["outcome"]
        if outcome not in {"confirmed", "corrected", "rolled_back", "superseded"}:
            raise ValueError("outcome is not supported")
        summary = self._trimmed_string(data["summary"], "summary", 2000)
        external_ref = data.get("external_ref")
        if external_ref is not None:
            external_ref = self._https_url(external_ref)
        receipt = self.feedback.record_outcome(
            trace_id,
            node_ids,
            outcome,
            summary,
            idempotency_key=key,
            external_ref=external_ref,
        )
        return {
            "contract_version": CONTRACT_VERSION,
            "outcome_id": receipt.outcome_id,
            "trace_id": receipt.trace_id,
            "node_ids": list(receipt.node_ids),
            "outcome": receipt.outcome,
            "recorded_at": receipt.recorded_at,
            "reinforcement_applied": receipt.reinforcement_applied,
        }

    @staticmethod
    def _success(output: dict[str, Any]) -> types.CallToolResult:
        text = json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return types.CallToolResult(
            content=[types.TextContent(text=text)],
            structured_content=output,
            is_error=False,
            result_type="complete",
        )

    @staticmethod
    def _error(code: str, message: str, retryable: bool) -> types.CallToolResult:
        text = json.dumps(
            {"code": code, "message": message, "retryable": retryable},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return types.CallToolResult(
            content=[types.TextContent(text=text)], is_error=True, result_type="complete"
        )

    @staticmethod
    def _safe_validation_message(error: Exception) -> str:
        message = str(error).strip()
        return message if message and len(message) <= 240 else "invalid tool argument"

    @staticmethod
    def _require_object(value: Any) -> None:
        if not isinstance(value, dict):
            raise TypeError("arguments must be an object")

    @staticmethod
    def _keys(value: dict[str, Any], allowed: set[str], required: set[str]) -> None:
        missing = required - value.keys()
        extra = value.keys() - allowed
        if missing:
            raise ValueError("missing required field")
        if extra:
            raise ValueError("unknown field")

    @staticmethod
    def _version(data: dict[str, Any]) -> None:
        if data["contract_version"] != CONTRACT_VERSION:
            raise FeedbackContractError(
                "unsupported_contract_version", "contract_version is not supported"
            )

    @staticmethod
    def _trimmed_string(value: Any, name: str, maximum: int) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        trimmed = value.strip()
        if not trimmed or len(trimmed) > maximum:
            raise ValueError(f"{name} length is invalid")
        return trimmed

    @staticmethod
    def _trace_id(value: Any) -> str:
        if not isinstance(value, str) or _TRACE.fullmatch(value) is None:
            raise ValueError("trace_id must be 32 lowercase hexadecimal characters")
        return value

    @staticmethod
    def _node_id(value: Any) -> str:
        if not isinstance(value, str) or not 1 <= len(value) <= 512:
            raise ValueError("node_id length is invalid")
        if any(ord(character) < 32 for character in value):
            raise ValueError("node_id contains a control character")
        return value

    @staticmethod
    def _idempotency_key(value: Any) -> str:
        if (
            not isinstance(value, str)
            or not 1 <= len(value) <= 128
            or _IDEMPOTENCY.fullmatch(value) is None
        ):
            raise ValueError("idempotency_key format is invalid")
        return value

    @staticmethod
    def _https_url(value: Any) -> str:
        if not isinstance(value, str) or len(value) > 2048:
            raise ValueError("external_ref must be an absolute HTTPS URL")
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("external_ref must be an absolute HTTPS URL")
        return value


def create_server(
    database: str | Path, *, config: EngineConfig | None = None
) -> tuple[Server[Any], FeedbackMCPAdapter]:
    adapter = FeedbackMCPAdapter(database, config=config)
    server: Server[Any] = Server(
        "neuron-graph-rag",
        version="0.1.0",
        instructions=(
            "Search local Neuron Graph RAG sources, then report ordered source-use "
            "transitions and delayed outcomes with the returned trace_id."
        ),
        on_list_tools=adapter.list_tools,
        on_call_tool=adapter.call_tool,
    )
    return server, adapter


async def _run(database: str | Path) -> None:
    server, adapter = create_server(database)
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="neuron-graph-rag",
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                    instructions=server.instructions,
                ),
            )
    finally:
        adapter.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Neuron Graph RAG MCP server")
    parser.add_argument("--database", required=True, help="Path to the local SQLite database")
    arguments = parser.parse_args()
    asyncio.run(_run(arguments.database))
