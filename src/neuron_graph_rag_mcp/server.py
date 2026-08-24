from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
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
from neuron_graph_rag.database_home import prepare_database, resolve_database
from neuron_graph_rag.config_provenance import (
    effective_config_provenance,
    effective_search_surface,
    search_with_surface,
)
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
CONFIRMED_SOURCE_USE_DESCRIPTION = (
    "Record ordered source-use transitions for candidates from one Neuron Graph RAG "
    "search trace. Use selected, validated, and used in order. This server uses the "
    "confirmed-outcome candidate: retrieved through used records provenance only and "
    "never changes graph weights. Retain the trace_id and call record_outcome with "
    "confirmed only when later evidence or an operational result supports the judgment. "
    "Retries and duplicate stages do not change state twice."
)
CONFIRMED_OUTCOME_DESCRIPTION = (
    "Record a delayed outcome for sources already marked used. This server uses the "
    "confirmed-outcome candidate: a new confirmed outcome can reinforce only the saved "
    "credited relation path. The first independent confirmation uses multiplier 1.0 and "
    "later independent traces use the configured geometric decay; duplicate traces and "
    "idempotency retries do not reinforce twice. Corrected, rolled_back, and superseded "
    "remain audit-only and never subtract or roll back weights."
)
SOFT_START_SOURCE_USE_DESCRIPTION = (
    "Record ordered source-use transitions for candidates from one Neuron Graph RAG "
    "search trace. Use selected, validated, and used in order. This server uses the "
    "soft-start candidate: the first newly used relation trace can apply the configured "
    "provisional fraction of one bounded update. Later used traces add provenance only; "
    "retrieved, selected, validated, retries, and duplicate stages never reinforce. "
    "Retain trace_id and record a confirmed outcome when later evidence supports it."
)
SOFT_START_OUTCOME_DESCRIPTION = (
    "Record a delayed outcome for sources already marked used. This server uses the "
    "soft-start candidate: the first independent confirmed outcome completes at most "
    "the remainder of one normal bounded update, and later independent confirmations "
    "use the configured geometric decay. Sibling normalization applies only to each "
    "confirmed outcome's actual delta. Duplicate traces and idempotency retries do not "
    "reinforce twice; negative outcomes remain audit-only."
)
DEACTIVATION_OUTCOME_DESCRIPTION = (
    "Record a delayed outcome for sources already marked used. This server uses the "
    "outcome-driven deactivation candidate: confirmed follows the soft-start schedule; "
    "causally attributed corrected and rolled_back outcomes exactly reverse each active "
    "credited contribution together with its same-source sibling normalization mutations. "
    "Superseded makes the saved relation path dormant without deleting evidence, and a "
    "later confirmed outcome on that saved path reactivates it. Unattributed, duplicate, "
    "lexical, and zero-hop outcomes remain non-mutating."
)
JUDGMENT_SEARCH_DESCRIPTION = (
    "Search only canonical SQLite judgments without creating a retrieval trace or changing "
    "activation, feedback, nodes, edges, revisions, lifecycle, or relations. Results include "
    "the current revision, lifecycle, statement, rationale, provenance, typed relations, "
    "and score explanation. Active judgments are the default; include archived judgments "
    "only when explicitly requested."
)
JUDGMENT_GET_DESCRIPTION = (
    "Get the current state of one canonical SQLite judgment by exact stable identity. This "
    "operation is read-only and returns archived judgments as well as active judgments."
)
JUDGMENT_TRAVERSE_DESCRIPTION = (
    "Traverse canonical SQLite judgment relations by type, direction, and finite hop count. "
    "Traversal is cycle-safe, deterministic, and read-only. Active judgments are the default; "
    "include archived judgments only when explicitly requested."
)
RELATION_TYPE_REGISTRY_READ_DESCRIPTION = (
    "List, get, or validate canonical judgment relation type definitions without changing "
    "any persistent table. Unknown and deprecated relation types return structured advisory "
    "warnings; validation never rejects a free-form relation assertion."
)
RELATION_TYPE_REGISTRY_WRITE_DESCRIPTION = (
    "Register or revise one canonical judgment relation type through the atomic domain API. "
    "Use expected_revision 0 to create a type and the exact current revision to update it. "
    "Stale writes fail closed; deprecation remains advisory and does not reject assertions."
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
JUDGMENT_WRITE_INPUT = {
    "type": "object",
    "properties": {
        "contract_version": {"type": "string", "const": CONTRACT_VERSION},
        "action": {"type": "string", "enum": ["add", "update", "supersede", "archive", "restore", "hard_delete"]},
        "judgment_id": {"type": "string", "minLength": 1, "maxLength": 128},
        "successor_id": {"type": "string", "minLength": 1, "maxLength": 128},
        "statement": {"type": "string", "minLength": 1},
        "rationale": {"type": "string", "minLength": 1},
        "provenance": {"type": "object"},
        "expected_revision": {"type": "integer", "minimum": 1},
        "relations": {
            "type": "array",
            "items": _object(
                {"target_id": {"type": "string"}, "relation_type": {"type": "string"}},
                ["target_id", "relation_type"],
            ),
        },
    },
    "required": ["contract_version", "action", "judgment_id"],
    "additionalProperties": False,
}
JUDGMENT_SEARCH_INPUT = _object(
    {
        "contract_version": {"type": "string", "const": CONTRACT_VERSION},
        "query": {"type": "string", "minLength": 1, "maxLength": 8192},
        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 5},
        "include_archived": {"type": "boolean", "default": False},
        "repository": {"type": "string", "minLength": 1, "maxLength": 256},
    },
    ["contract_version", "query"],
)
JUDGMENT_GET_INPUT = _object(
    {
        "contract_version": {"type": "string", "const": CONTRACT_VERSION},
        "judgment_id": {"type": "string", "minLength": 1, "maxLength": 128},
    },
    ["contract_version", "judgment_id"],
)
JUDGMENT_TRAVERSE_INPUT = _object(
    {
        "contract_version": {"type": "string", "const": CONTRACT_VERSION},
        "judgment_id": {"type": "string", "minLength": 1, "maxLength": 128},
        "direction": {
            "type": "string", "enum": ["incoming", "outgoing", "both"],
            "default": "outgoing",
        },
        "relation_type": {"type": "string", "minLength": 1, "maxLength": 64},
        "max_hops": {"type": "integer", "minimum": 1, "maximum": 32, "default": 1},
        "include_archived": {"type": "boolean", "default": False},
    },
    ["contract_version", "judgment_id"],
)
RELATION_TYPE_REGISTRY_READ_INPUT = _object(
    {
        "contract_version": {"type": "string", "const": CONTRACT_VERSION},
        "action": {"type": "string", "enum": ["list", "get", "validate"]},
        "relation_type": {"type": "string", "minLength": 1, "maxLength": 64},
        "revision": {"type": "integer", "minimum": 1},
        "include_deprecated": {"type": "boolean", "default": True},
    },
    ["contract_version", "action"],
)
RELATION_TYPE_REGISTRY_WRITE_INPUT = _object(
    {
        "contract_version": {"type": "string", "const": CONTRACT_VERSION},
        "relation_type": {"type": "string", "minLength": 1, "maxLength": 64},
        "definition": {"type": "string", "minLength": 1},
        "namespace": {"type": "string", "minLength": 1, "maxLength": 128},
        "provenance": {"type": "object"},
        "expected_revision": {"type": "integer", "minimum": 0},
        "lifecycle": {
            "type": "string", "enum": ["active", "deprecated"], "default": "active"
        },
    },
    [
        "contract_version", "relation_type", "definition", "namespace",
        "provenance", "expected_revision",
    ],
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
_CONFIRMATION_OUTPUT = _object(
    {
        "source_id": {"type": "string"},
        "target_id": {"type": "string"},
        "edge_type": {"type": "string"},
        "confirmation_count": {"type": "integer", "minimum": 1},
        "multiplier": {"type": "number", "exclusiveMinimum": 0.0},
        "actual_delta": {"type": "number", "minimum": 0.0},
        "old_weight": {"type": "number", "minimum": 0.0},
        "new_weight": {"type": "number", "minimum": 0.0},
    },
    [
        "source_id", "target_id", "edge_type", "confirmation_count",
        "multiplier", "actual_delta", "old_weight", "new_weight",
    ],
)
_CREDITED_PATH_OUTPUT = _object(
    {
        "node_id": {"type": "string"},
        "steps": {"type": "array", "items": _STEP_OUTPUT},
    },
    ["node_id", "steps"],
)
_DORMANCY_OUTPUT = _object(
    {
        "source_id": {"type": "string"},
        "target_id": {"type": "string"},
        "edge_type": {"type": "string"},
        "old_dormant": {"type": "boolean"},
        "new_dormant": {"type": "boolean"},
    },
    ["source_id", "target_id", "edge_type", "old_dormant", "new_dormant"],
)
_CONTRIBUTION_MUTATION_OUTPUT = _object(
    {
        "mutation_role": {"type": "string", "enum": ["credited", "sibling"]},
        "source_id": {"type": "string"},
        "target_id": {"type": "string"},
        "edge_type": {"type": "string"},
        "actual_delta": {"type": "number"},
        "old_weight": {"type": "number"},
        "new_weight": {"type": "number"},
    },
    [
        "mutation_role", "source_id", "target_id", "edge_type",
        "actual_delta", "old_weight", "new_weight",
    ],
)
_REVERSED_CONTRIBUTION_OUTPUT = _object(
    {
        "contribution_id": {"type": "string"},
        "contribution_kind": {
            "type": "string",
            "enum": ["soft_start_provisional", "soft_start_confirmation"],
        },
        "source_record_id": {"type": "string"},
        "source_id": {"type": "string"},
        "target_id": {"type": "string"},
        "edge_type": {"type": "string"},
        "credited_delta": {"type": "number"},
        "mutations": {"type": "array", "items": _CONTRIBUTION_MUTATION_OUTPUT},
    },
    [
        "contribution_id", "contribution_kind", "source_record_id", "source_id",
        "target_id", "edge_type", "credited_delta", "mutations",
    ],
)
SEARCH_OUTPUT = _object(
    {
        "contract_version": {"type": "string", "const": CONTRACT_VERSION},
        "trace_id": {"type": "string", "pattern": _TRACE.pattern},
        "query": {"type": "string"},
        "created_at": {"type": "number"},
        "trace_expires_at": {"type": ["number", "null"]},
        "effective_config_provenance": _object(
            {
                "effective_config": _object(
                    {
                        "retrieval": {"type": "object"},
                        "feedback": {"type": "object"},
                    },
                    ["retrieval", "feedback"],
                ),
                "search_surface": {
                    "type": "string",
                    "enum": ["combined", "relation"],
                },
                "retrieval_config_fingerprint": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
                "feedback_config_fingerprint": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
                "full_config_fingerprint": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
            },
            [
                "effective_config", "search_surface", "retrieval_config_fingerprint",
                "feedback_config_fingerprint", "full_config_fingerprint",
            ],
        ),
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
    [
        "contract_version", "trace_id", "query", "created_at",
        "trace_expires_at", "effective_config_provenance", "hits",
    ],
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
        "reinforcement_applied": {"type": "boolean"},
        "confirmations": {"type": "array", "items": _CONFIRMATION_OUTPUT},
        "credited_paths": {"type": "array", "items": _CREDITED_PATH_OUTPUT},
        "normalized_sibling_edges": {"type": "array", "items": _EDGE_OUTPUT},
        "deactivation_applied": {"type": "boolean"},
        "reversed_contributions": {
            "type": "array", "items": _REVERSED_CONTRIBUTION_OUTPUT
        },
        "dormancy_changes": {"type": "array", "items": _DORMANCY_OUTPUT},
        "reactivated_edges": {"type": "array", "items": _DORMANCY_OUTPUT},
    },
    ["contract_version", "outcome_id", "trace_id", "node_ids", "outcome", "recorded_at", "reinforcement_applied"],
)

_JUDGMENT_RELATION_OUTPUT = _object(
    {
        "target_id": {"type": "string"},
        "relation_type": {"type": "string"},
        "relation_type_revision": {"type": ["integer", "null"], "minimum": 1},
        "assertion_kind": {"type": "string", "const": "explicit"},
    },
    ["target_id", "relation_type", "relation_type_revision", "assertion_kind"],
)
_ADVISORY_WARNING_OUTPUT = _object(
    {
        "code": {
            "type": "string",
            "enum": ["unknown_relation_type", "deprecated_relation_type"],
        },
        "message": {"type": "string"},
        "relation_type": {"type": "string"},
        "relation_type_revision": {"type": ["integer", "null"], "minimum": 1},
        "lifecycle": {"type": "string", "enum": ["unknown", "deprecated"]},
    },
    [
        "code", "message", "relation_type", "relation_type_revision", "lifecycle"
    ],
)
_JUDGMENT_PROPERTIES = {
    "judgment_id": {"type": "string"},
    "revision": {"type": "integer", "minimum": 1},
    "statement": {"type": "string"},
    "rationale": {"type": "string"},
    "provenance": {"type": "object"},
    "lifecycle": {"type": "string", "enum": ["active", "archived"]},
    "superseded_by": {"type": ["string", "null"]},
    "relations": {"type": "array", "items": _JUDGMENT_RELATION_OUTPUT},
    "advisory_warnings": {"type": "array", "items": _ADVISORY_WARNING_OUTPUT},
}
_JUDGMENT_REQUIRED = list(_JUDGMENT_PROPERTIES)
JUDGMENT_OUTPUT = _object(dict(_JUDGMENT_PROPERTIES), _JUDGMENT_REQUIRED)
JUDGMENT_SEARCH_RESULT_OUTPUT = _object(
    {
        **_JUDGMENT_PROPERTIES,
        "score": {"type": "number"},
        "explanation": _object(
            {
                "sparse_score": {"type": "number"},
                "dense_score": {"type": "number"},
                "sparse_weight": {"type": "number"},
                "dense_weight": {"type": "number"},
            },
            ["sparse_score", "dense_score", "sparse_weight", "dense_weight"],
        ),
    },
    [*_JUDGMENT_REQUIRED, "score", "explanation"],
)
JUDGMENT_SEARCH_OUTPUT = _object(
    {
        "contract_version": {"type": "string", "const": CONTRACT_VERSION},
        "judgments": {"type": "array", "items": JUDGMENT_SEARCH_RESULT_OUTPUT},
    },
    ["contract_version", "judgments"],
)
JUDGMENT_GET_OUTPUT = _object(
    {
        "contract_version": {"type": "string", "const": CONTRACT_VERSION},
        "judgment": JUDGMENT_OUTPUT,
    },
    ["contract_version", "judgment"],
)
_TRAVERSED_RELATION_OUTPUT = _object(
    {
        "source_id": {"type": "string"},
        "target_id": {"type": "string"},
        "relation_type": {"type": "string"},
        "relation_type_revision": {"type": ["integer", "null"], "minimum": 1},
        "assertion_kind": {"type": "string", "const": "explicit"},
    },
    [
        "source_id", "target_id", "relation_type",
        "relation_type_revision", "assertion_kind",
    ],
)
JUDGMENT_TRAVERSE_OUTPUT = _object(
    {
        "contract_version": {"type": "string", "const": CONTRACT_VERSION},
        "results": {
            "type": "array",
            "items": _object(
                {
                    "hop": {"type": "integer", "minimum": 1},
                    "direction": {"type": "string", "enum": ["incoming", "outgoing"]},
                    "relation": _TRAVERSED_RELATION_OUTPUT,
                    "judgment": JUDGMENT_OUTPUT,
                },
                ["hop", "direction", "relation", "judgment"],
            ),
        },
    },
    ["contract_version", "results"],
)

_RELATION_TYPE_OUTPUT = _object(
    {
        "relation_type": {"type": "string"},
        "revision": {"type": "integer", "minimum": 1},
        "definition": {"type": "string"},
        "namespace": {"type": "string"},
        "provenance": {"type": "object"},
        "lifecycle": {"type": "string", "enum": ["active", "deprecated"]},
        "created_at": {"type": "string"},
        "is_current": {"type": "boolean"},
    },
    [
        "relation_type", "revision", "definition", "namespace", "provenance",
        "lifecycle", "created_at", "is_current",
    ],
)
RELATION_TYPE_REGISTRY_READ_OUTPUT = _object(
    {
        "contract_version": {"type": "string", "const": CONTRACT_VERSION},
        "action": {"type": "string", "enum": ["list", "get", "validate"]},
        "relation_types": {"type": "array", "items": _RELATION_TYPE_OUTPUT},
        "advisory_warnings": {
            "type": "array", "items": _ADVISORY_WARNING_OUTPUT
        },
    },
    ["contract_version", "action", "relation_types", "advisory_warnings"],
)
RELATION_TYPE_REGISTRY_WRITE_OUTPUT = _object(
    {
        "contract_version": {"type": "string", "const": CONTRACT_VERSION},
        "relation_type": _RELATION_TYPE_OUTPUT,
        "advisory_warnings": {
            "type": "array", "items": _ADVISORY_WARNING_OUTPUT
        },
    },
    ["contract_version", "relation_type", "advisory_warnings"],
)


def _annotations(
    *, idempotent: bool, read_only: bool = False, destructive: bool = False
) -> types.ToolAnnotations:
    return types.ToolAnnotations(
        read_only_hint=read_only,
        destructive_hint=destructive,
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
    types.Tool(
        name="write_judgment",
        description=(
            "Atomically add, update, supersede, archive, restore, or explicitly hard-delete "
            "a canonical SQLite judgment. Never use raw SQL. Updates and lifecycle changes "
            "require expected_revision; hard delete is restricted to safe archived candidates."
        ),
        input_schema=JUDGMENT_WRITE_INPUT,
        output_schema=_object(
            {
                "contract_version": {"type": "string"},
                "action": {"type": "string"},
                "judgment": {"anyOf": [{"type": "null"}, JUDGMENT_OUTPUT]},
            },
            ["contract_version", "action", "judgment"],
        ),
        annotations=_annotations(idempotent=False, destructive=True),
    ),
    types.Tool(
        name="search_judgments",
        description=JUDGMENT_SEARCH_DESCRIPTION,
        input_schema=JUDGMENT_SEARCH_INPUT,
        output_schema=JUDGMENT_SEARCH_OUTPUT,
        annotations=_annotations(idempotent=True, read_only=True),
    ),
    types.Tool(
        name="get_judgment",
        description=JUDGMENT_GET_DESCRIPTION,
        input_schema=JUDGMENT_GET_INPUT,
        output_schema=JUDGMENT_GET_OUTPUT,
        annotations=_annotations(idempotent=True, read_only=True),
    ),
    types.Tool(
        name="traverse_judgments",
        description=JUDGMENT_TRAVERSE_DESCRIPTION,
        input_schema=JUDGMENT_TRAVERSE_INPUT,
        output_schema=JUDGMENT_TRAVERSE_OUTPUT,
        annotations=_annotations(idempotent=True, read_only=True),
    ),
    types.Tool(
        name="read_relation_type_registry",
        description=RELATION_TYPE_REGISTRY_READ_DESCRIPTION,
        input_schema=RELATION_TYPE_REGISTRY_READ_INPUT,
        output_schema=RELATION_TYPE_REGISTRY_READ_OUTPUT,
        annotations=_annotations(idempotent=True, read_only=True),
    ),
    types.Tool(
        name="write_relation_type_registry",
        description=RELATION_TYPE_REGISTRY_WRITE_DESCRIPTION,
        input_schema=RELATION_TYPE_REGISTRY_WRITE_INPUT,
        output_schema=RELATION_TYPE_REGISTRY_WRITE_OUTPUT,
        annotations=_annotations(idempotent=False),
    ),
)


def _tools(
    *,
    confirmed_outcome_reinforcement: bool,
    soft_start_feedback_reinforcement: bool,
    outcome_driven_feedback_deactivation: bool = False,
) -> tuple[types.Tool, ...]:
    if not confirmed_outcome_reinforcement and not soft_start_feedback_reinforcement:
        return TOOLS
    source_use_description = (
        SOFT_START_SOURCE_USE_DESCRIPTION
        if soft_start_feedback_reinforcement
        else CONFIRMED_SOURCE_USE_DESCRIPTION
    )
    outcome_description = (
        DEACTIVATION_OUTCOME_DESCRIPTION
        if outcome_driven_feedback_deactivation
        else (
            SOFT_START_OUTCOME_DESCRIPTION
            if soft_start_feedback_reinforcement
            else CONFIRMED_OUTCOME_DESCRIPTION
        )
    )
    return (
        TOOLS[0],
        types.Tool(
            name="record_source_use",
            description=source_use_description,
            input_schema=SOURCE_USE_INPUT,
            output_schema=SOURCE_USE_OUTPUT,
            annotations=_annotations(idempotent=True),
        ),
        types.Tool(
            name="record_outcome",
            description=outcome_description,
            input_schema=OUTCOME_INPUT,
            output_schema=OUTCOME_OUTPUT,
            annotations=_annotations(idempotent=True),
        ),
        *TOOLS[3:],
    )


class FeedbackMCPAdapter:
    def __init__(
        self, database: str | Path, *, config: EngineConfig | None = None
    ) -> None:
        self.engine = NeuronGraphRAG(database, config=config)
        self.feedback = FeedbackLedger(self.engine)
        self.tools = _tools(
            confirmed_outcome_reinforcement=(
                self.engine.config.confirmed_outcome_reinforcement
            ),
            soft_start_feedback_reinforcement=(
                self.engine.config.soft_start_feedback_reinforcement
            ),
            outcome_driven_feedback_deactivation=(
                self.engine.config.outcome_driven_feedback_deactivation
            ),
        )

    def close(self) -> None:
        self.engine.close()

    async def list_tools(self, *_: object) -> types.ListToolsResult:
        return types.ListToolsResult(tools=list(self.tools), result_type="complete")

    async def call_tool(self, _context: object, params: types.CallToolRequestParams) -> types.CallToolResult:
        if params.name not in {tool.name for tool in self.tools}:
            raise MCPError(-32601, "Unknown tool")
        arguments = params.arguments or {}
        try:
            self._require_object(arguments)
            if params.name == "search":
                output = self._search(arguments)
            elif params.name == "record_source_use":
                output = self._record_source_use(arguments)
            elif params.name == "record_outcome":
                output = self._record_outcome(arguments)
            elif params.name == "write_judgment":
                output = self._write_judgment(arguments)
            elif params.name == "search_judgments":
                output = self._search_judgments(arguments)
            elif params.name == "get_judgment":
                output = self._get_judgment(arguments)
            elif params.name == "traverse_judgments":
                output = self._traverse_judgments(arguments)
            elif params.name == "read_relation_type_registry":
                output = self._read_relation_type_registry(arguments)
            else:
                output = self._write_relation_type_registry(arguments)
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
            search_surface = effective_search_surface(self.engine.config)
            trace = search_with_surface(
                self.engine,
                query,
                limit=limit,
                search_surface=search_surface,
            )
        except ValueError as error:
            if str(error) == "Cannot search an empty corpus":
                raise FeedbackContractError("empty_corpus", "local NGR corpus is empty") from error
            raise
        hits = []
        for rank, hit in enumerate(trace.hits, start=1):
            final_score = (
                hit.channel_score if hasattr(hit, "channel_score") else hit.final_score
            )
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
                        "final": final_score,
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
            "effective_config_provenance": {
                **effective_config_provenance(self.engine.config),
                "search_surface": search_surface,
            },
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
            "confirmations": [
                {
                    "source_id": item.source_id,
                    "target_id": item.target_id,
                    "edge_type": item.edge_type,
                    "confirmation_count": item.confirmation_count,
                    "multiplier": item.multiplier,
                    "actual_delta": item.actual_delta,
                    "old_weight": item.old_weight,
                    "new_weight": item.new_weight,
                }
                for item in receipt.confirmations
            ],
            "credited_paths": [
                {
                    "node_id": path.node_id,
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
                for path in receipt.credited_paths
            ],
            "normalized_sibling_edges": [
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "edge_type": edge.edge_type,
                    "old_weight": edge.old_weight,
                    "new_weight": edge.new_weight,
                }
                for edge in receipt.normalized_sibling_edges
            ],
            "deactivation_applied": receipt.deactivation_applied,
            "reversed_contributions": [
                {
                    "contribution_id": item.contribution_id,
                    "contribution_kind": item.contribution_kind,
                    "source_record_id": item.source_record_id,
                    "source_id": item.source_id,
                    "target_id": item.target_id,
                    "edge_type": item.edge_type,
                    "credited_delta": item.credited_delta,
                    "mutations": [
                        {
                            "mutation_role": mutation.mutation_role,
                            "source_id": mutation.source_id,
                            "target_id": mutation.target_id,
                            "edge_type": mutation.edge_type,
                            "actual_delta": mutation.actual_delta,
                            "old_weight": mutation.old_weight,
                            "new_weight": mutation.new_weight,
                        }
                        for mutation in item.mutations
                    ],
                }
                for item in receipt.reversed_contributions
            ],
            "dormancy_changes": [
                {
                    "source_id": item.source_id,
                    "target_id": item.target_id,
                    "edge_type": item.edge_type,
                    "old_dormant": item.old_dormant,
                    "new_dormant": item.new_dormant,
                }
                for item in receipt.dormancy_changes
            ],
            "reactivated_edges": [
                {
                    "source_id": item.source_id,
                    "target_id": item.target_id,
                    "edge_type": item.edge_type,
                    "old_dormant": item.old_dormant,
                    "new_dormant": item.new_dormant,
                }
                for item in receipt.reactivated_edges
            ],
        }

    def _write_judgment(self, data: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "contract_version", "action", "judgment_id", "successor_id",
            "statement", "rationale", "provenance", "expected_revision", "relations",
        }
        self._keys(data, allowed, {"contract_version", "action", "judgment_id"})
        self._version(data)
        action = data["action"]
        identifier = data["judgment_id"]
        graph = self.engine.judgments
        if action == "add":
            required = {"statement", "rationale", "provenance"}
            if not required <= data.keys():
                raise ValueError("add requires statement, rationale, and provenance")
            judgment = graph.add(identifier, data["statement"], data["rationale"], data["provenance"], relations=data.get("relations", []))
        elif action == "update":
            required = {"statement", "rationale", "provenance", "expected_revision"}
            if not required <= data.keys():
                raise ValueError("update requires content, provenance, and expected_revision")
            judgment = graph.update(identifier, data["statement"], data["rationale"], data["provenance"], expected_revision=data["expected_revision"], relations=data.get("relations", []))
        elif action == "supersede":
            required = {"successor_id", "statement", "rationale", "provenance", "expected_revision"}
            if not required <= data.keys():
                raise ValueError("supersede requires successor content and expected_revision")
            judgment = graph.supersede(identifier, data["successor_id"], data["statement"], data["rationale"], data["provenance"], expected_revision=data["expected_revision"], relations=data.get("relations", []))
        elif action in {"archive", "restore"}:
            if "expected_revision" not in data:
                raise ValueError("lifecycle change requires expected_revision")
            judgment = getattr(graph, action)(identifier, expected_revision=data["expected_revision"])
        elif action == "hard_delete":
            if "expected_revision" not in data:
                raise ValueError("hard_delete requires expected_revision")
            graph.hard_delete(identifier, expected_revision=data["expected_revision"])
            judgment = None
        else:
            raise ValueError("unsupported judgment action")
        return {"contract_version": CONTRACT_VERSION, "action": action, "judgment": judgment}

    def _search_judgments(self, data: dict[str, Any]) -> dict[str, Any]:
        self._keys(
            data,
            {"contract_version", "query", "limit", "include_archived", "repository"},
            {"contract_version", "query"},
        )
        self._version(data)
        query = self._trimmed_string(data["query"], "query", 8192)
        limit = data.get("limit", 5)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 through 100")
        results = self.engine.judgments.search_judgments(
            query,
            limit=limit,
            include_archived=data.get("include_archived", False),
            repository=data.get("repository"),
        )
        return {"contract_version": CONTRACT_VERSION, "judgments": results}

    def _get_judgment(self, data: dict[str, Any]) -> dict[str, Any]:
        self._keys(
            data,
            {"contract_version", "judgment_id"},
            {"contract_version", "judgment_id"},
        )
        self._version(data)
        return {
            "contract_version": CONTRACT_VERSION,
            "judgment": self.engine.judgments.get_judgment(data["judgment_id"]),
        }

    def _traverse_judgments(self, data: dict[str, Any]) -> dict[str, Any]:
        self._keys(
            data,
            {
                "contract_version", "judgment_id", "direction", "relation_type",
                "max_hops", "include_archived",
            },
            {"contract_version", "judgment_id"},
        )
        self._version(data)
        results = self.engine.judgments.traverse_judgments(
            data["judgment_id"],
            direction=data.get("direction", "outgoing"),
            relation_type=data.get("relation_type"),
            max_hops=data.get("max_hops", 1),
            include_archived=data.get("include_archived", False),
        )
        return {"contract_version": CONTRACT_VERSION, "results": results}

    def _read_relation_type_registry(self, data: dict[str, Any]) -> dict[str, Any]:
        self._keys(
            data,
            {
                "contract_version", "action", "relation_type", "revision",
                "include_deprecated",
            },
            {"contract_version", "action"},
        )
        self._version(data)
        action = data["action"]
        graph = self.engine.judgments
        if action == "list":
            if "relation_type" in data or "revision" in data:
                raise ValueError("list does not accept relation_type or revision")
            relation_types = graph.list_relation_types(
                include_deprecated=data.get("include_deprecated", True)
            )
            warnings = [
                warning
                for record in relation_types
                for warning in graph.validate_relation_type(record["relation_type"])
            ]
        elif action == "get":
            if "relation_type" not in data or "include_deprecated" in data:
                raise ValueError("get requires relation_type and does not accept include_deprecated")
            record = graph.get_relation_type(
                data["relation_type"], revision=data.get("revision")
            )
            relation_types = [record]
            warnings = graph.validate_relation_type(data["relation_type"])
        elif action == "validate":
            if (
                "relation_type" not in data
                or "revision" in data
                or "include_deprecated" in data
            ):
                raise ValueError("validate requires only relation_type")
            warnings = graph.validate_relation_type(data["relation_type"])
            try:
                relation_types = [graph.get_relation_type(data["relation_type"])]
            except KeyError:
                relation_types = []
        else:
            raise ValueError("unsupported relation type registry action")
        return {
            "contract_version": CONTRACT_VERSION,
            "action": action,
            "relation_types": relation_types,
            "advisory_warnings": warnings,
        }

    def _write_relation_type_registry(self, data: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "contract_version", "relation_type", "definition", "namespace",
            "provenance", "expected_revision", "lifecycle",
        }
        self._keys(
            data,
            allowed,
            {
                "contract_version", "relation_type", "definition", "namespace",
                "provenance", "expected_revision",
            },
        )
        self._version(data)
        record = self.engine.judgments.register_relation_type(
            data["relation_type"],
            data["definition"],
            data["namespace"],
            data["provenance"],
            expected_revision=data["expected_revision"],
            lifecycle=data.get("lifecycle", "active"),
        )
        return {
            "contract_version": CONTRACT_VERSION,
            "relation_type": record,
            "advisory_warnings": self.engine.judgments.validate_relation_type(
                data["relation_type"]
            ),
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


async def _run(
    database: str | Path, *, config: EngineConfig | None = None
) -> None:
    server, adapter = create_server(database, config=config)
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


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _unit_interval(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number from 0.0 through 1.0") from error
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be a number from 0.0 through 1.0")
    return parsed


def _open_unit_interval(value: str) -> float:
    parsed = _unit_interval(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must be a number strictly between 0.0 and 1.0")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Neuron Graph RAG MCP server")
    parser.add_argument(
        "--database",
        help=(
            "Path to the local SQLite database "
            "(default: NGR_DATABASE or ~/.ngrdb/knowledge.db)"
        ),
    )
    parser.add_argument(
        "--relation-feedback-evidence-quorum",
        type=_positive_integer,
        default=1,
        help="Independent relation feedback items required before reinforcement (default: 1)",
    )
    parser.add_argument(
        "--sibling-feedback-normalization",
        type=_unit_interval,
        default=0.0,
        help="Same-source sibling normalization ratio from 0.0 through 1.0 (default: 0.0)",
    )
    parser.add_argument(
        "--confirmed-outcome-reinforcement",
        action="store_true",
        help="Move positive relation reinforcement from used to confirmed outcomes",
    )
    parser.add_argument(
        "--confirmation-decay-ratio",
        type=_open_unit_interval,
        default=None,
        help="Geometric decay ratio required by confirmed-only or soft-start reinforcement",
    )
    parser.add_argument(
        "--soft-start-feedback-reinforcement",
        action="store_true",
        help="Apply a provisional used update and complete it on first confirmation",
    )
    parser.add_argument(
        "--soft-start-feedback-ratio",
        type=_open_unit_interval,
        default=None,
        help="Provisional fraction required by soft-start feedback reinforcement",
    )
    parser.add_argument(
        "--outcome-driven-feedback-deactivation",
        action="store_true",
        help="Exactly reverse attributed soft-start contributions or make them dormant",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    arguments = parser.parse_args()
    if (
        arguments.confirmed_outcome_reinforcement
        and arguments.soft_start_feedback_reinforcement
    ):
        parser.error(
            "--confirmed-outcome-reinforcement and "
            "--soft-start-feedback-reinforcement are mutually exclusive"
        )
    candidate_enabled = (
        arguments.confirmed_outcome_reinforcement
        or arguments.soft_start_feedback_reinforcement
    )
    if candidate_enabled and arguments.confirmation_decay_ratio is None:
        parser.error(
            "confirmed-only and soft-start reinforcement require --confirmation-decay-ratio"
        )
    if not candidate_enabled and arguments.confirmation_decay_ratio is not None:
        parser.error(
            "--confirmation-decay-ratio requires confirmed-only or soft-start reinforcement"
        )
    if (
        arguments.soft_start_feedback_reinforcement
        and arguments.soft_start_feedback_ratio is None
    ):
        parser.error(
            "--soft-start-feedback-reinforcement requires --soft-start-feedback-ratio"
        )
    if (
        not arguments.soft_start_feedback_reinforcement
        and arguments.soft_start_feedback_ratio is not None
    ):
        parser.error(
            "--soft-start-feedback-ratio requires --soft-start-feedback-reinforcement"
        )
    if (
        arguments.outcome_driven_feedback_deactivation
        and not arguments.soft_start_feedback_reinforcement
    ):
        parser.error(
            "--outcome-driven-feedback-deactivation requires "
            "--soft-start-feedback-reinforcement"
        )
    config = EngineConfig(
        relation_feedback_evidence_quorum=(
            arguments.relation_feedback_evidence_quorum
        ),
        sibling_feedback_normalization=arguments.sibling_feedback_normalization,
        confirmed_outcome_reinforcement=arguments.confirmed_outcome_reinforcement,
        confirmation_decay_ratio=arguments.confirmation_decay_ratio,
        soft_start_feedback_reinforcement=(
            arguments.soft_start_feedback_reinforcement
        ),
        soft_start_feedback_ratio=arguments.soft_start_feedback_ratio,
        outcome_driven_feedback_deactivation=(
            arguments.outcome_driven_feedback_deactivation
        ),
    )
    try:
        database = prepare_database(
            resolve_database(arguments.database, environ=os.environ)
        )
    except ValueError as error:
        parser.error(str(error))
    asyncio.run(_run(database, config=config))
