from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


FIXTURE_SCHEMA_VERSION = 1
PROVENANCE_SCHEMA_VERSION = 1
SEARCH_DOC_COLUMNS = (
    "vector_id",
    "repo",
    "type",
    "state",
    "labels",
    "milestone",
    "assignees",
    "updated_at",
    "number",
    "tag_name",
    "doc_path",
    "commit_sha",
    "file_path",
    "file_status",
    "commit_date",
    "commit_author",
    "tokenizer_kind",
    "content",
    "indexed_at",
)
DOC_EDGE_COLUMNS = (
    "src_vector_id",
    "dst_vector_id",
    "repo",
    "src_slug",
    "dst_slug",
    "edge_kind",
    "updated_at",
)
EXPECTED_SCHEMA = {
    "search_docs": SEARCH_DOC_COLUMNS[:17] + ("content", "indexed_at", "content_fts"),
    "doc_edges": DOC_EDGE_COLUMNS,
}
FORBIDDEN_SQL = re.compile(
    r"\b(?:attach|alter|create|delete|detach|drop|insert|pragma|reindex|replace|"
    r"truncate|update|vacuum)\b",
    re.IGNORECASE,
)
SECRET_PATTERNS = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"(?i)\b(?:authorization\s*:\s*bearer|bearer)\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|client[_-]?secret|github_token|cloudflare_api_token)"
        r"\s*[:=]\s*[\"']?[A-Za-z0-9._~+/=-]{12,}[\"']?"
    ),
)


def validate_read_only_sql(sql: str) -> str:
    normalized = sql.strip()
    if normalized.endswith(";"):
        normalized = normalized[:-1].rstrip()
    if not normalized or ";" in normalized:
        raise ValueError("Exactly one SQL statement is allowed")
    executable = re.sub(r"'(?:''|[^'])*'", "''", normalized)
    if not re.match(r"^(?:SELECT|WITH)\b", executable, re.IGNORECASE):
        raise ValueError("Only a SELECT or WITH query is allowed")
    if FORBIDDEN_SQL.search(executable):
        raise ValueError("Mutating or administrative SQL is not allowed")
    return normalized


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def wrangler_select(
    database: str, sql: str, *, cwd: Path, wrangler_command: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query = validate_read_only_sql(sql)
    completed = subprocess.run(
        [
            *wrangler_command,
            "d1",
            "execute",
            database,
            "--remote",
            "--command",
            query,
            "--json",
        ],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(
            "Wrangler D1 query failed: " + completed.stderr.strip()
        )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError("Wrangler returned an unexpected result envelope")
    result = payload[0]
    meta = dict(result.get("meta", {}))
    if not result.get("success"):
        raise RuntimeError("D1 SELECT did not succeed")
    if (
        int(meta.get("rows_written", -1)) != 0
        or int(meta.get("changes", -1)) != 0
        or bool(meta.get("changed_db", True))
    ):
        raise RuntimeError("Read-only invariant failed: D1 reported a write")
    return list(result.get("results", [])), meta


def redact(value: Any) -> tuple[Any, int]:
    if isinstance(value, str):
        count = 0
        for pattern in SECRET_PATTERNS:
            value, replaced = pattern.subn("[REDACTED_SECRET]", value)
            count += replaced
        return value, count
    if isinstance(value, list):
        output: list[Any] = []
        count = 0
        for item in value:
            redacted, replaced = redact(item)
            output.append(redacted)
            count += replaced
        return output, count
    if isinstance(value, dict):
        output_dict: dict[str, Any] = {}
        count = 0
        for key, item in value.items():
            redacted, replaced = redact(item)
            output_dict[str(key)] = redacted
            count += replaced
        return output_dict, count
    return value, 0


def source_url(row: dict[str, Any]) -> str:
    repo = str(row["repo"])
    doc_type = str(row["type"])
    number = row.get("number")
    if doc_type in {"issue", "issue_comment"} and number is not None:
        return f"https://github.com/{repo}/issues/{number}"
    if doc_type in {"pull_request", "pr_review", "pr_review_comment"} and number is not None:
        return f"https://github.com/{repo}/pull/{number}"
    if doc_type == "release" and row.get("tag_name"):
        return f"https://github.com/{repo}/releases/tag/{row['tag_name']}"
    if doc_type == "wiki_doc" and row.get("doc_path"):
        return f"https://github.com/{repo}/wiki/{row['doc_path']}"
    if row.get("commit_sha") and (row.get("file_path") or row.get("doc_path")):
        path = row.get("file_path") or row.get("doc_path")
        return f"https://github.com/{repo}/blob/{row['commit_sha']}/{path}"
    if row.get("commit_sha"):
        return f"https://github.com/{repo}/commit/{row['commit_sha']}"
    return f"https://github.com/{repo}"


def build_coverage_query(
    repositories: Sequence[str], types: Sequence[str]
) -> str:
    repo_filter = ", ".join(sql_literal(value) for value in repositories)
    type_filter = ", ".join(sql_literal(value) for value in types)
    return validate_read_only_sql(
        "SELECT repo, type, COUNT(*) AS source_count, "
        "MIN(updated_at) AS oldest_updated_at, MAX(updated_at) AS newest_updated_at, "
        "COUNT(DISTINCT NULLIF(commit_sha, '')) AS distinct_commit_count "
        f"FROM search_docs WHERE repo IN ({repo_filter}) AND type IN ({type_filter}) "
        "GROUP BY repo, type ORDER BY repo, type"
    )


def transform(
    search_rows: list[dict[str, Any]], edge_rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, int]]:
    nodes: list[dict[str, Any]] = []
    node_ids: set[str] = set()
    for row in search_rows:
        node_id = str(row["vector_id"])
        if node_id in node_ids:
            raise ValueError(f"Duplicate source vector_id: {node_id}")
        node_ids.add(node_id)
        metadata = {
            key: row.get(key)
            for key in SEARCH_DOC_COLUMNS
            if key != "content"
        }
        metadata["source_table"] = "search_docs"
        metadata["source_url"] = source_url(row)
        node = {
            "node_id": node_id,
            "text": str(row.get("content") or ""),
            "metadata": metadata,
            "confidence": 1.0,
        }
        if not str(node["text"]).strip():
            raise ValueError(f"Source row has empty content: {node_id}")
        nodes.append(node)

    edges: list[dict[str, Any]] = []
    excluded = Counter()
    for row in edge_rows:
        source_id = str(row["src_vector_id"])
        target_id = str(row["dst_vector_id"])
        missing = int(source_id not in node_ids) + int(target_id not in node_ids)
        if missing:
            excluded["one_endpoint_missing" if missing == 1 else "both_endpoints_missing"] += 1
            continue
        edge = {
            "source_id": source_id,
            "target_id": target_id,
            "edge_type": str(row["edge_kind"]),
            "weight": 1.0,
            "factuality": 1.0,
            "metadata": {
                "source_table": "doc_edges",
                "source_record": {key: row.get(key) for key in DOC_EDGE_COLUMNS},
            },
        }
        edges.append(edge)

    fixture = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "nodes": nodes,
        "edges": edges,
    }
    statistics = {
        "nodes_included": len(nodes),
        "edges_included": len(edges),
        "edges_one_endpoint_missing": excluded["one_endpoint_missing"],
        "edges_both_endpoints_missing": excluded["both_endpoints_missing"],
        "redactions": 0,
        "fixture_redactions": 0,
        "provenance_redactions": 0,
    }
    return fixture, statistics


def redact_final_payloads(
    fixture: dict[str, Any], report: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    redacted_fixture, fixture_redactions = redact(fixture)
    redacted_report, provenance_redactions = redact(report)
    result = redacted_report.setdefault("result", {})
    result["fixture_redactions"] = fixture_redactions
    result["provenance_redactions"] = provenance_redactions
    result["redactions"] = fixture_redactions + provenance_redactions
    return redacted_fixture, redacted_report


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def acquire(args: argparse.Namespace) -> None:
    repositories = sorted(set(args.repo))
    types = sorted(set(args.type))
    doc_paths = sorted(set(getattr(args, "doc_path", [])))
    if not repositories or not types or args.per_type_limit < 1:
        raise ValueError("At least one repo/type and a positive limit are required")
    if doc_paths and (len(repositories) != 1 or types != ["wiki_doc"]):
        raise ValueError(
            "--doc-path requires exactly one repository and --type wiki_doc"
        )
    repo_filter = ", ".join(sql_literal(value) for value in repositories)
    type_filter = ", ".join(sql_literal(value) for value in types)
    command = tuple(args.wrangler_command)
    cwd = args.wrangler_project.resolve()
    metas: list[dict[str, Any]] = []

    schema: dict[str, list[dict[str, Any]]] = {}
    for table in ("search_docs", "doc_edges"):
        rows, meta = wrangler_select(
            args.database,
            'SELECT cid, name, type, "notnull" AS not_null, dflt_value, pk '
            f"FROM pragma_table_info({sql_literal(table)}) ORDER BY cid",
            cwd=cwd,
            wrangler_command=command,
        )
        metas.append(meta)
        names = tuple(str(row["name"]) for row in rows)
        if names != EXPECTED_SCHEMA[table]:
            raise RuntimeError(
                f"Live {table} schema differs from the acquisition contract: {names!r}"
            )
        schema[table] = rows

    selected_columns = ", ".join(f'"{column}"' for column in SEARCH_DOC_COLUMNS)
    search_rows: list[dict[str, Any]] = []
    if doc_paths:
        doc_path_filter = ", ".join(sql_literal(value) for value in doc_paths)
        rows, meta = wrangler_select(
            args.database,
            f"SELECT {selected_columns} FROM search_docs "
            f"WHERE \"repo\" = {sql_literal(repositories[0])} "
            "AND \"type\" = 'wiki_doc' "
            f"AND \"doc_path\" IN ({doc_path_filter}) "
            "ORDER BY \"doc_path\", \"vector_id\"",
            cwd=cwd,
            wrangler_command=command,
        )
        metas.append(meta)
        observed_paths = {str(row["doc_path"]) for row in rows}
        missing_paths = sorted(set(doc_paths) - observed_paths)
        if missing_paths:
            raise RuntimeError(f"Requested wiki documents are missing: {missing_paths!r}")
        search_rows.extend(rows)
    else:
        for repository in repositories:
            for doc_type in types:
                rows, meta = wrangler_select(
                    args.database,
                    f"SELECT {selected_columns} FROM search_docs "
                    f"WHERE \"repo\" = {sql_literal(repository)} "
                    f"AND \"type\" = {sql_literal(doc_type)} "
                    "ORDER BY \"repo\", \"type\", \"updated_at\", \"vector_id\" "
                    f"LIMIT {args.per_type_limit}",
                    cwd=cwd,
                    wrangler_command=command,
                )
                metas.append(meta)
                search_rows.extend(rows)

    edge_columns = ", ".join(f'"{column}"' for column in DOC_EDGE_COLUMNS)
    edge_rows, meta = wrangler_select(
        args.database,
        f"SELECT {edge_columns} FROM doc_edges "
        f"WHERE repo IN ({repo_filter}) "
        "ORDER BY repo, edge_kind, updated_at, src_vector_id, dst_vector_id",
        cwd=cwd,
        wrangler_command=command,
    )
    metas.append(meta)

    coverage_rows, meta = wrangler_select(
        args.database,
        build_coverage_query(repositories, types),
        cwd=cwd,
        wrangler_command=command,
    )
    metas.append(meta)

    fixture, statistics = transform(search_rows, edge_rows)
    if getattr(args, "require_connected", False):
        assert_connected(fixture)
    schema_bytes = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    schema_fingerprint = "sha256:" + hashlib.sha256(schema_bytes).hexdigest()
    selection_order = (
        ["repo", "type", "doc_path", "vector_id"]
        if doc_paths
        else ["repo", "type", "updated_at", "vector_id"]
    )
    fixture["source"] = {
        "database": args.database,
        "repositories": repositories,
        "types": types,
        "per_type_limit": args.per_type_limit,
        "selection_order": selection_order,
        "schema_fingerprint": schema_fingerprint,
    }
    if doc_paths:
        fixture["source"]["doc_paths"] = doc_paths
    report = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "acquired_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {
            "database": args.database,
            "repositories": repositories,
            "types": types,
            "schema_fingerprint": schema_fingerprint,
            "authoritative_history": "GitHub",
        },
        "selection": {
            "per_type_limit": args.per_type_limit,
            "order": selection_order,
        },
        "coverage": coverage_rows,
        "result": statistics,
        "read_only_evidence": {
            "query_count": len(metas),
            "rows_written": [int(meta["rows_written"]) for meta in metas],
            "changes": [int(meta["changes"]) for meta in metas],
            "changed_db": [bool(meta["changed_db"]) for meta in metas],
        },
        "known_gaps": list(args.known_gap),
        "limits": [
            "D1 is a lossy search snapshot; content may be truncated.",
            "Binary and patchless files can be absent from diff indexing.",
            "Use GitHub for byte-exact historical reconstruction.",
        ],
    }
    if doc_paths:
        report["selection"]["doc_paths"] = doc_paths
    fixture, report = redact_final_payloads(fixture, report)
    write_json(args.output, fixture)
    write_json(args.provenance_output, report)


def assert_connected(fixture: dict[str, Any]) -> None:
    node_ids = {str(node["node_id"]) for node in fixture["nodes"]}
    if not node_ids:
        raise RuntimeError("Connected fixture must contain at least one node")
    neighbors = {node_id: set() for node_id in node_ids}
    for edge in fixture["edges"]:
        source_id = str(edge["source_id"])
        target_id = str(edge["target_id"])
        neighbors[source_id].add(target_id)
        neighbors[target_id].add(source_id)
    visited: set[str] = set()
    pending = [min(node_ids)]
    while pending:
        node_id = pending.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        pending.extend(sorted(neighbors[node_id] - visited, reverse=True))
    if visited != node_ids:
        raise RuntimeError(
            "Selected fixture is not weakly connected; unreachable nodes: "
            f"{sorted(node_ids - visited)!r}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire a deterministic, read-only D1 fixture for NGR validation."
    )
    parser.add_argument("--database", default="github-rag-fts")
    parser.add_argument("--repo", action="append", required=True)
    parser.add_argument("--type", action="append", required=True)
    parser.add_argument("--per-type-limit", type=int, default=3)
    parser.add_argument(
        "--doc-path",
        action="append",
        default=[],
        help="Select an exact wiki doc_path (repeatable; requires one repo/wiki_doc).",
    )
    parser.add_argument(
        "--require-connected",
        action="store_true",
        help="Fail unless the selected fixture is weakly connected.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--known-gap", action="append", default=[])
    parser.add_argument("--wrangler-project", type=Path, required=True)
    parser.add_argument(
        "--wrangler-command",
        nargs="+",
        default=[shutil.which("npx") or "npx", "wrangler"],
        help="Command prefix used to invoke Wrangler (default: npx wrangler).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    acquire(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
