"""Compare a fixed GitHub source snapshot with github-rag-mcp source contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from neuron_graph_rag import EngineConfig, NeuronGraphRAG
from neuron_graph_rag.github_source import (
    GitHubSnapshot,
    changed_paths,
    index_github_snapshot,
)


def run_compatibility(
    snapshot_path: str | Path,
    cases_path: str | Path,
    *,
    github_rag_mcp_capture_path: str | Path | None = None,
    updated_snapshot_path: str | Path | None = None,
) -> dict[str, Any]:
    snapshot = GitHubSnapshot.read(snapshot_path)
    cases = _read_cases(cases_path)
    capture = _read_github_rag_mcp_capture(
        github_rag_mcp_capture_path, cases, snapshot.repository
    )
    config = EngineConfig(
        sparse_weight=1.0,
        dense_weight=0.0,
        entry_weight=1.0,
        graph_weight=0.0,
        use_dense_retrieval=False,
        use_graph_propagation=False,
    )
    with NeuronGraphRAG(config=config) as engine:
        receipt = index_github_snapshot(engine, snapshot)
        comparisons = [
            _compare_case(engine, case, capture[case["id"]], snapshot.repository)
            if capture
            else _compare_case(engine, case, repository=snapshot.repository)
            for case in cases
        ]
        update = _update_followup(engine, snapshot, updated_snapshot_path)

    all_source_matches = capture is not None and all(
        item["source_match"] for item in comparisons
    )
    if capture is None:
        verdict = "inconclusive"
    elif update["provided"] and all_source_matches and update["followed"]:
        verdict = "continue_candidate"
    elif not update["provided"]:
        verdict = "inconclusive"
    else:
        verdict = "incompatible"
    return {
        "schema_version": 1,
        "source": {
            "repository": snapshot.repository,
            "commit": snapshot.commit,
            "fingerprint": receipt.fingerprint,
            "read_only": True,
            "source_urls": list(receipt.source_urls),
        },
        "comparisons": comparisons,
        "comparison_status": "compared" if capture is not None else "adapter_pipeline_only",
        "source_update": update,
        "verdict": verdict,
        "limitations": [
            "continue_candidate requires a preserved github-rag-mcp search capture for every query.",
            "This does not implement an MCP server, authentication, transport, or remote deployment.",
            "The adapter indexes only acquired file content; it does not ingest issues, pull requests, comments, releases, or commit diffs.",
        ],
    }


def _read_cases(path: str | Path) -> list[dict[str, str]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise ValueError("Unsupported GitHub compatibility case schema version")
    rows = value.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("GitHub compatibility cases must not be empty")
    cases: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("GitHub compatibility cases must be objects")
        case = {key: _required_string(row, key) for key in ("id", "query")}
        cases.append(case)
    return cases


def _read_github_rag_mcp_capture(
    path: str | Path | None,
    cases: Sequence[Mapping[str, str]],
    repository: str,
) -> dict[str, dict[str, Any]] | None:
    if path is None:
        return None
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise ValueError("Unsupported github-rag-mcp capture schema version")
    captured_at = _required_string(value, "captured_at")
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("github-rag-mcp capture provenance must be an object")
    provenance_value = {
        key: _required_string(provenance, key)
        for key in ("service", "tool", "capture_reference")
    }
    if provenance_value["service"] != "github-rag-mcp" or provenance_value["tool"] != "search":
        raise ValueError("capture must be from github-rag-mcp search")
    expected_queries = {case["id"]: case["query"] for case in cases}
    rows = value.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("github-rag-mcp capture cases must not be empty")
    captured: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("github-rag-mcp capture cases must be objects")
        case_id = _required_string(row, "id")
        query = _required_string(row, "query")
        if case_id not in expected_queries or query != expected_queries[case_id]:
            raise ValueError("capture case must match a fixed compatibility query")
        if case_id in captured:
            raise ValueError(f"Duplicate github-rag-mcp capture case: {case_id}")
        request = row.get("request")
        if not isinstance(request, Mapping):
            raise ValueError("github-rag-mcp capture request must be an object")
        if request != {"repo": repository, "type": "doc", "top_k": 10}:
            raise ValueError("capture request must use the fixed repository, doc type, and top_k=10")
        raw_result = row.get("raw_result")
        if not isinstance(raw_result, Mapping):
            raise ValueError("github-rag-mcp capture raw_result must be an object")
        results = raw_result.get("results")
        if not isinstance(results, list):
            raise ValueError("github-rag-mcp capture raw_result results must be a list")
        source_urls = []
        for item in results:
            if not isinstance(item, Mapping):
                raise ValueError("github-rag-mcp capture results must be objects")
            url = _required_string(item, "url")
            if _source_identity(url, repository) is None:
                raise ValueError("capture result URLs must belong to the fixed source repository")
            source_urls.append(url)
        captured[case_id] = {
            "captured_at": captured_at,
            "provenance": provenance_value,
            "request": dict(request),
            "raw_result": raw_result,
            "source_urls": source_urls,
        }
    if set(captured) != set(expected_queries):
        raise ValueError("github-rag-mcp capture must cover every fixed compatibility query")
    return captured


def _compare_case(
    engine: NeuronGraphRAG,
    case: Mapping[str, str],
    captured: Mapping[str, Any] | None = None,
    repository: str = "",
) -> dict[str, Any]:
    trace = engine.search(case["query"], limit=5, now=0.0)
    hits = []
    for rank, hit in enumerate(trace.hits, start=1):
        explanation = hit.explain()
        hits.append(
            {
                "rank": rank,
                "source_url": hit.node.metadata["source_url"],
                "node_id": hit.node.node_id,
                "rationale": explanation,
            }
        )
    result: dict[str, Any] = {
        "id": case["id"],
        "query": case["query"],
        "ngr": {"hits": hits},
    }
    if captured is None:
        result["github_rag_mcp"] = {"captured": False}
        result["source_match"] = None
    else:
        result["github_rag_mcp"] = {"captured": True, **captured}
        result["source_match"] = bool(
            {_source_identity(hit["source_url"], repository) for hit in hits}
            & {_source_identity(url, repository) for url in captured["source_urls"]}
        )
    return result


def _source_identity(url: str, repository: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "github.com":
        return None
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 5 or parts[2] != "blob" or "/".join(parts[:2]) != repository:
        return None
    return "/".join((repository, *parts[4:]))


def _update_followup(
    engine: NeuronGraphRAG,
    snapshot: GitHubSnapshot,
    updated_snapshot_path: str | Path | None,
) -> dict[str, Any]:
    if updated_snapshot_path is None:
        return {"provided": False, "followed": False, "changed_paths": []}
    updated = GitHubSnapshot.read(updated_snapshot_path)
    paths = changed_paths(snapshot, updated)
    receipt = index_github_snapshot(engine, updated)
    indexed = {node.node_id: node for node in engine.store.list_nodes()}
    followed = bool(paths) and all(
        indexed[updated.document_id(document)].metadata["blob_sha"] == document.blob_sha
        for document in updated.documents
        if document.path in paths
    )
    return {
        "provided": True,
        "before_commit": snapshot.commit,
        "after_commit": updated.commit,
        "changed_paths": list(paths),
        "reindexed_node_ids": list(receipt.node_ids),
        "followed": followed,
    }


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"Case {key} must be a non-empty string")
    return item


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--github-rag-mcp-capture", type=Path)
    parser.add_argument("--updated-snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_compatibility(
        args.snapshot,
        args.cases,
        github_rag_mcp_capture_path=args.github_rag_mcp_capture,
        updated_snapshot_path=args.updated_snapshot,
    )
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
