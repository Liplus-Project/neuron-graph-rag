"""Compare a fixed GitHub source snapshot with github-rag-mcp source contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    updated_snapshot_path: str | Path | None = None,
) -> dict[str, Any]:
    snapshot = GitHubSnapshot.read(snapshot_path)
    cases = _read_cases(cases_path, snapshot.repository)
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
        comparisons = [_compare_case(engine, case) for case in cases]
        update = _update_followup(engine, snapshot, updated_snapshot_path)

    all_expected_found = all(item["expected_source_found"] for item in comparisons)
    if update["provided"] and all_expected_found and update["followed"]:
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
        "source_update": update,
        "verdict": verdict,
        "limitations": [
            "github-rag-mcp production search is not called; expected sources are an explicit comparison contract.",
            "This does not implement an MCP server, authentication, transport, or remote deployment.",
            "The adapter indexes only acquired file content; it does not ingest issues, pull requests, comments, releases, or commit diffs.",
        ],
    }


def _read_cases(path: str | Path, repository: str) -> list[dict[str, str]]:
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
        case = {key: _required_string(row, key) for key in ("id", "query", "expected_source_url")}
        if f"https://github.com/{repository}/blob/" not in case["expected_source_url"]:
            raise ValueError("expected_source_url must belong to the fixed source repository")
        cases.append(case)
    return cases


def _compare_case(engine: NeuronGraphRAG, case: Mapping[str, str]) -> dict[str, Any]:
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
    return {
        "id": case["id"],
        "query": case["query"],
        "github_rag_mcp_contract": {"expected_source_url": case["expected_source_url"]},
        "ngr": {"hits": hits},
        "expected_source_found": case["expected_source_url"] in {
            hit["source_url"] for hit in hits
        },
    }


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
    parser.add_argument("--updated-snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_compatibility(
        args.snapshot,
        args.cases,
        updated_snapshot_path=args.updated_snapshot,
    )
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
