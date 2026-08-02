from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _canonical_sha256(path: Path) -> str:
    value = _read_json(path)
    canonical = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _fixture_identifiers(path: Path) -> dict[str, set[str]]:
    fixture = _read_json(path)
    nodes = fixture.get("nodes")
    edges = fixture.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError(f"Fixture lacks nodes or edges: {path}")
    return {
        "doc_paths": {
            str(node.get("metadata", {}).get("doc_path", ""))
            for node in nodes
            if str(node.get("metadata", {}).get("doc_path", ""))
        },
        "file_paths": {
            str(node.get("metadata", {}).get("file_path", ""))
            for node in nodes
            if str(node.get("metadata", {}).get("file_path", ""))
        },
        "node_ids": {str(node["node_id"]) for node in nodes},
        "source_urls": {
            str(node.get("metadata", {}).get("source_url", ""))
            for node in nodes
            if str(node.get("metadata", {}).get("source_url", ""))
        },
        "relation_endpoints": {
            str(edge[field])
            for edge in edges
            for field in ("source_id", "target_id")
        },
    }


def _gold_identifiers(path: Path) -> dict[str, set[str]]:
    gold = _read_json(path)
    cases = gold.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"Gold lacks cases: {path}")
    relation_endpoints: set[str] = set()
    for case in cases:
        for step in case.get("expected_path", []):
            relation_endpoints.update(
                (str(step["source_id"]), str(step["target_id"]))
            )
    return {
        "normalized_queries": {
            re.sub(r"\s+", " ", str(case["query"]).strip().lower())
            for case in cases
        },
        "expected_node_ids": {str(case["expected_node_id"]) for case in cases},
        "relation_endpoints": relation_endpoints,
        "source_urls": {str(case["source_url"]) for case in cases},
    }


def _overlap(
    left: dict[str, set[str]], right: dict[str, set[str]]
) -> dict[str, list[str]]:
    return {
        key: sorted(left[key] & right[key])
        for key in sorted(set(left) & set(right))
    }


def build_audit(
    *,
    development_fixture: Path,
    development_gold: Path,
    holdout_fixture: Path,
    holdout_gold: Path,
    prior_fixtures: list[Path],
) -> dict[str, Any]:
    development = _fixture_identifiers(development_fixture)
    holdout = _fixture_identifiers(holdout_fixture)
    development_gold_ids = _gold_identifiers(development_gold)
    holdout_gold_ids = _gold_identifiers(holdout_gold)
    checks = {
        "new_split_fixture_overlap": _overlap(development, holdout),
        "new_split_gold_overlap": _overlap(
            development_gold_ids, holdout_gold_ids
        ),
    }
    prior_inputs: list[dict[str, str]] = []
    for index, prior_path in enumerate(prior_fixtures, start=1):
        prior = _fixture_identifiers(prior_path)
        checks[f"development_vs_prior_{index}"] = _overlap(development, prior)
        checks[f"holdout_vs_prior_{index}"] = _overlap(holdout, prior)
        prior_inputs.append(
            {
                "fixture": prior_path.name,
                "fixture_sha256": _canonical_sha256(prior_path),
            }
        )
    passed = all(
        not values for check in checks.values() for values in check.values()
    )
    return {
        "schema_version": 1,
        "passed": passed,
        "inputs": {
            "development_fixture_sha256": _canonical_sha256(development_fixture),
            "development_gold_sha256": _canonical_sha256(development_gold),
            "holdout_fixture_sha256": _canonical_sha256(holdout_fixture),
            "holdout_gold_sha256": _canonical_sha256(holdout_gold),
            "prior_fixtures": prior_inputs,
        },
        "checks": checks,
        "prior_usage": (
            "fixture identifiers only; prior gold and result artifacts are not loaded"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit independent-channel fixtures for contamination."
    )
    parser.add_argument("--development-fixture", type=Path, required=True)
    parser.add_argument("--development-gold", type=Path, required=True)
    parser.add_argument("--holdout-fixture", type=Path, required=True)
    parser.add_argument("--holdout-gold", type=Path, required=True)
    parser.add_argument(
        "--prior-fixture", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite an audit artifact: {args.output}")
    audit = build_audit(
        development_fixture=args.development_fixture,
        development_gold=args.development_gold,
        holdout_fixture=args.holdout_fixture,
        holdout_gold=args.holdout_gold,
        prior_fixtures=args.prior_fixture,
    )
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if not audit["passed"]:
        raise SystemExit("Contamination audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
