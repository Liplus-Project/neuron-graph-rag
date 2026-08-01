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
    if not isinstance(nodes, list):
        raise ValueError(f"Fixture lacks nodes: {path}")
    return {
        "doc_paths": {
            str(node.get("metadata", {}).get("doc_path", ""))
            for node in nodes
            if str(node.get("metadata", {}).get("doc_path", ""))
        },
        "node_ids": {str(node["node_id"]) for node in nodes},
        "source_urls": {
            str(node.get("metadata", {}).get("source_url", ""))
            for node in nodes
            if str(node.get("metadata", {}).get("source_url", ""))
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
            relation_endpoints.add(str(step["source_id"]))
            relation_endpoints.add(str(step["target_id"]))
    return {
        "normalized_queries": {
            re.sub(r"\s+", " ", str(case["query"]).strip().lower())
            for case in cases
        },
        "expected_node_ids": {str(case["expected_node_id"]) for case in cases},
        "relation_endpoints": relation_endpoints,
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
    prior_development_fixture: Path,
    prior_holdout_fixture: Path,
) -> dict[str, Any]:
    development = _fixture_identifiers(development_fixture)
    holdout = _fixture_identifiers(holdout_fixture)
    prior_development = _fixture_identifiers(prior_development_fixture)
    prior_holdout = _fixture_identifiers(prior_holdout_fixture)
    development_gold_ids = _gold_identifiers(development_gold)
    holdout_gold_ids = _gold_identifiers(holdout_gold)
    checks = {
        "new_split_fixture_overlap": _overlap(development, holdout),
        "new_split_gold_overlap": _overlap(
            development_gold_ids, holdout_gold_ids
        ),
        "development_vs_prior_development": _overlap(
            development, prior_development
        ),
        "development_vs_prior_holdout": _overlap(development, prior_holdout),
        "holdout_vs_prior_development": _overlap(holdout, prior_development),
        "holdout_vs_prior_holdout": _overlap(holdout, prior_holdout),
    }
    passed = all(
        not values
        for check in checks.values()
        for values in check.values()
    )
    return {
        "schema_version": 1,
        "passed": passed,
        "inputs": {
            "development_fixture_sha256": _canonical_sha256(development_fixture),
            "development_gold_sha256": _canonical_sha256(development_gold),
            "holdout_fixture_sha256": _canonical_sha256(holdout_fixture),
            "holdout_gold_sha256": _canonical_sha256(holdout_gold),
            "prior_development_fixture_sha256": _canonical_sha256(
                prior_development_fixture
            ),
            "prior_holdout_fixture_sha256": _canonical_sha256(
                prior_holdout_fixture
            ),
        },
        "checks": checks,
        "old_holdout_usage": (
            "fixture identifiers only; old holdout gold and result are not loaded"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit frozen local-competition fixtures for contamination."
    )
    parser.add_argument("--development-fixture", type=Path, required=True)
    parser.add_argument("--development-gold", type=Path, required=True)
    parser.add_argument("--holdout-fixture", type=Path, required=True)
    parser.add_argument("--holdout-gold", type=Path, required=True)
    parser.add_argument("--prior-development-fixture", type=Path, required=True)
    parser.add_argument("--prior-holdout-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite an audit artifact: {args.output}")
    audit = build_audit(
        development_fixture=args.development_fixture,
        development_gold=args.development_gold,
        holdout_fixture=args.holdout_fixture,
        holdout_gold=args.holdout_gold,
        prior_development_fixture=args.prior_development_fixture,
        prior_holdout_fixture=args.prior_holdout_fixture,
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
