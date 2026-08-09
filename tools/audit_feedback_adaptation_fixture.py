from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256((json.dumps(_read(path), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()


def _identifiers(fixture: Path, gold: Path) -> dict[str, set[str]]:
    corpus = _read(fixture)
    schedule = _read(gold)
    cases = schedule.get("scoring_cases")
    if not isinstance(cases, list):
        raise ValueError(f"Gold lacks scoring_cases: {gold}")
    return {
        "doc_paths": {str(node.get("metadata", {}).get("doc_path", "")) for node in corpus["nodes"]},
        "node_ids": {str(node["node_id"]) for node in corpus["nodes"]},
        "source_urls": {str(case["source_url"]) for case in cases},
        "queries": {str(case["query"]).strip().lower() for case in cases} | {str(schedule["feedback"]["query"]).strip().lower()},
        "endpoints": {str(value) for edge in corpus["edges"] for value in (edge["source_id"], edge["target_id"])},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit frozen feedback adaptation split contamination.")
    parser.add_argument("--development-fixture", type=Path, required=True)
    parser.add_argument("--development-gold", type=Path, required=True)
    parser.add_argument("--holdout-fixture", type=Path, required=True)
    parser.add_argument("--holdout-gold", type=Path, required=True)
    parser.add_argument("--prior-fixture", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite an audit artifact: {args.output}")
    development = _identifiers(args.development_fixture, args.development_gold)
    holdout = _identifiers(args.holdout_fixture, args.holdout_gold)
    checks: dict[str, dict[str, list[str]]] = {
        "development_vs_holdout": {key: sorted(development[key] & holdout[key]) for key in development},
    }
    prior_inputs: list[dict[str, str]] = []
    for index, path in enumerate(args.prior_fixture, start=1):
        corpus = _read(path)
        prior = {
            "doc_paths": {str(node.get("metadata", {}).get("doc_path", "")) for node in corpus["nodes"]},
            "node_ids": {str(node["node_id"]) for node in corpus["nodes"]},
            "source_urls": {str(node.get("metadata", {}).get("source_url", "")) for node in corpus["nodes"]},
            "queries": set(),
            "endpoints": {str(value) for edge in corpus["edges"] for value in (edge["source_id"], edge["target_id"])},
        }
        checks[f"development_vs_prior_{index}"] = {key: sorted(development[key] & prior[key]) for key in prior}
        checks[f"holdout_vs_prior_{index}"] = {key: sorted(holdout[key] & prior[key]) for key in prior}
        prior_inputs.append({"fixture": path.name, "fixture_sha256": _sha(path)})
    result = {
        "schema_version": 1,
        "passed": all(not values for check in checks.values() for values in check.values()),
        "inputs": {"development_fixture_sha256": _sha(args.development_fixture), "development_gold_sha256": _sha(args.development_gold), "holdout_fixture_sha256": _sha(args.holdout_fixture), "holdout_gold_sha256": _sha(args.holdout_gold), "prior_fixtures": prior_inputs},
        "checks": checks,
        "prior_usage": "fixture identifiers only; prior gold and result artifacts are not loaded",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    if not result["passed"]:
        raise SystemExit("Contamination audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
