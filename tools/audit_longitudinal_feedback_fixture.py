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
    return "sha256:" + hashlib.sha256(
        (json.dumps(_read(path), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ).hexdigest()


def _identifiers(fixture: Path, gold: Path | None = None) -> dict[str, set[str]]:
    corpus = _read(fixture)
    values = {
        "doc_paths": {str(node.get("metadata", {}).get("doc_path", "")) for node in corpus["nodes"]},
        "node_ids": {str(node["node_id"]) for node in corpus["nodes"]},
        "source_urls": {str(node.get("metadata", {}).get("source_url", "")) for node in corpus["nodes"]},
        "endpoints": {str(value) for edge in corpus["edges"] for value in (edge["source_id"], edge["target_id"])},
        "edge_identities": {"\u0000".join((str(edge["source_id"]), str(edge["target_id"]), str(edge["edge_type"]))) for edge in corpus["edges"]},
        "queries": set(),
    }
    if gold is not None:
        schedule = _read(gold)
        values["source_urls"] = {str(case["source_url"]) for case in schedule["scoring_cases"]}
        values["queries"] = {str(case["query"]).strip().lower() for case in schedule["scoring_cases"]}
        values["queries"].update(str(event["query"]).strip().lower() for event in schedule["feedback_schedule"])
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit disjoint longitudinal feedback corpus clusters.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prior-fixture", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite an audit artifact: {args.output}")
    manifest = _read(args.manifest)
    base = args.manifest.parent
    current: dict[str, dict[str, set[str]]] = {}
    inputs: list[dict[str, str]] = []
    for stage in ("development", "holdout"):
        for cluster in manifest[stage]["clusters"]:
            fixture = base / str(cluster["fixture"])
            gold = base / str(cluster["gold"])
            key = f"{stage}:{cluster['cluster_id']}"
            current[key] = _identifiers(fixture, gold)
            inputs.append({"cluster": key, "fixture_sha256": _sha(fixture), "gold_sha256": _sha(gold)})
    checks: dict[str, dict[str, list[str]]] = {}
    keys = sorted(current)
    for index, left in enumerate(keys):
        for right in keys[index + 1:]:
            checks[f"{left}_vs_{right}"] = {name: sorted(current[left][name] & current[right][name]) for name in current[left]}
    prior_inputs: list[dict[str, str]] = []
    for number, fixture in enumerate(args.prior_fixture, start=1):
        prior = _identifiers(fixture)
        prior_inputs.append({"fixture": fixture.name, "fixture_sha256": _sha(fixture)})
        for key in keys:
            checks[f"{key}_vs_prior_{number}"] = {name: sorted(current[key][name] & prior[name]) for name in prior}
    result: dict[str, Any] = {
        "schema_version": 1,
        "passed": all(not overlap for check in checks.values() for overlap in check.values()),
        "inputs": {"clusters": inputs, "prior_fixtures": prior_inputs},
        "checks": checks,
        "prior_usage": "fixture identifiers only; prior gold and observed result artifacts are not loaded",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    if not result["passed"]:
        raise SystemExit("Contamination audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
