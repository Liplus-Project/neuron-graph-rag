"""Frozen, create-only longitudinal feedback trajectory evaluator for corpus v3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .corpus_integrity import verify_source_sha256


PROTOCOL = "longitudinal-feedback-trajectory-v1"
ARTIFACTS = (
    "longitudinal_feedback_trajectory_v3.fixture.json",
    "longitudinal_feedback_trajectory_v3.gold.json",
    "longitudinal_feedback_trajectory_v3.schedule.json",
    "longitudinal_feedback_trajectory_v3.gate.json",
)
MANIFEST = "longitudinal_feedback_trajectory_v3.manifest.json"
OBSERVED = {
    "development": "longitudinal_feedback_trajectory_v3.development.observed.json",
    "holdout": "longitudinal_feedback_trajectory_v3.holdout.observed.json",
}


class ProtocolError(ValueError):
    """The frozen protocol is malformed, altered, or cannot be safely run."""


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_protocol(root: Path) -> dict[str, Any]:
    fixture_root = root / "tests" / "fixtures"
    artifacts = {
        name: _load_json(fixture_root / name)
        for name in (*ARTIFACTS, MANIFEST)
    }
    for name, artifact in artifacts.items():
        if artifact.get("protocol") != PROTOCOL:
            raise ProtocolError(f"{name} does not declare {PROTOCOL}")
    _verify_manifest_hashes(fixture_root, artifacts[MANIFEST])
    _verify_source_integrity(root, artifacts["longitudinal_feedback_trajectory_v3.fixture.json"])
    _verify_identity_isolation(artifacts)
    artifacts[MANIFEST]["_raw_sha256"] = sha256_file(fixture_root / MANIFEST)
    return artifacts


def run_registered_split(root: Path, split: str) -> dict[str, Any]:
    """Run one registered split exactly once, saving either PASS or FAIL evidence."""

    if split not in OBSERVED:
        raise ProtocolError("split must be development or holdout")
    artifacts = load_frozen_protocol(root)
    fixture_root = root / "tests" / "fixtures"
    output_path = fixture_root / OBSERVED[split]
    if output_path.exists():
        raise ProtocolError(f"exclusive output already exists: {output_path.name}")

    handoff: dict[str, Any] | None = None
    if split == "holdout":
        handoff = _verify_development_handoff(fixture_root, artifacts[MANIFEST])

    result = _evaluate_split(split, artifacts, handoff)
    _write_create_only(output_path, result)
    return result


def _evaluate_split(
    split: str, artifacts: dict[str, Any], handoff: dict[str, Any] | None
) -> dict[str, Any]:
    fixture = artifacts["longitudinal_feedback_trajectory_v3.fixture.json"]
    gold = artifacts["longitudinal_feedback_trajectory_v3.gold.json"]
    schedule = artifacts["longitudinal_feedback_trajectory_v3.schedule.json"]
    gate = artifacts["longitudinal_feedback_trajectory_v3.gate.json"]
    manifest = artifacts[MANIFEST]
    role = fixture["roles"][split]
    target = gold["targets"][split]
    counts = schedule["feedback_counts"]
    increment = schedule["credited_edge_increment"]
    edge = role["credited_edge"]
    candidates = [
        document
        for document in fixture["source_documents"]
        if document["cluster"] == role["cluster"]
        and "-credit-" in document["node_id"]
        and document["credit_ceiling"] in counts
    ]

    points = []
    for count in counts:
        control_rank = _rank_target(candidates, target, None, 0.0)
        treatment_rank = _rank_target(candidates, target, edge, count * increment)
        points.append(
            {
                "credit_count": count,
                "control": {
                    "mrr": 1.0 / control_rank,
                    "target_rank": control_rank,
                    "mutated_edges": [],
                },
                "treatment": {
                    "mrr": 1.0 / treatment_rank,
                    "target_rank": treatment_rank,
                    "mutated_edges": [] if count == 0 else [edge["identity"]],
                },
            }
        )
    gate_evidence = _apply_gate(points, edge["identity"], gate)
    result: dict[str, Any] = {
        "protocol": PROTOCOL,
        "split": split,
        "status": "passed" if gate_evidence["passed"] else "failed",
        "freeze_manifest_hash": manifest["_raw_sha256"],
        "source_commit": fixture["source_commit"],
        "source_hash_verification": "passed",
        "points": points,
        "gate_evidence": gate_evidence,
    }
    if handoff is not None:
        result["development_to_holdout_handoff"] = handoff
    return result


def _rank_target(
    candidates: list[dict[str, Any]],
    target: str,
    credited_edge: dict[str, str] | None,
    mutation: float,
) -> int:
    scores: list[tuple[float, str]] = []
    ordered_ceilings = sorted({document["credit_ceiling"] for document in candidates})
    for document in candidates:
        score = float(len(ordered_ceilings) - 1 - ordered_ceilings.index(document["credit_ceiling"]))
        if credited_edge and document["node_id"] == credited_edge["target_node"]:
            score += mutation
        scores.append((score, document["node_id"]))
    ranked = sorted(scores, key=lambda item: (-item[0], item[1]))
    return next(index for index, (_, node_id) in enumerate(ranked, start=1) if node_id == target)


def _apply_gate(points: list[dict[str, Any]], edge: str, gate: dict[str, Any]) -> dict[str, Any]:
    control = [point["control"]["mrr"] for point in points]
    treatment = [point["treatment"]["mrr"] for point in points]
    expected_counts = gate["feedback_counts"]
    count_match = [point["credit_count"] for point in points] == expected_counts
    control_non_regression = all(value >= control[0] for value in control)
    treatment_non_regression = all(
        right >= left for left, right in zip(treatment, treatment[1:])
    )
    strict_headroom = treatment[-1] > treatment[0]
    ceiling_respected = all(value <= gate["mrr_ceiling"] for value in (*control, *treatment))
    mutation_scope = all(
        point["control"]["mutated_edges"] == []
        and point["treatment"]["mutated_edges"]
        == ([] if point["credit_count"] == 0 else [edge])
        for point in points
    )
    checks = {
        "feedback_counts": count_match,
        "control_non_regression": control_non_regression,
        "treatment_non_regression": treatment_non_regression,
        "treatment_strict_0_to_10_improvement": strict_headroom,
        "ceiling_respected": ceiling_respected,
        "credited_path_only_mutation": mutation_scope,
    }
    return {"checks": checks, "passed": all(checks.values())}


def _verify_development_handoff(fixture_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    development = _load_json(fixture_root / OBSERVED["development"])
    if development.get("protocol") != PROTOCOL or development.get("split") != "development":
        raise ProtocolError("development output is not this protocol's development run")
    if development.get("status") != "passed":
        raise ProtocolError("holdout is forbidden because development did not pass")
    if development.get("freeze_manifest_hash") != manifest["_raw_sha256"]:
        raise ProtocolError("development output does not hand off this frozen manifest")
    return {
        "development_status": development["status"],
        "development_manifest_hash": development["freeze_manifest_hash"],
    }


def _verify_manifest_hashes(fixture_root: Path, manifest: dict[str, Any]) -> None:
    expected = manifest.get("artifact_hashes")
    if set(expected or {}) != set(ARTIFACTS):
        raise ProtocolError("manifest must hash every non-manifest frozen artifact")
    for name, expected_hash in expected.items():
        if sha256_file(fixture_root / name) != expected_hash:
            raise ProtocolError(f"frozen artifact hash mismatch: {name}")


def _verify_source_integrity(root: Path, fixture: dict[str, Any]) -> None:
    documents = fixture.get("source_documents", [])
    if len(documents) != 15:
        raise ProtocolError("fixture must identify all 15 v3 source documents")
    for document in documents:
        path = root / document["path"]
        verification = verify_source_sha256(path, document["sha256"])
        if not verification.accepted:
            raise ProtocolError(f"source integrity failed: {document['path']}")


def _verify_identity_isolation(artifacts: dict[str, Any]) -> None:
    fixture = artifacts["longitudinal_feedback_trajectory_v3.fixture.json"]
    roles = fixture.get("roles", {})
    if set(roles) != {"development", "holdout", "trajectory_audit"}:
        raise ProtocolError("fixture must define development, holdout, and trajectory_audit")
    clusters = [role["cluster"] for role in roles.values()]
    if len(set(clusters)) != 3:
        raise ProtocolError("development, holdout, and trajectory clusters must differ")
    source_by_node = {document["node_id"]: document for document in fixture["source_documents"]}
    identities: list[set[str]] = []
    for role in roles.values():
        edge = role["credited_edge"]
        relevant = [
            document for document in source_by_node.values() if document["cluster"] == role["cluster"]
        ]
        role_identities = {
            *(document["node_id"] for document in relevant),
            *(document["path"] for document in relevant),
            *(document["source_url"] for document in relevant),
            edge["identity"],
        }
        if edge["source_node"] not in source_by_node or edge["target_node"] not in source_by_node:
            raise ProtocolError("credited edge references an unknown node")
        identities.append(role_identities)
    if any(left & right for index, left in enumerate(identities) for right in identities[index + 1 :]):
        raise ProtocolError("role identities are not mutually disjoint")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot load {path.name}: {error}") from error


def _write_create_only(path: Path, result: dict[str, Any]) -> None:
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as output:
            output.write(payload)
    except FileExistsError as error:
        raise ProtocolError(f"exclusive output already exists: {path.name}") from error
