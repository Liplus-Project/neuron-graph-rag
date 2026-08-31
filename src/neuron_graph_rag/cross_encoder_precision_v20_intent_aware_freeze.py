from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .intent_aware_rank_fusion import PRODUCTION_SIGNAL_KEYS

PROTOCOL_ID = "github-ngr-cross-encoder-precision-v20"
PREDECESSOR_MERGE_COMMIT = "2f5f5d7d658681a495bcaab05e8729b567db9fc7"
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path("tests/fixtures/github_cross_encoder_precision_v20.manifest.json")
INTENT_CONTRACT = Path(
    "tests/fixtures/github_cross_encoder_precision_v20.intent-aware-contract.json"
)
GATE_OWNERSHIP = Path(
    "tests/fixtures/github_cross_encoder_precision_v20.gate-ownership.json"
)
FUTURE_IDENTITIES = Path(
    "tests/fixtures/github_cross_encoder_precision_v20.future-identities.json"
)
RESULT_FREE_AUDIT = Path(
    "tests/fixtures/github_cross_encoder_precision_v20.result-free-audit.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_bytes().decode("utf-8", errors="strict"))


def _object(root: Path, relative: Path, label: str) -> dict[str, Any]:
    value = read_json(root / relative)
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _verify_registry(
    root: Path, registry: object, *, label: str, expected_count: int | None = None
) -> dict[str, str]:
    if not isinstance(registry, dict):
        raise TypeError(f"{label} must be an object")
    if expected_count is not None and len(registry) != expected_count:
        raise ValueError(f"{label} must contain exactly {expected_count} files")
    observed = {}
    for relative, expected in registry.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise TypeError(f"{label} entries must be strings")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"{label} file missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{label} file changed: {relative}")
        observed[relative] = actual
    return observed


def _assert_no_ranking_leakage(value: object, *, path: str = "contract") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("gold", "expected", "forbidden")):
                raise ValueError(f"evaluation-only ranking key at {path}.{key}")
            _assert_no_ranking_leakage(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_ranking_leakage(child, path=f"{path}[{index}]")


def validate_prebuild(root: Path = ROOT) -> dict[str, Any]:
    manifest = _object(root, MANIFEST, "v20 manifest")
    contract = _object(root, INTENT_CONTRACT, "v20 intent contract")
    ownership = _object(root, GATE_OWNERSHIP, "v20 gate ownership")
    identities = _object(root, FUTURE_IDENTITIES, "v20 future identities")
    audit = _object(root, RESULT_FREE_AUDIT, "v20 result-free audit")
    expected_manifest = {
        "protocol_id": PROTOCOL_ID,
        "phase": "intent-aware-rank-gate-freeze",
        "predecessor_merge_commit": PREDECESSOR_MERGE_COMMIT,
        "result_free_only": True,
        "model_execution_allowed": False,
        "performance": "not assessed",
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(f"v20 manifest mismatch: {key}")
    predecessor = _verify_registry(
        root,
        manifest.get("predecessor_immutable_sha256"),
        label="v19 predecessor registry",
        expected_count=30,
    )
    artifacts = _verify_registry(
        root,
        manifest.get("protocol_artifact_sha256"),
        label="v20 protocol registry",
    )
    if (
        contract.get("protocol_id") != PROTOCOL_ID
        or contract.get("helper_module")
        != "neuron_graph_rag.intent_aware_rank_fusion"
    ):
        raise ValueError("v20 intent contract protocol mismatch")
    if contract.get("production_signal_keys") != [
        "source_path",
        "prefilter_rank",
        "prefilter_score",
        "positive_logit",
        "exclusion_logits",
        "relation_paths",
    ] or set(contract["production_signal_keys"]) != PRODUCTION_SIGNAL_KEYS:
        raise ValueError("v20 production signal registry mismatch")
    ranking_contract = contract.get("ranking_contract")
    expected_ranking_contract = {
        "query_intent": {
            "input": "query_text",
            "positive_clause_required": True,
            "exclusion_clause_logits": "one per decomposed exclusion clause",
            "languages": ["en", "ja"],
        },
        "fusion": {
            "prefilter_rank_weight": 0.15,
            "prefilter_score_weight": 0.15,
            "positive_logit_weight": 1.0,
            "exclusion_penalty_weight": 1.0,
            "relation_path_bonus": 0.25,
            "rrf_constant": 60,
            "top_k": 5,
            "tie_break": "source_path ascending",
        },
        "relation_processing": {
            "bonus_requires_query_relation_intent": True,
            "path_target_must_equal_candidate_source": True,
            "returned_paths_preserved_byte-for-byte": True,
        },
    }
    if ranking_contract != expected_ranking_contract:
        raise ValueError("v20 ranking contract literal mismatch")
    _assert_no_ranking_leakage(ranking_contract)
    protocol_gates = ownership.get("protocol_validity_gates")
    candidate_gates = ownership.get("candidate_controllable_gates")
    if ownership.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("v20 gate ownership protocol mismatch")
    if protocol_gates != [
        "protocol-source-contract-integrity",
        "identity-separation",
        "baseline-prefilter-validity",
        "relation-source-edge-only-provenance",
        "production-signal-only",
        "default-surface-immutability",
    ] or candidate_gates != [
        "positive-case-rank-non-regression",
        "positive-cohort-mrr-hit-at-5-non-regression",
        "negative-non-worsening-and-aggregate-strict-improvement",
        "positive-expected-source-top-5-completeness",
        "intent-aware-fusion-rank-only-recomputation",
        "relation-path-preservation",
    ]:
        raise ValueError("v20 gate ownership registry mismatch")
    if set(protocol_gates) & set(candidate_gates):
        raise ValueError("protocol and candidate gate ownership must be disjoint")
    if "relation-source-edge-only-provenance" not in protocol_gates:
        raise ValueError("baseline relation provenance must be a protocol gate")
    if "relation-source-edge-only-provenance" in candidate_gates:
        raise ValueError("baseline relation provenance cannot be candidate-owned")
    if "positive-case-rank-non-regression" not in candidate_gates:
        raise ValueError("positive per-case non-regression cannot be relaxed")
    rule = ownership.get("positive_case_non_regression")
    if rule != {
        "comparator": "candidate_rank <= baseline_rank",
        "when": "baseline expected source is in top_k",
        "missing_candidate": "fail",
        "cohort_average_can_mask_case_regression": False,
    }:
        raise ValueError("positive per-case non-regression rule changed")
    if (
        identities.get("protocol_id") != PROTOCOL_ID
        or identities.get("identity_schema")
        != "ngr.intent-aware-next-observation/v1"
    ):
        raise ValueError("v20 future identity schema mismatch")
    development = identities.get("development")
    holdout = identities.get("holdout")
    if not isinstance(development, str) or not isinstance(holdout, str):
        raise TypeError("v20 future identities must be strings")
    if development == holdout or not all(
        value.startswith("github-ngr-cross-encoder-precision-v20-")
        for value in (development, holdout)
    ):
        raise ValueError("v20 future identities must be fresh and separated")
    if identities.get("v19_case_role") != "diagnostic-design-input-only":
        raise ValueError("v19 cases cannot be future performance evidence")
    zero_counts = (
        "model_import_count",
        "model_load_count",
        "model_forward_inference_count",
        "registered_query_execution_count",
        "development_claim_count",
        "holdout_claim_count",
        "worker_process_count",
        "observed_result_count",
        "shared_database_open_count",
        "runtime_volume_create_count",
        "retry_count",
    )
    for key in zero_counts:
        if audit.get(key) != 0:
            raise ValueError(f"v20 result-free count must be zero: {key}")
    if (
        audit.get("protocol_id") != PROTOCOL_ID
        or audit.get("status") != "pass"
        or audit.get("performance") != "not assessed"
        or audit.get("v19_artifacts_unchanged") is not True
        or audit.get("v19_results_reused_as_performance_evidence") is not False
    ):
        raise ValueError("v20 result-free audit boundary mismatch")
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "pass",
        "phase": "intent-aware-rank-gate-freeze",
        "predecessor_artifact_count": len(predecessor),
        "protocol_artifact_count": len(artifacts),
        "protocol_validity_gate_count": len(protocol_gates),
        "candidate_controllable_gate_count": len(candidate_gates),
        "model_forward_inference_count": 0,
        "observed_result_count": 0,
        "performance": "not assessed",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "validate-prebuild"))
    args = parser.parse_args(argv)
    if args.command in {"audit", "validate-prebuild"}:
        print(json.dumps(validate_prebuild(), sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
