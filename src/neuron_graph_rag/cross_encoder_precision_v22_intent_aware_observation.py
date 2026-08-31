from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from . import intent_aware_observation_engine as shared
from .intent_aware_rank_fusion import IntentAwareFusionConfig

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = Path(
    "tests/fixtures/github_cross_encoder_precision_v22.engine-contract.json"
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes().decode("utf-8", errors="strict"))
    if not isinstance(value, dict):
        raise TypeError(f"v22 engine scaffold JSON must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_spec(contract: dict[str, Any]) -> shared.IntentAwareObservationSpec:
    paths = contract["fixture_paths"]
    config = contract["fusion_config"]
    return shared.IntentAwareObservationSpec(
        protocol_id=str(contract["protocol_id"]),
        fixture_paths=shared.ObservationFixturePaths(
            corpus=Path(paths["corpus"]),
            queries=Path(paths["queries"]),
            gold=Path(paths["gold"]),
            model_registry=Path(paths["model_registry"]),
        ),
        stage_identities=tuple(contract["stage_identities"].items()),
        models=tuple(shared.ModelIdentity(**row) for row in contract["models"]),
        baseline_evidence_id=str(contract["baseline_evidence_id"]),
        protocol_gate_ids=tuple(contract["protocol_gate_ids"]),
        candidate_gate_ids=tuple(contract["candidate_gate_ids"]),
        fusion_config=IntentAwareFusionConfig(**config),
    )


def _verify_v21_registries(root: Path, contract: dict[str, Any]) -> dict[str, int]:
    registries = contract["immutable_v21_registries"]
    for relative, expected in registries.items():
        if _sha256_file(root / relative) != expected:
            raise ValueError(f"immutable v21 registry changed: {relative}")
    protocol_manifest = _read_object(
        root / "tests/fixtures/github_cross_encoder_precision_v21_observation.manifest.json"
    )
    protocol_files = protocol_manifest["v21_protocol_artifact_sha256"]
    for relative, expected in protocol_files.items():
        if _sha256_file(root / relative) != expected:
            raise ValueError(f"immutable v21 protocol artifact changed: {relative}")
    evidence_root = root / "tests/evidence/github_cross_encoder_precision_v21_observation"
    terminal_manifest = _read_object(evidence_root / "terminal-evidence-manifest.json")
    evidence_files = terminal_manifest["files_sha256"]
    for relative, expected in evidence_files.items():
        if _sha256_file(evidence_root / relative) != expected:
            raise ValueError(f"immutable v21 evidence artifact changed: {relative}")
    return {
        "v21_protocol_artifact_count": len(protocol_files),
        "v21_evidence_artifact_count": len(evidence_files),
    }


def validate_result_free(root: Path = ROOT) -> dict[str, Any]:
    contract = _read_object(root / CONTRACT)
    spec = build_spec(contract)
    spec.validate()
    if (
        contract.get("phase") != "result-free-engine-scaffold"
        or contract.get("performance") != "not assessed"
        or contract.get("model_execution_count") != 0
        or contract.get("result_count") != 0
    ):
        raise ValueError("v22 scaffold is not result-free")
    evidence = root / str(contract["evidence_path"])
    if evidence.exists():
        raise FileExistsError("v22 scaffold must not contain performance evidence")
    registries = _verify_v21_registries(root, contract)
    return {
        "protocol_id": spec.protocol_id,
        "status": "result-free-engine-scaffold-valid",
        "performance": "not assessed",
        "model_execution_count": 0,
        "result_count": 0,
        "v21_source_and_evidence_immutable": True,
        **registries,
    }


ENGINE_SPEC = build_spec(_read_object(ROOT / CONTRACT))
ENGINE = shared.IntentAwareObservationEngine(ENGINE_SPEC)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate result-free intent-aware observation scaffold v22"
    )
    parser.add_argument("command", choices=("validate",))
    arguments = parser.parse_args(argv)
    if arguments.command != "validate":
        raise ValueError("v22 scaffold exposes no performance command")
    print(json.dumps(validate_result_free(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
