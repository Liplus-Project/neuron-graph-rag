"""Corrected-hash adapter for the #240 finalizer-only recovery."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import e5_structural_centroid_recovery as impl

PROTOCOL_ID = "github-retrieval-parity-v5-e5-structural-centroid-finalizer-recovery-v2"
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path("tests/fixtures/e5_structural_centroid_recovery_v2.manifest.json")
SCHEMA = Path("tests/fixtures/e5_structural_centroid_recovery_v2.schema.json")
RECOVERY_EVIDENCE = Path("tests/evidence/e5_structural_centroid_finalizer_recovery_v2")
CLAIM = RECOVERY_EVIDENCE / "development.claim.json"
RESULT = RECOVERY_EVIDENCE / "development.recovered.json"
V1_RECOVERY_EVIDENCE = Path(
    "tests/evidence/e5_structural_centroid_finalizer_recovery_v1"
)
V1_RECOVERY_CLAIM = V1_RECOVERY_EVIDENCE / "development.claim.json"
V1_RECOVERY_ERROR = V1_RECOVERY_EVIDENCE / "development.error.json"


def _bind() -> None:
    impl.PROTOCOL_ID = PROTOCOL_ID
    impl.MANIFEST = MANIFEST
    impl.SCHEMA = SCHEMA
    impl.RECOVERY_EVIDENCE = RECOVERY_EVIDENCE
    impl.CLAIM = CLAIM
    impl.RESULT = RESULT


def _verify_v1_recovery_failure(root: Path) -> None:
    manifest = impl.read_json(root / MANIFEST)
    expected = manifest["recovery_v1_evidence_sha256"]
    actual = {relative: impl.sha256_file(root / relative) for relative in expected}
    if actual != expected:
        raise ValueError("recovery-v1 failure evidence hash mismatch")
    claim_value = impl.read_json(root / V1_RECOVERY_CLAIM)
    error_value = impl.read_json(root / V1_RECOVERY_ERROR)
    if claim_value.get("status") != "worker_packet_complete_gold_absent":
        raise ValueError("recovery-v1 claim did not prove packet completeness")
    if (
        claim_value.get("worker_packet_sha256")
        != manifest["source_evidence_sha256"][impl.WORKER.as_posix()]
    ):
        raise ValueError("recovery-v1 claim/worker binding mismatch")
    if error_value.get("status") != "failed_pre_rank_gold_hash_gate":
        raise ValueError("recovery-v1 failure status mismatch")
    if error_value.get("result_present") is not False:
        raise ValueError("recovery-v1 failure evidence claims a result")
    if (
        error_value.get("registered_query_execution_count") != 0
        or error_value.get("model_forward_inference_count") != 0
    ):
        raise ValueError("recovery-v1 failure performed registered inference")
    if (root / V1_RECOVERY_EVIDENCE / "development.recovered.json").exists():
        raise ValueError("recovery-v1 result must remain absent")


def claim(root: Path = ROOT) -> dict[str, Any]:
    _bind()
    _verify_v1_recovery_failure(root)
    return impl.claim(root)


def finalize(root: Path = ROOT) -> dict[str, Any]:
    _bind()
    _verify_v1_recovery_failure(root)
    return impl.finalize(root)


def audit(root: Path = ROOT) -> dict[str, Any]:
    _bind()
    _verify_v1_recovery_failure(root)
    return impl.audit(root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("claim", "finalize", "audit"):
        command = sub.add_parser(name)
        command.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.command == "claim":
        result = claim(args.root)
    elif args.command == "finalize":
        result = finalize(args.root)
    else:
        result = audit(args.root)
    sys.stdout.buffer.write(impl.canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
