"""Verification bridge for the one-time retrieval parity v5 observation."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from . import github_retrieval_parity_v5 as parity

ROOT = Path(__file__).resolve().parents[2]
FREEZE_COMMIT = "ef1c14d573fde04913b86b9ef2f8ad8b8621497c"
EVIDENCE_ROOT = Path("tests/evidence/github_retrieval_parity_v5")
EVIDENCE_SHA256 = {
    "development.preflight.json": (
        "fbc74c135127ad972142e67a521cfd450050e1ea1569b8ae77e657596ee65f19"
    ),
    "development.capture.json": (
        "92b93069fb3434d9f89e8c3dd7f444f503c77b9ca1a44a92ac08a32d97e56600"
    ),
    "development.claim.json": (
        "9739ccb18d6eb2d33a6fdfcf31820ec663fe7c5b479609dccd4aaae65d51b7f8"
    ),
    "development.observed.json": (
        "9fdb094f246dcd2293481c46635cedd068fc0790c986a20163519e7933632e7c"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _assert_exact_registry(evidence: Path) -> None:
    expected = set(EVIDENCE_SHA256)
    actual = {path.name for path in evidence.iterdir() if path.is_file()}
    if actual != expected:
        raise ValueError(
            f"v5 observation registry mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def extract_freeze_checkout(destination: Path, root: Path = ROOT) -> None:
    """Materialize the immutable freeze commit without creating a worktree."""

    archive = subprocess.run(
        ["git", "archive", "--format=tar", FREEZE_COMMIT],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        bundle.extractall(destination)


def run_frozen_v5_tests(root: Path = ROOT) -> str:
    """Run every frozen v5 protocol test in an immutable temporary checkout."""

    with TemporaryDirectory() as directory:
        frozen_root = Path(directory)
        extract_freeze_checkout(frozen_root, root)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(frozen_root / "src")
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "tests.test_github_retrieval_parity_v5",
                "-v",
            ],
            cwd=frozen_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
        )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0 or "Ran 11 tests" not in output:
        raise AssertionError(
            "frozen v5 test suite did not pass all 11 tests\n" + output
        )
    return output


def verify_observation(root: Path = ROOT) -> dict[str, Any]:
    """Verify the recorded observation against untouched freeze bytes."""

    evidence = root / EVIDENCE_ROOT
    _assert_exact_registry(evidence)
    actual_hashes = {
        name: _sha256(evidence / name) for name in EVIDENCE_SHA256
    }
    if actual_hashes != EVIDENCE_SHA256:
        raise ValueError("v5 observation evidence SHA-256 mismatch")

    parity.load_protocol(root, require_result_free=False)
    parity.verify_registered_preflight("development", root)
    parity.verify_registered_result("development", root)
    current_lifecycle = parity.audit_repository_lifecycle(root)
    if current_lifecycle["phase"] != "development-closed" or any(
        current_lifecycle["holdout"].values()
    ):
        raise ValueError("current v5 registry is not closed development-only evidence")

    with TemporaryDirectory() as directory:
        frozen_root = Path(directory)
        extract_freeze_checkout(frozen_root, root)
        frozen_evidence = frozen_root / EVIDENCE_ROOT
        frozen_evidence.mkdir(parents=True, exist_ok=True)
        for name in EVIDENCE_SHA256:
            shutil.copy2(evidence / name, frozen_evidence / name)

        parity.verify_registered_preflight("development", frozen_root)
        parity.verify_registered_result("development", frozen_root)
        frozen_lifecycle = parity.audit_repository_lifecycle(frozen_root)
        preflight = _json(frozen_evidence / "development.preflight.json")
        capture = _json(frozen_evidence / "development.capture.json")
        claim = _json(frozen_evidence / "development.claim.json")
        result = _json(frozen_evidence / "development.observed.json")

    if frozen_lifecycle != current_lifecycle:
        raise ValueError("current and frozen-root lifecycle verification diverged")
    if frozen_lifecycle["phase"] != "development-closed":
        raise ValueError("v5 development observation is not closed")
    if any(frozen_lifecycle["holdout"].values()):
        raise ValueError("v5 holdout must remain unopened")
    if len(preflight["documents"]) != 93 or len(capture["cases"]) != 5:
        raise ValueError("v5 development evidence count mismatch")
    if claim["protocol_commit"] != FREEZE_COMMIT:
        raise ValueError("v5 claim does not identify the freeze commit")
    passed_gates = sum(bool(gate["passed"]) for gate in result["gates"])
    if (
        result["status"] != "failed"
        or result["failure_code"] != "hard-gate-failed"
        or result["all_hard_gates_pass"] is not False
        or len(result["gates"]) != 10
        or passed_gates != 6
    ):
        raise ValueError("v5 development hard-gate outcome mismatch")

    return {
        "protocol_id": parity.PROTOCOL_ID,
        "freeze_commit": FREEZE_COMMIT,
        "phase": frozen_lifecycle["phase"],
        "preflight_document_count": len(preflight["documents"]),
        "registered_query_execution_count": len(capture["cases"]),
        "hard_gates_passed": passed_gates,
        "hard_gates_total": len(result["gates"]),
        "all_hard_gates_pass": result["all_hard_gates_pass"],
        "holdout_opened": any(frozen_lifecycle["holdout"].values()),
        "evidence_sha256": actual_hashes,
    }
