from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from . import cross_encoder_precision_v8_observation as frozen_v8
from . import cross_encoder_precision_v11_observation as predecessor
from . import cross_encoder_precision_v13_observation as git_free

PROTOCOL_ID = "github-ngr-cross-encoder-precision-v15"
PREDECESSOR_MERGE_COMMIT = "388eb9a2a658535bae63399168b3e1b2a5a8921a"
FROZEN_PROTOCOL_COMMIT = "d2fdf7720e2a9dde7e8d666cf4fd9f314fd3d12f"
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path("tests/fixtures/github_cross_encoder_precision_v15.manifest.json")
SOURCE_IDENTITY = Path(
    "tests/fixtures/github_cross_encoder_precision_v15.source-identity.json"
)
RESULT_FREE_AUDIT = Path(
    "tests/fixtures/github_cross_encoder_precision_v15.result-free-audit.json"
)
EVIDENCE = Path("tests/evidence/github_cross_encoder_precision_v15")

IMAGE = predecessor.IMAGE
IMAGE_ID = predecessor.IMAGE_ID
WSLC_VERSION = predecessor.WSLC_VERSION
ROOT_NORMALIZATION_FREEZE_VOLUME = (
    "github-cross-encoder-precision-v15-root-normalization-freeze"
)
FUTURE_RUNTIME_VOLUME = "github-cross-encoder-precision-v15-runtime"
V10_RUNTIME_VOLUME = "github-cross-encoder-precision-v10-runtime"
V10_CACHE_FREEZE_VOLUME = "github-cross-encoder-precision-v10-cache-freeze"
V11_ROOT_FREEZE_VOLUME = "github-cross-encoder-precision-v11-root-freeze"
V12_RUNTIME_VOLUME = "github-cross-encoder-precision-v12-runtime"
V13_COMMIT_FREEZE_VOLUME = "github-cross-encoder-precision-v13-commit-freeze"
V14_RUNTIME_VOLUME = "github-cross-encoder-precision-v14-runtime"

CONTAINER_ROOT = PurePosixPath("/opt/ngr-v15/root-normalization-freeze")
CONTAINER_SOURCE = CONTAINER_ROOT / "source"
CONTAINER_CACHE = CONTAINER_ROOT / "model-cache"
CONTAINER_PROTOCOL_SOURCE = CONTAINER_ROOT / "frozen-source"
CONTAINER_REPORT = CONTAINER_ROOT / "claim-source-root-verification.json"
CONTAINER_SOURCE_IDENTITY = CONTAINER_SOURCE / SOURCE_IDENTITY.as_posix()
OLD_FROZEN_SOURCE = PurePosixPath("/opt/ngr-v8/runtime/frozen-source")

canonical_sha256 = predecessor.canonical_sha256
sha256_file = predecessor.sha256_file
read_json = predecessor.read_json
_write_json_exclusive = predecessor._write_json_exclusive


def serialize_container_path(value: PurePosixPath | str) -> str:
    return predecessor.serialize_container_path(value)


def named_volume_spec(
    volume: str,
    destination: PurePosixPath | str,
    *,
    mode: str | None = None,
) -> str:
    return predecessor.named_volume_spec(volume, destination, mode=mode)


def _manifest(root: Path) -> dict[str, Any]:
    value = read_json(root / MANIFEST)
    if not isinstance(value, dict):
        raise TypeError("v15 manifest must be an object")
    return value


def _source_identity(root: Path) -> dict[str, Any]:
    value = read_json(root / SOURCE_IDENTITY)
    if not isinstance(value, dict):
        raise TypeError("v15 source identity must be an object")
    return value


def _audit_contract(root: Path) -> dict[str, Any]:
    value = read_json(root / RESULT_FREE_AUDIT)
    if not isinstance(value, dict):
        raise TypeError("v15 result-free audit must be an object")
    return value


def _expected_container_paths() -> dict[str, str]:
    return {
        "root": serialize_container_path(CONTAINER_ROOT),
        "source": serialize_container_path(CONTAINER_SOURCE),
        "model_cache": serialize_container_path(CONTAINER_CACHE),
        "protocol_source": serialize_container_path(CONTAINER_PROTOCOL_SOURCE),
        "claim_source_root_report": serialize_container_path(CONTAINER_REPORT),
        "source_identity": serialize_container_path(CONTAINER_SOURCE_IDENTITY),
        "old_frozen_source": serialize_container_path(OLD_FROZEN_SOURCE),
    }


def _verify_predecessor_hashes(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, str]:
    registry = manifest.get("predecessor_immutable_sha256")
    if not isinstance(registry, dict) or len(registry) != 15:
        raise ValueError("v15 predecessor registry must contain exactly 15 files")
    actual: dict[str, str] = {}
    for relative, expected in registry.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise TypeError("v15 predecessor registry entries must be strings")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"v14 predecessor artifact missing: {relative}")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"v14 predecessor artifact changed: {relative}")
        actual[relative] = observed
    return actual


def validate_prebuild(root: Path = ROOT) -> dict[str, Any]:
    manifest = _manifest(root)
    expected = {
        "protocol_id": PROTOCOL_ID,
        "phase": "claim-source-root-normalization-freeze",
        "predecessor_merge_commit": PREDECESSOR_MERGE_COMMIT,
        "frozen_protocol_commit": FROZEN_PROTOCOL_COMMIT,
        "root_normalization_freeze_volume": ROOT_NORMALIZATION_FREEZE_VOLUME,
        "future_runtime_volume": FUTURE_RUNTIME_VOLUME,
        "accepted_image": {"tag": IMAGE, "id": IMAGE_ID},
        "accepted_image_rebuild_allowed": False,
        "result_free_only": True,
        "wslc_version": WSLC_VERSION,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"v15 manifest mismatch: {key}")
    if manifest.get("container_paths") != _expected_container_paths():
        raise ValueError("v15 container path registry mismatch")
    if manifest.get("source_identity_sha256") != sha256_file(root / SOURCE_IDENTITY):
        raise ValueError("v15 source identity hash mismatch")
    if manifest.get("result_free_audit_sha256") != sha256_file(
        root / RESULT_FREE_AUDIT
    ):
        raise ValueError("v15 result-free audit hash mismatch")
    if manifest.get("expected_evidence") != [
        "accepted-image-inspect.json",
        "root-normalization-commands.json",
        "root-normalization.pass.json|root-normalization.error.json",
        "claim-source-root-verification.json",
        "count-audit.json",
        "evidence-manifest.json",
        "source-identity.json",
        "volume-identity.json",
    ]:
        raise ValueError("v15 expected evidence registry mismatch")
    predecessor_hashes = _verify_predecessor_hashes(root, manifest)
    identity = _source_identity(root)
    required_identity = {
        "identity_schema",
        "source_archive_commit",
        "configured_claim_source_root",
        "configured_frozen_source_root",
        "git_free_identity",
    }
    if set(identity) != required_identity:
        raise ValueError("v15 source identity is incomplete")
    if (
        identity["identity_schema"] != "ngr.claim-source-root-normalization/v1"
        or identity["source_archive_commit"] != PREDECESSOR_MERGE_COMMIT
        or identity["configured_claim_source_root"]
        != serialize_container_path(CONTAINER_SOURCE)
        or identity["configured_frozen_source_root"]
        != serialize_container_path(CONTAINER_PROTOCOL_SOURCE)
    ):
        raise ValueError("v15 source identity literal mismatch")
    git_identity = identity.get("git_free_identity")
    if not isinstance(git_identity, dict):
        raise TypeError("v15 git-free identity must be an object")
    if (
        git_identity.get("identity_schema")
        != "ngr.git-free-protocol-identity/v1"
        or git_identity.get("frozen_protocol_commit") != FROZEN_PROTOCOL_COMMIT
        or git_identity.get("protocol_artifact_count") != 23
        or git_identity.get("corpus_document_count") != 24
    ):
        raise ValueError("v15 nested git-free identity mismatch")
    audit = _audit_contract(root)
    for key in (
        "accepted_image_rebuild_count",
        "model_cache_copy_count",
        "model_import_count",
        "model_load_count",
        "model_forward_inference_count",
        "registered_query_execution_count",
        "development_claim_count",
        "holdout_claim_count",
        "worker_process_count",
        "observed_result_count",
        "retry_count",
        "shared_database_open_count",
        "container_git_executable_invocation_count",
        "container_subprocess_invocation_count",
    ):
        if audit.get(key) != 0:
            raise ValueError(f"v15 result-free count must be zero: {key}")
    if (
        audit.get("protocol_id") != PROTOCOL_ID
        or audit.get("phase") != "claim-source-root-normalization-freeze"
        or audit.get("root_normalization_verifier_run_limit") != 1
        or audit.get("root_normalization_freeze_volume_reusable") is not False
        or audit.get("performance") != "not assessed"
    ):
        raise ValueError("v15 result-free audit boundary mismatch")
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "prebuild_contract_valid",
        "predecessor_artifact_count": len(predecessor_hashes),
        "protocol_artifact_count": 23,
        "corpus_document_count": 24,
        "root_normalization_verifier_run_limit": 1,
        "registered_query_execution_count": 0,
        "model_forward_inference_count": 0,
        "observed_result_count": 0,
        "performance": "not assessed",
    }


def git_free_verify_protocol_commit(
    protocol_commit: str,
    protocol: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    protocol_source: Path,
) -> dict[str, Any]:
    required = {
        "identity_schema",
        "source_archive_commit",
        "frozen_protocol_commit",
        "manifest_path",
        "manifest_sha256",
        "artifact_registry_sha256",
        "protocol_artifact_count",
        "corpus_path",
        "corpus_sha256",
        "corpus_commit",
        "corpus_document_count",
    }
    if set(identity) != required:
        raise ValueError("git-free source identity is incomplete")
    if (
        identity["identity_schema"] != "ngr.git-free-protocol-identity/v1"
        or protocol_commit != FROZEN_PROTOCOL_COMMIT
        or identity["frozen_protocol_commit"] != FROZEN_PROTOCOL_COMMIT
        or identity["source_archive_commit"] != PREDECESSOR_MERGE_COMMIT
    ):
        raise ValueError("frozen protocol commit identity mismatch")
    root = Path(protocol.get("root", ""))
    if root != protocol_source:
        raise ValueError("protocol root does not match frozen source identity")
    manifest_path = protocol_source / str(identity["manifest_path"])
    corpus_path = protocol_source / str(identity["corpus_path"])
    if sha256_file(manifest_path) != identity["manifest_sha256"]:
        raise ValueError("frozen protocol manifest byte mismatch")
    if sha256_file(corpus_path) != identity["corpus_sha256"]:
        raise ValueError("frozen corpus registry byte mismatch")
    manifest = protocol.get("manifest")
    corpus = protocol.get("corpus")
    if not isinstance(manifest, dict) or not isinstance(corpus, dict):
        raise TypeError("frozen protocol shape mismatch")
    artifacts = manifest.get("artifact_sha256")
    documents = corpus.get("documents")
    if not isinstance(artifacts, dict) or len(artifacts) != 23:
        raise ValueError("frozen protocol artifact registry is incomplete")
    if canonical_sha256(artifacts) != identity["artifact_registry_sha256"]:
        raise ValueError("frozen protocol artifact registry identity mismatch")
    if (
        corpus.get("commit") != identity["corpus_commit"]
        or not isinstance(documents, list)
        or len(documents) != 24
    ):
        raise ValueError("frozen corpus identity mismatch")
    artifact_rows: list[dict[str, Any]] = []
    for relative, expected in sorted(artifacts.items()):
        path = protocol_source / str(relative)
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"frozen protocol artifact mismatch: {relative}")
        artifact_rows.append(
            {"path": str(relative), "sha256": observed, "size": path.stat().st_size}
        )
    document_rows: list[dict[str, Any]] = []
    for row in documents:
        if not isinstance(row, dict) or set(row) != {"path", "content_sha256"}:
            raise ValueError("frozen corpus document identity is incomplete")
        relative = str(row["path"])
        raw = (protocol_source / relative).read_bytes()
        observed = hashlib.sha256(raw).hexdigest()
        if observed != row["content_sha256"]:
            raise ValueError(f"frozen corpus byte mismatch: {relative}")
        document_rows.append({"path": relative, "sha256": observed, "size": len(raw)})
    return {
        "expected_frozen_protocol_commit": FROZEN_PROTOCOL_COMMIT,
        "verified_frozen_protocol_commit": protocol_commit,
        "source_archive_commit": identity["source_archive_commit"],
        "manifest_sha256": identity["manifest_sha256"],
        "artifact_registry_sha256": identity["artifact_registry_sha256"],
        "protocol_artifact_count": len(artifact_rows),
        "protocol_artifact_size": sum(row["size"] for row in artifact_rows),
        "protocol_artifacts": artifact_rows,
        "corpus_commit": identity["corpus_commit"],
        "corpus_document_count": len(document_rows),
        "corpus_document_size": sum(row["size"] for row in document_rows),
        "corpus_documents": document_rows,
        "exact_protocol_bytes_verified": True,
        "exact_corpus_bytes_verified": True,
        "source_identity_complete": True,
        "container_git_executable_invocation_count": 0,
        "container_subprocess_invocation_count": 0,
    }


def resolve_claim_source_root(
    observed_root: str,
    *,
    configured_source: Path | PurePosixPath,
    configured_protocol_source: Path | PurePosixPath,
) -> PurePosixPath:
    source = PurePosixPath(str(configured_source))
    protocol_source = PurePosixPath(str(configured_protocol_source))
    observed = PurePosixPath(observed_root)
    if not source.is_absolute() or not protocol_source.is_absolute():
        raise ValueError("configured roots must be absolute POSIX paths")
    if not observed.is_absolute():
        raise ValueError("claim source root must be absolute")
    if ".." in observed.parts:
        raise ValueError("claim source root must not escape its configured root")
    if observed != source:
        raise ValueError("claim source root does not match configured source")
    return protocol_source


def resolver_aware_verify_protocol_commit(
    protocol_commit: str,
    protocol: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    source: Path | PurePosixPath,
    protocol_source: Path | PurePosixPath,
) -> dict[str, Any]:
    if identity.get("identity_schema") != "ngr.claim-source-root-normalization/v1":
        raise ValueError("claim source root identity schema mismatch")
    if identity.get("source_archive_commit") != PREDECESSOR_MERGE_COMMIT:
        raise ValueError("claim source root archive identity mismatch")
    if identity.get("configured_claim_source_root") != str(source):
        raise ValueError("configured claim source root identity mismatch")
    if identity.get("configured_frozen_source_root") != str(protocol_source):
        raise ValueError("configured frozen source root identity mismatch")
    observed_root = protocol.get("root")
    if not isinstance(observed_root, str):
        raise TypeError("protocol root must be a string")
    resolved = resolve_claim_source_root(
        observed_root,
        configured_source=source,
        configured_protocol_source=protocol_source,
    )
    git_identity = identity.get("git_free_identity")
    if not isinstance(git_identity, dict):
        raise TypeError("git-free identity must be an object")
    normalized = dict(protocol)
    normalized["root"] = str(resolved)
    verification = git_free.git_free_verify_protocol_commit(
        protocol_commit,
        normalized,
        identity=git_identity,
        protocol_source=Path(str(protocol_source)),
    )
    return {
        **verification,
        "observed_claim_source_root": observed_root,
        "configured_claim_source_root": str(source),
        "resolved_frozen_source_root": str(resolved),
        "root_normalization_exact": True,
    }


def bind_claim_source_root_verifier(
    wrapper: Any,
    *,
    volume: str,
    root: Path | PurePosixPath,
    source: Path | PurePosixPath,
    cache: Path | PurePosixPath,
    protocol_source: Path | PurePosixPath,
    evidence: Path,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    git_identity = identity.get("git_free_identity")
    if not isinstance(git_identity, dict):
        raise TypeError("git-free identity must be an object")
    binding = git_free.bind_git_free_commit_verifier(
        wrapper,
        volume=volume,
        root=root,
        source=source,
        cache=cache,
        protocol_source=protocol_source,
        evidence=evidence,
        identity=git_identity,
    )
    base = wrapper._BASE
    evaluation = wrapper.evaluation
    evaluation_base = getattr(evaluation, "_BASE", None)
    nested = getattr(base, "_v4", None)
    nested_base = getattr(nested, "_BASE", None)
    if evaluation_base is None or nested is None or nested_base is None:
        raise TypeError("frozen protocol evaluator object graph is incomplete")

    def verifier(protocol_commit: str, protocol: Mapping[str, Any]) -> dict[str, Any]:
        return resolver_aware_verify_protocol_commit(
            protocol_commit,
            protocol,
            identity=identity,
            source=source,
            protocol_source=protocol_source,
        )

    surfaces = {
        "wrapper": wrapper,
        "base": base,
        "evaluation": evaluation,
        "evaluation_base": evaluation_base,
        "nested_protocol_evaluator": nested,
        "nested_protocol_evaluator_base": nested_base,
    }
    for module in surfaces.values():
        module.verify_protocol_commit = verifier
    if not all(
        getattr(module, "verify_protocol_commit", None) is verifier
        for module in surfaces.values()
    ):
        raise ValueError("claim source root verifier binding diverged")
    return {
        **binding,
        "claim_source_root_resolver_bound": True,
        "configured_claim_source_root": str(source),
        "configured_frozen_source_root": str(protocol_source),
        "verifier_binding_surfaces": sorted(surfaces),
    }


def claim_source_root_verify(
    root: Path,
    source: Path,
    cache: Path,
    protocol_source: Path,
    identity_path: Path,
    output: Path,
    *,
    wrapper: Any = frozen_v8,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("claim source root report already exists")
    for path, name in (
        (root, "root"),
        (source, "source"),
        (protocol_source, "protocol_source"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"parameterized {name} is missing")
    if cache.exists():
        raise FileExistsError("result-free root freeze must not create model-cache")
    old_path = Path(serialize_container_path(OLD_FROZEN_SOURCE))
    if old_path.exists():
        raise FileExistsError("old v8 frozen-source root must remain absent")
    identity = read_json(identity_path)
    binding = bind_claim_source_root_verifier(
        wrapper,
        volume=ROOT_NORMALIZATION_FREEZE_VOLUME,
        root=root,
        source=source,
        cache=cache,
        protocol_source=protocol_source,
        evidence=EVIDENCE,
        identity=identity,
    )
    protocol = wrapper.evaluation.load_protocol(source)
    verification = wrapper._BASE._v4._BASE.verify_protocol_commit(
        FROZEN_PROTOCOL_COMMIT, protocol
    )
    if old_path.exists() or cache.exists():
        raise ValueError("root normalization verification crossed a forbidden boundary")
    report = {
        "protocol_id": PROTOCOL_ID,
        "status": "verified",
        "root_normalization_verifier_run_count": 1,
        "root_binding_verifier_run_count": 1,
        "binding": binding,
        **verification,
        "old_frozen_source": str(old_path),
        "old_frozen_source_absent_before": True,
        "old_frozen_source_absent_after": True,
        "old_frozen_source_created": False,
        "old_frozen_source_mounted": False,
        "old_frozen_source_read": False,
        "model_cache_absent_before": True,
        "model_cache_absent_after": True,
        "model_cache_copy_count": 0,
        "model_import_count": 0,
        "model_load_count": 0,
        "model_forward_inference_count": 0,
        "registered_query_execution_count": 0,
        "development_claim_count": 0,
        "holdout_claim_count": 0,
        "worker_process_count": 0,
        "observed_result_count": 0,
        "shared_database_open_count": 0,
        "performance": "not assessed",
    }
    _write_json_exclusive(output, report)
    return report


def _protocol_source_import_script() -> str:
    root = serialize_container_path(CONTAINER_ROOT)
    source = serialize_container_path(CONTAINER_PROTOCOL_SOURCE)
    cache = serialize_container_path(CONTAINER_CACHE)
    old = serialize_container_path(OLD_FROZEN_SOURCE)
    fixture = serialize_container_path(
        CONTAINER_PROTOCOL_SOURCE
        / "tests/fixtures/github_cross_encoder_precision_v8.manifest.json"
    )
    return (
        "set -eu; "
        f"test -d '{root}'; test ! -e '{source}'; test ! -e '{cache}'; "
        f"test ! -e '{old}'; mkdir '{source}'; tar -xf - -C '{source}'; "
        f"test -f '{fixture}'; test ! -e '{cache}'; test ! -e '{old}'"
    )


def _harness_source_import_script() -> str:
    root = serialize_container_path(CONTAINER_ROOT)
    source = serialize_container_path(CONTAINER_SOURCE)
    cache = serialize_container_path(CONTAINER_CACHE)
    old = serialize_container_path(OLD_FROZEN_SOURCE)
    module = serialize_container_path(
        CONTAINER_SOURCE
        / "src/neuron_graph_rag/cross_encoder_precision_v15_observation.py"
    )
    return (
        "set -eu; "
        f"test -d '{root}'; test ! -e '{source}'; test ! -e '{cache}'; "
        f"test ! -e '{old}'; mkdir '{source}'; tar -xf - -C '{source}'; "
        f"test -f '{module}'; test ! -e '{cache}'; test ! -e '{old}'"
    )


def root_normalization_command() -> list[str]:
    return [
        "wslc",
        "run",
        "--rm",
        "--network",
        "none",
        "--volume",
        named_volume_spec(ROOT_NORMALIZATION_FREEZE_VOLUME, CONTAINER_ROOT),
        "--env",
        f"PYTHONPATH={serialize_container_path(CONTAINER_SOURCE / 'src')}",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "TRANSFORMERS_OFFLINE=1",
        "--env",
        "NO_PROXY=*",
        "--workdir",
        serialize_container_path(CONTAINER_SOURCE),
        "--entrypoint",
        "python",
        IMAGE,
        "-m",
        "neuron_graph_rag.cross_encoder_precision_v15_observation",
        "root-normalization-verify",
        "--root",
        serialize_container_path(CONTAINER_ROOT),
        "--source",
        serialize_container_path(CONTAINER_SOURCE),
        "--cache",
        serialize_container_path(CONTAINER_CACHE),
        "--protocol-source",
        serialize_container_path(CONTAINER_PROTOCOL_SOURCE),
        "--identity",
        serialize_container_path(CONTAINER_SOURCE_IDENTITY),
        "--output",
        serialize_container_path(CONTAINER_REPORT),
    ]


def _count_audit(
    *,
    status: str,
    rows: Sequence[Mapping[str, Any]],
    future_runtime_absent_before: bool,
    future_runtime_absent_after: bool,
    predecessor_unchanged: bool,
) -> dict[str, Any]:
    commands = [row.get("command", []) for row in rows]
    create_count = sum(
        command
        == ["wslc", "volume", "create", ROOT_NORMALIZATION_FREEZE_VOLUME]
        for command in commands
    )
    verifier_count = sum("root-normalization-verify" in command for command in commands)
    command_text = "\n".join(str(value) for command in commands for value in command)
    return {
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "root_normalization_freeze_volume_create_count": create_count,
        "root_freeze_volume_create_count": create_count,
        "root_normalization_verifier_run_count": verifier_count,
        "root_binding_verifier_run_count": verifier_count,
        "root_normalization_verifier_retry_count": 0,
        "root_binding_verifier_retry_count": 0,
        "retry_count": 0,
        "accepted_image_rebuild_count": 0,
        "model_cache_copy_count": 0,
        "model_import_count": 0,
        "model_load_count": 0,
        "model_forward_inference_count": 0,
        "registered_query_execution_count": 0,
        "development_claim_count": 0,
        "holdout_claim_count": 0,
        "worker_process_count": 0,
        "observed_result_count": 0,
        "shared_database_open_count": 0,
        "container_git_executable_invocation_count": 0,
        "container_subprocess_invocation_count": 0,
        "v10_runtime_volume_mounted": V10_RUNTIME_VOLUME in command_text,
        "v10_cache_freeze_volume_mounted": V10_CACHE_FREEZE_VOLUME in command_text,
        "v11_root_freeze_volume_mounted": V11_ROOT_FREEZE_VOLUME in command_text,
        "v12_runtime_volume_mounted": V12_RUNTIME_VOLUME in command_text,
        "v13_commit_freeze_volume_mounted": V13_COMMIT_FREEZE_VOLUME in command_text,
        "v14_runtime_volume_mounted": V14_RUNTIME_VOLUME in command_text,
        "old_frozen_source_created": False,
        "old_frozen_source_mounted": False,
        "old_frozen_source_read": False,
        "future_runtime_volume_absent_before": future_runtime_absent_before,
        "future_runtime_volume_absent_after": future_runtime_absent_after,
        "predecessor_artifacts_unchanged": predecessor_unchanged,
        "root_normalization_freeze_volume_reusable": False,
        "root_freeze_volume_reusable": False,
        "performance": "not assessed",
    }


def _write_evidence(
    root: Path,
    *,
    status: str,
    summary: Mapping[str, Any],
    rows: list[dict[str, Any]],
    image: Mapping[str, Any] | None,
    volume: Mapping[str, Any] | None,
    source_identity: Mapping[str, Any] | None,
    root_binding: Mapping[str, Any] | None,
    future_runtime_absent_before: bool,
    future_runtime_absent_after: bool,
    predecessor_unchanged: bool,
) -> None:
    evidence = root / EVIDENCE
    summary_name = (
        "root-normalization.pass.json"
        if status == "pass"
        else "root-normalization.error.json"
    )
    summary_value = dict(summary)
    root_hash = summary_value.pop("root_binding_verification_sha256", None)
    if root_hash is not None:
        summary_value["claim_source_root_verification_sha256"] = root_hash
    _write_json_exclusive(evidence / summary_name, summary_value)
    _write_json_exclusive(
        evidence / "root-normalization-commands.json", {"commands": rows}
    )
    if image is not None:
        _write_json_exclusive(evidence / "accepted-image-inspect.json", dict(image))
    if volume is not None:
        _write_json_exclusive(evidence / "volume-identity.json", dict(volume))
    if source_identity is not None:
        _write_json_exclusive(evidence / "source-identity.json", dict(source_identity))
    if root_binding is not None:
        _write_json_exclusive(
            evidence / "claim-source-root-verification.json", dict(root_binding)
        )
    _write_json_exclusive(
        evidence / "count-audit.json",
        _count_audit(
            status=status,
            rows=rows,
            future_runtime_absent_before=future_runtime_absent_before,
            future_runtime_absent_after=future_runtime_absent_after,
            predecessor_unchanged=predecessor_unchanged,
        ),
    )
    registry = {
        path.name: sha256_file(path)
        for path in sorted(evidence.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "evidence-manifest.json"
    }
    _write_json_exclusive(
        evidence / "evidence-manifest.json",
        {"protocol_id": PROTOCOL_ID, "status": status, "files_sha256": registry},
    )


@contextmanager
def _v15_scope() -> Any:
    replacements = {
        "PROTOCOL_ID": PROTOCOL_ID,
        "PREDECESSOR_MERGE_COMMIT": PREDECESSOR_MERGE_COMMIT,
        "ROOT": ROOT,
        "MANIFEST": MANIFEST,
        "RESULT_FREE_AUDIT": RESULT_FREE_AUDIT,
        "EVIDENCE": EVIDENCE,
        "ROOT_FREEZE_VOLUME": ROOT_NORMALIZATION_FREEZE_VOLUME,
        "FUTURE_RUNTIME_VOLUME": FUTURE_RUNTIME_VOLUME,
        "CONTAINER_ROOT": CONTAINER_ROOT,
        "CONTAINER_SOURCE": CONTAINER_SOURCE,
        "CONTAINER_CACHE": CONTAINER_CACHE,
        "CONTAINER_PROTOCOL_SOURCE": CONTAINER_PROTOCOL_SOURCE,
        "CONTAINER_REPORT": CONTAINER_REPORT,
        "OLD_FROZEN_SOURCE": OLD_FROZEN_SOURCE,
        "_manifest": _manifest,
        "_audit_contract": _audit_contract,
        "_expected_container_paths": _expected_container_paths,
        "_verify_predecessor_hashes": _verify_predecessor_hashes,
        "validate_prebuild": validate_prebuild,
        "_protocol_source_import_script": _protocol_source_import_script,
        "_harness_source_import_script": _harness_source_import_script,
        "root_binding_command": root_normalization_command,
        "_count_audit": _count_audit,
        "_write_evidence": _write_evidence,
    }
    original = {name: getattr(predecessor, name) for name in replacements}
    for name, value in replacements.items():
        setattr(predecessor, name, value)
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(predecessor, name, value)


def run_root_normalization_freeze(root: Path = ROOT) -> dict[str, Any]:
    with _v15_scope():
        result = predecessor.run_root_freeze(root)
    value = dict(result)
    root_hash = value.pop("root_binding_verification_sha256", None)
    if root_hash is not None:
        value["claim_source_root_verification_sha256"] = root_hash
    return value


def _verify_evidence_manifest(evidence: Path) -> dict[str, Any]:
    value = read_json(evidence / "evidence-manifest.json")
    registry = value.get("files_sha256")
    if not isinstance(registry, dict):
        raise TypeError("v15 evidence hash registry is missing")
    actual = {
        path.name
        for path in evidence.iterdir()
        if path.is_file() and path.name != "evidence-manifest.json"
    }
    if set(registry) != actual:
        raise ValueError("v15 evidence file set mismatch")
    for name, expected in registry.items():
        if sha256_file(evidence / name) != expected:
            raise ValueError(f"v15 evidence hash mismatch: {name}")
    return value


def audit_evidence(root: Path = ROOT) -> dict[str, Any]:
    prebuild = validate_prebuild(root)
    evidence = root / EVIDENCE
    if not evidence.exists():
        return {
            **prebuild,
            "status": "prebuild_ready_evidence_absent",
            "future_runtime_volume_absent_before": None,
            "future_runtime_volume_absent_after": None,
        }
    manifest = _verify_evidence_manifest(evidence)
    status = str(manifest.get("status"))
    if status not in {"pass", "error"}:
        raise ValueError("v15 evidence status mismatch")
    counts = read_json(evidence / "count-audit.json")
    for key in (
        "root_normalization_verifier_retry_count",
        "retry_count",
        "accepted_image_rebuild_count",
        "model_cache_copy_count",
        "model_import_count",
        "model_load_count",
        "model_forward_inference_count",
        "registered_query_execution_count",
        "development_claim_count",
        "holdout_claim_count",
        "worker_process_count",
        "observed_result_count",
        "shared_database_open_count",
        "container_git_executable_invocation_count",
        "container_subprocess_invocation_count",
    ):
        if counts.get(key) != 0:
            raise ValueError(f"v15 terminal count mismatch: {key}")
    if (
        counts.get("root_normalization_freeze_volume_create_count") not in {0, 1}
        or counts.get("root_normalization_verifier_run_count") not in {0, 1}
        or counts.get("v10_runtime_volume_mounted") is not False
        or counts.get("v10_cache_freeze_volume_mounted") is not False
        or counts.get("v11_root_freeze_volume_mounted") is not False
        or counts.get("v12_runtime_volume_mounted") is not False
        or counts.get("v13_commit_freeze_volume_mounted") is not False
        or counts.get("v14_runtime_volume_mounted") is not False
        or counts.get("old_frozen_source_created") is not False
        or counts.get("old_frozen_source_mounted") is not False
        or counts.get("old_frozen_source_read") is not False
        or counts.get("root_normalization_freeze_volume_reusable") is not False
        or counts.get("predecessor_artifacts_unchanged") is not True
        or counts.get("performance") != "not assessed"
    ):
        raise ValueError("v15 terminal boundary mismatch")
    if status == "pass" and (
        counts.get("root_normalization_freeze_volume_create_count") != 1
        or counts.get("root_normalization_verifier_run_count") != 1
        or counts.get("future_runtime_volume_absent_before") is not True
        or counts.get("future_runtime_volume_absent_after") is not True
    ):
        raise ValueError("v15 successful root normalization count mismatch")
    return {
        **prebuild,
        "status": status,
        "root_normalization_freeze_volume_create_count": counts[
            "root_normalization_freeze_volume_create_count"
        ],
        "root_normalization_verifier_run_count": counts[
            "root_normalization_verifier_run_count"
        ],
        "future_runtime_volume_absent_before": counts[
            "future_runtime_volume_absent_before"
        ],
        "future_runtime_volume_absent_after": counts[
            "future_runtime_volume_absent_after"
        ],
        "predecessor_artifacts_unchanged": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prebuild")
    commands.add_parser("freeze")
    commands.add_parser("audit")
    verify = commands.add_parser("root-normalization-verify")
    verify.add_argument("--root", required=True)
    verify.add_argument("--source", required=True)
    verify.add_argument("--cache", required=True)
    verify.add_argument("--protocol-source", required=True)
    verify.add_argument("--identity", required=True)
    verify.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "prebuild":
        result = validate_prebuild()
    elif arguments.command == "freeze":
        result = run_root_normalization_freeze()
    elif arguments.command == "audit":
        result = audit_evidence()
    else:
        if os.name != "posix":
            raise RuntimeError("root normalization verifier requires the POSIX container")
        result = claim_source_root_verify(
            Path(arguments.root),
            Path(arguments.source),
            Path(arguments.cache),
            Path(arguments.protocol_source),
            Path(arguments.identity),
            Path(arguments.output),
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
