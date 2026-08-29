from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from . import cross_encoder_precision_v8_observation as frozen_v8
from . import cross_encoder_precision_v10_observation as v10_freeze

PROTOCOL_ID = "github-ngr-cross-encoder-precision-v11"
PREDECESSOR_MERGE_COMMIT = "6a511dbb3289dc83c6a7a7a1ec16593a0110b539"
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path("tests/fixtures/github_cross_encoder_precision_v11.manifest.json")
RESULT_FREE_AUDIT = Path(
    "tests/fixtures/github_cross_encoder_precision_v11.result-free-audit.json"
)
EVIDENCE = Path("tests/evidence/github_cross_encoder_precision_v11")

IMAGE = v10_freeze.IMAGE
IMAGE_ID = v10_freeze.IMAGE_ID
WSLC_VERSION = v10_freeze.WSLC_VERSION
ROOT_FREEZE_VOLUME = "github-cross-encoder-precision-v11-root-freeze"
FUTURE_RUNTIME_VOLUME = "github-cross-encoder-precision-v11-runtime"
V10_RUNTIME_VOLUME = "github-cross-encoder-precision-v10-runtime"
V10_CACHE_FREEZE_VOLUME = "github-cross-encoder-precision-v10-cache-freeze"

CONTAINER_ROOT = PurePosixPath("/opt/ngr-v11/root-freeze")
CONTAINER_SOURCE = CONTAINER_ROOT / "source"
CONTAINER_CACHE = CONTAINER_ROOT / "model-cache"
CONTAINER_PROTOCOL_SOURCE = CONTAINER_ROOT / "frozen-source"
CONTAINER_REPORT = CONTAINER_ROOT / "root-binding-verification.json"
OLD_FROZEN_SOURCE = PurePosixPath("/opt/ngr-v8/runtime/frozen-source")

V10_RAW_FAILURE_PATH = Path(
    "tests/evidence/github_cross_encoder_precision_v10_observation/preflight.error.json"
)
V10_RAW_FAILURE_SHA256 = (
    "88f8e0b71be42751ff5414a3c793d5f08e2c52d7c8689bc6e27e00cf20f4f038"
)
V10_TERMINAL_PATH = Path(
    "tests/evidence/github_cross_encoder_precision_v10_observation/"
    "preflight-terminal.json"
)
V10_TERMINAL_SHA256 = "135add111a07ab505a7efbed36937cd4409b88ffc5145b585ad6695056b4e060"
V10_EVIDENCE_MANIFEST_PATH = Path(
    "tests/evidence/github_cross_encoder_precision_v10_observation/"
    "observation-evidence-manifest.json"
)
V10_EVIDENCE_MANIFEST_SHA256 = (
    "0763fbdfcddf9d9e49dcb26c5e5889d0c8e414ac9d279667984f9b1ae430de0f"
)

canonical_sha256 = v10_freeze.canonical_sha256
sha256_file = v10_freeze.sha256_file
read_json = v10_freeze.read_json
_write_json_exclusive = v10_freeze._write_json_exclusive
_command_row = v10_freeze._command_row
_run_logged = v10_freeze._run_logged


def serialize_container_path(value: PurePosixPath | str) -> str:
    return v10_freeze.serialize_container_path(value)


def named_volume_spec(
    volume: str,
    destination: PurePosixPath | str,
    *,
    mode: str | None = None,
) -> str:
    return v10_freeze.named_volume_spec(volume, destination, mode=mode)


def _manifest(root: Path) -> dict[str, Any]:
    value = read_json(root / MANIFEST)
    if not isinstance(value, dict):
        raise TypeError("v11 manifest must be an object")
    return value


def _audit_contract(root: Path) -> dict[str, Any]:
    value = read_json(root / RESULT_FREE_AUDIT)
    if not isinstance(value, dict):
        raise TypeError("v11 result-free audit must be an object")
    return value


def _expected_container_paths() -> dict[str, str]:
    return {
        "root": serialize_container_path(CONTAINER_ROOT),
        "source": serialize_container_path(CONTAINER_SOURCE),
        "model_cache": serialize_container_path(CONTAINER_CACHE),
        "protocol_source": serialize_container_path(CONTAINER_PROTOCOL_SOURCE),
        "root_binding_report": serialize_container_path(CONTAINER_REPORT),
        "old_frozen_source": serialize_container_path(OLD_FROZEN_SOURCE),
    }


def _verify_predecessor_hashes(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, str]:
    registry = manifest.get("predecessor_immutable_sha256")
    if not isinstance(registry, dict) or len(registry) != 23:
        raise ValueError("v11 predecessor registry must contain exactly 23 files")
    actual: dict[str, str] = {}
    for relative, expected in registry.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise TypeError("v11 predecessor registry entries must be strings")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"v10 predecessor artifact missing: {relative}")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"v10 predecessor artifact changed: {relative}")
        actual[relative] = observed
    return actual


def validate_prebuild(root: Path = ROOT) -> dict[str, Any]:
    manifest = _manifest(root)
    expected_header = {
        "protocol_id": PROTOCOL_ID,
        "phase": "parameterized-root-freeze",
        "predecessor_merge_commit": PREDECESSOR_MERGE_COMMIT,
        "root_freeze_volume": ROOT_FREEZE_VOLUME,
        "future_runtime_volume": FUTURE_RUNTIME_VOLUME,
        "accepted_image": {"tag": IMAGE, "id": IMAGE_ID},
        "accepted_image_rebuild_allowed": False,
        "result_free_only": True,
        "wslc_version": WSLC_VERSION,
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            raise ValueError(f"v11 manifest mismatch: {key}")
    if manifest.get("container_paths") != _expected_container_paths():
        raise ValueError("v11 container path registry mismatch")
    if manifest.get("expected_evidence") != [
        "accepted-image-inspect.json",
        "count-audit.json",
        "evidence-manifest.json",
        "root-binding-verification.json",
        "root-freeze-commands.json",
        "root-freeze.pass.json|root-freeze.error.json",
        "source-identity.json",
        "volume-identity.json",
    ]:
        raise ValueError("v11 expected evidence registry mismatch")
    if manifest.get("result_free_audit_sha256") != sha256_file(
        root / RESULT_FREE_AUDIT
    ):
        raise ValueError("v11 result-free audit hash mismatch")
    predecessor = _verify_predecessor_hashes(root, manifest)
    anchors = {
        V10_RAW_FAILURE_PATH.as_posix(): V10_RAW_FAILURE_SHA256,
        V10_TERMINAL_PATH.as_posix(): V10_TERMINAL_SHA256,
        V10_EVIDENCE_MANIFEST_PATH.as_posix(): V10_EVIDENCE_MANIFEST_SHA256,
    }
    for relative, expected in anchors.items():
        if predecessor.get(relative) != expected:
            raise ValueError(f"v10 terminal anchor is not frozen: {relative}")
    audit = _audit_contract(root)
    expected_zero = (
        "accepted_image_rebuild_count",
        "model_cache_copy_count",
        "model_import_count",
        "model_load_count",
        "model_forward_inference_count",
        "registered_query_execution_count",
        "development_claim_count",
        "holdout_claim_count",
        "observed_result_count",
        "retry_count",
        "shared_database_open_count",
    )
    for key in expected_zero:
        if audit.get(key) != 0:
            raise ValueError(f"v11 result-free count must be zero: {key}")
    if (
        audit.get("protocol_id") != PROTOCOL_ID
        or audit.get("phase") != "parameterized-root-freeze"
        or audit.get("root_binding_verifier_run_limit") != 1
        or audit.get("root_freeze_volume_reusable") is not False
        or audit.get("v10_runtime_volume_reused") is not False
        or audit.get("v10_cache_freeze_volume_reused") is not False
        or audit.get("performance") != "not assessed"
    ):
        raise ValueError("v11 result-free audit boundary mismatch")
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "prebuild_contract_valid",
        "predecessor_artifact_count": len(predecessor),
        "old_frozen_source": serialize_container_path(OLD_FROZEN_SOURCE),
        "old_frozen_source_allowed": False,
        "protocol_artifact_count": 23,
        "corpus_document_count": 24,
        "root_binding_verifier_run_limit": 1,
        "registered_query_execution_count": 0,
        "model_cache_copy_count": 0,
        "model_import_count": 0,
        "model_load_count": 0,
        "model_forward_inference_count": 0,
        "development_claim_count": 0,
        "holdout_claim_count": 0,
        "observed_result_count": 0,
        "shared_database_open_count": 0,
        "performance": "not assessed",
    }


def _container_path(value: PurePosixPath | str) -> Path:
    return Path(serialize_container_path(value))


def _require_child(
    path: Path | PurePosixPath, root: Path | PurePosixPath, name: str
) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} must be below the parameterized root") from error
    if path == root:
        raise ValueError(f"{name} must be distinct from the parameterized root")


def bind_frozen_harness_root(
    wrapper: Any,
    *,
    volume: str,
    root: Path | PurePosixPath,
    source: Path | PurePosixPath,
    cache: Path | PurePosixPath,
    protocol_source: Path | PurePosixPath,
    evidence: Path,
) -> dict[str, Any]:
    if not root.is_absolute():
        raise ValueError("parameterized container root must be absolute")
    for path, name in (
        (source, "source"),
        (cache, "cache"),
        (protocol_source, "protocol_source"),
    ):
        if not path.is_absolute():
            raise ValueError(f"parameterized {name} must be absolute")
        _require_child(path, root, name)
    old_root = OLD_FROZEN_SOURCE.parent.as_posix()
    if any(
        path.as_posix() == old_root or path.as_posix().startswith(f"{old_root}/")
        for path in (root, source, cache, protocol_source)
    ):
        raise ValueError("old v8 runtime root is forbidden")
    base = getattr(wrapper, "_BASE", None)
    if base is None or base is wrapper:
        raise ValueError("frozen wrapper must expose a distinct _BASE module object")
    bindings = {
        "VOLUME": volume,
        "CONTAINER_ROOT": root,
        "CONTAINER_SOURCE": source,
        "CONTAINER_CACHE": cache,
        "CONTAINER_PROTOCOL_SOURCE": protocol_source,
        "ROOT": protocol_source,
        "EVIDENCE": evidence,
    }
    for module in (wrapper, base):
        for name, value in bindings.items():
            setattr(module, name, value)
    binder = getattr(base, "_bind_container_harness", None)
    if not callable(binder):
        raise TypeError("frozen _BASE does not expose its container binder")
    binder()
    for module in (wrapper, base):
        for name, value in bindings.items():
            setattr(module, name, value)
    evaluation = getattr(wrapper, "evaluation", None)
    evaluation_base = getattr(evaluation, "_BASE", None)
    if evaluation is None or evaluation_base is None:
        raise TypeError("frozen wrapper evaluation object graph is incomplete")
    evaluation.ROOT = protocol_source
    evaluation_base.ROOT = protocol_source
    return {
        "wrapper_base_distinct": wrapper is not base,
        "wrapper_binding": {name: str(getattr(wrapper, name)) for name in bindings},
        "base_binding": {name: str(getattr(base, name)) for name in bindings},
        "evaluation_root": str(evaluation.ROOT),
        "evaluation_base_root": str(evaluation_base.ROOT),
    }


def root_binding_verify(
    root: Path,
    source: Path,
    cache: Path,
    protocol_source: Path,
    output: Path,
    *,
    wrapper: Any = frozen_v8,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("root-binding report already exists")
    for path, name in (
        (root, "root"),
        (source, "source"),
        (protocol_source, "protocol_source"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"parameterized {name} is missing")
    if cache.exists():
        raise FileExistsError("result-free root freeze must not create model-cache")
    old_path = _container_path(OLD_FROZEN_SOURCE)
    if old_path.exists():
        raise FileExistsError("old v8 frozen-source root must remain absent")
    binding = bind_frozen_harness_root(
        wrapper,
        volume=ROOT_FREEZE_VOLUME,
        root=root,
        source=source,
        cache=cache,
        protocol_source=protocol_source,
        evidence=EVIDENCE,
    )
    protocol = wrapper.evaluation.load_protocol(protocol_source)
    manifest = protocol.get("manifest")
    corpus = protocol.get("corpus")
    if not isinstance(manifest, dict) or not isinstance(corpus, dict):
        raise TypeError("frozen v8 protocol shape mismatch")
    artifact_registry = manifest.get("artifact_sha256")
    documents = corpus.get("documents")
    commit = corpus.get("commit")
    if not isinstance(artifact_registry, dict) or len(artifact_registry) != 23:
        raise ValueError("frozen v8 protocol must contain exactly 23 artifacts")
    if not isinstance(documents, list) or len(documents) != 24:
        raise ValueError("frozen v8 corpus must contain exactly 24 documents")
    protocol_artifacts: list[dict[str, Any]] = []
    for relative, expected in sorted(artifact_registry.items()):
        path = protocol_source / str(relative)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"exact protocol byte mismatch: {relative}")
        protocol_artifacts.append(
            {"path": relative, "sha256": actual, "size": path.stat().st_size}
        )
    git_bytes = wrapper.evaluation._BASE._git_bytes
    corpus_documents: list[dict[str, Any]] = []
    for row in documents:
        if not isinstance(row, dict):
            raise TypeError("frozen v8 corpus row must be an object")
        relative = str(row.get("path"))
        raw = git_bytes(protocol_source, str(commit), relative)
        actual = hashlib.sha256(raw).hexdigest()
        if actual != row.get("content_sha256"):
            raise ValueError(f"exact corpus byte mismatch: {relative}")
        corpus_documents.append({"path": relative, "sha256": actual, "size": len(raw)})
    if old_path.exists() or cache.exists():
        raise ValueError("root-binding verification crossed a forbidden boundary")
    expected_root = str(root)
    for surface in (binding["wrapper_binding"], binding["base_binding"]):
        if (
            surface["CONTAINER_ROOT"] != expected_root
            or surface["CONTAINER_SOURCE"] != str(source)
            or surface["CONTAINER_CACHE"] != str(cache)
            or surface["CONTAINER_PROTOCOL_SOURCE"] != str(protocol_source)
            or surface["ROOT"] != str(protocol_source)
        ):
            raise ValueError("wrapper and _BASE root bindings diverged")
    report = {
        "protocol_id": PROTOCOL_ID,
        "status": "verified",
        "root_binding_verifier_run_count": 1,
        "binding": binding,
        "old_frozen_source": str(old_path),
        "old_frozen_source_absent_before": True,
        "old_frozen_source_absent_after": True,
        "old_frozen_source_created": False,
        "old_frozen_source_mounted": False,
        "old_frozen_source_read": False,
        "model_cache_absent_before": True,
        "model_cache_absent_after": True,
        "model_cache_copy_count": 0,
        "protocol_artifact_count": len(protocol_artifacts),
        "protocol_artifact_size": sum(row["size"] for row in protocol_artifacts),
        "protocol_artifacts": protocol_artifacts,
        "corpus_document_count": len(corpus_documents),
        "corpus_document_size": sum(row["size"] for row in corpus_documents),
        "corpus_documents": corpus_documents,
        "exact_protocol_bytes_verified": True,
        "exact_corpus_bytes_verified": True,
        "registered_query_execution_count": 0,
        "model_import_count": 0,
        "model_load_count": 0,
        "model_forward_inference_count": 0,
        "development_claim_count": 0,
        "holdout_claim_count": 0,
        "observed_result_count": 0,
        "shared_database_open_count": 0,
        "performance": "not assessed",
    }
    _write_json_exclusive(output, report)
    return report


def _archive_import_command(script: str) -> list[str]:
    return [
        "wslc",
        "run",
        "--rm",
        "--interactive",
        "--network",
        "none",
        "--volume",
        named_volume_spec(ROOT_FREEZE_VOLUME, CONTAINER_ROOT),
        "--entrypoint",
        "/bin/sh",
        IMAGE,
        "-c",
        script,
    ]


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
        / "src/neuron_graph_rag/cross_encoder_precision_v11_observation.py"
    )
    return (
        "set -eu; "
        f"test -d '{root}'; test ! -e '{source}'; test ! -e '{cache}'; "
        f"test ! -e '{old}'; mkdir '{source}'; tar -xf - -C '{source}'; "
        f"test -f '{module}'; test ! -e '{cache}'; test ! -e '{old}'"
    )


def root_binding_command() -> list[str]:
    return [
        "wslc",
        "run",
        "--rm",
        "--network",
        "none",
        "--volume",
        named_volume_spec(ROOT_FREEZE_VOLUME, CONTAINER_ROOT),
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
        "neuron_graph_rag.cross_encoder_precision_v11_observation",
        "root-binding-verify",
        "--root",
        serialize_container_path(CONTAINER_ROOT),
        "--source",
        serialize_container_path(CONTAINER_SOURCE),
        "--cache",
        serialize_container_path(CONTAINER_CACHE),
        "--protocol-source",
        serialize_container_path(CONTAINER_PROTOCOL_SOURCE),
        "--output",
        serialize_container_path(CONTAINER_REPORT),
    ]


def _volume_identity(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw = _run_logged(["wslc", "volume", "inspect", ROOT_FREEZE_VOLUME], root, rows)
    value = json.loads(raw)
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError("root-freeze volume inspection shape mismatch")
    volume = value[0]
    return {
        "name": volume.get("Name"),
        "driver": volume.get("Driver"),
        "scope": volume.get("Scope"),
        "mountpoint_sha256": hashlib.sha256(
            str(volume.get("Mountpoint", "")).encode("utf-8")
        ).hexdigest(),
    }


def _count_audit(
    *,
    status: str,
    rows: Sequence[Mapping[str, Any]],
    future_runtime_absent_before: bool,
    future_runtime_absent_after: bool,
    predecessor_unchanged: bool,
) -> dict[str, Any]:
    commands = [row.get("command", []) for row in rows]
    volume_create_count = sum(
        command == ["wslc", "volume", "create", ROOT_FREEZE_VOLUME]
        for command in commands
    )
    verifier_count = sum("root-binding-verify" in command for command in commands)
    command_text = "\n".join(str(value) for command in commands for value in command)
    return {
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "root_freeze_volume_create_count": volume_create_count,
        "root_binding_verifier_run_count": verifier_count,
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
        "observed_result_count": 0,
        "shared_database_open_count": 0,
        "v10_runtime_volume_mounted": V10_RUNTIME_VOLUME in command_text,
        "v10_cache_freeze_volume_mounted": V10_CACHE_FREEZE_VOLUME in command_text,
        "old_frozen_source_created": False,
        "old_frozen_source_mounted": False,
        "old_frozen_source_read": False,
        "future_runtime_volume_absent_before": future_runtime_absent_before,
        "future_runtime_volume_absent_after": future_runtime_absent_after,
        "predecessor_artifacts_unchanged": predecessor_unchanged,
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
        "root-freeze.pass.json" if status == "pass" else "root-freeze.error.json"
    )
    _write_json_exclusive(evidence / summary_name, dict(summary))
    _write_json_exclusive(evidence / "root-freeze-commands.json", {"commands": rows})
    if image is not None:
        _write_json_exclusive(evidence / "accepted-image-inspect.json", dict(image))
    if volume is not None:
        _write_json_exclusive(evidence / "volume-identity.json", dict(volume))
    if source_identity is not None:
        _write_json_exclusive(evidence / "source-identity.json", dict(source_identity))
    if root_binding is not None:
        _write_json_exclusive(
            evidence / "root-binding-verification.json", dict(root_binding)
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


def run_root_freeze(root: Path = ROOT) -> dict[str, Any]:
    validate_prebuild(root)
    evidence = root / EVIDENCE
    if evidence.exists():
        raise FileExistsError(
            "v11 root freeze evidence already exists; retry forbidden"
        )
    evidence.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    image: dict[str, Any] | None = None
    volume: dict[str, Any] | None = None
    source_identity: dict[str, Any] | None = None
    root_binding: dict[str, Any] | None = None
    future_runtime_absent_before = False
    future_runtime_absent_after = False
    predecessor_unchanged = False
    manifest = _manifest(root)
    try:
        ci = v10_freeze._require_remote_ci_green(root, rows)
        predecessor_before = _verify_predecessor_hashes(root, manifest)
        version = _run_logged(["wslc", "--version"], root, rows).splitlines()[0]
        if version.removeprefix("wslc ").strip() != WSLC_VERSION:
            raise ValueError("WSLC version mismatch")
        if not v10_freeze._inspect_absent(ROOT_FREEZE_VOLUME, root, rows):
            raise FileExistsError("v11 root-freeze volume already exists")
        future_runtime_absent_before = v10_freeze._inspect_absent(
            FUTURE_RUNTIME_VOLUME, root, rows
        )
        if not future_runtime_absent_before:
            raise FileExistsError("v11 future runtime volume already exists")
        image = v10_freeze._image_identity(root, rows)
        _run_logged(["wslc", "volume", "create", ROOT_FREEZE_VOLUME], root, rows)
        predecessor_archive = v10_freeze._archive_bytes(
            PREDECESSOR_MERGE_COMMIT, root, rows
        )
        _run_logged(
            _archive_import_command(_protocol_source_import_script()),
            root,
            rows,
            input_bytes=predecessor_archive,
        )
        current_archive = v10_freeze._archive_bytes(ci["commit"], root, rows)
        _run_logged(
            _archive_import_command(_harness_source_import_script()),
            root,
            rows,
            input_bytes=current_archive,
        )
        source_identity = {
            "implementation_commit": ci["commit"],
            "predecessor_merge_commit": PREDECESSOR_MERGE_COMMIT,
            "harness_source_archive_sha256": hashlib.sha256(
                current_archive
            ).hexdigest(),
            "protocol_source_archive_sha256": hashlib.sha256(
                predecessor_archive
            ).hexdigest(),
            "harness_source": serialize_container_path(CONTAINER_SOURCE),
            "protocol_source": serialize_container_path(CONTAINER_PROTOCOL_SOURCE),
        }
        root_binding = json.loads(_run_logged(root_binding_command(), root, rows))
        expected_report = {
            "status": "verified",
            "root_binding_verifier_run_count": 1,
            "protocol_artifact_count": 23,
            "corpus_document_count": 24,
            "exact_protocol_bytes_verified": True,
            "exact_corpus_bytes_verified": True,
            "old_frozen_source_created": False,
            "old_frozen_source_mounted": False,
            "old_frozen_source_read": False,
            "model_cache_copy_count": 0,
            "model_import_count": 0,
            "model_load_count": 0,
            "model_forward_inference_count": 0,
            "registered_query_execution_count": 0,
            "development_claim_count": 0,
            "holdout_claim_count": 0,
            "observed_result_count": 0,
            "shared_database_open_count": 0,
            "performance": "not assessed",
        }
        for key, expected in expected_report.items():
            if root_binding.get(key) != expected:
                raise ValueError(f"v11 root-binding report mismatch: {key}")
        volume = _volume_identity(root, rows)
        future_runtime_absent_after = v10_freeze._inspect_absent(
            FUTURE_RUNTIME_VOLUME, root, rows
        )
        if not future_runtime_absent_after:
            raise ValueError("future v11 runtime volume was created during freeze")
        predecessor_after = _verify_predecessor_hashes(root, manifest)
        predecessor_unchanged = predecessor_after == predecessor_before
        if not predecessor_unchanged:
            raise ValueError("v10 predecessor artifacts changed during v11 freeze")
        counts = _count_audit(
            status="pass",
            rows=rows,
            future_runtime_absent_before=True,
            future_runtime_absent_after=True,
            predecessor_unchanged=True,
        )
        if (
            counts["root_freeze_volume_create_count"] != 1
            or counts["root_binding_verifier_run_count"] != 1
            or counts["v10_runtime_volume_mounted"] is not False
            or counts["v10_cache_freeze_volume_mounted"] is not False
        ):
            raise ValueError("v11 exactly-once or predecessor-volume boundary mismatch")
        summary = {
            "protocol_id": PROTOCOL_ID,
            "status": "pass",
            "implementation_commit": ci["commit"],
            "remote_ci": ci,
            "wslc_version": WSLC_VERSION,
            "accepted_image": image,
            "volume_identity": volume,
            "source_identity": source_identity,
            "root_binding_verification_sha256": canonical_sha256(root_binding),
            "root_freeze_volume_absent_before_create": True,
            "predecessor_sha256_before": predecessor_before,
            "predecessor_sha256_after": predecessor_after,
            "future_runtime_volume_absent_before": True,
            "future_runtime_volume_absent_after": True,
            "development_claim_count": 0,
            "holdout_claim_count": 0,
            "registered_query_execution_count": 0,
            "model_cache_copy_count": 0,
            "model_import_count": 0,
            "model_load_count": 0,
            "model_forward_inference_count": 0,
            "observed_result_count": 0,
            "shared_database_open_count": 0,
            "retry_count": 0,
            "performance": "not assessed",
        }
        _write_evidence(
            root,
            status="pass",
            summary=summary,
            rows=rows,
            image=image,
            volume=volume,
            source_identity=source_identity,
            root_binding=root_binding,
            future_runtime_absent_before=True,
            future_runtime_absent_after=True,
            predecessor_unchanged=True,
        )
        return summary
    except BaseException as error:
        if not future_runtime_absent_after:
            try:
                future_runtime_absent_after = v10_freeze._inspect_absent(
                    FUTURE_RUNTIME_VOLUME, root, rows
                )
            except (OSError, RuntimeError, ValueError):
                future_runtime_absent_after = False
        if volume is None:
            try:
                if not v10_freeze._inspect_absent(ROOT_FREEZE_VOLUME, root, rows):
                    volume = _volume_identity(root, rows)
            except (OSError, RuntimeError, ValueError):
                volume = None
        try:
            predecessor_unchanged = bool(_verify_predecessor_hashes(root, manifest))
        except (OSError, RuntimeError, ValueError):
            predecessor_unchanged = False
        summary = {
            "protocol_id": PROTOCOL_ID,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "future_runtime_volume_absent_before": future_runtime_absent_before,
            "future_runtime_volume_absent_after": future_runtime_absent_after,
            "development_claim_count": 0,
            "holdout_claim_count": 0,
            "registered_query_execution_count": 0,
            "model_cache_copy_count": 0,
            "model_import_count": 0,
            "model_load_count": 0,
            "model_forward_inference_count": 0,
            "observed_result_count": 0,
            "shared_database_open_count": 0,
            "retry_count": 0,
            "performance": "not assessed",
        }
        _write_evidence(
            root,
            status="error",
            summary=summary,
            rows=rows,
            image=image,
            volume=volume,
            source_identity=source_identity,
            root_binding=root_binding,
            future_runtime_absent_before=future_runtime_absent_before,
            future_runtime_absent_after=future_runtime_absent_after,
            predecessor_unchanged=predecessor_unchanged,
        )
        raise


def _verify_evidence_manifest(evidence: Path) -> dict[str, Any]:
    value = read_json(evidence / "evidence-manifest.json")
    registry = value.get("files_sha256")
    if not isinstance(registry, dict):
        raise TypeError("v11 evidence hash registry is missing")
    actual_names = {
        path.name
        for path in evidence.iterdir()
        if path.is_file() and path.name != "evidence-manifest.json"
    }
    if set(registry) != actual_names:
        raise ValueError("v11 evidence file set mismatch")
    for name, expected in registry.items():
        if sha256_file(evidence / name) != expected:
            raise ValueError(f"v11 evidence hash mismatch: {name}")
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
        raise ValueError("v11 evidence status mismatch")
    counts = read_json(evidence / "count-audit.json")
    for key in (
        "root_binding_verifier_retry_count",
        "retry_count",
        "accepted_image_rebuild_count",
        "model_cache_copy_count",
        "model_import_count",
        "model_load_count",
        "model_forward_inference_count",
        "registered_query_execution_count",
        "development_claim_count",
        "holdout_claim_count",
        "observed_result_count",
        "shared_database_open_count",
    ):
        if counts.get(key) != 0:
            raise ValueError(f"v11 terminal count mismatch: {key}")
    if (
        counts.get("root_freeze_volume_create_count") not in {0, 1}
        or counts.get("root_binding_verifier_run_count") not in {0, 1}
        or counts.get("v10_runtime_volume_mounted") is not False
        or counts.get("v10_cache_freeze_volume_mounted") is not False
        or counts.get("old_frozen_source_created") is not False
        or counts.get("old_frozen_source_mounted") is not False
        or counts.get("old_frozen_source_read") is not False
        or counts.get("root_freeze_volume_reusable") is not False
        or counts.get("predecessor_artifacts_unchanged") is not True
        or counts.get("performance") != "not assessed"
    ):
        raise ValueError("v11 terminal boundary mismatch")
    if status == "pass" and (
        counts.get("root_freeze_volume_create_count") != 1
        or counts.get("root_binding_verifier_run_count") != 1
        or counts.get("future_runtime_volume_absent_before") is not True
        or counts.get("future_runtime_volume_absent_after") is not True
    ):
        raise ValueError("v11 successful root freeze count mismatch")
    return {
        **prebuild,
        "status": status,
        "root_freeze_volume_create_count": counts["root_freeze_volume_create_count"],
        "root_binding_verifier_run_count": counts["root_binding_verifier_run_count"],
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
    verify = commands.add_parser("root-binding-verify")
    verify.add_argument("--root", required=True)
    verify.add_argument("--source", required=True)
    verify.add_argument("--cache", required=True)
    verify.add_argument("--protocol-source", required=True)
    verify.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "prebuild":
        result = validate_prebuild()
    elif arguments.command == "freeze":
        result = run_root_freeze()
    elif arguments.command == "audit":
        result = audit_evidence()
    else:
        if os.name != "posix":
            raise RuntimeError("root-binding verifier requires the POSIX container")
        result = root_binding_verify(
            Path(arguments.root),
            Path(arguments.source),
            Path(arguments.cache),
            Path(arguments.protocol_source),
            Path(arguments.output),
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
