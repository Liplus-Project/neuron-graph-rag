from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

_V3_SOURCE = Path(__file__).with_name("cross_encoder_precision_v3_evaluation.py")
_SPEC = importlib.util.spec_from_file_location(
    "neuron_graph_rag._cross_encoder_precision_v5_frozen_base", _V3_SOURCE
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("unable to load frozen v3 evaluator")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

STEM = "github_cross_encoder_precision_v5"
PROTOCOL_ID = "github-ngr-cross-encoder-precision-v5"
ARTIFACT_KINDS = (
    "corpus",
    "queries",
    "gold",
    "models",
    "candidates",
    "gate",
    "result-schema",
    "result-free-audit",
    "platform",
    "dependency-artifacts",
    "requirements.lock",
)

# The v3 evaluator is the literal rank-only semantic base for both v4 and v5.
# Only protocol identity and the new container/dependency contract are rebound.
_BASE.STEM = STEM
_BASE.PROTOCOL_ID = PROTOCOL_ID
_BASE.ARTIFACT_KINDS = ARTIFACT_KINDS
for _name in dir(_BASE):
    if _name.startswith("__") or _name in {
        "STEM",
        "PROTOCOL_ID",
        "ARTIFACT_KINDS",
        "main",
    }:
        continue
    globals()[_name] = getattr(_BASE, _name)

_mapping = _BASE._mapping
_path = _BASE._path
_string = _BASE._string
_verify_hash_registry = _BASE._verify_hash_registry
load_protocol = _BASE.load_protocol
verify_phase_state = _BASE.verify_phase_state
prove_archive_round_trip = _BASE.prove_archive_round_trip
_VALIDATE_V3_SEMANTICS = _BASE.validate_protocol

RUNTIME_METADATA_FIELDS = {
    "protocol_id",
    "wslc_version",
    "image_id",
    "os",
    "architecture",
    "python_version",
    "versions",
    "cuda",
    "network",
    "filesystem_probe",
    "registered_query_execution_count",
    "model_forward_count",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _locked_versions(raw: str) -> dict[str, str]:
    versions = {
        name.replace("_", "-"): version
        for name, version in re.findall(
            r"^([a-z0-9][a-z0-9._-]*)==([^ \\\n]+)", raw, flags=re.MULTILINE
        )
    }
    direct = re.search(
        r"^torch @ https://download\.pytorch\.org/whl/cpu/"
        r"torch-([0-9.]+)%2Bcpu-cp311-cp311-linux_x86_64\.whl",
        raw,
        flags=re.MULTILINE,
    )
    if direct:
        versions["torch"] = f"{direct.group(1)}+cpu"
    return versions


def _validate_dependency_contract(protocol: Mapping[str, Any]) -> None:
    root = _path(protocol, "root")
    platform = _mapping(protocol, "platform")
    resolver = _mapping(platform, "resolver")
    if resolver != {
        "name": "uv",
        "version": "0.12.3",
        "build": "507230998",
        "host": "x86_64-pc-windows-msvc",
        "default_index": "https://pypi.org/simple",
        "torch_route": "direct-url-only",
        "torch_source": "https://download.pytorch.org/whl/cpu",
        "index_fallback": False,
        "compile_command": [
            "uv",
            "pip",
            "compile",
            "--generate-hashes",
            "--emit-index-url",
            "--emit-index-annotation",
            "--python-version",
            "3.11.15",
            "--python-platform",
            "x86_64-unknown-linux-gnu",
            "--constraint",
            "tests/fixtures/github_cross_encoder_precision_v4.requirements.lock",
            "--output-file",
            "tests/fixtures/github_cross_encoder_precision_v5.requirements.lock",
            "tests/fixtures/github_cross_encoder_precision_v5.requirements.in",
        ],
    }:
        raise ValueError("resolver and package routing contract mismatch")
    if _mapping(platform, "installer") != {
        "name": "pip",
        "version": "24.0",
        "only_binary": True,
        "require_hashes": True,
        "ignore_installed": True,
        "report_path": "/opt/ngr-v5/dependency-report.json",
    }:
        raise ValueError("container installer contract mismatch")

    lock_contract = _mapping(platform, "dependency_lock")
    for path_key, hash_key in (
        ("input_path", "input_sha256"),
        ("lock_path", "lock_sha256"),
        ("artifacts_path", "artifacts_sha256"),
    ):
        relative = _string(lock_contract, path_key)
        if _sha256(root / relative) != lock_contract.get(hash_key):
            raise ValueError(f"dependency {path_key} hash mismatch")

    artifacts = _BASE._object_list(
        _mapping(protocol, "dependency_artifacts"), "artifacts"
    )
    if len(artifacts) != 26 or lock_contract.get("artifact_count") != len(artifacts):
        raise ValueError("exact Linux artifact count mismatch")
    names: set[str] = set()
    lock = _string(protocol, "requirements_lock")
    for row in artifacts:
        if set(row) != {"name", "version", "filename", "url", "sha256"}:
            raise ValueError("dependency artifact shape mismatch")
        name = _string(row, "name")
        if name in names:
            raise ValueError("duplicate dependency artifact")
        names.add(name)
        filename = _string(row, "filename")
        url = _string(row, "url")
        digest = _string(row, "sha256")
        if unquote(PurePosixPath(urlparse(url).path).name) != filename:
            raise ValueError("artifact filename and URL mismatch")
        if not filename.endswith(".whl") or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("only exact wheel SHA-256 artifacts are allowed")
        if digest not in lock:
            raise ValueError("selected artifact hash is absent from lock")
        host = urlparse(url).hostname
        if name == "torch":
            if row != {
                "name": "torch",
                "version": "2.4.1+cpu",
                "filename": "torch-2.4.1+cpu-cp311-cp311-linux_x86_64.whl",
                "url": (
                    "https://download.pytorch.org/whl/cpu/"
                    "torch-2.4.1%2Bcpu-cp311-cp311-linux_x86_64.whl"
                ),
                "sha256": (
                    "2b03e20f37557d211d14e3fb3f71709325336402db132a1e0dd8b47392185baf"
                ),
            }:
                raise ValueError("official PyTorch CPU wheel contract mismatch")
        elif host != "files.pythonhosted.org":
            raise ValueError("non-torch dependency must use the frozen PyPI artifact")

    v4_lock = (
        root / "tests/fixtures/github_cross_encoder_precision_v4.requirements.lock"
    ).read_text(encoding="utf-8", errors="strict")
    if _locked_versions(lock) != _locked_versions(v4_lock):
        raise ValueError("v4/v5 dependency version drift")
    if lock.count("--index-url https://pypi.org/simple") != 1:
        raise ValueError("PyPI default index contract mismatch")
    if "--extra-index-url" in lock or lock.count("download.pytorch.org") != 1:
        raise ValueError("dependency index fallback or routing mismatch")


def _validate_container_contract(protocol: Mapping[str, Any]) -> None:
    root = _path(protocol, "root")
    platform = _mapping(protocol, "platform")
    if platform.get("substrate") != "WSLC" or _mapping(platform, "wslc") != {
        "version": "2.9.4.0",
        "platform": "windows-wsl-built-in-container",
    }:
        raise ValueError("WSLC substrate contract mismatch")
    if _mapping(platform, "python") != {
        "implementation": "CPython",
        "version": "3.11.15",
        "abi": "cp311",
    }:
        raise ValueError("container Python contract mismatch")

    container = _mapping(platform, "container")
    base = _mapping(container, "base_image")
    image = _mapping(container, "image")
    if (
        container.get("os") != "linux"
        or container.get("architecture") != "amd64"
        or base.get("tag") != "python:3.11.15-slim-bookworm"
        or base.get("digest")
        != "sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3"
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(base.get("local_image_id")))
        or image.get("tag") != "ngr-cross-encoder-precision-v5:freeze"
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(image.get("id")))
    ):
        raise ValueError("pinned Linux amd64 image identity mismatch")
    containerfile = root / _string(container, "containerfile")
    validator = root / _string(container, "validation_script")
    if _sha256(containerfile) != container.get("containerfile_sha256"):
        raise ValueError("Containerfile hash mismatch")
    if _sha256(validator) != container.get("validation_script_sha256"):
        raise ValueError("container validation script hash mismatch")
    first_line = containerfile.read_text(
        encoding="utf-8", errors="strict"
    ).splitlines()[0]
    if first_line != f"FROM {base['tag']}@{base['digest']}":
        raise ValueError("base image must include both tag and digest")
    if container.get("build_command") != [
        "wslc",
        "build",
        "--no-cache",
        "--file",
        "containers/github_cross_encoder_precision_v5/Containerfile",
        "--tag",
        "ngr-cross-encoder-precision-v5:freeze",
        ".",
    ]:
        raise ValueError("WSLC build command mismatch")
    if container.get("validation_command") != [
        "wslc",
        "run",
        "--rm",
        "--network",
        "none",
        "ngr-cross-encoder-precision-v5:freeze",
    ]:
        raise ValueError("offline WSLC validation command mismatch")
    run_root = _mapping(platform, "run_root")
    if run_root != {
        "path": "/opt/ngr-v5/runtime",
        "filesystem": "container",
        "exclusive_create": True,
        "host_bind_mount": False,
        "shared_windows_database": False,
        "future_observation_volume": "github-cross-encoder-precision-v5-runtime",
    }:
        raise ValueError("container runtime isolation mismatch")
    execution = _mapping(platform, "execution")
    if execution != {
        "device": "cpu",
        "dtype": "float32",
        "eval": True,
        "inference_mode": True,
        "batch_size": 8,
        "fresh_process": True,
        "fresh_database": True,
        "offline_validation": True,
        "local_files_only": True,
    }:
        raise ValueError("container execution contract mismatch")


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    audit = _mapping(protocol, "result_free_audit")
    if audit.get("count_scope") != (
        "v5 result-free freeze only; historical v1/v2/v3/v4 observations excluded"
    ):
        raise ValueError("result-free count scope mismatch")
    for key in (
        "freeze_registered_query_execution_count",
        "freeze_model_inference_count",
        "freeze_observed_result_count",
    ):
        if audit.get(key) != 0:
            raise ValueError(f"{key} must remain zero at freeze")
    for key in (
        "predecessor_evidence_semantic_content_opened",
        "shared_database_opened",
        "existing_experiment_database_opened",
        "github_rag_mcp_called",
        "feedback_or_outcome_recorded",
        "model_cache_opened",
        "model_weights_opened",
        "model_weights_downloaded",
        "model_forward_executed",
        "default_search_changed",
        "default_dependencies_changed",
        "mcp_config_changed",
        "predecessor_evidence_changed",
    ):
        if audit.get(key) is not False:
            raise ValueError(f"{key} must remain false at freeze")
    if (
        audit.get("wslc_image_build_count") != 1
        or audit.get("offline_synthetic_validation_count") != 1
    ):
        raise ValueError("freeze build/validation count mismatch")

    adapted = dict(protocol)
    adapted_audit = dict(audit)
    adapted_audit["count_scope"] = (
        "v3 result-free freeze only; historical v1/v2 observations excluded"
    )
    adapted["result_free_audit"] = adapted_audit
    _VALIDATE_V3_SEMANTICS(adapted)

    root = _path(protocol, "root")
    manifest = _mapping(protocol, "manifest")
    _verify_hash_registry(root, _mapping(manifest, "v3_immutable_sha256"))
    _verify_hash_registry(root, _mapping(manifest, "v4_immutable_sha256"))
    outputs = [
        path
        for stage in _mapping(manifest, "outputs").values()
        for path in stage.values()
    ]
    if (
        len(outputs) != len(set(outputs))
        or any(STEM not in path for path in outputs)
        or any(
            f"github_cross_encoder_precision_v{version}" in path
            for version in range(1, 5)
            for path in outputs
        )
    ):
        raise ValueError("v5 output path isolation mismatch")
    _validate_container_contract(protocol)
    _validate_dependency_contract(protocol)


_BASE.validate_protocol = validate_protocol


def _with_current_loader(function: Any, *args: Any, **kwargs: Any) -> Any:
    previous = _BASE.load_protocol
    _BASE.load_protocol = globals()["load_protocol"]
    try:
        return function(*args, **kwargs)
    finally:
        _BASE.load_protocol = previous


def write_stage_result(stage: str, payload: Mapping[str, Any]) -> Path:
    return _with_current_loader(_BASE.write_stage_result, stage, payload)


def write_stage_error(stage: str, message: str) -> Path:
    return _with_current_loader(_BASE.write_stage_error, stage, message)


def build_synthetic_runtime_metadata(protocol: Mapping[str, Any]) -> dict[str, Any]:
    platform = _mapping(protocol, "platform")
    artifacts = _BASE._object_list(
        _mapping(protocol, "dependency_artifacts"), "artifacts"
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "wslc_version": _mapping(platform, "wslc")["version"],
        "image_id": _mapping(_mapping(platform, "container"), "image")["id"],
        "os": "linux",
        "architecture": "amd64",
        "python_version": "3.11.15",
        "versions": {row["name"]: row["version"] for row in artifacts},
        "cuda": None,
        "network": "disabled",
        "filesystem_probe": "exclusive-create",
        "registered_query_execution_count": 0,
        "model_forward_count": 0,
    }


def validate_runtime_metadata(
    protocol: Mapping[str, Any], metadata: Mapping[str, Any]
) -> None:
    if set(metadata) != RUNTIME_METADATA_FIELDS:
        raise ValueError("runtime metadata fields mismatch")
    if dict(metadata) != build_synthetic_runtime_metadata(protocol):
        raise ValueError("runtime metadata does not match frozen WSLC image")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit the result-free WSLC cross-encoder precision v5 protocol"
    )
    parser.add_argument("command", choices=("audit", "probe"))
    args = parser.parse_args(argv)
    protocol = load_protocol()
    runtime = build_synthetic_runtime_metadata(protocol)
    validate_runtime_metadata(protocol, runtime)
    output: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "freeze_registered_query_execution_count": 0,
        "freeze_model_inference_count": 0,
        "freeze_observed_result_count": 0,
        "historical_v1_v2_v3_v4_observations_included": False,
        "image_id": runtime["image_id"],
        "platform_contract_sha256": hashlib.sha256(
            json.dumps(protocol["platform"], ensure_ascii=False, sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest(),
        "dependency_artifacts_sha256": hashlib.sha256(
            json.dumps(
                protocol["dependency_artifacts"], ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
        ).hexdigest(),
        "phase": verify_phase_state(protocol),
    }
    if args.command == "probe":
        output["archive_round_trip"] = prove_archive_round_trip()
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
