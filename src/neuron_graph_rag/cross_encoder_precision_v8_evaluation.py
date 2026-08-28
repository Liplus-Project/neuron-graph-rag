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
_CONTENT_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "containers"
    / "github_cross_encoder_precision_v8"
    / "runtime_content.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "neuron_graph_rag._cross_encoder_precision_v8_frozen_base", _V3_SOURCE
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("unable to load frozen v3 evaluator")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)
_CONTENT_SPEC = importlib.util.spec_from_file_location(
    "neuron_graph_rag._cross_encoder_precision_v8_runtime_content", _CONTENT_SOURCE
)
if _CONTENT_SPEC is None or _CONTENT_SPEC.loader is None:
    raise RuntimeError("unable to load frozen runtime content fingerprint tool")
_CONTENT = importlib.util.module_from_spec(_CONTENT_SPEC)
_CONTENT_SPEC.loader.exec_module(_CONTENT)

STEM = "github_cross_encoder_precision_v8"
PROTOCOL_ID = "github-ngr-cross-encoder-precision-v8"
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
    "expected-distributions",
    "requirements.lock",
)

# The v3 evaluator is the literal rank-only semantic base for v4, v5, and v8.
# Only protocol identity/path and the image equivalence contract are rebound.
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

ATTESTATION_FIELDS = {
    "os",
    "architecture",
    "python",
    "distributions",
    "expected_distribution_registry_sha256",
    "torch_cuda",
    "forbidden_distributions",
    "network",
    "filesystem_probe",
    "synthetic_tensor_probe",
    "registered_query_count",
    "model_forward_inference_count",
    "observed_result_count",
}

PENDING_OUTCOME = "pending_one_shot_wslc"
SUCCESS_OUTCOME = "accepted_exact_installed_distribution_freeze"
FAILURE_OUTCOME = "fail_closed_exact_installed_distribution_freeze"


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


def _expected_distribution_rows(
    protocol: Mapping[str, Any],
) -> list[dict[str, str]]:
    registry = _mapping(protocol, "expected_distributions")
    if set(registry) != {
        "canonicalization",
        "distributions",
        "expected_count",
        "protocol_id",
    }:
        raise ValueError("expected distribution registry fields mismatch")
    if (
        registry.get("canonicalization") != "pep503-lowercase-replace-runs"
        or registry.get("expected_count") != 29
        or registry.get("protocol_id") != PROTOCOL_ID
    ):
        raise ValueError("expected distribution registry identity mismatch")
    rows = _BASE._object_list(registry, "distributions")
    names: set[str] = set()
    toolchain: set[str] = set()
    normalized: list[dict[str, str]] = []
    for row in rows:
        if set(row) != {"canonical_name", "version", "origin_class"}:
            raise ValueError("expected distribution row shape mismatch")
        name = _CONTENT.canonicalize_name(_string(row, "canonical_name"))
        _string(row, "version")
        origin = _string(row, "origin_class")
        if name != row["canonical_name"] or name in names:
            raise ValueError("duplicate or non-canonical expected distribution")
        if origin not in {"ml-runtime-artifact", "image-toolchain"}:
            raise ValueError("expected distribution origin class mismatch")
        names.add(name)
        if origin == "image-toolchain":
            toolchain.add(name)
        normalized.append(dict(row))
    if len(normalized) != 29 or toolchain != {"pip", "setuptools", "wheel"}:
        raise ValueError("exact 29 expected distribution registry mismatch")
    return sorted(normalized, key=lambda row: row["canonical_name"])


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
            "tests/fixtures/github_cross_encoder_precision_v8.requirements.lock",
            "tests/fixtures/github_cross_encoder_precision_v8.requirements.in",
        ],
    }:
        raise ValueError("resolver and package routing contract mismatch")
    if _mapping(platform, "installer") != {
        "name": "pip",
        "version": "24.0",
        "only_binary": True,
        "require_hashes": True,
        "ignore_installed": True,
        "report_path": "/opt/ngr-v8/dependency-report.json",
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

    v7_lock = (
        root / "tests/fixtures/github_cross_encoder_precision_v7.requirements.lock"
    ).read_text(encoding="utf-8", errors="strict")
    if _locked_versions(lock) != _locked_versions(v7_lock):
        raise ValueError("v7/v8 dependency version drift")
    if lock.count("--index-url https://pypi.org/simple") != 1:
        raise ValueError("PyPI default index contract mismatch")
    if "--extra-index-url" in lock or lock.count("download.pytorch.org") != 1:
        raise ValueError("dependency index fallback or routing mismatch")
    expected_rows = _expected_distribution_rows(protocol)
    ml_rows = {
        row["canonical_name"]: row["version"]
        for row in expected_rows
        if row["origin_class"] == "ml-runtime-artifact"
    }
    artifact_versions = {
        _CONTENT.canonicalize_name(_string(row, "name")): _string(row, "version")
        for row in artifacts
    }
    if ml_rows != artifact_versions:
        raise ValueError("expected registry ML/runtime artifacts drifted from v6")

    expected_contract = _mapping(platform, "expected_distribution_registry")
    expected_path = (
        "tests/fixtures/github_cross_encoder_precision_v8.expected-distributions.json"
    )
    if (
        expected_contract.get("path") != expected_path
        or expected_contract.get("count") != 29
        or _sha256(root / expected_path) != expected_contract.get("sha256")
    ):
        raise ValueError("expected distribution registry hash contract mismatch")


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
    images = _mapping(container, "images")
    build_a = _mapping(images, "build_a")
    build_b = _mapping(images, "build_b")
    outcome = _mapping(platform, "content_equivalence").get("freeze_outcome")
    if (
        container.get("os") != "linux"
        or container.get("architecture") != "amd64"
        or base.get("tag") != "python:3.11.15-slim-bookworm"
        or base.get("digest")
        != "sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3"
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(base.get("local_image_id")))
        or build_a.get("tag") != "ngr-cross-encoder-precision-v8:freeze"
        or build_b.get("tag") != "ngr-cross-encoder-precision-v8:rebuild-check"
    ):
        raise ValueError("pinned Linux amd64 image identity mismatch")
    if outcome == PENDING_OUTCOME:
        if any(row.get("id") is not None for row in (build_a, build_b)):
            raise ValueError("pending build image IDs must be absent")
        if container.get("accepted_image") is not None:
            raise ValueError("pending freeze cannot accept an image")
    elif outcome in {SUCCESS_OUTCOME, FAILURE_OUTCOME}:
        if any(
            not re.fullmatch(r"sha256:[0-9a-f]{64}", str(row.get("id")))
            for row in (build_a, build_b)
        ):
            raise ValueError("completed build image identity mismatch")
        expected_accepted = (
            {"build": "build_a", "id": build_a.get("id"), "tag": build_a.get("tag")}
            if outcome == SUCCESS_OUTCOME
            else None
        )
        if container.get("accepted_image") != expected_accepted:
            raise ValueError("accepted build A identity mismatch")
    else:
        raise ValueError("unsupported container freeze outcome")
    containerfile = root / _string(container, "containerfile")
    validator = root / _string(container, "validation_script")
    fingerprint_tool = root / _string(container, "fingerprint_tool")
    if _sha256(containerfile) != container.get("containerfile_sha256"):
        raise ValueError("Containerfile hash mismatch")
    if _sha256(validator) != container.get("validation_script_sha256"):
        raise ValueError("container validation script hash mismatch")
    if _sha256(fingerprint_tool) != container.get("fingerprint_tool_sha256"):
        raise ValueError("runtime content fingerprint tool hash mismatch")
    first_line = containerfile.read_text(
        encoding="utf-8", errors="strict"
    ).splitlines()[0]
    if first_line != f"FROM {base['tag']}@{base['digest']}":
        raise ValueError("base image must include both tag and digest")
    builds = _mapping(container, "builds")
    for name, tag in (
        ("build_a", "ngr-cross-encoder-precision-v8:freeze"),
        ("build_b", "ngr-cross-encoder-precision-v8:rebuild-check"),
    ):
        record = _mapping(builds, name)
        expected_record = {
            "command": [
                "wslc",
                "build",
                "--no-cache",
                "--file",
                "containers/github_cross_encoder_precision_v8/Containerfile",
                "--tag",
                tag,
                ".",
            ],
            "return_code": (
                0 if outcome in {SUCCESS_OUTCOME, FAILURE_OUTCOME} else None
            ),
            "wslc_version": "2.9.4.0",
        }
        if record != expected_record:
            raise ValueError(f"WSLC {name} one-shot build record mismatch")
    expected_runs = {
        "attestation": ["wslc", "run", "--rm", "--network", "none", "{tag}"],
        "runtime_content": [
            "wslc",
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "python",
            "{tag}",
            "/opt/ngr-v8/runtime_content.py",
        ],
    }
    if _mapping(container, "run_command_templates") != expected_runs:
        raise ValueError("offline WSLC report command mismatch")
    run_root = _mapping(platform, "run_root")
    if run_root != {
        "path": "/opt/ngr-v8/runtime",
        "filesystem": "container",
        "exclusive_create": True,
        "host_bind_mount": False,
        "shared_windows_database": False,
        "future_observation_volume": "github-cross-encoder-precision-v8-runtime",
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
    _validate_content_equivalence(protocol, build_a.get("id"), build_b.get("id"))


def _read_canonical_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8", errors="strict"))
    if not isinstance(value, dict) or raw != _CONTENT.canonical_json_bytes(value):
        raise ValueError(f"report is not canonical JSON: {path}")
    return value


def validate_attestation(value: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    if set(value) != ATTESTATION_FIELDS:
        raise ValueError("offline runtime attestation fields mismatch")
    expected_rows = _expected_distribution_rows(protocol)
    expected_versions = {row["canonical_name"]: row["version"] for row in expected_rows}
    distributions: list[dict[str, str]] = []
    actual: dict[str, str] = {}
    for row in value.get("distributions", []):
        if not isinstance(row, Mapping) or set(row) != {
            "canonical_name",
            "name",
            "version",
        }:
            raise ValueError("actual distribution row shape mismatch")
        name = _string(row, "name")
        version = _string(row, "version")
        canonical = _CONTENT.canonicalize_name(name)
        if row.get("canonical_name") != canonical or canonical in actual:
            raise ValueError("duplicate or mismatched actual canonical name")
        actual[canonical] = version
        distributions.append(dict(row))
    distributions.sort(key=lambda row: row["canonical_name"])
    if actual != expected_versions or len(distributions) != 29:
        raise ValueError("actual installed distributions do not match exact registry")
    expected_registry_path = (
        _path(protocol, "root")
        / "tests/fixtures/github_cross_encoder_precision_v8.expected-distributions.json"
    )
    expected = {
        "architecture": "amd64",
        "distributions": distributions,
        "expected_distribution_registry_sha256": _sha256(expected_registry_path),
        "filesystem_probe": "exclusive-create",
        "forbidden_distributions": [],
        "model_forward_inference_count": 0,
        "network": "disabled",
        "observed_result_count": 0,
        "os": "linux",
        "python": {"abi": "cp311", "implementation": "CPython", "version": "3.11.15"},
        "registered_query_count": 0,
        "synthetic_tensor_probe": {
            "device": "cpu",
            "dtype": "float32",
            "output": [-1.0, 4.0],
        },
        "torch_cuda": None,
    }
    if dict(value) != expected:
        raise ValueError("offline runtime attestation mismatch")


def validate_exact_installed_distributions(
    report: Mapping[str, Any],
    attestation: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    expected = {
        row["canonical_name"]: row["version"]
        for row in _expected_distribution_rows(protocol)
    }
    actual = {
        row["canonical_name"]: row["version"]
        for row in attestation.get("distributions", [])
    }
    filesystem = {
        row["canonical_name"]: row["version"]
        for row in report.get("filesystem_distributions", [])
    }
    if len(expected) != 29 or actual != expected or filesystem != expected:
        raise ValueError("expected/importlib/filesystem exact 29 inventories differ")
    diagnostic = _mapping(report, "metadata_correspondence")
    if (
        diagnostic.get("expected_distribution_count") != 29
        or diagnostic.get("actual_distribution_count") != 29
        or diagnostic.get("filesystem_distribution_count") != 29
    ):
        raise ValueError("METADATA diagnostic exact 29 counts differ")
    registry_sha = _sha256(
        _path(protocol, "root")
        / "tests/fixtures/github_cross_encoder_precision_v8.expected-distributions.json"
    )
    if (
        report.get("expected_distribution_registry_sha256") != registry_sha
        or attestation.get("expected_distribution_registry_sha256") != registry_sha
    ):
        raise ValueError("expected distribution registry attestation mismatch")


def validate_content_equivalence(
    report_a: Mapping[str, Any],
    report_b: Mapping[str, Any],
    attestation_a: Mapping[str, Any],
    attestation_b: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    image_id_a: str,
    image_id_b: str,
) -> None:
    del image_id_a, image_id_b
    _CONTENT.validate_report(report_a)
    _CONTENT.validate_report(report_b)
    validate_attestation(attestation_a, protocol)
    validate_attestation(attestation_b, protocol)
    if dict(report_a) != dict(report_b):
        raise ValueError("build A/B runtime content fingerprints differ")
    if dict(attestation_a) != dict(attestation_b):
        raise ValueError("build A/B offline runtime attestations differ")
    validate_exact_installed_distributions(report_a, attestation_a, protocol)
    validate_exact_installed_distributions(report_b, attestation_b, protocol)


def _validate_content_equivalence(
    protocol: Mapping[str, Any], image_id_a: str | None, image_id_b: str | None
) -> None:
    root = _path(protocol, "root")
    contract = _mapping(_mapping(protocol, "platform"), "content_equivalence")
    expected_paths = {
        "runtime_content_build_a": "tests/fixtures/github_cross_encoder_precision_v8.runtime-content.build-a.json",
        "runtime_content_build_b": "tests/fixtures/github_cross_encoder_precision_v8.runtime-content.build-b.json",
        "attestation_build_a": "tests/fixtures/github_cross_encoder_precision_v8.attestation.build-a.json",
        "attestation_build_b": "tests/fixtures/github_cross_encoder_precision_v8.attestation.build-b.json",
    }
    if contract.get("algorithm_version") != _CONTENT.ALGORITHM_VERSION:
        raise ValueError("runtime content algorithm version mismatch")
    if (
        contract.get("exclusion_registry_sha256")
        != _CONTENT.exclusion_registry_sha256()
    ):
        raise ValueError("runtime content exclusion registry mismatch")
    expected_registry_sha = _sha256(
        root
        / "tests/fixtures/github_cross_encoder_precision_v8.expected-distributions.json"
    )
    if contract.get("expected_distribution_registry_sha256") != expected_registry_sha:
        raise ValueError("content equivalence expected registry mismatch")
    outcome = contract.get("freeze_outcome")
    if outcome == PENDING_OUTCOME:
        for key, expected_path in expected_paths.items():
            record = _mapping(contract, key)
            if record != {"path": expected_path, "sha256": None}:
                raise ValueError("pending runtime report contract mismatch")
            if (root / expected_path).exists():
                raise ValueError("pending one-shot report path must not exist")
        if any(
            contract.get(key) is not None
            for key in (
                "fingerprint_sha256",
                "attestation_sha256",
                "metadata_correspondence_sha256",
            )
        ) or any(
            contract.get(key) is not False
            for key in (
                "fingerprint_reports_equal",
                "attestation_reports_equal",
                "filesystem_inventory_reports_equal",
                "metadata_correspondence_reports_equal",
                "exact_installed_distribution_set_attested",
                "successor_observation_allowed",
            )
        ):
            raise ValueError("pending content equivalence outcome mismatch")
        return
    if outcome == FAILURE_OUTCOME:
        failure_path = (
            "tests/evidence/github_cross_encoder_precision_v8/"
            "freeze-runtime-content.error.json"
        )
        partial_path = (
            "tests/evidence/github_cross_encoder_precision_v8/"
            "runtime-content.build-a.partial"
        )
        partial_sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        expected_failure_records = {
            "runtime_content_build_a": {
                "path": partial_path,
                "sha256": partial_sha,
                "run_count": 1,
                "return_code": 1,
            },
            "runtime_content_build_b": {
                "path": expected_paths["runtime_content_build_b"],
                "sha256": None,
                "run_count": 0,
                "return_code": None,
            },
            "attestation_build_a": {
                "path": expected_paths["attestation_build_a"],
                "sha256": None,
                "run_count": 0,
                "return_code": None,
            },
            "attestation_build_b": {
                "path": expected_paths["attestation_build_b"],
                "sha256": None,
                "run_count": 0,
                "return_code": None,
            },
        }
        for key, expected_record in expected_failure_records.items():
            if _mapping(contract, key) != expected_record:
                raise ValueError("failed runtime report contract mismatch")
        if _sha256(root / partial_path) != partial_sha:
            raise ValueError("partial runtime report evidence hash mismatch")
        for key in (
            "runtime_content_build_b",
            "attestation_build_a",
            "attestation_build_b",
        ):
            if (root / expected_paths[key]).exists():
                raise ValueError("unattempted offline report path must not exist")
        if any(
            contract.get(key) is not None
            for key in (
                "fingerprint_sha256",
                "attestation_sha256",
                "metadata_correspondence_sha256",
            )
        ) or any(
            contract.get(key) is not False
            for key in (
                "fingerprint_reports_equal",
                "attestation_reports_equal",
                "filesystem_inventory_reports_equal",
                "metadata_correspondence_reports_equal",
                "exact_installed_distribution_set_attested",
                "successor_observation_allowed",
            )
        ):
            raise ValueError("v8 failure equivalence outcome mismatch")
        failure = _mapping(contract, "failure_evidence")
        if failure.get("path") != failure_path or _sha256(
            root / failure_path
        ) != failure.get("sha256"):
            raise ValueError("v8 failure evidence registry mismatch")
        failure_value = _BASE.read_json(root / failure_path)
        failure_builds = _mapping(failure_value, "builds")
        reports = _mapping(failure_value, "reports")
        runtime_a = _mapping(reports, "runtime_content_build_a")
        exact_inventory = _mapping(failure_value, "exact_inventory")
        if (
            failure_value.get("outcome") != FAILURE_OUTCOME
            or failure_value.get("accepted_image") is not None
            or failure_value.get("successor_observation_allowed") is not False
            or failure_value.get("performance") != "not assessed"
            or failure_value.get("retry_allowed") is not False
            or _mapping(failure_value, "freeze_counts")
            != {
                "registered_query_count": 0,
                "model_forward_inference_count": 0,
                "observed_result_count": 0,
            }
            or _mapping(failure_builds, "build_a")
            != {
                "count": 1,
                "return_code": 0,
                "local_image_id": image_id_a,
            }
            or _mapping(failure_builds, "build_b")
            != {
                "count": 1,
                "return_code": 0,
                "local_image_id": image_id_b,
            }
            or failure_builds.get("additional_build_count") != 0
            or runtime_a
            != {
                "count": 1,
                "return_code": 1,
                "partial_output_path": partial_path,
                "partial_output_sha256": partial_sha,
                "partial_output_size": 0,
            }
            or any(
                _mapping(reports, key).get("count") != 0
                for key in (
                    "runtime_content_build_b",
                    "attestation_build_a",
                    "attestation_build_b",
                )
            )
            or reports.get("additional_report_run_count") != 0
            or exact_inventory
            != {
                "expected_distribution_count": 29,
                "actual_distribution_count": None,
                "filesystem_distribution_count": None,
                "three_way_match_established": False,
            }
        ):
            raise ValueError("v8 failure evidence content mismatch")
        return
    if outcome != SUCCESS_OUTCOME:
        raise ValueError("unsupported v8 freeze outcome")
    reports: dict[str, dict[str, Any]] = {}
    for key, expected_path in expected_paths.items():
        record = _mapping(contract, key)
        if record.get("path") != expected_path:
            raise ValueError("runtime report path mismatch")
        path = root / expected_path
        if _sha256(path) != record.get("sha256"):
            raise ValueError("runtime report registry hash mismatch")
        reports[key] = _read_canonical_json(path)
    validate_content_equivalence(
        reports["runtime_content_build_a"],
        reports["runtime_content_build_b"],
        reports["attestation_build_a"],
        reports["attestation_build_b"],
        protocol,
        image_id_a=image_id_a,
        image_id_b=image_id_b,
    )
    if contract.get("fingerprint_sha256") != reports["runtime_content_build_a"].get(
        "fingerprint_sha256"
    ):
        raise ValueError("recorded runtime fingerprint registry mismatch")
    attestation_sha = hashlib.sha256(
        _CONTENT.canonical_json_bytes(reports["attestation_build_a"])
    ).hexdigest()
    if contract.get("attestation_sha256") != attestation_sha:
        raise ValueError("recorded runtime attestation registry mismatch")
    diagnostic = _mapping(reports["runtime_content_build_a"], "metadata_correspondence")
    if contract.get("metadata_correspondence_sha256") != hashlib.sha256(
        _CONTENT.canonical_json_bytes(diagnostic)
    ).hexdigest():
        raise ValueError("recorded METADATA correspondence registry mismatch")
    if (
        contract.get("fingerprint_reports_equal") is not True
        or contract.get("attestation_reports_equal") is not True
        or contract.get("filesystem_inventory_reports_equal") is not True
        or contract.get("metadata_correspondence_reports_equal") is not True
        or contract.get("exact_installed_distribution_set_attested") is not True
        or contract.get("successor_observation_allowed") is not True
    ):
        raise ValueError("v8 success outcome contract mismatch")


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    audit = _mapping(protocol, "result_free_audit")
    if audit.get("count_scope") != (
        "v8 registered query/model inference/observed result counts only; "
        "historical v1/v2/v3/v4/v5/v6/v7 observations excluded"
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
    outcome = audit.get("freeze_outcome")
    if outcome not in {PENDING_OUTCOME, SUCCESS_OUTCOME, FAILURE_OUTCOME}:
        raise ValueError("v8 freeze outcome mismatch")
    if audit.get("container_attempt_count_scope") != (
        "v8 one-shot build A/B and their original offline report collection only"
    ):
        raise ValueError("container attempt count scope mismatch")
    completed = outcome == SUCCESS_OUTCOME
    failed = outcome == FAILURE_OUTCOME
    for key, expected in (
        ("one_shot_wslc_image_build_count", 2 if completed or failed else 0),
        ("runtime_content_report_count", 2 if completed else (1 if failed else 0)),
        ("offline_attestation_report_count", 2 if completed else 0),
        ("additional_wslc_image_build_count", 0),
        ("additional_offline_report_run_count", 0),
    ):
        if audit.get(key) != expected:
            raise ValueError(f"{key} mismatch")
    if (
        audit.get("accepted_image") is not completed
        or audit.get("successor_observation_allowed") is not completed
        or audit.get("performance") != "not assessed"
        or audit.get("failure_evidence")
        != (
            "tests/evidence/github_cross_encoder_precision_v8/"
            "freeze-runtime-content.error.json"
            if failed
            else None
        )
    ):
        raise ValueError("v8 audit outcome mismatch")

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
    _verify_hash_registry(root, _mapping(manifest, "v5_immutable_sha256"))
    _verify_hash_registry(root, _mapping(manifest, "v6_immutable_sha256"))
    _verify_hash_registry(root, _mapping(manifest, "v7_immutable_sha256"))
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
            for version in range(1, 8)
            for path in outputs
        )
    ):
        raise ValueError("v8 output path isolation mismatch")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit the result-free WSLC cross-encoder precision v8 protocol"
    )
    parser.add_argument("command", choices=("audit", "probe"))
    args = parser.parse_args(argv)
    protocol = load_protocol()
    container = _mapping(_mapping(protocol, "platform"), "container")
    output: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "freeze_registered_query_execution_count": 0,
        "freeze_model_inference_count": 0,
        "freeze_observed_result_count": 0,
        "historical_v1_v2_v3_v4_v5_v6_v7_observations_included": False,
        "freeze_outcome": protocol["result_free_audit"]["freeze_outcome"],
        "accepted_image_id": (
            container["accepted_image"]["id"]
            if container["accepted_image"] is not None
            else None
        ),
        "successor_observation_allowed": protocol["result_free_audit"][
            "successor_observation_allowed"
        ],
        "performance": "not assessed",
        "fingerprint_sha256": _mapping(
            _mapping(protocol, "platform"), "content_equivalence"
        )["fingerprint_sha256"],
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
