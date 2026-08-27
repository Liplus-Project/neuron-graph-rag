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
    / "github_cross_encoder_precision_v6"
    / "runtime_content.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "neuron_graph_rag._cross_encoder_precision_v6_frozen_base", _V3_SOURCE
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("unable to load frozen v3 evaluator")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)
_CONTENT_SPEC = importlib.util.spec_from_file_location(
    "neuron_graph_rag._cross_encoder_precision_v6_runtime_content", _CONTENT_SOURCE
)
if _CONTENT_SPEC is None or _CONTENT_SPEC.loader is None:
    raise RuntimeError("unable to load frozen runtime content fingerprint tool")
_CONTENT = importlib.util.module_from_spec(_CONTENT_SPEC)
_CONTENT_SPEC.loader.exec_module(_CONTENT)

STEM = "github_cross_encoder_precision_v6"
PROTOCOL_ID = "github-ngr-cross-encoder-precision-v6"
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

# The v3 evaluator is the literal rank-only semantic base for v4, v5, and v6.
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
    "torch_cuda",
    "forbidden_distributions",
    "network",
    "filesystem_probe",
    "synthetic_tensor_probe",
    "registered_query_count",
    "model_forward_inference_count",
    "observed_result_count",
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
            "tests/fixtures/github_cross_encoder_precision_v6.requirements.lock",
            "tests/fixtures/github_cross_encoder_precision_v6.requirements.in",
        ],
    }:
        raise ValueError("resolver and package routing contract mismatch")
    if _mapping(platform, "installer") != {
        "name": "pip",
        "version": "24.0",
        "only_binary": True,
        "require_hashes": True,
        "ignore_installed": True,
        "report_path": "/opt/ngr-v6/dependency-report.json",
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

    v5_lock = (
        root / "tests/fixtures/github_cross_encoder_precision_v5.requirements.lock"
    ).read_text(encoding="utf-8", errors="strict")
    if _locked_versions(lock) != _locked_versions(v5_lock):
        raise ValueError("v5/v6 dependency version drift")
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
    images = _mapping(container, "images")
    build_a = _mapping(images, "build_a")
    build_b = _mapping(images, "build_b")
    if (
        container.get("os") != "linux"
        or container.get("architecture") != "amd64"
        or base.get("tag") != "python:3.11.15-slim-bookworm"
        or base.get("digest")
        != "sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3"
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(base.get("local_image_id")))
        or build_a.get("tag") != "ngr-cross-encoder-precision-v6:freeze"
        or build_b.get("tag") != "ngr-cross-encoder-precision-v6:rebuild-check"
        or any(
            not re.fullmatch(r"sha256:[0-9a-f]{64}", str(row.get("id")))
            for row in (build_a, build_b)
        )
        or container.get("accepted_image") is not None
    ):
        raise ValueError("pinned Linux amd64 image identity mismatch")
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
        ("build_a", "ngr-cross-encoder-precision-v6:freeze"),
        ("build_b", "ngr-cross-encoder-precision-v6:rebuild-check"),
    ):
        record = _mapping(builds, name)
        if record != {
            "command": [
                "wslc",
                "build",
                "--no-cache",
                "--file",
                "containers/github_cross_encoder_precision_v6/Containerfile",
                "--tag",
                tag,
                ".",
            ],
            "return_code": 0,
            "wslc_version": "2.9.4.0",
        }:
            raise ValueError(f"WSLC {name} one-shot build record mismatch")
    expected_runs = {
        "attestation": [
            "wslc", "run", "--rm", "--network", "none", "{tag}"
        ],
        "runtime_content": [
            "wslc", "run", "--rm", "--network", "none", "--entrypoint",
            "python", "{tag}", "/opt/ngr-v6/runtime_content.py"
        ],
    }
    if _mapping(container, "run_command_templates") != expected_runs:
        raise ValueError("offline WSLC report command mismatch")
    run_root = _mapping(platform, "run_root")
    if run_root != {
        "path": "/opt/ngr-v6/runtime",
        "filesystem": "container",
        "exclusive_create": True,
        "host_bind_mount": False,
        "shared_windows_database": False,
        "future_observation_volume": "github-cross-encoder-precision-v6-runtime",
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
    _validate_content_equivalence(protocol, build_a["id"], build_b["id"])


def _read_canonical_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8", errors="strict"))
    if not isinstance(value, dict) or raw != _CONTENT.canonical_json_bytes(value):
        raise ValueError(f"report is not canonical JSON: {path}")
    return value


def validate_attestation(value: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    if set(value) != ATTESTATION_FIELDS:
        raise ValueError("offline runtime attestation fields mismatch")
    expected_versions = {
        row["name"]: row["version"]
        for row in _BASE._object_list(
            _mapping(protocol, "dependency_artifacts"), "artifacts"
        )
    }
    expected = {
        "architecture": "amd64",
        "distributions": [
            {"name": name, "version": expected_versions[name]}
            for name in sorted(expected_versions)
        ],
        "filesystem_probe": "exclusive-create",
        "forbidden_distributions": [],
        "model_forward_inference_count": 0,
        "network": "disabled",
        "observed_result_count": 0,
        "os": "linux",
        "python": {"abi": "cp311", "implementation": "CPython", "version": "3.11.15"},
        "registered_query_count": 0,
        "synthetic_tensor_probe": {
            "device": "cpu", "dtype": "float32", "output": [-1.0, 4.0]
        },
        "torch_cuda": None,
    }
    if dict(value) != expected:
        raise ValueError("offline runtime attestation mismatch")


class ExactInstalledDistributionError(ValueError):
    def __init__(self, *, extras: list[str], missing: list[str]) -> None:
        self.extras = extras
        self.missing = missing
        super().__init__(
            "offline attestation is not the exact installed distribution set"
        )


def installed_distribution_delta(
    report: Mapping[str, Any], attestation: Mapping[str, Any]
) -> tuple[list[str], list[str]]:
    installed: dict[str, str] = {}
    for entry in report.get("normalized_entries", []):
        path = str(entry.get("path", ""))
        parts = path.split("/")
        if (
            len(parts) == 3
            and parts[0] == "site-packages"
            and parts[1].endswith(".dist-info")
            and parts[2] == "METADATA"
        ):
            stem = parts[1][: -len(".dist-info")]
            name, separator, _version = stem.rpartition("-")
            if not separator:
                raise ValueError("installed dist-info identity is malformed")
            canonical = re.sub(r"[-_.]+", "-", name).lower()
            if canonical in installed:
                raise ValueError("duplicate installed distribution identity")
            installed[canonical] = parts[1]
    declared = {
        re.sub(r"[-_.]+", "-", str(row.get("name", ""))).lower()
        for row in attestation.get("distributions", [])
    }
    extras = sorted(installed[name] for name in set(installed) - declared)
    missing = sorted(declared - set(installed))
    return extras, missing


def validate_exact_installed_distributions(
    report: Mapping[str, Any], attestation: Mapping[str, Any]
) -> None:
    extras, missing = installed_distribution_delta(report, attestation)
    if extras or missing:
        raise ExactInstalledDistributionError(extras=extras, missing=missing)


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
    validate_exact_installed_distributions(report_a, attestation_a)
    validate_exact_installed_distributions(report_b, attestation_b)


def _validate_content_equivalence(
    protocol: Mapping[str, Any], image_id_a: str, image_id_b: str
) -> None:
    root = _path(protocol, "root")
    contract = _mapping(_mapping(protocol, "platform"), "content_equivalence")
    expected_paths = {
        "runtime_content_build_a": "tests/fixtures/github_cross_encoder_precision_v6.runtime-content.build-a.json",
        "runtime_content_build_b": "tests/fixtures/github_cross_encoder_precision_v6.runtime-content.build-b.json",
        "attestation_build_a": "tests/fixtures/github_cross_encoder_precision_v6.attestation.build-a.json",
        "attestation_build_b": "tests/fixtures/github_cross_encoder_precision_v6.attestation.build-b.json",
    }
    if contract.get("algorithm_version") != _CONTENT.ALGORITHM_VERSION:
        raise ValueError("runtime content algorithm version mismatch")
    if contract.get("exclusion_registry_sha256") != _CONTENT.exclusion_registry_sha256():
        raise ValueError("runtime content exclusion registry mismatch")
    reports: dict[str, dict[str, Any]] = {}
    for key, expected_path in expected_paths.items():
        record = _mapping(contract, key)
        if record.get("path") != expected_path:
            raise ValueError("runtime report path mismatch")
        path = root / expected_path
        if _sha256(path) != record.get("sha256"):
            raise ValueError("runtime report registry hash mismatch")
        reports[key] = _read_canonical_json(path)
    try:
        validate_content_equivalence(
            reports["runtime_content_build_a"],
            reports["runtime_content_build_b"],
            reports["attestation_build_a"],
            reports["attestation_build_b"],
            protocol,
            image_id_a=image_id_a,
            image_id_b=image_id_b,
        )
    except ExactInstalledDistributionError as error:
        if error.extras != [
            "pip-24.0.dist-info",
            "setuptools-79.0.1.dist-info",
            "wheel-0.46.3.dist-info",
        ] or error.missing:
            raise ValueError("unexpected exact-distribution attestation failure") from error
    else:
        raise ValueError("v6 must remain failed after incomplete attestation")
    if contract.get("fingerprint_sha256") != reports["runtime_content_build_a"].get("fingerprint_sha256"):
        raise ValueError("recorded runtime fingerprint registry mismatch")
    attestation_sha = hashlib.sha256(
        _CONTENT.canonical_json_bytes(reports["attestation_build_a"])
    ).hexdigest()
    if contract.get("attestation_sha256") != attestation_sha:
        raise ValueError("recorded runtime attestation registry mismatch")
    if (
        contract.get("fingerprint_reports_equal") is not True
        or contract.get("attestation_reports_equal") is not True
        or contract.get("exact_installed_distribution_set_attested") is not False
        or contract.get("freeze_outcome")
        != "fail_closed_offline_attestation_not_exact"
        or contract.get("successor_observation_allowed") is not False
    ):
        raise ValueError("v6 failure outcome contract mismatch")
    failure = _mapping(contract, "failure_evidence")
    failure_path = "tests/evidence/github_cross_encoder_precision_v6/freeze-attestation.error.json"
    if failure.get("path") != failure_path or _sha256(root / failure_path) != failure.get("sha256"):
        raise ValueError("v6 failure evidence registry mismatch")
    failure_value = _BASE.read_json(root / failure_path)
    if (
        failure_value.get("outcome") != "fail_closed_offline_attestation_not_exact"
        or failure_value.get("accepted_image") is not None
        or failure_value.get("successor_observation_allowed") is not False
        or failure_value.get("performance") != "not assessed"
        or _mapping(failure_value, "freeze_counts")
        != {
            "registered_query_count": 0,
            "model_forward_inference_count": 0,
            "observed_result_count": 0,
        }
        or _mapping(_mapping(failure_value, "builds"), "build_a").get("count") != 1
        or _mapping(_mapping(failure_value, "builds"), "build_b").get("count") != 1
        or _mapping(failure_value, "builds").get("additional_build_count") != 0
        or _mapping(failure_value, "offline_attestation").get(
            "additional_report_run_count"
        )
        != 0
    ):
        raise ValueError("v6 failure evidence content mismatch")


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    audit = _mapping(protocol, "result_free_audit")
    if audit.get("count_scope") != (
        "v6 registered query/model inference/observed result counts only; "
        "historical v1/v2/v3/v4/v5 observations excluded"
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
    if audit.get("freeze_outcome") != "fail_closed_offline_attestation_not_exact":
        raise ValueError("v6 freeze outcome mismatch")
    if audit.get("container_attempt_count_scope") != (
        "v6 one-shot build A/B and their original offline report collection only"
    ):
        raise ValueError("container attempt count scope mismatch")
    for key, expected in (
        ("one_shot_wslc_image_build_count", 2),
        ("runtime_content_report_count", 2),
        ("offline_attestation_report_count", 2),
        ("additional_wslc_image_build_count", 0),
        ("additional_offline_report_run_count", 0),
    ):
        if audit.get(key) != expected:
            raise ValueError(f"{key} mismatch")
    if (
        audit.get("accepted_image") is not False
        or audit.get("successor_observation_allowed") is not False
        or audit.get("performance") != "not assessed"
        or audit.get("failure_evidence")
        != "tests/evidence/github_cross_encoder_precision_v6/freeze-attestation.error.json"
    ):
        raise ValueError("fail-closed audit outcome mismatch")

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
            for version in range(1, 6)
            for path in outputs
        )
    ):
        raise ValueError("v6 output path isolation mismatch")
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
        description="Audit the result-free WSLC cross-encoder precision v6 protocol"
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
        "historical_v1_v2_v3_v4_v5_observations_included": False,
        "freeze_outcome": "fail_closed_offline_attestation_not_exact",
        "accepted_image_id": None,
        "successor_observation_allowed": False,
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
