from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

_V3_SOURCE = Path(__file__).with_name("cross_encoder_precision_v3_evaluation.py")
_SPEC = importlib.util.spec_from_file_location(
    "neuron_graph_rag._cross_encoder_precision_v4_frozen_base", _V3_SOURCE
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("unable to load frozen v3 evaluator")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

STEM = "github_cross_encoder_precision_v4"
PROTOCOL_ID = "github-ngr-cross-encoder-precision-v4"
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
    "requirements.lock",
)

# Load the frozen v3 evaluator into an isolated module namespace, then bind only
# the v4 identity and artifact surface. Rank-only scoring, candidate ordering,
# hard gates, lifecycle, and tamper rejection therefore remain literally v3.
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


PLATFORM_METADATA_FIELDS = {
    "protocol_id",
    "substrate",
    "wsl_version",
    "os",
    "architecture",
    "libc",
    "platform_tag",
    "python_version",
    "python_artifact_sha256",
    "resolver",
    "resolver_version",
    "dependency_lock_sha256",
    "run_root",
    "run_root_filesystem",
    "paths",
}
RUNTIME_PATH_FIELDS = {
    "environment",
    "database",
    "worker_output",
    "runtime_claim",
    "runtime_result",
    "runtime_error",
    "archive_claim",
    "archive_result",
    "archive_error",
    "transport",
}
_VALIDATE_V3_SEMANTICS = _BASE.validate_protocol


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    audit = _mapping(protocol, "result_free_audit")
    if audit.get("count_scope") != (
        "v4 result-free freeze only; historical v1/v2/v3 observations excluded"
    ):
        raise ValueError("result-free count scope mismatch")
    for key in (
        "model_cache_opened",
        "model_weights_opened",
        "venv_opened",
        "wsl_observation_executed",
    ):
        if audit.get(key) is not False:
            raise ValueError(f"{key} must remain false at freeze")

    # The frozen v3 validator owns every unchanged semantic. Adapt only its
    # versioned audit literal before invoking it.
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
    platform = _mapping(protocol, "platform")
    expected_platform = {
        "protocol_id": PROTOCOL_ID,
        "substrate": "WSL2",
        "wsl_version": 2,
        "distribution": "Ubuntu",
        "os": "linux",
        "architecture": "x86_64",
        "libc": "gnu",
        "platform_tag": "x86_64-unknown-linux-gnu",
    }
    for key, expected in expected_platform.items():
        if platform.get(key) != expected:
            raise ValueError(f"platform contract {key} mismatch")
    python = _mapping(platform, "python")
    if python != {
        "implementation": "CPython",
        "version": "3.11.15",
        "uv_key": "cpython-3.11.15-linux-x86_64-gnu",
        "source": "python-build-standalone",
        "release": "20260807",
        "artifact_url": (
            "https://github.com/astral-sh/python-build-standalone/releases/"
            "download/20260807/cpython-3.11.15%2B20260807-"
            "x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
        ),
        "artifact_size": 30939767,
        "artifact_sha256": (
            "69dfac9d0f15a0b9281a38486f212cbf76421609228c184dc0d34a0533d57ba6"
        ),
    }:
        raise ValueError("Python artifact contract mismatch")
    resolver = _mapping(platform, "resolver")
    if (
        resolver.get("name") != "uv"
        or resolver.get("version") != "0.12.3"
        or resolver.get("build") != "507230998"
        or resolver.get("host") != "x86_64-pc-windows-msvc"
    ):
        raise ValueError("resolver contract mismatch")
    lock = _mapping(platform, "dependency_lock")
    lock_bytes = _string(protocol, "requirements_lock").encode("utf-8")
    if hashlib.sha256(lock_bytes).hexdigest() != lock.get("sha256"):
        raise ValueError("Linux dependency lock hash mismatch")
    input_path = root / f"tests/fixtures/{STEM}.requirements.in"
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != lock.get("input_sha256"):
        raise ValueError("dependency input hash mismatch")
    outputs = [
        path
        for stage in _mapping(manifest, "outputs").values()
        for path in stage.values()
    ]
    if (
        len(outputs) != len(set(outputs))
        or any(STEM not in path for path in outputs)
        or any("github_cross_encoder_precision_v3" in path for path in outputs)
    ):
        raise ValueError("v4 output path isolation mismatch")


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


def validate_runtime_platform(
    protocol: Mapping[str, Any], metadata: Mapping[str, Any]
) -> None:
    if set(metadata) != PLATFORM_METADATA_FIELDS:
        raise ValueError("runtime platform metadata fields mismatch")
    contract = _mapping(protocol, "platform")
    expected = {
        "protocol_id": PROTOCOL_ID,
        "substrate": contract["substrate"],
        "wsl_version": contract["wsl_version"],
        "os": contract["os"],
        "architecture": contract["architecture"],
        "libc": contract["libc"],
        "platform_tag": contract["platform_tag"],
        "python_version": contract["python"]["version"],
        "python_artifact_sha256": contract["python"]["artifact_sha256"],
        "resolver": contract["resolver"]["name"],
        "resolver_version": contract["resolver"]["version"],
        "dependency_lock_sha256": contract["dependency_lock"]["sha256"],
        "run_root_filesystem": contract["run_root"]["filesystem"],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"runtime platform {key} mismatch")

    run_root_raw = metadata.get("run_root")
    if not isinstance(run_root_raw, str):
        raise TypeError("run_root must be a POSIX path")
    run_root = PurePosixPath(run_root_raw)
    if not run_root.is_absolute() or run_root == PurePosixPath("/"):
        raise ValueError("run_root must be a dedicated absolute path")
    if run_root == PurePosixPath("/mnt") or PurePosixPath("/mnt") in run_root.parents:
        raise ValueError("run_root must not use a WSL Windows mount")

    paths = metadata.get("paths")
    if not isinstance(paths, Mapping) or set(paths) != RUNTIME_PATH_FIELDS:
        raise ValueError("runtime path fields mismatch")
    if len(set(paths.values())) != len(RUNTIME_PATH_FIELDS):
        raise ValueError("runtime paths must be exclusive")
    for key, value in paths.items():
        if not isinstance(value, str):
            raise TypeError(f"runtime path {key} must be a POSIX path")
        path = PurePosixPath(value)
        try:
            path.relative_to(run_root)
        except ValueError as exc:
            raise ValueError(f"runtime path {key} must stay under run_root") from exc


def build_synthetic_platform_metadata(
    protocol: Mapping[str, Any], run_root: str = "/home/ngr/v4-run"
) -> dict[str, Any]:
    contract = _mapping(protocol, "platform")
    root = PurePosixPath(run_root)
    relative_paths = {
        "environment": ".venv",
        "database": "db/knowledge.db",
        "worker_output": "worker/development",
        "runtime_claim": "runtime/development.claim.json",
        "runtime_result": "runtime/development.observed.json",
        "runtime_error": "runtime/development.error.json",
        "archive_claim": "archive/development.claim.json",
        "archive_result": "archive/development.observed.json",
        "archive_error": "archive/development.error.json",
        "transport": "transport/development.transport.json",
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "substrate": contract["substrate"],
        "wsl_version": contract["wsl_version"],
        "os": contract["os"],
        "architecture": contract["architecture"],
        "libc": contract["libc"],
        "platform_tag": contract["platform_tag"],
        "python_version": contract["python"]["version"],
        "python_artifact_sha256": contract["python"]["artifact_sha256"],
        "resolver": contract["resolver"]["name"],
        "resolver_version": contract["resolver"]["version"],
        "dependency_lock_sha256": contract["dependency_lock"]["sha256"],
        "run_root": str(root),
        "run_root_filesystem": contract["run_root"]["filesystem"],
        "paths": {key: str(root / value) for key, value in relative_paths.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit the result-free Linux cross-encoder precision protocol"
    )
    parser.add_argument("command", choices=("audit", "probe"))
    args = parser.parse_args(argv)
    protocol = load_protocol()
    synthetic_platform = build_synthetic_platform_metadata(protocol)
    validate_runtime_platform(protocol, synthetic_platform)
    output: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "freeze_registered_query_execution_count": 0,
        "freeze_model_inference_count": 0,
        "freeze_observed_result_count": 0,
        "historical_v1_v2_v3_observations_included": False,
        "platform_contract_sha256": hashlib.sha256(
            json.dumps(
                protocol["platform"], ensure_ascii=False, sort_keys=True
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
