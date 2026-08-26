from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .cross_encoder_precision_v4_evaluation import (
    PROTOCOL_ID,
    ROOT,
    archive_stage,
    evaluate_result_payload,
    load_protocol,
    project_passages,
    read_json,
    register_stage_claim,
    sha256_bytes,
    verify_phase_state,
    verify_protocol_commit,
    verify_result_payload,
    write_json_exclusive,
    write_stage_error,
    write_stage_result,
)

_V3_SOURCE = Path(__file__).with_name("cross_encoder_precision_v3_observation.py")
_SPEC = importlib.util.spec_from_file_location(
    "neuron_graph_rag._cross_encoder_precision_v4_observation_base", _V3_SOURCE
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("unable to load frozen v3 observation harness")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

PROTOCOL_COMMIT = "a79e801483d656d401336198a5cc56887a286842"
SOURCE_COMMIT = "c32b3049fd3daaa2190faf5e3e85955a195ee88c"
RUN_ROOT = Path("/home/hal/ngr-experiments/github_cross_encoder_precision_v4")
EVIDENCE = Path("tests/evidence/github_cross_encoder_precision_v4")
BATCH_SIZE = 8
PYTHON_ARTIFACT = Path("downloads/cpython-3.11.15.tar.gz")
BOOTSTRAP_LOG = Path("bootstrap-commands.tsv")
EXCLUSIVE_MARKER = Path("exclusive-create.json")
MODEL_REPORT = "model-verification.json"
PREFLIGHT_FILES = (
    "preflight.json",
    MODEL_REPORT,
    "dependency-report.json",
    "platform-report.json",
    "preflight-commands.json",
)
PRECLAIM = "preclaim.json"
SHARED_BEFORE_PREFLIGHT = "shared-db-before-preflight.sha256"
CI_GREEN = "preflight-ci-green.json"

_ORIGINAL_RUN_STAGE_ONCE = _BASE._run_stage_once
_ORIGINAL_WORKER_ENVIRONMENT = _BASE._worker_environment


def canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def default_external_root() -> Path:
    return RUN_ROOT


def default_model_cache() -> Path:
    return RUN_ROOT / "model-cache"


def shared_database_path() -> Path:
    return Path("/mnt/c/Users/smile/.ngrdb/knowledge.db")


def experiment_python(external_root: Path) -> Path:
    return external_root / ".venv" / "bin" / "python"


def _bind_frozen_base() -> None:
    bindings = {
        "PROTOCOL_ID": PROTOCOL_ID,
        "PROTOCOL_COMMIT": PROTOCOL_COMMIT,
        "SOURCE_COMMIT": SOURCE_COMMIT,
        "ROOT": ROOT,
        "EVIDENCE": EVIDENCE,
        "BATCH_SIZE": BATCH_SIZE,
        "load_protocol": load_protocol,
        "archive_stage": archive_stage,
        "evaluate_result_payload": evaluate_result_payload,
        "project_passages": project_passages,
        "read_json": read_json,
        "register_stage_claim": register_stage_claim,
        "sha256_bytes": sha256_bytes,
        "verify_phase_state": verify_phase_state,
        "verify_protocol_commit": verify_protocol_commit,
        "verify_result_payload": verify_result_payload,
        "write_json_exclusive": write_json_exclusive,
        "write_stage_error": write_stage_error,
        "write_stage_result": write_stage_result,
        "default_external_root": default_external_root,
        "default_model_cache": default_model_cache,
        "shared_database_path": shared_database_path,
        "experiment_python": experiment_python,
        "canonical_sha256": canonical_sha256,
    }
    for name, value in bindings.items():
        setattr(_BASE, name, value)


_bind_frozen_base()


def _require_linux_run_root(root: Path, external: Path) -> dict[str, Any]:
    if os.name != "posix" or sys.platform != "linux":
        raise OSError("v4 observation is Linux-only")
    external = external.resolve()
    if external != RUN_ROOT:
        raise ValueError("external root must equal the frozen WSL run root")
    if root.resolve() != (external / "source").resolve():
        raise ValueError("frozen source checkout must be run-root/source")
    if PurePosixPath(str(external)) == PurePosixPath("/mnt") or PurePosixPath(
        "/mnt"
    ) in PurePosixPath(str(external)).parents:
        raise ValueError("run root must not use a WSL Windows mount")
    marker = external / EXCLUSIVE_MARKER
    if not marker.is_file():
        raise FileNotFoundError("exclusive-create marker is unavailable")
    marker_value = read_json(marker)
    if (
        marker_value.get("protocol_id") != PROTOCOL_ID
        or marker_value.get("run_root") != str(RUN_ROOT)
        or marker_value.get("absent_before_create") is not True
        or marker_value.get("exclusive_create_returncode") != 0
    ):
        raise ValueError("exclusive-create marker mismatch")
    findmnt = subprocess.run(
        ["findmnt", "-n", "-o", "FSTYPE", "--target", str(external)],
        check=False,
        capture_output=True,
        text=True,
    )
    filesystem = findmnt.stdout.strip()
    if findmnt.returncode != 0 or filesystem != "ext4":
        raise ValueError("run root filesystem must be ext4")
    source_findmnt = subprocess.run(
        ["findmnt", "-n", "-o", "FSTYPE", "--target", str(root)],
        check=False,
        capture_output=True,
        text=True,
    )
    if source_findmnt.returncode != 0 or source_findmnt.stdout.strip() != "ext4":
        raise ValueError("source checkout filesystem must be ext4")
    return {
        "run_root": str(external),
        "run_root_filesystem": filesystem,
        "source_root": str(root.resolve()),
        "source_filesystem": source_findmnt.stdout.strip(),
        "exclusive_create": marker_value,
    }


def _parse_bootstrap_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != 6:
            raise ValueError("bootstrap command log row shape mismatch")
        sequence, command, returncode, stdout_sha256, stderr_sha256, command_sha256 = (
            fields
        )
        row = {
            "sequence": int(sequence),
            "command": command,
            "returncode": int(returncode),
            "stdout_sha256": stdout_sha256,
            "stderr_sha256": stderr_sha256,
            "command_sha256": command_sha256,
        }
        if row["command_sha256"] != sha256_bytes(command.encode("utf-8")):
            raise ValueError("bootstrap command hash mismatch")
        rows.append(row)
    if not rows or [row["sequence"] for row in rows] != list(
        range(1, len(rows) + 1)
    ):
        raise ValueError("bootstrap command sequence mismatch")
    if any(row["returncode"] != 0 for row in rows):
        raise ValueError("bootstrap command did not complete successfully")
    return rows


def _runtime_platform_report(root: Path, external: Path) -> dict[str, Any]:
    contract = load_protocol(root)["platform"]
    implementation = platform.python_implementation()
    version = platform.python_version()
    architecture = platform.machine()
    libc_name, libc_version = platform.libc_ver()
    if implementation != "CPython" or version != "3.11.15":
        raise ValueError("exact CPython 3.11.15 is required")
    if architecture != "x86_64" or libc_name != "glibc":
        raise ValueError("Linux x86_64 GNU runtime is required")
    proc_version = Path("/proc/version").read_text(encoding="utf-8").strip()
    if "microsoft-standard-WSL2" not in proc_version:
        raise ValueError("WSL2 kernel identity mismatch")
    artifact = external / PYTHON_ARTIFACT
    artifact_raw = artifact.read_bytes()
    expected_python = contract["python"]
    if len(artifact_raw) != expected_python["artifact_size"]:
        raise ValueError("Python artifact size mismatch")
    artifact_sha = sha256_bytes(artifact_raw)
    if artifact_sha != expected_python["artifact_sha256"]:
        raise ValueError("Python artifact SHA-256 mismatch")
    uv = external / "tools" / "uv"
    uv_result = subprocess.run(
        [str(uv), "--version"], check=False, capture_output=True, text=True
    )
    if uv_result.returncode != 0 or uv_result.stdout.strip() != "uv 0.12.3":
        raise ValueError("uv 0.12.3 runtime mismatch")
    path_report = _require_linux_run_root(root, external)
    runtime_paths = {
        "environment": external / ".venv",
        "database": external / "databases" / "development",
        "worker_output": external / "runs" / "development",
        "runtime_claim": root / "runtime/github_cross_encoder_precision_v4/development.claim.json",
        "runtime_result": root / "runtime/github_cross_encoder_precision_v4/development.observed.json",
        "runtime_error": root / "runtime/github_cross_encoder_precision_v4/development.error.json",
        "archive_claim": root / "archive/github_cross_encoder_precision_v4/development.claim.json",
        "archive_result": root / "archive/github_cross_encoder_precision_v4/development.observed.json",
        "archive_error": root / "archive/github_cross_encoder_precision_v4/development.error.json",
        "transport": root / "transport/github_cross_encoder_precision_v4/development.transport.json",
    }
    resolved = {key: str(value.resolve()) for key, value in runtime_paths.items()}
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("runtime path isolation mismatch")
    for value in resolved.values():
        Path(value).relative_to(external)
    return {
        "protocol_id": PROTOCOL_ID,
        "substrate": "WSL2",
        "wsl_version": 2,
        "os": "linux",
        "architecture": architecture,
        "libc": "gnu",
        "libc_runtime": libc_name,
        "libc_version": libc_version,
        "platform_tag": contract["platform_tag"],
        "python_implementation": implementation,
        "python_version": version,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_artifact_size": len(artifact_raw),
        "python_artifact_sha256": artifact_sha,
        "uv_version": uv_result.stdout.strip().removeprefix("uv "),
        "kernel": platform.release(),
        "proc_version_sha256": sha256_bytes(proc_version.encode("utf-8")),
        "paths": resolved,
        **path_report,
    }


def _locked_versions(lock: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in lock.read_text(encoding="utf-8").splitlines():
        if not line or line[0].isspace() or line.startswith("#") or "==" not in line:
            continue
        name, value = line.split("==", 1)
        result[name.lower().replace("_", "-")] = value.split(" ", 1)[0].rstrip("\\")
    return result


def _dependency_report(root: Path) -> dict[str, Any]:
    lock = root / "tests/fixtures/github_cross_encoder_precision_v4.requirements.lock"
    lock_sha = sha256_bytes(lock.read_bytes())
    if lock_sha != "db3310ea9f1b27b63d0c4e4085223502e3353787835c6233355ae3c23bff6df4":
        raise ValueError("Linux dependency lock SHA-256 mismatch")
    locked = _locked_versions(lock)
    installed = {
        dist.metadata["Name"].lower().replace("_", "-"): dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    }
    missing = {
        name: version
        for name, version in locked.items()
        if installed.get(name) != version
    }
    if missing:
        raise ValueError(f"locked distribution mismatch: {sorted(missing)}")
    if installed.get("torch") != "2.4.1+cpu":
        raise ValueError("CPU-only torch 2.4.1+cpu is required")
    forbidden = sorted(
        name for name in installed if name == "triton" or name.startswith("nvidia-")
    )
    if forbidden:
        raise ValueError("CUDA/triton/nvidia distributions must be absent")
    import torch

    if torch.version.cuda is not None or torch.cuda.is_available():
        raise ValueError("torch runtime must be CPU-only")
    return {
        "protocol_id": PROTOCOL_ID,
        "lock_path": "tests/fixtures/github_cross_encoder_precision_v4.requirements.lock",
        "lock_sha256": lock_sha,
        "install_mode": "require-hashes",
        "locked_distributions": locked,
        "installed_distributions": dict(sorted(installed.items())),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
        "forbidden_distributions": forbidden,
    }


def model_copy_verify(source_cache: Path, cache: Path, output: Path) -> None:
    if cache.exists():
        raise FileExistsError("dedicated ext4 model cache already exists")
    protocol = load_protocol()
    source_rows = []
    destination_rows = []
    cache.mkdir(parents=True, exist_ok=False)
    for spec in _BASE._rows(_BASE._mapping(protocol, "models"), "models"):
        model_id = _BASE._string(spec, "model_id")
        revision = _BASE._string(spec, "revision")
        source = _BASE._snapshot_path(source_cache, model_id, revision)
        source_report = _BASE._verify_snapshot(spec, source)
        destination = cache / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision
        for frozen in _BASE._rows(spec, "required_files"):
            relative = _BASE._string(frozen, "path")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with (source / relative).open("rb") as source_handle, target.open("xb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
        destination_report = _BASE._verify_snapshot(spec, destination)
        for frozen in _BASE._rows(spec, "required_files"):
            relative = _BASE._string(frozen, "path")
            if sha256_bytes((source / relative).read_bytes()) != sha256_bytes(
                (destination / relative).read_bytes()
            ):
                raise ValueError("model cache copy byte identity mismatch")
        source_rows.append(source_report)
        destination_rows.append(destination_report)
    write_json_exclusive(
        output,
        {
            "protocol_id": PROTOCOL_ID,
            "protocol_commit": PROTOCOL_COMMIT,
            "source_cache_path": str(source_cache.resolve()),
            "cache_path": str(cache.resolve()),
            "source_models": source_rows,
            "models": destination_rows,
            "all_required_files_byte_identical": True,
            "source_cache_read_only": True,
        },
    )


def _verify_model_report(
    protocol: Mapping[str, Any], report: Mapping[str, Any], cache: Path
) -> None:
    if (
        report.get("protocol_id") != PROTOCOL_ID
        or report.get("protocol_commit") != PROTOCOL_COMMIT
        or report.get("cache_path") != str(cache.resolve())
        or report.get("all_required_files_byte_identical") is not True
        or report.get("source_cache_read_only") is not True
    ):
        raise ValueError("model verification report identity mismatch")
    frozen = _BASE._rows(_BASE._mapping(protocol, "models"), "models")
    rows = _BASE._rows(report, "models")
    if len(rows) != len(frozen):
        raise ValueError("model verification report cardinality mismatch")
    for spec, row in zip(frozen, rows, strict=True):
        if row != _BASE._verify_snapshot(spec, Path(_BASE._string(row, "snapshot_path"))):
            raise ValueError("model verification report is not reproducible")


def _verify_preclaim(root: Path) -> dict[str, Any]:
    value = read_json(root / EVIDENCE / PRECLAIM)
    previous = value.get("clean_handoff", {}).get("previous_executor_state", {})
    successor = value.get("successor_executor", {})
    zero_state = value.get("preclaim_state", {})
    if (
        value.get("protocol_id") != PROTOCOL_ID
        or value.get("issue") != 151
        or value.get("clean_handoff", {}).get("source_comment_url")
        != "https://github.com/Liplus-Project/neuron-graph-rag/issues/151#issuecomment-5424136323"
        or previous
        != {
            "development_claim_count": 0,
            "holdout_claim_count": 0,
            "run_root_absent": True,
            "model_copy_count": 0,
            "model_load_count": 0,
            "model_inference_count": 0,
            "result_count": 0,
            "error_count": 0,
        }
        or successor.get("task_identity")
        != "/root/observe_linux_rankonly_v4_clean"
        or successor.get("distinct_from_previous_executor") is not True
        or successor.get("forbidden_semantic_content_opened") is not False
        or successor.get("semantic_unread_attested") is not True
        or zero_state.get("development_claim_count") != 0
        or zero_state.get("holdout_claim_count") != 0
        or zero_state.get("run_root_absent") is not True
        or zero_state.get("model_activity_count") != 0
        or zero_state.get("result_count") != 0
        or zero_state.get("error_count") != 0
    ):
        raise ValueError("preclaim provenance mismatch")
    return value


def _verify_ci_green(
    root: Path, external: Path, preflight_report: Mapping[str, Any]
) -> dict[str, Any]:
    value = read_json(external / CI_GREEN)
    checks = value.get("check_runs")
    allowed = {"success", "skipped", "neutral"}
    if (
        value.get("repository") != "Liplus-Project/neuron-graph-rag"
        or value.get("branch") != "experiment/151-linux-rank-only-v4"
        or value.get("preflight_implementation_commit")
        != preflight_report.get("implementation_commit")
        or value.get("remote_ci_green") is not True
        or not isinstance(value.get("preflight_evidence_commit"), str)
        or len(value["preflight_evidence_commit"]) != 40
        or not isinstance(checks, list)
        or not checks
        or any(
            not isinstance(row, Mapping)
            or row.get("status") != "completed"
            or row.get("conclusion") not in allowed
            for row in checks
        )
    ):
        raise ValueError("remote preflight CI green evidence mismatch")
    expected_hashes = value.get("evidence_sha256")
    if not isinstance(expected_hashes, Mapping) or set(expected_hashes) != {
        PRECLAIM,
        *PREFLIGHT_FILES,
    }:
        raise ValueError("remote preflight evidence file set mismatch")
    for name, expected in expected_hashes.items():
        if sha256_bytes((root / EVIDENCE / name).read_bytes()) != expected:
            raise ValueError("remote preflight evidence byte identity mismatch")
    return value


def preflight(
    root: Path = ROOT,
    external_root: Path | None = None,
    model_cache: Path | None = None,
) -> dict[str, Any]:
    external = (external_root or default_external_root()).resolve()
    cache = (model_cache or default_model_cache()).resolve()
    preclaim = _verify_preclaim(root)
    protocol = load_protocol(root)
    verify_protocol_commit(PROTOCOL_COMMIT, protocol)
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
    ).returncode:
        raise ValueError("preflight HEAD must contain the exact freeze merge commit")
    if verify_phase_state(protocol) != {
        "development": "unobserved",
        "holdout": "unobserved",
    }:
        raise ValueError("preflight requires an unobserved protocol")
    evidence_root = root / EVIDENCE
    if any((evidence_root / name).exists() for name in PREFLIGHT_FILES):
        raise FileExistsError("preflight evidence already exists")
    platform_report = _runtime_platform_report(root, external)
    dependency_report = _dependency_report(root)
    model_report_path = external / MODEL_REPORT
    model_report = read_json(model_report_path)
    _verify_model_report(protocol, model_report, cache)
    bootstrap_rows = _parse_bootstrap_log(external / BOOTSTRAP_LOG)
    shared = shared_database_path()
    if not shared.is_file():
        raise FileNotFoundError("shared Windows database path is unavailable")
    recorded_before = (external / SHARED_BEFORE_PREFLIGHT).read_text(
        encoding="utf-8"
    ).strip()
    if len(recorded_before) != 64:
        raise ValueError("shared database preflight hash record mismatch")
    shared_before = _BASE.hash_file_shared(shared)
    if shared_before != recorded_before:
        raise ValueError("shared database changed since preflight began")
    offline = _worker_environment(root, offline=True)
    command_rows: list[dict[str, Any]] = []
    python = experiment_python(external)
    probe_output = _BASE._run_logged(
        [
            str(python),
            "-m",
            "neuron_graph_rag.cross_encoder_precision_v4_observation",
            "model-probe",
            "--cache",
            str(cache),
        ],
        root,
        offline,
        command_rows,
    )
    probe = json.loads(probe_output)
    if probe.get("forward_inference_count") != 2 or probe.get("batch_size") != BATCH_SIZE:
        raise ValueError("offline model probe did not cover both frozen models")
    for arguments in (
        [
            str(external / "tools" / "ruff"),
            "check",
            "src/neuron_graph_rag/cross_encoder_precision_v4_observation.py",
            "tests/test_cross_encoder_precision_v4_observation.py",
        ],
        [str(python), "-m", "unittest", "tests.test_cross_encoder_precision_v4"],
        [
            str(python),
            "-m",
            "unittest",
            "tests.test_cross_encoder_precision_v4_observation",
        ],
        [
            str(python),
            "-m",
            "neuron_graph_rag.cross_encoder_precision_v4_evaluation",
            "audit",
        ],
        [
            str(python),
            "-m",
            "neuron_graph_rag.cross_encoder_precision_v4_evaluation",
            "probe",
        ],
        [
            str(python),
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
        ],
    ):
        _BASE._run_logged(arguments, root, offline, command_rows)
    shared_after = _BASE.hash_file_shared(shared)
    if shared_after != shared_before:
        raise ValueError("shared Windows database changed during preflight")
    implementation_commit = _BASE._git_output(root, "rev-parse", "HEAD")
    preflight_report = {
        "protocol_id": PROTOCOL_ID,
        "protocol_commit": PROTOCOL_COMMIT,
        "implementation_commit": implementation_commit,
        "source_commit": SOURCE_COMMIT,
        "completed_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "run_root": str(external),
        "cache_path": str(cache),
        "cache_reused_as_verified_model_bytes_only": True,
        "predecessor_evidence_used_as_protocol_input": False,
        "predecessor_evidence_used_for_execution_design": False,
        "predecessor_worker_packet_reused": False,
        "offline": True,
        "local_files_only": True,
        "trust_remote_code": False,
        "batch_size": BATCH_SIZE,
        "preflight_forward_inference_count": 2,
        "claim_count": 0,
        "registered_query_execution_count": 0,
        "observed_stage_inference_count": 0,
        "result_count": 0,
        "phase": {"development": "unobserved", "holdout": "unobserved"},
        "preclaim_sha256": canonical_sha256(preclaim),
        "model_report_sha256": canonical_sha256(model_report),
        "dependency_report_sha256": canonical_sha256(dependency_report),
        "platform_report_sha256": canonical_sha256(platform_report),
        "shared_database_path": str(shared),
        "shared_database_sha256_before_preflight": shared_before,
        "shared_database_sha256_after_preflight": shared_after,
    }
    write_json_exclusive(evidence_root / MODEL_REPORT, model_report)
    write_json_exclusive(evidence_root / "dependency-report.json", dependency_report)
    write_json_exclusive(evidence_root / "platform-report.json", platform_report)
    write_json_exclusive(
        evidence_root / "preflight-commands.json",
        {"bootstrap_commands": bootstrap_rows, "verification_commands": command_rows},
    )
    write_json_exclusive(evidence_root / "preflight.json", preflight_report)
    verify_preflight(root, external, cache)
    return preflight_report


def verify_preflight(
    root: Path = ROOT,
    external_root: Path | None = None,
    model_cache: Path | None = None,
) -> dict[str, Any]:
    external = (external_root or default_external_root()).resolve()
    cache = (model_cache or default_model_cache()).resolve()
    preclaim = _verify_preclaim(root)
    protocol = load_protocol(root)
    verify_protocol_commit(PROTOCOL_COMMIT, protocol)
    _require_linux_run_root(root, external)
    evidence_root = root / EVIDENCE
    report = read_json(evidence_root / "preflight.json")
    model_report = read_json(evidence_root / MODEL_REPORT)
    dependency = read_json(evidence_root / "dependency-report.json")
    platform_value = read_json(evidence_root / "platform-report.json")
    commands = read_json(evidence_root / "preflight-commands.json")
    exact = {
        "protocol_id": PROTOCOL_ID,
        "protocol_commit": PROTOCOL_COMMIT,
        "source_commit": SOURCE_COMMIT,
        "run_root": str(external),
        "cache_path": str(cache),
        "cache_reused_as_verified_model_bytes_only": True,
        "predecessor_evidence_used_as_protocol_input": False,
        "predecessor_evidence_used_for_execution_design": False,
        "predecessor_worker_packet_reused": False,
        "offline": True,
        "local_files_only": True,
        "trust_remote_code": False,
        "batch_size": BATCH_SIZE,
        "preflight_forward_inference_count": 2,
        "claim_count": 0,
        "registered_query_execution_count": 0,
        "observed_stage_inference_count": 0,
        "result_count": 0,
        "phase": {"development": "unobserved", "holdout": "unobserved"},
    }
    if any(report.get(key) != value for key, value in exact.items()):
        raise ValueError("preflight report identity mismatch")
    if report.get("implementation_commit") != _BASE._git_output(root, "rev-parse", "HEAD"):
        raise ValueError("preflight implementation commit mismatch")
    if report.get("model_report_sha256") != canonical_sha256(model_report):
        raise ValueError("preflight model report binding mismatch")
    if report.get("dependency_report_sha256") != canonical_sha256(dependency):
        raise ValueError("preflight dependency report binding mismatch")
    if report.get("platform_report_sha256") != canonical_sha256(platform_value):
        raise ValueError("preflight platform report binding mismatch")
    if report.get("preclaim_sha256") != canonical_sha256(preclaim):
        raise ValueError("preflight preclaim provenance binding mismatch")
    _verify_model_report(protocol, model_report, cache)
    if dependency != _dependency_report(root):
        raise ValueError("dependency report is not reproducible")
    if platform_value != _runtime_platform_report(root, external):
        raise ValueError("platform report is not reproducible")
    if not commands.get("bootstrap_commands") or not commands.get("verification_commands"):
        raise ValueError("preflight command evidence is missing")
    shared_hash = _BASE.hash_file_shared(shared_database_path())
    if (
        report.get("shared_database_sha256_before_preflight") != shared_hash
        or report.get("shared_database_sha256_after_preflight") != shared_hash
    ):
        raise ValueError("shared database hash no longer matches preflight")
    return report


def _worker_environment(root: Path, *, offline: bool) -> dict[str, str]:
    environment = _ORIGINAL_WORKER_ENVIRONMENT(root, offline=offline)
    environment["HF_HOME"] = str(RUN_ROOT / "model-cache")
    environment["HF_HUB_CACHE"] = str(RUN_ROOT / "model-cache")
    environment["TORCH_HOME"] = str(RUN_ROOT / "torch-cache")
    environment["UV_CACHE_DIR"] = str(RUN_ROOT / "uv-cache")
    environment["NO_PROXY"] = "*"
    if offline:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    return environment


def _run_worker(
    root: Path,
    external: Path,
    cache: Path,
    stage: str,
    worker_kind: str,
    replay_kind: str,
    stage_root: Path,
    command_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    python = experiment_python(external)
    output = stage_root / f"{worker_kind}-{replay_kind}.json"
    database = external / "databases" / stage / f"{worker_kind}-{replay_kind}.sqlite3"
    if output.exists() or database.exists():
        raise FileExistsError("worker DB/output must be fresh")
    database.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        "-m",
        "neuron_graph_rag.cross_encoder_precision_v4_observation",
        "worker",
        "--stage",
        stage,
        "--kind",
        worker_kind,
        "--cache",
        str(cache),
        "--database",
        str(database),
        "--output",
        str(output),
    ]
    _BASE._run_logged(command, root, _worker_environment(root, offline=True), command_rows)
    return read_json(output)


def _copy_stage_evidence(stage: str, root: Path) -> None:
    protocol = load_protocol(root)
    outputs = protocol["manifest"]["outputs"][stage]
    evidence_root = root / EVIDENCE
    for key in ("archive_claim", "archive_result", "archive_error", "transport"):
        source = root / outputs[key]
        if not source.is_file():
            continue
        if key == "archive_claim":
            name = f"{stage}.claim.json"
        elif key == "archive_result":
            name = f"{stage}.observed.json"
        elif key == "archive_error":
            name = f"{stage}.error.json"
        else:
            name = f"{stage}.transport.json"
        destination = evidence_root / name
        raw = source.read_bytes()
        _BASE._write_bytes_exclusive(destination, raw)
        if destination.read_bytes() != raw:
            raise ValueError("stage evidence byte identity mismatch")


def _run_stage_once(
    stage: str,
    root: Path,
    external: Path,
    cache: Path,
    command_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        return _ORIGINAL_RUN_STAGE_ONCE(stage, root, external, cache, command_rows)
    finally:
        _copy_stage_evidence(stage, root)


_BASE._worker_environment = _worker_environment
_BASE._run_worker = _run_worker
_BASE._run_stage_once = _run_stage_once


def run_conditional(
    root: Path = ROOT,
    external_root: Path | None = None,
    model_cache: Path | None = None,
) -> dict[str, Any]:
    external = (external_root or default_external_root()).resolve()
    cache = (model_cache or default_model_cache()).resolve()
    preflight_report = verify_preflight(root, external, cache)
    ci_green = _verify_ci_green(root, external, preflight_report)
    ci_green_sha256 = canonical_sha256(ci_green)
    before = preflight_report["shared_database_sha256_before_preflight"]
    claim_before = _BASE.hash_file_shared(shared_database_path())
    if claim_before != before:
        raise ValueError("shared database changed before development claim")
    execution_rows: list[dict[str, Any]] = []
    development: dict[str, Any] | None = None
    holdout: dict[str, Any] | None = None
    try:
        development = _run_stage_once("development", root, external, cache, execution_rows)
        if development.get("all_hard_gates_pass") is True:
            holdout = _run_stage_once("holdout", root, external, cache, execution_rows)
    except BaseException as error:
        after = _BASE.hash_file_shared(shared_database_path())
        phases = verify_phase_state(load_protocol(root))
        development_claim_count = int(phases.get("development") != "unobserved")
        holdout_claim_count = int(phases.get("holdout") != "unobserved")
        failure = {
            "protocol_id": PROTOCOL_ID,
            "protocol_commit": PROTOCOL_COMMIT,
            "implementation_commit": preflight_report["implementation_commit"],
            "shared_database_sha256_before_preflight": before,
            "shared_database_sha256_before_claim": claim_before,
            "shared_database_sha256_after_observation": after,
            "shared_database_unchanged": before == claim_before == after,
            "phase": phases,
            "error": f"{type(error).__name__}: {error}",
            "commands": execution_rows,
            "retry_count": 0,
            "claim_count": development_claim_count + holdout_claim_count,
            "development_claim_count": development_claim_count,
            "holdout_claim_count": holdout_claim_count,
            "preflight_ci_green_sha256": ci_green_sha256,
            "preflight_evidence_commit": ci_green["preflight_evidence_commit"],
        }
        write_json_exclusive(root / EVIDENCE / "execution-error.json", failure)
        raise
    after = _BASE.hash_file_shared(shared_database_path())
    if before != claim_before or claim_before != after:
        raise ValueError("shared database changed during observation")
    phases = verify_phase_state(load_protocol(root))
    report = {
        "protocol_id": PROTOCOL_ID,
        "protocol_commit": PROTOCOL_COMMIT,
        "implementation_commit": preflight_report["implementation_commit"],
        "shared_database_sha256_before_preflight": before,
        "shared_database_sha256_before_claim": claim_before,
        "shared_database_sha256_after_observation": after,
        "shared_database_unchanged": True,
        "claim_count": 1 + int(holdout is not None),
        "development_claim_count": 1,
        "holdout_claim_count": int(holdout is not None),
        "retry_count": 0,
        "preflight_ci_green_sha256": ci_green_sha256,
        "preflight_evidence_commit": ci_green["preflight_evidence_commit"],
        "stage_process_count": 6 * (1 + int(holdout is not None)),
        "phase": phases,
        "selected_candidate": {
            "development": development.get("selected_candidate_id"),
            "holdout": None if holdout is None else holdout.get("selected_candidate_id"),
        },
        "commands": execution_rows,
    }
    write_json_exclusive(root / EVIDENCE / "execution.json", report)
    return {"development": development, "holdout": holdout, "execution": report}


def model_probe(cache: Path) -> dict[str, Any]:
    return _BASE.model_probe(cache)


def worker(stage: str, kind: str, cache: Path, database: Path, output: Path) -> None:
    for path in (cache, database.parent, output.parent):
        path.resolve().relative_to(RUN_ROOT)
    _BASE.worker(stage, kind, cache, database, output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run frozen v4 Linux observation once")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "verify-preflight", "run"):
        command = commands.add_parser(name)
        command.add_argument("--external-root", type=Path, default=default_external_root())
        command.add_argument("--model-cache", type=Path, default=default_model_cache())
    copy_command = commands.add_parser("model-copy-verify")
    copy_command.add_argument("--source-cache", type=Path, required=True)
    copy_command.add_argument("--cache", type=Path, required=True)
    copy_command.add_argument("--output", type=Path, required=True)
    probe = commands.add_parser("model-probe")
    probe.add_argument("--cache", type=Path, required=True)
    worker_command = commands.add_parser("worker")
    worker_command.add_argument("--stage", required=True)
    worker_command.add_argument("--kind", required=True)
    worker_command.add_argument("--cache", type=Path, required=True)
    worker_command.add_argument("--database", type=Path, required=True)
    worker_command.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "preflight":
        result = preflight(ROOT, arguments.external_root, arguments.model_cache)
    elif arguments.command == "verify-preflight":
        result = verify_preflight(ROOT, arguments.external_root, arguments.model_cache)
    elif arguments.command == "run":
        result = run_conditional(ROOT, arguments.external_root, arguments.model_cache)
    elif arguments.command == "model-copy-verify":
        model_copy_verify(arguments.source_cache, arguments.cache, arguments.output)
        result = {"status": "verified"}
    elif arguments.command == "model-probe":
        result = model_probe(arguments.cache)
    else:
        worker(arguments.stage, arguments.kind, arguments.cache, arguments.database, arguments.output)
        result = {"status": "completed"}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
