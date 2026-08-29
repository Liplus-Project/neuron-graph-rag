from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from . import cross_encoder_precision_v8_observation as predecessor
from . import cross_encoder_precision_v9_observation as path_freeze

PROTOCOL_ID = path_freeze.PROTOCOL_ID
FREEZE_COMMIT = "25790b5218ccc7a5741dbdf6a19d1f7723d7afeb"
V8_PROTOCOL_COMMIT = predecessor.PROTOCOL_COMMIT
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(
    "tests/fixtures/github_cross_encoder_precision_v9_observation.manifest.json"
)
EVIDENCE = Path("tests/evidence/github_cross_encoder_precision_v9_observation")

IMAGE = path_freeze.IMAGE
IMAGE_ID = path_freeze.IMAGE_ID
WSLC_VERSION = path_freeze.WSLC_VERSION
VOLUME = path_freeze.FUTURE_RUNTIME_VOLUME
PATH_FREEZE_VOLUME = path_freeze.PATH_FREEZE_VOLUME
CONTAINER_ROOT = PurePosixPath("/opt/ngr-v9/runtime")
CONTAINER_SOURCE = CONTAINER_ROOT / "source"
CONTAINER_CACHE = CONTAINER_ROOT / "model-cache"
CONTAINER_DATABASES = CONTAINER_ROOT / "databases"
CONTAINER_RUNS = CONTAINER_ROOT / "runs"
CONTAINER_ARCHIVE = CONTAINER_ROOT / "archive"
CONTAINER_TRANSPORT = CONTAINER_ROOT / "transport"
BATCH_SIZE = predecessor.BATCH_SIZE
SHARED_DATABASE = predecessor.SHARED_DATABASE
MODEL_CACHE = predecessor.MODEL_CACHE
WORKERS = predecessor.WORKERS

PATH_SMOKE_SHA256 = (
    "0f2d679a5f78f6f7eec5ca7f838fcae1ad013208a2591870ce7cacc2b0f1d0de"
)
COUNT_AUDIT_SHA256 = (
    "8626767ae3e777a68db180c44efbdb2627fa20aecb03c016f054cfd25f747a81"
)
PATH_EVIDENCE_MANIFEST_SHA256 = (
    "358be7af7afa81d5771a8a52fc36307e79319f84780af2121c07afc030fd7fc7"
)

canonical_sha256 = predecessor.canonical_sha256
sha256_file = predecessor.sha256_file
_write_json_exclusive = predecessor._write_json_exclusive
_git_output = predecessor._git_output
_hash_shared_database = predecessor._hash_shared_database
_command_row = predecessor._command_row
_run_logged = predecessor._run_logged


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def serialize_container_path(value: PurePosixPath | str) -> str:
    return path_freeze.serialize_container_path(value)


def named_volume_spec(
    volume: str,
    destination: PurePosixPath | str,
    *,
    mode: str | None = None,
) -> str:
    return path_freeze.named_volume_spec(volume, destination, mode=mode)


def host_bind_spec(
    source: Path,
    destination: PurePosixPath | str,
    *,
    mode: str,
) -> str:
    if mode not in {"ro", "rw"}:
        raise ValueError("host bind mode must be ro or rw")
    resolved = source.resolve()
    if not resolved.is_absolute():
        raise ValueError("host bind source must be absolute")
    return f"{resolved}:{serialize_container_path(destination)}:{mode}"


def _manifest(root: Path) -> dict[str, Any]:
    value = read_json(root / MANIFEST)
    if not isinstance(value, dict):
        raise TypeError("v9 observation manifest must be an object")
    return value


def _verify_path_freeze_inputs(root: Path) -> dict[str, Any]:
    manifest = _manifest(root)
    expected_header = {
        "protocol_id": PROTOCOL_ID,
        "phase": "performance-observation",
        "freeze_commit": FREEZE_COMMIT,
        "v8_protocol_commit": V8_PROTOCOL_COMMIT,
        "runtime_volume": VOLUME,
        "path_freeze_volume_reusable": False,
        "accepted_image": {"tag": IMAGE, "id": IMAGE_ID},
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            raise ValueError(f"v9 observation manifest mismatch: {key}")
    expected_paths = {
        "root": serialize_container_path(CONTAINER_ROOT),
        "source": serialize_container_path(CONTAINER_SOURCE),
        "model_cache": serialize_container_path(CONTAINER_CACHE),
        "databases": serialize_container_path(CONTAINER_DATABASES),
        "runs": serialize_container_path(CONTAINER_RUNS),
        "archive": serialize_container_path(CONTAINER_ARCHIVE),
        "transport": serialize_container_path(CONTAINER_TRANSPORT),
    }
    if manifest.get("container_paths") != expected_paths:
        raise ValueError("v9 observation container path registry mismatch")
    registry = manifest.get("path_freeze_immutable_sha256")
    if not isinstance(registry, dict) or len(registry) != 12:
        raise ValueError("v9 path-freeze registry must contain exactly 12 files")
    for relative, expected in registry.items():
        path = root / str(relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"v9 path-freeze artifact changed: {relative}")
    if registry.get(
        "tests/evidence/github_cross_encoder_precision_v9/path-smoke.pass.json"
    ) != PATH_SMOKE_SHA256:
        raise ValueError("v9 path smoke hash is not frozen")
    if registry.get(
        "tests/evidence/github_cross_encoder_precision_v9/count-audit.json"
    ) != COUNT_AUDIT_SHA256:
        raise ValueError("v9 path count audit hash is not frozen")
    if registry.get(
        "tests/evidence/github_cross_encoder_precision_v9/evidence-manifest.json"
    ) != PATH_EVIDENCE_MANIFEST_SHA256:
        raise ValueError("v9 path evidence manifest hash is not frozen")
    prebuild = path_freeze.validate_prebuild(root)
    path_evidence = path_freeze.audit_evidence(root)
    if path_evidence.get("status") != "pass":
        raise ValueError("successful v9 path-freeze evidence is required")
    if path_evidence.get("future_runtime_volume_absent") is not True:
        raise ValueError("v9 path-freeze did not preserve runtime volume absence")
    return {
        "path_freeze_artifact_count": len(registry),
        "v8_predecessor_artifact_count": prebuild["predecessor_artifact_count"],
        "path_smoke_sha256": PATH_SMOKE_SHA256,
        "count_audit_sha256": COUNT_AUDIT_SHA256,
        "path_evidence_manifest_sha256": PATH_EVIDENCE_MANIFEST_SHA256,
    }


def _stored_freeze_contract(root: Path) -> dict[str, Any]:
    path_contract = _verify_path_freeze_inputs(root)
    image_contract = predecessor._stored_freeze_contract(root)
    return {
        **path_contract,
        "accepted_image": image_contract["accepted_image"],
        "runtime_content_sha256": image_contract["runtime_content_sha256"],
        "attestation_sha256": image_contract["attestation_sha256"],
        "fingerprint_sha256": image_contract["fingerprint_sha256"],
        "metadata_correspondence_sha256": image_contract[
            "metadata_correspondence_sha256"
        ],
        "expected_distribution_count": image_contract[
            "expected_distribution_count"
        ],
        "additional_image_build_count": 0,
        "additional_runtime_content_report_count": 0,
        "additional_attestation_report_count": 0,
        "path_freeze_volume_mounted": False,
        "path_freeze_volume_read": False,
    }


def _container_command(
    *arguments: str,
    extra_volumes: Sequence[str] = (),
    name: str | None = None,
) -> list[str]:
    command = ["wslc", "run", "--rm", "--network", "none"]
    if name is not None:
        command.extend(["--name", name])
    command.extend(["--volume", named_volume_spec(VOLUME, CONTAINER_ROOT)])
    for volume in extra_volumes:
        command.extend(["--volume", volume])
    command.extend(
        [
            "--env",
            f"PYTHONPATH={serialize_container_path(CONTAINER_SOURCE / 'src')}",
            "--env",
            "HF_HUB_OFFLINE=1",
            "--env",
            "TRANSFORMERS_OFFLINE=1",
            "--env",
            f"HF_HOME={serialize_container_path(CONTAINER_CACHE)}",
            "--env",
            f"HF_HUB_CACHE={serialize_container_path(CONTAINER_CACHE)}",
            "--env",
            "NO_PROXY=*",
            "--workdir",
            serialize_container_path(CONTAINER_SOURCE),
            "--entrypoint",
            "python",
            IMAGE,
            "-m",
            "neuron_graph_rag.cross_encoder_precision_v9_performance_observation",
            *arguments,
        ]
    )
    return command


def _container_path(value: str) -> Path:
    return Path(serialize_container_path(value))


def _configure_container_harness() -> None:
    container_root = _container_path(serialize_container_path(CONTAINER_ROOT))
    container_source = _container_path(serialize_container_path(CONTAINER_SOURCE))
    container_cache = _container_path(serialize_container_path(CONTAINER_CACHE))
    predecessor.PROTOCOL_COMMIT = V8_PROTOCOL_COMMIT
    predecessor.ROOT = container_source
    predecessor.VOLUME = VOLUME
    predecessor.CONTAINER_ROOT = container_root
    predecessor.CONTAINER_SOURCE = container_source
    predecessor.CONTAINER_CACHE = container_cache
    predecessor.EVIDENCE = EVIDENCE
    predecessor._bind_container_harness()


def _dependency_report() -> dict[str, Any]:
    _configure_container_harness()
    report = predecessor._dependency_report()
    report["observation_protocol_id"] = PROTOCOL_ID
    return report


def _container_model_copy(source: str, cache: str, output: str) -> dict[str, Any]:
    _configure_container_harness()
    predecessor._container_model_copy(
        _container_path(source), _container_path(cache), _container_path(output)
    )
    return {"status": "verified"}


def _container_model_probe(cache: str) -> dict[str, Any]:
    _configure_container_harness()
    return predecessor._container_model_probe(_container_path(cache))


def _container_worker(
    stage: str,
    kind: str,
    replay: str,
    database: str,
    output: str,
) -> dict[str, Any]:
    _configure_container_harness()
    return predecessor._container_worker(
        stage,
        kind,
        replay,
        _container_path(database),
        _container_path(output),
    )


def _container_claim(stage: str) -> dict[str, Any]:
    _configure_container_harness()
    return predecessor._container_claim(stage)


def _container_finalize(stage: str) -> dict[str, Any]:
    _configure_container_harness()
    return predecessor._container_finalize(stage)


def _container_fail_stage(stage: str, message: str) -> dict[str, Any]:
    _configure_container_harness()
    return predecessor._container_fail_stage(stage, message)


def _source_initialization_script() -> str:
    root = serialize_container_path(CONTAINER_ROOT)
    source = serialize_container_path(CONTAINER_SOURCE)
    paths = " ".join(
        serialize_container_path(path)
        for path in (
            CONTAINER_CACHE,
            CONTAINER_DATABASES,
            CONTAINER_RUNS,
            CONTAINER_ARCHIVE,
            CONTAINER_TRANSPORT,
        )
    )
    fixture = serialize_container_path(
        CONTAINER_SOURCE
        / "tests/fixtures/github_cross_encoder_precision_v8.manifest.json"
    )
    return (
        "set -eu; "
        f"test -d '{root}'; test ! -e '{source}'; "
        f"mkdir -p {paths}; mkdir '{source}'; "
        f"cp -a /input/source/. '{source}/'; "
        f"rm -rf '{source}/.git'; test -f '{fixture}'"
    )


def _freeze_export_script() -> str:
    target = serialize_container_path(CONTAINER_ROOT / "frozen-source")
    fixture = serialize_container_path(
        CONTAINER_ROOT
        / "frozen-source/tests/fixtures/github_cross_encoder_precision_v8.manifest.json"
    )
    return (
        "set -eu; "
        f"test ! -e '{target}'; mkdir '{target}'; "
        f"tar -xf - -C '{target}'; test -f '{fixture}'"
    )


def _verify_host_source(root: Path) -> str:
    if root.resolve() != ROOT.resolve():
        raise ValueError("v9 observation source must be the delegated checkout")
    if _git_output(root, "status", "--porcelain"):
        raise ValueError("v9 preflight source must be committed and clean")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", FREEZE_COMMIT, "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
    ).returncode:
        raise ValueError("HEAD must contain the exact v9 path-freeze merge commit")
    return _git_output(root, "rev-parse", "HEAD")


def _runtime_volume_absent(
    root: Path, rows: list[dict[str, Any]]
) -> bool:
    return path_freeze._inspect_absent(VOLUME, root, rows)


def _wslc_version(root: Path, rows: list[dict[str, Any]]) -> str:
    first = _run_logged(["wslc", "--version"], root, rows).splitlines()[0]
    version = first.removeprefix("wslc ").strip()
    if version != WSLC_VERSION:
        raise ValueError("WSLC version mismatch")
    return version


def _image_identity(root: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    return path_freeze._image_identity(root, rows)


def _write_preflight_manifest(evidence: Path) -> None:
    names = (
        "dependency-report.json",
        "model-verification.json",
        "platform-report.json",
        "preclaim.json",
        "preflight-commands.json",
        "preflight.json",
    )
    registry = {name: sha256_file(evidence / name) for name in names}
    _write_json_exclusive(
        evidence / "preflight-evidence-manifest.json",
        {"protocol_id": PROTOCOL_ID, "files_sha256": registry},
    )


def _verification_commands(root: Path) -> tuple[list[str], ...]:
    python = root / ".venv" / "Scripts" / "python.exe"
    return (
        [
            "uvx",
            "--offline",
            "ruff",
            "check",
            "src/neuron_graph_rag/cross_encoder_precision_v9_performance_observation.py",
            "tests/test_cross_encoder_precision_v9_performance_observation.py",
        ],
        [
            str(python),
            "-m",
            "unittest",
            "tests.test_cross_encoder_precision_v8",
            "tests.test_cross_encoder_precision_v8_observation",
            "tests.test_cross_encoder_precision_v9",
            "tests.test_cross_encoder_precision_v9_performance_observation",
        ],
        [
            str(python),
            "-m",
            "neuron_graph_rag.cross_encoder_precision_v8_evaluation",
            "audit",
        ],
        [
            str(python),
            "-m",
            "neuron_graph_rag.cross_encoder_precision_v8_evaluation",
            "probe",
        ],
        [
            str(python),
            "-m",
            "neuron_graph_rag.cross_encoder_precision_v9_observation",
            "audit",
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
    )


def preflight(root: Path = ROOT, model_cache: Path = MODEL_CACHE) -> dict[str, Any]:
    evidence = root / EVIDENCE
    if evidence.exists():
        raise FileExistsError("v9 observation preflight evidence already exists")
    rows: list[dict[str, Any]] = []
    shared_before = ""
    implementation_commit = ""
    runtime_volume_create_count = 0
    preflight_forward_inference_count = 0
    try:
        implementation_commit = _verify_host_source(root)
        freeze = _stored_freeze_contract(root)
        if not _runtime_volume_absent(root, rows):
            raise FileExistsError("v9 runtime volume already exists; retry forbidden")
        shared_before = _hash_shared_database()
        _wslc_version(root, rows)
        image = _image_identity(root, rows)
        _run_logged(["wslc", "volume", "create", VOLUME], root, rows)
        runtime_volume_create_count = 1
        archive_process = subprocess.run(
            ["git", "archive", "--format=tar", FREEZE_COMMIT],
            cwd=root,
            check=False,
            capture_output=True,
        )
        archive_command = ["git", "archive", "--format=tar", FREEZE_COMMIT]
        archive_row = _command_row(archive_command, archive_process)
        archive_row["binary_export"] = True
        rows.append(archive_row)
        if archive_process.returncode:
            raise RuntimeError("exact v9 freeze export failed")
        frozen_archive = archive_process.stdout
        _run_logged(
            [
                "wslc",
                "run",
                "--rm",
                "--interactive",
                "--network",
                "none",
                "--volume",
                named_volume_spec(VOLUME, CONTAINER_ROOT),
                "--entrypoint",
                "/bin/sh",
                IMAGE,
                "-c",
                _freeze_export_script(),
            ],
            root,
            rows,
            input_bytes=frozen_archive,
        )
        _run_logged(
            [
                "wslc",
                "run",
                "--rm",
                "--network",
                "none",
                "--volume",
                named_volume_spec(VOLUME, CONTAINER_ROOT),
                "--volume",
                host_bind_spec(root, "/input/source", mode="ro"),
                "--entrypoint",
                "/bin/sh",
                IMAGE,
                "-c",
                _source_initialization_script(),
            ],
            root,
            rows,
        )
        model_output = CONTAINER_ROOT / "model-verification.json"
        _run_logged(
            _container_command(
                "model-copy-verify",
                "--source-cache",
                "/input/models",
                "--cache",
                serialize_container_path(CONTAINER_CACHE),
                "--output",
                serialize_container_path(model_output),
                extra_volumes=(
                    host_bind_spec(model_cache, "/input/models", mode="ro"),
                ),
            ),
            root,
            rows,
        )
        probe = json.loads(
            _run_logged(
                _container_command(
                    "model-probe",
                    "--cache",
                    serialize_container_path(CONTAINER_CACHE),
                ),
                root,
                rows,
            )
        )
        if probe.get("forward_inference_count") != 2:
            raise ValueError("synthetic model preflight probe count mismatch")
        if probe.get("batch_size") != BATCH_SIZE:
            raise ValueError("synthetic model preflight batch mismatch")
        preflight_forward_inference_count = 2
        dependency = json.loads(
            _run_logged(_container_command("dependency-report"), root, rows)
        )
        if dependency.get("artifact_registry_sha256") != predecessor.DEPENDENCY_ARTIFACT_SHA256:
            raise ValueError("dependency artifact registry mismatch")
        if dependency.get("lock_sha256") != predecessor.DEPENDENCY_LOCK_SHA256:
            raise ValueError("dependency lock mismatch")
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONPATH": str(root / "src"),
                "PYTHONUTF8": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        for command in _verification_commands(root):
            _run_logged(command, root, rows, environment=environment)
        shared_after = _hash_shared_database()
        if shared_after != shared_before:
            raise ValueError("shared Windows database changed during preflight")
        model_report = json.loads(
            _run_logged(
                _container_command(
                    "read-json", serialize_container_path(model_output)
                ),
                root,
                rows,
            )
        )
        platform_report = {
            "protocol_id": PROTOCOL_ID,
            "wslc_version": WSLC_VERSION,
            "image": image,
            "frozen_source_commit": FREEZE_COMMIT,
            "frozen_source_export_sha256": hashlib.sha256(frozen_archive).hexdigest(),
            "stored_freeze_contract": freeze,
            "runtime_volume": VOLUME,
            "runtime_volume_absent_before_create": True,
            "path_freeze_volume": PATH_FREEZE_VOLUME,
            "path_freeze_volume_mounted": False,
            "path_freeze_volume_read": False,
            "container_root": serialize_container_path(CONTAINER_ROOT),
            "host_bind_shared_database": False,
            "fresh_worker_container_process": True,
            "fresh_worker_database": True,
        }
        preclaim = {
            "protocol_id": PROTOCOL_ID,
            "freeze_commit": FREEZE_COMMIT,
            "v8_protocol_commit": V8_PROTOCOL_COMMIT,
            "implementation_commit": implementation_commit,
            "development_claim_count": 0,
            "holdout_claim_count": 0,
            "registered_query_execution_count": 0,
            "observed_stage_inference_count": 0,
            "result_count": 0,
            "phase": {"development": "unobserved", "holdout": "unobserved"},
            "predecessor_evidence_semantic_content_opened": False,
            "predecessor_packet_reused": False,
            "accepted_image_rebuilt": False,
            "additional_runtime_report_run_count": 0,
            "additional_attestation_run_count": 0,
            "retry_count": 0,
        }
        report = {
            **preclaim,
            "completed_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "shared_database_sha256_before_preflight": shared_before,
            "shared_database_sha256_after_preflight": shared_after,
            "batch_size": BATCH_SIZE,
            "offline": True,
            "local_files_only": True,
            "trust_remote_code": False,
            "preflight_forward_inference_count": 2,
            "model_report_sha256": canonical_sha256(model_report),
            "dependency_report_sha256": canonical_sha256(dependency),
            "platform_report_sha256": canonical_sha256(platform_report),
            "preclaim_sha256": canonical_sha256(preclaim),
        }
        for name, value in (
            ("preclaim.json", preclaim),
            ("model-verification.json", model_report),
            ("dependency-report.json", dependency),
            ("platform-report.json", platform_report),
            ("preflight-commands.json", {"commands": rows}),
            ("preflight.json", report),
        ):
            _write_json_exclusive(evidence / name, value)
        _write_preflight_manifest(evidence)
        return report
    except BaseException as error:
        evidence.mkdir(parents=True, exist_ok=True)
        _write_json_exclusive(
            evidence / "preflight.error.json",
            {
                "protocol_id": PROTOCOL_ID,
                "freeze_commit": FREEZE_COMMIT,
                "implementation_commit": implementation_commit or None,
                "error": f"{type(error).__name__}: {error}",
                "commands": rows,
                "development_claim_count": 0,
                "holdout_claim_count": 0,
                "registered_query_execution_count": 0,
                "preflight_forward_inference_count": preflight_forward_inference_count,
                "observed_stage_inference_count": 0,
                "result_count": 0,
                "retry_count": 0,
                "runtime_volume_create_count": runtime_volume_create_count,
                "shared_database_sha256_before_preflight": shared_before or None,
            },
        )
        raise


def _verify_hash_manifest(
    evidence: Path, name: str, *, exact: bool = False
) -> dict[str, Any]:
    manifest = read_json(evidence / name)
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"{name} protocol mismatch")
    registry = manifest.get("files_sha256")
    if not isinstance(registry, dict) or not registry:
        raise ValueError(f"{name} registry is missing")
    for relative, expected in registry.items():
        if sha256_file(evidence / str(relative)) != expected:
            raise ValueError(f"{name} hash mismatch: {relative}")
    if exact:
        actual = {
            path.relative_to(evidence).as_posix()
            for path in evidence.rglob("*")
            if path.is_file() and path.name != name
        }
        if actual != set(registry):
            raise ValueError(f"{name} file set mismatch")
    return manifest


def verify_preflight(root: Path = ROOT) -> dict[str, Any]:
    evidence = root / EVIDENCE
    report = read_json(evidence / "preflight.json")
    preclaim = read_json(evidence / "preclaim.json")
    model = read_json(evidence / "model-verification.json")
    dependency = read_json(evidence / "dependency-report.json")
    platform_report = read_json(evidence / "platform-report.json")
    commands = read_json(evidence / "preflight-commands.json")
    _verify_hash_manifest(evidence, "preflight-evidence-manifest.json")
    if report.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("v9 preflight identity mismatch")
    if report.get("freeze_commit") != FREEZE_COMMIT:
        raise ValueError("v9 preflight freeze commit mismatch")
    for key in (
        "development_claim_count",
        "holdout_claim_count",
        "registered_query_execution_count",
        "observed_stage_inference_count",
        "result_count",
        "retry_count",
    ):
        if report.get(key) != 0:
            raise ValueError("v9 preflight count is not result-free")
    if report.get("phase") != {
        "development": "unobserved",
        "holdout": "unobserved",
    }:
        raise ValueError("v9 preflight phase mismatch")
    if (
        report.get("accepted_image_rebuilt") is not False
        or report.get("additional_runtime_report_run_count") != 0
        or report.get("additional_attestation_run_count") != 0
    ):
        raise ValueError("v9 preflight reran an accepted freeze operation")
    for key, value in (
        ("preclaim_sha256", preclaim),
        ("model_report_sha256", model),
        ("dependency_report_sha256", dependency),
        ("platform_report_sha256", platform_report),
    ):
        if report.get(key) != canonical_sha256(value):
            raise ValueError(f"v9 preflight binding mismatch: {key}")
    if platform_report.get("stored_freeze_contract") != _stored_freeze_contract(root):
        raise ValueError("v9 stored freeze contract changed")
    if not commands.get("commands") or any(
        row.get("returncode") != 0
        for row in commands["commands"]
        if not (
            row.get("command") == ["wslc", "volume", "inspect", VOLUME]
            and row.get("returncode") != 0
        )
    ):
        raise ValueError("v9 preflight command evidence mismatch")
    current_shared = _hash_shared_database()
    if current_shared != report.get(
        "shared_database_sha256_before_preflight"
    ) or current_shared != report.get("shared_database_sha256_after_preflight"):
        raise ValueError("shared database no longer matches v9 preflight")
    return report


def _remote_ci_green(
    root: Path, commit: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    if _git_output(root, "status", "--porcelain"):
        raise ValueError("working tree must be clean before v9 one-shot")
    branch = _git_output(root, "branch", "--show-current")
    remote = subprocess.run(
        ["git", "ls-remote", "origin", f"refs/heads/{branch}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    if not remote or remote[0] != commit:
        raise ValueError("v9 one-shot requires pushed remote HEAD")
    raw = _run_logged(
        [
            "gh",
            "api",
            f"repos/Liplus-Project/neuron-graph-rag/commits/{commit}/check-runs",
        ],
        root,
        rows,
    )
    checks = json.loads(raw).get("check_runs", [])
    if not checks or any(row.get("status") != "completed" for row in checks):
        raise ValueError("v9 preflight evidence commit CI is not complete")
    if any(
        row.get("conclusion") not in {"success", "skipped", "neutral"}
        for row in checks
    ):
        raise ValueError("v9 preflight evidence commit CI is not green")
    return {
        "protocol_id": PROTOCOL_ID,
        "preflight_evidence_commit": commit,
        "checks": [
            {
                "name": row["name"],
                "status": row["status"],
                "conclusion": row["conclusion"],
            }
            for row in checks
        ],
    }


def _sync_preflight_evidence(root: Path, rows: list[dict[str, Any]]) -> None:
    source = (root / EVIDENCE).resolve()
    destination = serialize_container_path(
        CONTAINER_SOURCE / "tests/evidence/github_cross_encoder_precision_v9_observation"
    )
    command = (
        f"mkdir -p '{destination}' && "
        f"cp -a /input/evidence/. '{destination}/'"
    )
    _run_logged(
        [
            "wslc",
            "run",
            "--rm",
            "--network",
            "none",
            "--volume",
            named_volume_spec(VOLUME, CONTAINER_ROOT),
            "--volume",
            host_bind_spec(source, "/input/evidence", mode="ro"),
            "--entrypoint",
            "/bin/sh",
            IMAGE,
            "-c",
            command,
        ],
        root,
        rows,
    )


def _export_volume_evidence(root: Path, rows: list[dict[str, Any]]) -> None:
    evidence = (root / EVIDENCE).resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    source = serialize_container_path(
        CONTAINER_SOURCE / "tests/evidence/github_cross_encoder_precision_v9_observation"
    )
    _run_logged(
        [
            "wslc",
            "run",
            "--rm",
            "--network",
            "none",
            "--volume",
            named_volume_spec(VOLUME, CONTAINER_ROOT, mode="ro"),
            "--volume",
            host_bind_spec(evidence, "/output", mode="rw"),
            "--entrypoint",
            "/bin/sh",
            IMAGE,
            "-c",
            f"cp -an '{source}/.' /output/",
        ],
        root,
        rows,
    )


def _run_stage_host(
    stage: str,
    root: Path,
    rows: list[dict[str, Any]],
    claim_counts: dict[str, int],
) -> dict[str, Any]:
    _run_logged(_container_command("claim", "--stage", stage), root, rows)
    claim_counts[stage] += 1
    stage_root = CONTAINER_RUNS / stage
    database_root = CONTAINER_DATABASES / stage
    for kind, replay in WORKERS:
        identity = f"ngr-v9-{stage}-{kind}-{replay}"
        command = _container_command(
            "worker",
            "--stage",
            stage,
            "--kind",
            kind,
            "--replay",
            replay,
            "--database",
            serialize_container_path(database_root / f"{kind}-{replay}.sqlite3"),
            "--output",
            serialize_container_path(stage_root / f"{kind}-{replay}.json"),
            name=identity,
        )
        insert_at = command.index("--workdir")
        command[insert_at:insert_at] = [
            "--env",
            f"NGR_V8_CONTAINER_IDENTITY={identity}",
        ]
        _run_logged(command, root, rows)
    result = json.loads(
        _run_logged(_container_command("finalize", "--stage", stage), root, rows)
    )
    _export_volume_evidence(root, rows)
    return result


def _write_terminal_manifest(evidence: Path, status: str) -> None:
    registry = {
        path.relative_to(evidence).as_posix(): sha256_file(path)
        for path in sorted(evidence.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and path.name != "observation-evidence-manifest.json"
    }
    _write_json_exclusive(
        evidence / "observation-evidence-manifest.json",
        {"protocol_id": PROTOCOL_ID, "status": status, "files_sha256": registry},
    )


def finalize_preflight_error(root: Path = ROOT) -> dict[str, Any]:
    evidence = root / EVIDENCE
    raw_path = evidence / "preflight.error.json"
    terminal_path = evidence / "preflight-terminal.json"
    if not raw_path.is_file():
        raise FileNotFoundError("v9 raw preflight error evidence is missing")
    if (evidence / "preflight.json").exists():
        raise ValueError("successful preflight evidence cannot be finalized as error")
    if terminal_path.exists() or (
        evidence / "observation-evidence-manifest.json"
    ).exists():
        raise FileExistsError("v9 preflight error is already terminal")
    raw = read_json(raw_path)
    for key in (
        "development_claim_count",
        "holdout_claim_count",
        "registered_query_execution_count",
        "preflight_forward_inference_count",
        "observed_stage_inference_count",
        "result_count",
        "retry_count",
    ):
        if raw.get(key) != 0:
            raise ValueError("v9 raw preflight error count mismatch")
    if raw.get("runtime_volume_create_count") != 1:
        raise ValueError("v9 raw preflight error volume count mismatch")
    error = str(raw.get("error", ""))
    if "dedicated ext4 model cache already exists" not in error:
        raise ValueError("v9 raw preflight error cause mismatch")
    before = raw.get("shared_database_sha256_before_preflight")
    after = _hash_shared_database()
    terminal = {
        "protocol_id": PROTOCOL_ID,
        "status": "error",
        "phase": "preflight",
        "implementation_commit": raw.get("implementation_commit"),
        "raw_failure_sha256": sha256_file(raw_path),
        "failure_point": "model-copy-verify",
        "failure_cause": "dedicated model cache existed before exclusive copy",
        "next_candidate_axis": (
            "source initialization must leave /opt/ngr-v9/runtime/model-cache "
            "absent until frozen model-copy verification exclusively creates it"
        ),
        "runtime_volume_create_count": 1,
        "development_claim_count": 0,
        "holdout_claim_count": 0,
        "registered_query_execution_count": 0,
        "preflight_forward_inference_count": 0,
        "observed_stage_inference_count": 0,
        "result_count": 0,
        "retry_count": 0,
        "same_protocol_retry_allowed": False,
        "accepted_image_rebuild_count": 0,
        "runtime_report_rerun_count": 0,
        "attestation_rerun_count": 0,
        "path_freeze_volume_mounted": False,
        "path_freeze_volume_read": False,
        "path_freeze_volume_reused": False,
        "shared_database_sha256_before_preflight": before,
        "shared_database_sha256_after_error": after,
        "shared_database_post_error_hash_recorded": True,
        "shared_database_unchanged": before == after,
        "performance": "not assessed",
    }
    _write_json_exclusive(terminal_path, terminal)
    _write_terminal_manifest(evidence, "preflight-error")
    return terminal


def run_once(root: Path = ROOT) -> dict[str, Any]:
    report = verify_preflight(root)
    evidence = root / EVIDENCE
    if (evidence / "execution.json").exists() or (
        evidence / "execution-error.json"
    ).exists():
        raise FileExistsError("v9 observation already has terminal execution evidence")
    rows: list[dict[str, Any]] = []
    head = _git_output(root, "rev-parse", "HEAD")
    ci = _remote_ci_green(root, head, rows)
    _write_json_exclusive(evidence / "preflight-ci-green.json", ci)
    before = report["shared_database_sha256_before_preflight"]
    claim_before: str | None = None
    after: str | None = None
    development: dict[str, Any] | None = None
    holdout: dict[str, Any] | None = None
    current_stage = "development"
    claim_counts = {"development": 0, "holdout": 0}
    try:
        _sync_preflight_evidence(root, rows)
        claim_before = _hash_shared_database()
        if claim_before != before:
            raise ValueError("shared database changed before development claim")
        development = _run_stage_host(
            "development", root, rows, claim_counts
        )
        if (
            development.get("all_hard_gates_pass") is True
            and development.get("selected_candidate_id") is not None
        ):
            current_stage = "holdout"
            holdout = _run_stage_host("holdout", root, rows, claim_counts)
        after = _hash_shared_database()
        if before != claim_before or claim_before != after:
            raise ValueError("shared database changed during v9 observation")
    except BaseException as error:
        recovery_errors: list[str] = []
        if claim_counts[current_stage] == 1:
            try:
                _run_logged(
                    _container_command(
                        "fail-stage",
                        "--stage",
                        current_stage,
                        "--message",
                        f"{type(error).__name__}: {error}",
                    ),
                    root,
                    rows,
                )
            except (OSError, RuntimeError, ValueError) as archive_error:
                recovery_errors.append(
                    f"fail-stage: {type(archive_error).__name__}: {archive_error}"
                )
        try:
            _export_volume_evidence(root, rows)
        except (OSError, RuntimeError, ValueError) as export_error:
            recovery_errors.append(
                f"export: {type(export_error).__name__}: {export_error}"
            )
        try:
            after = _hash_shared_database()
        except (OSError, RuntimeError, ValueError) as hash_error:
            recovery_errors.append(
                f"shared-db-hash: {type(hash_error).__name__}: {hash_error}"
            )
            after = None
        _write_json_exclusive(
            evidence / "execution-error.json",
            {
                "protocol_id": PROTOCOL_ID,
                "freeze_commit": FREEZE_COMMIT,
                "preflight_evidence_commit": head,
                "error": f"{type(error).__name__}: {error}",
                "recovery_errors": recovery_errors,
                "retry_count": 0,
                "development_claim_count": claim_counts["development"],
                "holdout_claim_count": claim_counts["holdout"],
                "shared_database_sha256_before_preflight": before,
                "shared_database_sha256_before_claim": claim_before,
                "shared_database_sha256_after_observation": after,
                "shared_database_unchanged": (
                    claim_before is not None and before == claim_before == after
                ),
                "commands": rows,
            },
        )
        _write_terminal_manifest(evidence, "error")
        raise
    execution = {
        "protocol_id": PROTOCOL_ID,
        "freeze_commit": FREEZE_COMMIT,
        "v8_protocol_commit": V8_PROTOCOL_COMMIT,
        "preflight_evidence_commit": head,
        "retry_count": 0,
        "development_claim_count": claim_counts["development"],
        "holdout_claim_count": claim_counts["holdout"],
        "claim_count": sum(claim_counts.values()),
        "stage_process_count": 6 * sum(claim_counts.values()),
        "shared_database_sha256_before_preflight": before,
        "shared_database_sha256_before_claim": claim_before,
        "shared_database_sha256_after_observation": after,
        "shared_database_unchanged": True,
        "selected_candidate": {
            "development": development.get("selected_candidate_id"),
            "holdout": None
            if holdout is None
            else holdout.get("selected_candidate_id"),
        },
        "all_hard_gates_pass": {
            "development": development.get("all_hard_gates_pass"),
            "holdout": None
            if holdout is None
            else holdout.get("all_hard_gates_pass"),
        },
        "commands": rows,
    }
    _write_json_exclusive(evidence / "execution.json", execution)
    _write_terminal_manifest(evidence, "complete")
    return {"development": development, "holdout": holdout, "execution": execution}


def audit_evidence(root: Path = ROOT) -> dict[str, Any]:
    freeze = _stored_freeze_contract(root)
    evidence = root / EVIDENCE
    if not evidence.exists():
        return {
            "protocol_id": PROTOCOL_ID,
            "status": "preflight-not-run",
            "freeze": freeze,
        }
    if (evidence / "preflight.error.json").is_file():
        error = read_json(evidence / "preflight.error.json")
        if any(
            error.get(key) != 0
            for key in (
                "development_claim_count",
                "holdout_claim_count",
                "registered_query_execution_count",
                "preflight_forward_inference_count",
                "observed_stage_inference_count",
                "result_count",
                "retry_count",
            )
        ):
            raise ValueError("v9 preflight error counts are not result-free")
        terminal_path = evidence / "preflight-terminal.json"
        manifest_path = evidence / "observation-evidence-manifest.json"
        if not terminal_path.is_file() or not manifest_path.is_file():
            return {
                "protocol_id": PROTOCOL_ID,
                "status": "preflight-error-unfinalized",
            }
        terminal = read_json(terminal_path)
        manifest = _verify_hash_manifest(
            evidence, "observation-evidence-manifest.json", exact=True
        )
        if (
            terminal.get("raw_failure_sha256") != sha256_file(evidence / "preflight.error.json")
            or terminal.get("shared_database_unchanged") is not True
            or terminal.get("retry_count") != 0
            or terminal.get("performance") != "not assessed"
            or manifest.get("status") != "preflight-error"
        ):
            raise ValueError("v9 terminal preflight error evidence mismatch")
        return {
            "protocol_id": PROTOCOL_ID,
            "status": "preflight-error",
            "raw_failure_sha256": terminal["raw_failure_sha256"],
            "shared_database_unchanged": True,
            "retry_count": 0,
            "performance": "not assessed",
        }
    report = verify_preflight(root)
    terminal = evidence / "observation-evidence-manifest.json"
    if not terminal.exists():
        return {
            "protocol_id": PROTOCOL_ID,
            "status": "preflight-complete",
            "implementation_commit": report["implementation_commit"],
        }
    manifest = _verify_hash_manifest(
        evidence, "observation-evidence-manifest.json", exact=True
    )
    execution_path = evidence / "execution.json"
    error_path = evidence / "execution-error.json"
    if execution_path.is_file() == error_path.is_file():
        raise ValueError("v9 terminal evidence must contain one outcome")
    terminal_report = read_json(execution_path if execution_path.is_file() else error_path)
    if terminal_report.get("retry_count") != 0:
        raise ValueError("v9 retry count changed")
    if terminal_report.get("development_claim_count") != 1:
        raise ValueError("v9 development claim count mismatch")
    if terminal_report.get("holdout_claim_count") not in {0, 1}:
        raise ValueError("v9 holdout claim count mismatch")
    return {
        "protocol_id": PROTOCOL_ID,
        "status": manifest["status"],
        "development_claim_count": terminal_report["development_claim_count"],
        "holdout_claim_count": terminal_report["holdout_claim_count"],
        "retry_count": 0,
        "shared_database_unchanged": terminal_report.get(
            "shared_database_unchanged"
        ),
    }


def _read_json_command(path: str) -> dict[str, Any]:
    value = read_json(_container_path(path))
    if not isinstance(value, dict):
        raise TypeError("container JSON command expects an object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Observe frozen WSLC rank-only benchmark v9 once"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    commands.add_parser("verify-preflight")
    commands.add_parser("run")
    commands.add_parser("audit")
    commands.add_parser("finalize-preflight-error")
    copy = commands.add_parser("model-copy-verify")
    copy.add_argument("--source-cache", required=True)
    copy.add_argument("--cache", required=True)
    copy.add_argument("--output", required=True)
    probe = commands.add_parser("model-probe")
    probe.add_argument("--cache", required=True)
    commands.add_parser("dependency-report")
    read = commands.add_parser("read-json")
    read.add_argument("path")
    claim = commands.add_parser("claim")
    claim.add_argument("--stage", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--stage", required=True)
    worker.add_argument("--kind", required=True)
    worker.add_argument("--replay", required=True)
    worker.add_argument("--database", required=True)
    worker.add_argument("--output", required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--stage", required=True)
    failure = commands.add_parser("fail-stage")
    failure.add_argument("--stage", required=True)
    failure.add_argument("--message", required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "preflight":
        result = preflight()
    elif arguments.command == "verify-preflight":
        result = verify_preflight()
    elif arguments.command == "run":
        result = run_once()
    elif arguments.command == "audit":
        result = audit_evidence()
    elif arguments.command == "finalize-preflight-error":
        result = finalize_preflight_error()
    elif arguments.command == "model-copy-verify":
        result = _container_model_copy(
            arguments.source_cache, arguments.cache, arguments.output
        )
    elif arguments.command == "model-probe":
        result = _container_model_probe(arguments.cache)
    elif arguments.command == "dependency-report":
        result = _dependency_report()
    elif arguments.command == "read-json":
        result = _read_json_command(arguments.path)
    elif arguments.command == "claim":
        result = _container_claim(arguments.stage)
    elif arguments.command == "worker":
        result = _container_worker(
            arguments.stage,
            arguments.kind,
            arguments.replay,
            arguments.database,
            arguments.output,
        )
    elif arguments.command == "finalize":
        result = _container_finalize(arguments.stage)
    else:
        result = _container_fail_stage(arguments.stage, arguments.message)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
