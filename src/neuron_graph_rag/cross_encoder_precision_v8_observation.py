from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import cross_encoder_precision_v8_evaluation as evaluation

_V5_SOURCE = Path(__file__).with_name("cross_encoder_precision_v5_observation.py")
_SPEC = importlib.util.spec_from_file_location(
    "neuron_graph_rag._cross_encoder_precision_v8_observation_base", _V5_SOURCE
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("unable to load frozen v5 observation harness")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

PROTOCOL_ID = evaluation.PROTOCOL_ID
PROTOCOL_COMMIT = "d2fdf7720e2a9dde7e8d666cf4fd9f314fd3d12f"
SOURCE_COMMIT = _BASE.SOURCE_COMMIT
ROOT = evaluation.ROOT
IMAGE = "ngr-cross-encoder-precision-v8:freeze"
IMAGE_ID = "sha256:136ad9466799109bf32b4b96b611c9db9a099bcc47cf78243f26c7227bc16742"
BASE_IMAGE_ID = (
    "sha256:f0c05afecbd16040caff4c000954567c7e3b56fc6c1f783fa10a55cba3ccfbfc"
)
WSLC_VERSION = "2.9.4.0"
VOLUME = "github-cross-encoder-precision-v8-runtime"
CONTAINER_ROOT = Path("/opt/ngr-v8/runtime")
CONTAINER_SOURCE = CONTAINER_ROOT / "source"
CONTAINER_CACHE = CONTAINER_ROOT / "model-cache"
EVIDENCE = Path("tests/evidence/github_cross_encoder_precision_v8")
BATCH_SIZE = 8
SHARED_DATABASE = _BASE.SHARED_DATABASE
MODEL_CACHE = _BASE.MODEL_CACHE
WORKERS = _BASE.WORKERS
DEPENDENCY_ARTIFACT_SHA256 = (
    "a805625a67cc30ac36b1ddcbefd0f1efcfbb579ea6011d9b25ea82f98d207bfb"
)
DEPENDENCY_LOCK_SHA256 = (
    "0734b63356d6a0c102c87fcb014495fb251352e32d77b18fffad82c170f2efd4"
)
RUNTIME_CONTENT_SHA256 = (
    "7d754e4e1713f90654ae05c749379d08920e31fede89ab25ba075c0b582bcee8"
)
ATTESTATION_SHA256 = "045c813894bed25e3eae29a38fa6366013a0600ce8d0952a40ae29008747a50b"
FINGERPRINT_SHA256 = "8969a259ffdfe822a70ac8bd8ce52dc7223b6e6a2b51ca21c095fe14a388b2bc"
METADATA_CORRESPONDENCE_SHA256 = (
    "30348d7b8a352e0ab15d98871c71df5240ffbf1547c58564fd951cd451c377c4"
)

for _name, _value in {
    "evaluation": evaluation,
    "PROTOCOL_ID": PROTOCOL_ID,
    "PROTOCOL_COMMIT": PROTOCOL_COMMIT,
    "ROOT": ROOT,
    "IMAGE": IMAGE,
    "IMAGE_ID": IMAGE_ID,
    "BASE_IMAGE_ID": BASE_IMAGE_ID,
    "WSLC_VERSION": WSLC_VERSION,
    "VOLUME": VOLUME,
    "CONTAINER_ROOT": CONTAINER_ROOT,
    "CONTAINER_SOURCE": CONTAINER_SOURCE,
    "CONTAINER_CACHE": CONTAINER_CACHE,
    "EVIDENCE": EVIDENCE,
}.items():
    setattr(_BASE, _name, _value)

canonical_sha256 = _BASE.canonical_sha256
sha256_file = _BASE.sha256_file
_write_json_exclusive = _BASE._write_json_exclusive
_git_output = _BASE._git_output
_hash_shared_database = _BASE._hash_shared_database
_command_row = _BASE._command_row
_run_logged = _BASE._run_logged
_wslc_json = _BASE._wslc_json
_volume_exists = _BASE._volume_exists
_bind_container_harness = _BASE._bind_container_harness
_container_model_copy = _BASE._container_model_copy
_container_model_probe = _BASE._container_model_probe
_stage_paths = _BASE._stage_paths
_container_claim = _BASE._container_claim
_copy_stage_evidence = _BASE._copy_stage_evidence
_container_finalize = _BASE._container_finalize
_container_fail_stage = _BASE._container_fail_stage
_verify_host_source = _BASE._verify_host_source
_wslc_version = _BASE._wslc_version
_image_identity = _BASE._image_identity
_remote_ci_green = _BASE._remote_ci_green


def _container_command(
    *arguments: str,
    extra_volumes: Sequence[str] = (),
    name: str | None = None,
) -> list[str]:
    command = ["wslc", "run", "--rm", "--network", "none"]
    if name is not None:
        command.extend(["--name", name])
    command.extend(["--volume", f"{VOLUME}:{CONTAINER_ROOT}"])
    for volume in extra_volumes:
        command.extend(["--volume", volume])
    command.extend(
        [
            "--env",
            f"PYTHONPATH={CONTAINER_SOURCE / 'src'}",
            "--env",
            "HF_HUB_OFFLINE=1",
            "--env",
            "TRANSFORMERS_OFFLINE=1",
            "--env",
            f"HF_HOME={CONTAINER_CACHE}",
            "--env",
            f"HF_HUB_CACHE={CONTAINER_CACHE}",
            "--env",
            "NO_PROXY=*",
            "--workdir",
            str(CONTAINER_SOURCE),
            "--entrypoint",
            "python",
            IMAGE,
            "-m",
            "neuron_graph_rag.cross_encoder_precision_v8_observation",
            *arguments,
        ]
    )
    return command


def _dependency_report() -> dict[str, Any]:
    _bind_container_harness()
    protocol = evaluation.load_protocol(CONTAINER_SOURCE)
    expected = {
        row["name"]: row["version"]
        for row in protocol["dependency_artifacts"]["artifacts"]
    }
    import importlib.metadata
    from urllib.parse import unquote, urlparse

    installed = {
        dist.metadata["Name"].lower().replace("_", "-"): dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    }
    mismatched = {
        name: {"expected": version, "actual": installed.get(name)}
        for name, version in expected.items()
        if installed.get(name) != version
    }
    if mismatched or len(expected) != 26:
        raise ValueError(f"exact dependency mismatch: {mismatched}")
    forbidden = sorted(
        name for name in installed if name == "triton" or name.startswith("nvidia-")
    )
    if forbidden:
        raise ValueError("accelerator dependency is forbidden")
    frozen_rows = {
        row["name"]: row for row in protocol["dependency_artifacts"]["artifacts"]
    }
    install_report = evaluation.read_json(Path("/opt/ngr-v8/dependency-report.json"))
    if install_report.get("pip_version") != "24.0":
        raise ValueError("pip report version mismatch")
    installed_artifacts = []
    for row in install_report.get("install", []):
        metadata = row.get("metadata", {})
        name = str(metadata.get("name", "")).lower().replace("_", "-")
        download = row.get("download_info", {})
        url = download.get("url")
        actual = {
            "name": name,
            "version": metadata.get("version"),
            "filename": unquote(Path(urlparse(str(url)).path).name),
            "url": url,
            "sha256": download.get("archive_info", {}).get("hashes", {}).get("sha256"),
        }
        if actual != frozen_rows.get(name):
            raise ValueError(f"pip artifact report mismatch: {name}")
        installed_artifacts.append(actual)
    if len(installed_artifacts) != 26 or set(frozen_rows) != {
        row["name"] for row in installed_artifacts
    }:
        raise ValueError("pip artifact report cardinality mismatch")
    return {
        "protocol_id": PROTOCOL_ID,
        "artifact_registry_sha256": sha256_file(
            CONTAINER_SOURCE
            / "tests/fixtures/github_cross_encoder_precision_v8.dependency-artifacts.json"
        ),
        "lock_sha256": sha256_file(
            CONTAINER_SOURCE
            / "tests/fixtures/github_cross_encoder_precision_v8.requirements.lock"
        ),
        "artifact_count": len(expected),
        "installed": {name: installed[name] for name in sorted(expected)},
        "pip_version": install_report["pip_version"],
        "artifacts": sorted(installed_artifacts, key=lambda row: row["name"]),
        "forbidden": forbidden,
    }


def _container_worker(
    stage: str, kind: str, replay: str, database: Path, output: Path
) -> dict[str, Any]:
    _bind_container_harness()
    for path in (database.parent, output.parent, CONTAINER_CACHE):
        path.resolve().relative_to(CONTAINER_ROOT)
    if (kind, replay) not in WORKERS:
        raise ValueError("worker order identity is not frozen")
    _BASE._v4._BASE.worker(stage, kind, CONTAINER_CACHE, database, output)
    payload = evaluation.read_json(output)
    payload["container_id"] = os.environ.get(
        "NGR_V8_CONTAINER_IDENTITY", socket.gethostname()
    )
    payload["container_process_pid"] = os.getpid()
    temporary = output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, output)
    return payload


def _source_initialization_script() -> str:
    return (
        "test ! -e /opt/ngr-v8/runtime/source && "
        "mkdir -p /opt/ngr-v8/runtime/logs /opt/ngr-v8/runtime/databases "
        "/opt/ngr-v8/runtime/runs /opt/ngr-v8/runtime/archive && "
        "cp -a /input/source /opt/ngr-v8/runtime/source && "
        "rm -f /opt/ngr-v8/runtime/source/.git && "
        "test -f /opt/ngr-v8/runtime/source/tests/fixtures/"
        "github_cross_encoder_precision_v8.manifest.json"
    )


def _freeze_export_script() -> str:
    return (
        "test ! -e /opt/ngr-v8/runtime/frozen-source && "
        "mkdir /opt/ngr-v8/runtime/frozen-source && "
        "tar -xf - -C /opt/ngr-v8/runtime/frozen-source && "
        "test -f /opt/ngr-v8/runtime/frozen-source/tests/fixtures/"
        "github_cross_encoder_precision_v8.manifest.json"
    )


def _stored_freeze_contract(root: Path) -> dict[str, Any]:
    protocol = evaluation.load_protocol(root)
    platform = protocol["platform"]
    accepted = platform["container"]["accepted_image"]
    content = platform["content_equivalence"]
    if accepted != {"build": "build_a", "id": IMAGE_ID, "tag": IMAGE}:
        raise ValueError("accepted image registry mismatch")
    expected = {
        "runtime_content_build_a": RUNTIME_CONTENT_SHA256,
        "runtime_content_build_b": RUNTIME_CONTENT_SHA256,
        "attestation_build_a": ATTESTATION_SHA256,
        "attestation_build_b": ATTESTATION_SHA256,
    }
    for key, digest in expected.items():
        record = content.get(key, {})
        path = root / str(record.get("path", ""))
        if record.get("sha256") != digest or sha256_file(path) != digest:
            raise ValueError(f"stored freeze report mismatch: {key}")
    if (
        content.get("fingerprint_sha256") != FINGERPRINT_SHA256
        or content.get("metadata_correspondence_sha256")
        != METADATA_CORRESPONDENCE_SHA256
        or content.get("exact_installed_distribution_set_attested") is not True
        or content.get("successor_observation_allowed") is not True
        or content.get("freeze_outcome")
        != "accepted_exact_installed_distribution_freeze"
    ):
        raise ValueError("stored freeze acceptance contract mismatch")
    return {
        "accepted_image": dict(accepted),
        "runtime_content_sha256": RUNTIME_CONTENT_SHA256,
        "attestation_sha256": ATTESTATION_SHA256,
        "fingerprint_sha256": FINGERPRINT_SHA256,
        "metadata_correspondence_sha256": METADATA_CORRESPONDENCE_SHA256,
        "expected_distribution_count": 29,
        "importlib_distribution_count": 29,
        "filesystem_distribution_count": 29,
        "nested_metadata_path_count": 16,
        "additional_image_build_count": 0,
        "additional_runtime_content_report_count": 0,
        "additional_attestation_report_count": 0,
    }


def _export_volume_evidence(root: Path, rows: list[dict[str, Any]]) -> None:
    evidence = (root / EVIDENCE).resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    command = [
        "wslc",
        "run",
        "--rm",
        "--network",
        "none",
        "--volume",
        f"{VOLUME}:{CONTAINER_ROOT}:ro",
        "--volume",
        f"{evidence}:/output",
        "--entrypoint",
        "/bin/sh",
        IMAGE,
        "-c",
        (
            "cp -an /opt/ngr-v8/runtime/source/tests/evidence/"
            "github_cross_encoder_precision_v8/. /output/"
        ),
    ]
    _run_logged(command, root, rows)


def preflight(root: Path = ROOT, model_cache: Path = MODEL_CACHE) -> dict[str, Any]:
    evidence = root / EVIDENCE
    if evidence.exists():
        raise FileExistsError("v8 preflight evidence already exists")
    rows: list[dict[str, Any]] = []
    shared_before = ""
    implementation_commit = ""
    try:
        implementation_commit = _verify_host_source(root)
        if _volume_exists(root, rows):
            raise FileExistsError(
                "v8 volume already exists; preflight is not retryable"
            )
        shared_before = _hash_shared_database()
        _wslc_version(root, rows)
        image = _image_identity(root, rows)
        freeze = _stored_freeze_contract(root)
        _run_logged(["wslc", "volume", "create", VOLUME], root, rows)
        archive_process = subprocess.run(
            ["git", "archive", "--format=tar", PROTOCOL_COMMIT],
            cwd=root,
            check=False,
            capture_output=True,
        )
        archive_command = ["git", "archive", "--format=tar", PROTOCOL_COMMIT]
        archive_row = _command_row(archive_command, archive_process)
        archive_row["binary_export"] = True
        rows.append(archive_row)
        if archive_process.returncode:
            raise RuntimeError("exact freeze export failed")
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
                f"{VOLUME}:{CONTAINER_ROOT}",
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
                f"{VOLUME}:{CONTAINER_ROOT}",
                "--volume",
                f"{root.resolve()}:/input/source:ro",
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
                str(CONTAINER_CACHE),
                "--output",
                str(model_output),
                extra_volumes=(f"{model_cache.resolve()}:/input/models:ro",),
            ),
            root,
            rows,
        )
        probe = json.loads(
            _run_logged(
                _container_command("model-probe", "--cache", str(CONTAINER_CACHE)),
                root,
                rows,
            )
        )
        if probe.get("forward_inference_count") != 2 or probe.get("batch_size") != 8:
            raise ValueError("synthetic model preflight probe mismatch")
        dependency = json.loads(
            _run_logged(_container_command("dependency-report"), root, rows)
        )
        if dependency.get("artifact_registry_sha256") != DEPENDENCY_ARTIFACT_SHA256:
            raise ValueError("dependency artifact registry mismatch")
        if dependency.get("lock_sha256") != DEPENDENCY_LOCK_SHA256:
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
        python = root / ".venv" / "Scripts" / "python.exe"
        for command in (
            [
                "uvx",
                "--offline",
                "ruff",
                "check",
                "src/neuron_graph_rag/cross_encoder_precision_v8_observation.py",
                "tests/test_cross_encoder_precision_v8_observation.py",
            ],
            [
                str(python),
                "-m",
                "unittest",
                "tests.test_cross_encoder_precision_v8",
                "tests.test_cross_encoder_precision_v8_observation",
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
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "test_*.py",
            ],
        ):
            _run_logged(command, root, rows, environment=environment)
        shared_after = _hash_shared_database()
        if shared_after != shared_before:
            raise ValueError("shared Windows database changed during preflight")
        platform_report = {
            "protocol_id": PROTOCOL_ID,
            "wslc_version": WSLC_VERSION,
            "image": image,
            "base_image_id_from_freeze_registry": BASE_IMAGE_ID,
            "base_digest": "sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3",
            "frozen_source_export_sha256": hashlib.sha256(frozen_archive).hexdigest(),
            "stored_freeze_contract": freeze,
            "volume": VOLUME,
            "volume_absent_before_create": True,
            "container_root": str(CONTAINER_ROOT),
            "host_bind_shared_database": False,
            "fresh_worker_container_process": True,
            "fresh_worker_database": True,
        }
        model_report = json.loads(
            _run_logged(_container_command("read-json", str(model_output)), root, rows)
        )
        preclaim = {
            "protocol_id": PROTOCOL_ID,
            "protocol_commit": PROTOCOL_COMMIT,
            "implementation_commit": implementation_commit,
            "development_claim_count": 0,
            "registered_query_execution_count": 0,
            "observed_stage_inference_count": 0,
            "result_count": 0,
            "phase": {"development": "unobserved", "holdout": "unobserved"},
            "predecessor_evidence_semantic_content_opened": False,
            "predecessor_packet_reused": False,
            "accepted_image_rebuilt": False,
            "additional_runtime_report_run_count": 0,
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
        return report
    except BaseException as error:
        evidence.mkdir(parents=True, exist_ok=True)
        _write_json_exclusive(
            evidence / "preflight.error.json",
            {
                "protocol_id": PROTOCOL_ID,
                "protocol_commit": PROTOCOL_COMMIT,
                "implementation_commit": implementation_commit or None,
                "error": f"{type(error).__name__}: {error}",
                "commands": rows,
                "development_claim_count": 0,
                "registered_query_execution_count": 0,
                "observed_stage_inference_count": 0,
                "result_count": 0,
                "retry_count": 0,
                "shared_database_sha256_before_preflight": shared_before or None,
            },
        )
        raise


def verify_preflight(root: Path = ROOT) -> dict[str, Any]:
    evidence = root / EVIDENCE
    report = evaluation.read_json(evidence / "preflight.json")
    preclaim = evaluation.read_json(evidence / "preclaim.json")
    model = evaluation.read_json(evidence / "model-verification.json")
    dependency = evaluation.read_json(evidence / "dependency-report.json")
    platform_report = evaluation.read_json(evidence / "platform-report.json")
    commands = evaluation.read_json(evidence / "preflight-commands.json")
    if (
        report.get("protocol_id") != PROTOCOL_ID
        or report.get("protocol_commit") != PROTOCOL_COMMIT
    ):
        raise ValueError("preflight identity mismatch")
    for key in (
        "development_claim_count",
        "registered_query_execution_count",
        "observed_stage_inference_count",
        "result_count",
    ):
        if report.get(key) != 0:
            raise ValueError("preflight count is not result-free")
    if report.get("phase") != {"development": "unobserved", "holdout": "unobserved"}:
        raise ValueError("preflight phase mismatch")
    if (
        report.get("accepted_image_rebuilt") is not False
        or report.get("additional_runtime_report_run_count") != 0
    ):
        raise ValueError("preflight reused the accepted freeze incorrectly")
    for key, value in (
        ("preclaim_sha256", preclaim),
        ("model_report_sha256", model),
        ("dependency_report_sha256", dependency),
        ("platform_report_sha256", platform_report),
    ):
        if report.get(key) != canonical_sha256(value):
            raise ValueError(f"{key} binding mismatch")
    if platform_report.get("stored_freeze_contract") != _stored_freeze_contract(root):
        raise ValueError("stored freeze contract is no longer reproducible")
    if not commands.get("commands") or any(
        row.get("returncode") != 0
        for row in commands["commands"]
        if not (
            row.get("command") == ["wslc", "volume", "inspect", VOLUME]
            and row.get("returncode") != 0
        )
    ):
        raise ValueError("preflight command evidence mismatch")
    current_shared = _hash_shared_database()
    if current_shared not in {
        report.get("shared_database_sha256_before_preflight"),
        report.get("shared_database_sha256_after_preflight"),
    } or report.get("shared_database_sha256_before_preflight") != report.get(
        "shared_database_sha256_after_preflight"
    ):
        raise ValueError("shared database no longer matches preflight")
    return report


def _sync_preflight_evidence(root: Path, rows: list[dict[str, Any]]) -> None:
    source = (root / EVIDENCE).resolve()
    _run_logged(
        [
            "wslc",
            "run",
            "--rm",
            "--network",
            "none",
            "--volume",
            f"{VOLUME}:{CONTAINER_ROOT}",
            "--volume",
            f"{source}:/input/evidence:ro",
            "--entrypoint",
            "/bin/sh",
            IMAGE,
            "-c",
            (
                "mkdir -p /opt/ngr-v8/runtime/source/tests/evidence/"
                "github_cross_encoder_precision_v8 && cp -a /input/evidence/. "
                "/opt/ngr-v8/runtime/source/tests/evidence/"
                "github_cross_encoder_precision_v8/"
            ),
        ],
        root,
        rows,
    )


def _run_stage_host(
    stage: str, root: Path, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    _run_logged(_container_command("claim", "--stage", stage), root, rows)
    stage_root = CONTAINER_ROOT / "runs" / stage
    database_root = CONTAINER_ROOT / "databases" / stage
    for kind, replay in WORKERS:
        identity = f"ngr-v8-{stage}-{kind}-{replay}"
        command = _container_command(
            "worker",
            "--stage",
            stage,
            "--kind",
            kind,
            "--replay",
            replay,
            "--database",
            str(database_root / f"{kind}-{replay}.sqlite3"),
            "--output",
            str(stage_root / f"{kind}-{replay}.json"),
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


def run_once(root: Path = ROOT) -> dict[str, Any]:
    report = verify_preflight(root)
    evidence = root / EVIDENCE
    if (evidence / "execution.json").exists() or (
        evidence / "execution-error.json"
    ).exists():
        raise FileExistsError("v8 observation already has terminal execution evidence")
    rows: list[dict[str, Any]] = []
    implementation_head = _git_output(root, "rev-parse", "HEAD")
    ci = _remote_ci_green(root, implementation_head, rows)
    _write_json_exclusive(evidence / "preflight-ci-green.json", ci)
    _sync_preflight_evidence(root, rows)
    before = report["shared_database_sha256_before_preflight"]
    claim_before = _hash_shared_database()
    if claim_before != before:
        raise ValueError("shared database changed before development claim")
    development: dict[str, Any] | None = None
    holdout: dict[str, Any] | None = None
    current_stage = "development"
    try:
        development = _run_stage_host("development", root, rows)
        if development.get("all_hard_gates_pass") is True:
            current_stage = "holdout"
            holdout = _run_stage_host("holdout", root, rows)
    except BaseException as error:
        recovery_errors: list[str] = []
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
        after = _hash_shared_database()
        _write_json_exclusive(
            evidence / "execution-error.json",
            {
                "protocol_id": PROTOCOL_ID,
                "protocol_commit": PROTOCOL_COMMIT,
                "preflight_evidence_commit": implementation_head,
                "error": f"{type(error).__name__}: {error}",
                "recovery_errors": recovery_errors,
                "retry_count": 0,
                "development_claim_count": 1,
                "holdout_claim_count": int(current_stage == "holdout"),
                "shared_database_sha256_before_preflight": before,
                "shared_database_sha256_before_claim": claim_before,
                "shared_database_sha256_after_observation": after,
                "shared_database_unchanged": before == claim_before == after,
                "commands": rows,
            },
        )
        raise
    after = _hash_shared_database()
    if before != claim_before or claim_before != after:
        raise ValueError("shared database changed during observation")
    execution = {
        "protocol_id": PROTOCOL_ID,
        "protocol_commit": PROTOCOL_COMMIT,
        "preflight_evidence_commit": implementation_head,
        "retry_count": 0,
        "development_claim_count": 1,
        "holdout_claim_count": int(holdout is not None),
        "claim_count": 1 + int(holdout is not None),
        "stage_process_count": 6 * (1 + int(holdout is not None)),
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
        "commands": rows,
    }
    _write_json_exclusive(evidence / "execution.json", execution)
    return {"development": development, "holdout": holdout, "execution": execution}


def _read_json_command(path: Path) -> dict[str, Any]:
    path.resolve().relative_to(CONTAINER_ROOT)
    return evaluation.read_json(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Observe frozen WSLC rank-only v8 once"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    commands.add_parser("verify-preflight")
    commands.add_parser("run")
    copy = commands.add_parser("model-copy-verify")
    copy.add_argument("--source-cache", type=Path, required=True)
    copy.add_argument("--cache", type=Path, required=True)
    copy.add_argument("--output", type=Path, required=True)
    probe = commands.add_parser("model-probe")
    probe.add_argument("--cache", type=Path, required=True)
    commands.add_parser("dependency-report")
    read = commands.add_parser("read-json")
    read.add_argument("path", type=Path)
    claim = commands.add_parser("claim")
    claim.add_argument("--stage", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--stage", required=True)
    worker.add_argument("--kind", required=True)
    worker.add_argument("--replay", required=True)
    worker.add_argument("--database", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
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
    elif arguments.command == "model-copy-verify":
        _container_model_copy(arguments.source_cache, arguments.cache, arguments.output)
        result = {"status": "verified"}
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
