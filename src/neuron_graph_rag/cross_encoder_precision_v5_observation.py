from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import socket
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from . import cross_encoder_precision_v4_observation as _v4
from . import cross_encoder_precision_v5_evaluation as evaluation

PROTOCOL_ID = evaluation.PROTOCOL_ID
PROTOCOL_COMMIT = "d5c25d7998d634cac0aa96511f59a9cce0b7725a"
SOURCE_COMMIT = _v4.SOURCE_COMMIT
ROOT = evaluation.ROOT
IMAGE = "ngr-cross-encoder-precision-v5:freeze"
IMAGE_ID = "sha256:bc105cebf12e144ef0e178b18b3ff95367bf7567113fdfe524c6c7c2de2b4dd2"
BASE_IMAGE_ID = "sha256:f0c05afecbd16040caff4c000954567c7e3b56fc6c1f783fa10a55cba3ccfbfc"
WSLC_VERSION = "2.9.4.0"
VOLUME = "github-cross-encoder-precision-v5-runtime"
CONTAINER_ROOT = Path("/opt/ngr-v5/runtime")
CONTAINER_SOURCE = CONTAINER_ROOT / "source"
CONTAINER_CACHE = CONTAINER_ROOT / "model-cache"
EVIDENCE = Path("tests/evidence/github_cross_encoder_precision_v5")
BATCH_SIZE = 8
SHARED_DATABASE = Path(r"C:\Users\smile\.ngrdb\knowledge.db")
MODEL_CACHE = Path(
    r"C:\Users\smile\Codex\workspace\experiments"
    r"\github_cross_encoder_precision_v1\model-cache"
)
WORKERS = tuple(
    (kind, replay)
    for kind in ("baseline", "base", "v2-m3")
    for replay in ("primary", "replay")
)


def canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(raw)


def _git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=root, check=False, capture_output=True
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))
    return completed.stdout.decode("ascii", errors="strict").strip()


def _hash_shared_database() -> str:
    # This reader requests FILE_SHARE_READ/WRITE/DELETE on Windows and never opens
    # SQLite. A live writer may therefore coexist without becoming an input.
    return _v4._BASE.hash_file_shared(SHARED_DATABASE)


def _command_row(command: Sequence[str], completed: subprocess.CompletedProcess[bytes]) -> dict[str, Any]:
    command_list = list(command)
    return {
        "command": command_list,
        "command_sha256": canonical_sha256(command_list),
        "returncode": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
    }


def _run_logged(
    command: Sequence[str],
    root: Path,
    rows: list[dict[str, Any]],
    *,
    environment: Mapping[str, str] | None = None,
    input_bytes: bytes | None = None,
) -> str:
    completed = subprocess.run(
        list(command),
        cwd=root,
        env=None if environment is None else dict(environment),
        check=False,
        capture_output=True,
        input=input_bytes,
    )
    row = _command_row(command, completed)
    if input_bytes is not None:
        row["stdin_sha256"] = hashlib.sha256(input_bytes).hexdigest()
        row["stdin_size"] = len(input_bytes)
    rows.append(row)
    if completed.returncode:
        message = completed.stderr.decode("utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"command failed ({completed.returncode}): {message}")
    return completed.stdout.decode("utf-8", errors="strict")


def _wslc_json(arguments: Sequence[str], root: Path, rows: list[dict[str, Any]]) -> Any:
    return json.loads(_run_logged(["wslc", *arguments], root, rows))


def _volume_exists(root: Path, rows: list[dict[str, Any]]) -> bool:
    completed = subprocess.run(
        ["wslc", "volume", "inspect", VOLUME],
        cwd=root,
        check=False,
        capture_output=True,
    )
    rows.append(_command_row(["wslc", "volume", "inspect", VOLUME], completed))
    if completed.returncode == 0:
        return True
    text = (completed.stdout + completed.stderr).decode("utf-8", errors="replace")
    if "not found" not in text.lower() and "見つかりません" not in text:
        raise RuntimeError(f"unable to inspect v5 volume: {text[-2000:]}")
    return False


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
            "neuron_graph_rag.cross_encoder_precision_v5_observation",
            *arguments,
        ]
    )
    return command


def _bind_container_harness() -> None:
    bindings = {
        "PROTOCOL_ID": PROTOCOL_ID,
        "PROTOCOL_COMMIT": PROTOCOL_COMMIT,
        "SOURCE_COMMIT": SOURCE_COMMIT,
        "ROOT": CONTAINER_SOURCE,
        "EVIDENCE": EVIDENCE,
        "BATCH_SIZE": BATCH_SIZE,
        "load_protocol": evaluation.load_protocol,
        "archive_stage": evaluation.archive_stage,
        "evaluate_result_payload": evaluation.evaluate_result_payload,
        "project_passages": evaluation.project_passages,
        "read_json": evaluation.read_json,
        "register_stage_claim": evaluation.register_stage_claim,
        "sha256_bytes": evaluation.sha256_bytes,
        "verify_phase_state": evaluation.verify_phase_state,
        "verify_protocol_commit": evaluation.verify_protocol_commit,
        "verify_result_payload": evaluation.verify_result_payload,
        "write_json_exclusive": evaluation.write_json_exclusive,
        "write_stage_error": evaluation.write_stage_error,
        "write_stage_result": evaluation.write_stage_result,
    }
    for module in (_v4, _v4._BASE):
        for name, value in bindings.items():
            setattr(module, name, value)
    evaluation.ROOT = CONTAINER_SOURCE
    evaluation._BASE.ROOT = CONTAINER_SOURCE

    def direct_git_bytes(root: Path, commit: str, path: str) -> bytes:
        if commit not in {SOURCE_COMMIT, PROTOCOL_COMMIT}:
            raise ValueError("unexpected frozen commit")
        return (CONTAINER_ROOT / "frozen-source" / path).read_bytes()

    _v4._BASE._git_bytes = direct_git_bytes
    evaluation._BASE._git_bytes = direct_git_bytes


def _dependency_report() -> dict[str, Any]:
    protocol = evaluation.load_protocol(CONTAINER_SOURCE)
    expected = {
        row["name"]: row["version"]
        for row in protocol["dependency_artifacts"]["artifacts"]
    }
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
        row["name"]: row
        for row in protocol["dependency_artifacts"]["artifacts"]
    }
    install_report = evaluation.read_json(Path("/opt/ngr-v5/dependency-report.json"))
    if install_report.get("pip_version") != "24.0":
        raise ValueError("pip report version mismatch")
    installed_artifacts = []
    for row in install_report.get("install", []):
        metadata = row.get("metadata", {})
        name = str(metadata.get("name", "")).lower().replace("_", "-")
        frozen = frozen_rows.get(name)
        download = row.get("download_info", {})
        archive = download.get("archive_info", {})
        url = download.get("url")
        digest = archive.get("hashes", {}).get("sha256")
        filename = unquote(Path(urlparse(str(url)).path).name)
        actual = {
            "name": name,
            "version": metadata.get("version"),
            "filename": filename,
            "url": url,
            "sha256": digest,
        }
        if actual != frozen:
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
            / "tests/fixtures/github_cross_encoder_precision_v5.dependency-artifacts.json"
        ),
        "lock_sha256": sha256_file(
            CONTAINER_SOURCE
            / "tests/fixtures/github_cross_encoder_precision_v5.requirements.lock"
        ),
        "artifact_count": len(expected),
        "installed": {name: installed[name] for name in sorted(expected)},
        "pip_version": install_report["pip_version"],
        "artifacts": sorted(installed_artifacts, key=lambda row: row["name"]),
        "forbidden": forbidden,
    }


def _container_model_copy(source: Path, cache: Path, output: Path) -> None:
    _bind_container_harness()
    _v4.model_copy_verify(source, cache, output)


def _container_model_probe(cache: Path) -> dict[str, Any]:
    _bind_container_harness()
    return _v4.model_probe(cache)


def _container_worker(
    stage: str, kind: str, replay: str, database: Path, output: Path
) -> dict[str, Any]:
    _bind_container_harness()
    for path in (database.parent, output.parent, CONTAINER_CACHE):
        path.resolve().relative_to(CONTAINER_ROOT)
    if (kind, replay) not in WORKERS:
        raise ValueError("worker order identity is not frozen")
    _v4._BASE.worker(stage, kind, CONTAINER_CACHE, database, output)
    payload = evaluation.read_json(output)
    payload["container_id"] = os.environ.get(
        "NGR_V5_CONTAINER_IDENTITY", socket.gethostname()
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


def _stage_paths(stage: str) -> tuple[Path, Path]:
    if stage not in evaluation.STAGES:
        raise ValueError("unknown stage")
    return CONTAINER_ROOT / "runs" / stage, CONTAINER_ROOT / "databases" / stage


def _container_claim(stage: str) -> dict[str, Any]:
    _bind_container_harness()
    path = evaluation.register_stage_claim(stage, PROTOCOL_COMMIT, CONTAINER_SOURCE)
    return {"stage": stage, "claim_sha256": sha256_file(path)}


def _copy_stage_evidence(stage: str) -> None:
    protocol = evaluation.load_protocol(CONTAINER_SOURCE)
    outputs = protocol["manifest"]["outputs"][stage]
    evidence = CONTAINER_SOURCE / EVIDENCE
    for key, suffix in (
        ("archive_claim", "claim"),
        ("archive_result", "observed"),
        ("archive_error", "error"),
        ("transport", "transport"),
    ):
        source = CONTAINER_SOURCE / outputs[key]
        if source.is_file():
            destination = evidence / f"{stage}.{suffix}.json"
            _v4._BASE._write_bytes_exclusive(destination, source.read_bytes())
            if destination.read_bytes() != source.read_bytes():
                raise ValueError("archive/git evidence byte identity mismatch")


def _container_finalize(stage: str) -> dict[str, Any]:
    _bind_container_harness()
    stage_root, _ = _stage_paths(stage)
    protocol = evaluation.load_protocol(CONTAINER_SOURCE)
    claim = CONTAINER_SOURCE / protocol["manifest"]["outputs"][stage]["runtime_claim"]
    claim_raw = claim.read_bytes()
    try:
        packets = {
            (kind, replay): evaluation.read_json(
                stage_root / f"{kind}-{replay}.json"
            )
            for kind, replay in WORKERS
        }
        baseline_primary = packets[("baseline", "primary")]
        baseline_replay = packets[("baseline", "replay")]
        baseline = {
            "baseline_id": "current-ngr",
            "cases": baseline_primary["cases"],
            "state": _v4._BASE._combine_state(
                baseline_primary, baseline_replay
            ),
        }
        models = []
        for kind in ("base", "v2-m3"):
            primary = packets[(kind, "primary")]
            replay = packets[(kind, "replay")]
            if primary["cases"] != replay["cases"]:
                raise ValueError(f"{kind} replay raw cases differ")
            models.append(
                {
                    "model_id": primary["model_id"],
                    "revision": primary["revision"],
                    "cases": primary["cases"],
                    "state": _v4._BASE._combine_state(primary, replay),
                    "metrics": primary["metrics"],
                }
            )
        _v4._BASE._archive_raw_workers(stage, stage_root, CONTAINER_SOURCE)
        result = evaluation.evaluate_result_payload(
            protocol, stage, claim_raw, baseline, models
        )
        evaluation.verify_result_payload(protocol, stage, result, claim_raw)
        evaluation.write_stage_result(stage, result)
        evaluation.archive_stage(stage, CONTAINER_SOURCE)
        evaluation.verify_phase_state(evaluation.load_protocol(CONTAINER_SOURCE))
        _copy_stage_evidence(stage)
        return result
    except BaseException as error:
        raw_manifest = CONTAINER_SOURCE / EVIDENCE / f"{stage}.raw-archive.json"
        if not raw_manifest.exists() and stage_root.exists():
            _v4._BASE._archive_raw_workers(stage, stage_root, CONTAINER_SOURCE)
        if claim.exists():
            evaluation.write_stage_error(stage, f"{type(error).__name__}: {error}")
            evaluation.archive_stage(stage, CONTAINER_SOURCE)
            _copy_stage_evidence(stage)
        raise


def _container_fail_stage(stage: str, message: str) -> dict[str, Any]:
    _bind_container_harness()
    protocol = evaluation.load_protocol(CONTAINER_SOURCE)
    claim = CONTAINER_SOURCE / protocol["manifest"]["outputs"][stage]["runtime_claim"]
    if not claim.is_file():
        raise FileNotFoundError("stage claim is unavailable")
    stage_root, _ = _stage_paths(stage)
    raw_manifest = CONTAINER_SOURCE / EVIDENCE / f"{stage}.raw-archive.json"
    if not raw_manifest.exists() and stage_root.exists():
        _v4._BASE._archive_raw_workers(stage, stage_root, CONTAINER_SOURCE)
    evaluation.write_stage_error(stage, message)
    evaluation.archive_stage(stage, CONTAINER_SOURCE)
    _copy_stage_evidence(stage)
    return {"stage": stage, "status": "archived-error"}


def _source_initialization_script() -> str:
    return (
        "test ! -e /opt/ngr-v5/runtime/source && "
        "mkdir -p /opt/ngr-v5/runtime/logs /opt/ngr-v5/runtime/databases "
        "/opt/ngr-v5/runtime/runs /opt/ngr-v5/runtime/archive && "
        "cp -a /input/source /opt/ngr-v5/runtime/source && "
        "rm -f /opt/ngr-v5/runtime/source/.git && "
        "test -f /opt/ngr-v5/runtime/source/tests/fixtures/"
        "github_cross_encoder_precision_v5.manifest.json"
    )


def _freeze_export_script() -> str:
    return (
        "test ! -e /opt/ngr-v5/runtime/frozen-source && "
        "mkdir /opt/ngr-v5/runtime/frozen-source && "
        "tar -xf - -C /opt/ngr-v5/runtime/frozen-source && "
        "test -f /opt/ngr-v5/runtime/frozen-source/tests/fixtures/"
        "github_cross_encoder_precision_v5.manifest.json"
    )


def _verify_host_source(root: Path) -> str:
    if root.resolve() != ROOT.resolve():
        raise ValueError("v5 host source must be the delegated checkout")
    if _git_output(root, "status", "--porcelain"):
        raise ValueError("preflight source must be committed and clean")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
    ).returncode:
        raise ValueError("HEAD must contain the exact v5 freeze merge commit")
    evaluation.verify_protocol_commit(PROTOCOL_COMMIT, evaluation.load_protocol(root))
    return _git_output(root, "rev-parse", "HEAD")


def _wslc_version(root: Path, rows: list[dict[str, Any]]) -> str:
    first = _run_logged(["wslc", "--version"], root, rows).splitlines()[0]
    version = first.removeprefix("wslc ").strip()
    if version != WSLC_VERSION:
        raise ValueError("WSLC version mismatch")
    return version


def _image_identity(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    value = _wslc_json(["image", "inspect", IMAGE], root, rows)
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError("built image inspection shape mismatch")
    image = value[0]
    if image.get("Id") != IMAGE_ID or image.get("Architecture") != "amd64":
        raise ValueError("rebuilt image identity mismatch")
    return {"tag": IMAGE, "id": image["Id"], "architecture": image["Architecture"]}


def _base_image_identity(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    tag = "python:3.11.15-slim-bookworm"
    value = _wslc_json(["image", "inspect", tag], root, rows)
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError("base image inspection shape mismatch")
    image = value[0]
    if image.get("Id") != BASE_IMAGE_ID or image.get("Architecture") != "amd64":
        raise ValueError("base image identity mismatch")
    return {"tag": tag, "id": image["Id"], "architecture": image["Architecture"]}


def _export_volume_evidence(root: Path, rows: list[dict[str, Any]]) -> None:
    evidence = (root / EVIDENCE).resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    copy_command = (
        "cp -an /opt/ngr-v5/runtime/source/tests/evidence/"
        "github_cross_encoder_precision_v5/. /output/"
    )
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
        copy_command,
    ]
    _run_logged(command, root, rows)


def preflight(
    root: Path = ROOT,
    model_cache: Path = MODEL_CACHE,
) -> dict[str, Any]:
    evidence = root / EVIDENCE
    if evidence.exists():
        raise FileExistsError("v5 preflight evidence already exists")
    rows: list[dict[str, Any]] = []
    shared_before = ""
    implementation_commit = ""
    try:
        implementation_commit = _verify_host_source(root)
        if _volume_exists(root, rows):
            raise FileExistsError("v5 volume already exists; preflight is not retryable")
        shared_before = _hash_shared_database()
        _wslc_version(root, rows)
        _run_logged(
            [
                "wslc",
                "build",
                "--no-cache",
                "--file",
                "containers/github_cross_encoder_precision_v5/Containerfile",
                "--tag",
                IMAGE,
                ".",
            ],
            root,
            rows,
        )
        image = _image_identity(root, rows)
        base_image = _base_image_identity(root, rows)
        validator = json.loads(
            _run_logged(
                ["wslc", "run", "--rm", "--network", "none", IMAGE], root, rows
            )
        )
        validator["protocol_id"] = PROTOCOL_ID
        validator["wslc_version"] = WSLC_VERSION
        validator["image_id"] = IMAGE_ID
        evaluation.validate_runtime_metadata(evaluation.load_protocol(root), validator)

        _run_logged(["wslc", "volume", "create", VOLUME], root, rows)
        # Stream the binary export directly to tar in the target volume.
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
        source_mount = f"{root.resolve()}:/input/source:ro"
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
                source_mount,
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
        if dependency.get("artifact_registry_sha256") != (
            "6e5a05a83b6fa914c7bdbff1983d2e7e9351ef9975cd25752ca4c60ce0c90a14"
        ):
            raise ValueError("dependency artifact registry mismatch")
        if dependency.get("lock_sha256") != (
            "94f39ace20ce53e82c2b85c050ba551af5639ff28eada9dcd8b6e378000fa981"
        ):
            raise ValueError("dependency lock mismatch")
        verification_environment = os.environ.copy()
        verification_environment["PYTHONPATH"] = str(root / "src")
        verification_environment["PYTHONUTF8"] = "1"
        verification_environment["HF_HUB_OFFLINE"] = "1"
        verification_environment["TRANSFORMERS_OFFLINE"] = "1"
        project_python = root / ".venv" / "Scripts" / "python.exe"
        for command in (
            [
                "uvx",
                "--offline",
                "ruff",
                "check",
                "src/neuron_graph_rag/cross_encoder_precision_v5_observation.py",
                "tests/test_cross_encoder_precision_v5_observation.py",
            ],
            [
                str(project_python),
                "-m",
                "unittest",
                "tests.test_cross_encoder_precision_v5",
                "tests.test_cross_encoder_precision_v5_observation",
            ],
            [
                str(project_python),
                "-m",
                "neuron_graph_rag.cross_encoder_precision_v5_evaluation",
                "audit",
            ],
            [
                str(project_python),
                "-m",
                "neuron_graph_rag.cross_encoder_precision_v5_evaluation",
                "probe",
            ],
            [
                str(project_python),
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "test_*.py",
            ],
        ):
            _run_logged(
                command,
                root,
                rows,
                environment=verification_environment,
            )
        shared_after = _hash_shared_database()
        if shared_after != shared_before:
            raise ValueError("shared Windows database changed during preflight")
        platform_report = {
            "protocol_id": PROTOCOL_ID,
            "wslc_version": WSLC_VERSION,
            "image": image,
            "base_image": base_image,
            "base_digest": (
                "sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3"
            ),
            "frozen_source_export_sha256": hashlib.sha256(frozen_archive).hexdigest(),
            "validator": validator,
            "volume": VOLUME,
            "volume_absent_before_create": True,
            "container_root": str(CONTAINER_ROOT),
            "host_bind_shared_database": False,
            "fresh_worker_container_process": True,
            "fresh_worker_database": True,
        }
        model_report = json.loads(
            _run_logged(
                _container_command("read-json", str(model_output)), root, rows
            )
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
        _write_json_exclusive(evidence / "preclaim.json", preclaim)
        _write_json_exclusive(evidence / "model-verification.json", model_report)
        _write_json_exclusive(evidence / "dependency-report.json", dependency)
        _write_json_exclusive(evidence / "platform-report.json", platform_report)
        _write_json_exclusive(evidence / "preflight-commands.json", {"commands": rows})
        _write_json_exclusive(evidence / "preflight.json", report)
        return report
    except BaseException as error:
        evidence.mkdir(parents=True, exist_ok=True)
        failure = {
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
        }
        _write_json_exclusive(evidence / "preflight.error.json", failure)
        raise


def verify_preflight(root: Path = ROOT) -> dict[str, Any]:
    evidence = root / EVIDENCE
    report = evaluation.read_json(evidence / "preflight.json")
    preclaim = evaluation.read_json(evidence / "preclaim.json")
    model = evaluation.read_json(evidence / "model-verification.json")
    dependency = evaluation.read_json(evidence / "dependency-report.json")
    platform_report = evaluation.read_json(evidence / "platform-report.json")
    commands = evaluation.read_json(evidence / "preflight-commands.json")
    if report.get("protocol_id") != PROTOCOL_ID or report.get("protocol_commit") != PROTOCOL_COMMIT:
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
    if report.get("preclaim_sha256") != canonical_sha256(preclaim):
        raise ValueError("preclaim binding mismatch")
    if report.get("model_report_sha256") != canonical_sha256(model):
        raise ValueError("model report binding mismatch")
    if report.get("dependency_report_sha256") != canonical_sha256(dependency):
        raise ValueError("dependency report binding mismatch")
    if report.get("platform_report_sha256") != canonical_sha256(platform_report):
        raise ValueError("platform report binding mismatch")
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
    if current_shared != report.get("shared_database_sha256_before_preflight") or current_shared != report.get("shared_database_sha256_after_preflight"):
        raise ValueError("shared database no longer matches preflight")
    return report


def _remote_ci_green(root: Path, commit: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
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
        raise ValueError("preflight evidence commit CI is not complete")
    if any(row.get("conclusion") not in {"success", "skipped", "neutral"} for row in checks):
        raise ValueError("preflight evidence commit CI is not green")
    return {
        "protocol_id": PROTOCOL_ID,
        "preflight_evidence_commit": commit,
        "checks": [
            {"name": row["name"], "status": row["status"], "conclusion": row["conclusion"]}
            for row in checks
        ],
    }


def _sync_preflight_evidence(root: Path, rows: list[dict[str, Any]]) -> None:
    source = (root / EVIDENCE).resolve()
    copy_command = (
        "mkdir -p /opt/ngr-v5/runtime/source/tests/evidence/"
        "github_cross_encoder_precision_v5 && cp -a /input/evidence/. "
        "/opt/ngr-v5/runtime/source/tests/evidence/"
        "github_cross_encoder_precision_v5/"
    )
    command = [
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
        copy_command,
    ]
    _run_logged(command, root, rows)


def _run_stage_host(stage: str, root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    _run_logged(_container_command("claim", "--stage", stage), root, rows)
    stage_root = CONTAINER_ROOT / "runs" / stage
    database_root = CONTAINER_ROOT / "databases" / stage
    for kind, replay in WORKERS:
        identity = f"ngr-v5-{stage}-{kind}-{replay}"
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
        command[insert_at:insert_at] = ["--env", f"NGR_V5_CONTAINER_IDENTITY={identity}"]
        _run_logged(command, root, rows)
    result = json.loads(
        _run_logged(_container_command("finalize", "--stage", stage), root, rows)
    )
    _export_volume_evidence(root, rows)
    return result


def run_once(root: Path = ROOT) -> dict[str, Any]:
    report = verify_preflight(root)
    evidence = root / EVIDENCE
    if (evidence / "execution.json").exists() or (evidence / "execution-error.json").exists():
        raise FileExistsError("v5 observation already has terminal execution evidence")
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
        failure = {
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
        }
        _write_json_exclusive(evidence / "execution-error.json", failure)
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
            "holdout": None if holdout is None else holdout.get("selected_candidate_id"),
        },
        "commands": rows,
    }
    _write_json_exclusive(evidence / "execution.json", execution)
    return {"development": development, "holdout": holdout, "execution": execution}


def _read_json_command(path: Path) -> dict[str, Any]:
    path.resolve().relative_to(CONTAINER_ROOT)
    return evaluation.read_json(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Observe frozen WSLC rank-only v5 once")
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
        _bind_container_harness()
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
