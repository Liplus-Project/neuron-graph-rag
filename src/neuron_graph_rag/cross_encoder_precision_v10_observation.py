from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

PROTOCOL_ID = "github-ngr-cross-encoder-precision-v10"
PREDECESSOR_MERGE_COMMIT = "aefe0123d48b762445c1a58e5ae6056cc02feab0"
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path("tests/fixtures/github_cross_encoder_precision_v10.manifest.json")
RESULT_FREE_AUDIT = Path(
    "tests/fixtures/github_cross_encoder_precision_v10.result-free-audit.json"
)
MODEL_REGISTRY = Path("tests/fixtures/github_cross_encoder_precision_v8.models.json")
EVIDENCE = Path("tests/evidence/github_cross_encoder_precision_v10")

IMAGE = "ngr-cross-encoder-precision-v8:freeze"
IMAGE_ID = "sha256:136ad9466799109bf32b4b96b611c9db9a099bcc47cf78243f26c7227bc16742"
WSLC_VERSION = "2.9.4.0"
CACHE_FREEZE_VOLUME = "github-cross-encoder-precision-v10-cache-freeze"
FUTURE_RUNTIME_VOLUME = "github-cross-encoder-precision-v10-runtime"
CONTAINER_ROOT = PurePosixPath("/opt/ngr-v10/cache-freeze")
CONTAINER_PREDECESSOR_SOURCE = CONTAINER_ROOT / "predecessor-source"
CONTAINER_SOURCE = CONTAINER_ROOT / "source"
CONTAINER_CACHE = CONTAINER_ROOT / "model-cache"
CONTAINER_DATABASES = CONTAINER_ROOT / "databases"
CONTAINER_RUNS = CONTAINER_ROOT / "runs"
CONTAINER_ARCHIVE = CONTAINER_ROOT / "archive"
CONTAINER_TRANSPORT = CONTAINER_ROOT / "transport"
CONTAINER_MODEL_REGISTRY = (
    CONTAINER_SOURCE / "tests/fixtures/github_cross_encoder_precision_v8.models.json"
)
CONTAINER_MODEL_REPORT = CONTAINER_ROOT / "model-verification.json"

V9_RAW_FAILURE_PATH = Path(
    "tests/evidence/github_cross_encoder_precision_v9_observation/preflight.error.json"
)
V9_RAW_FAILURE_SHA256 = (
    "cc3c57682dd25df86d8aa0122efee9ef081b18ae2d08f216e87672d2ffff4426"
)
V9_TERMINAL_PATH = Path(
    "tests/evidence/github_cross_encoder_precision_v9_observation/preflight-terminal.json"
)
V9_TERMINAL_SHA256 = "676480a3af07c09a0041623ed483d885884dea3d26a2c9f722e5d30a3ba0786e"
V9_EVIDENCE_MANIFEST_PATH = Path(
    "tests/evidence/github_cross_encoder_precision_v9_observation/"
    "observation-evidence-manifest.json"
)
V9_EVIDENCE_MANIFEST_SHA256 = (
    "75f581c1b07520c4c55cbcb9cd49805b2c29919d25fcb94204074038f8c292d6"
)

_VOLUME_NAME = re.compile(r"[a-z0-9][a-z0-9_.-]*\Z")
_DRIVE_PREFIX = re.compile(r"(?:^|/)[A-Za-z]:")


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value))


def serialize_container_path(value: PurePosixPath | str) -> str:
    if isinstance(value, Path):
        raise TypeError("host Path is forbidden for container paths")
    if not isinstance(value, (PurePosixPath, str)):
        raise TypeError("container path must be PurePosixPath or str")
    raw = str(value)
    if not raw or raw in {".", ".."}:
        raise ValueError("container path is empty or relative")
    if "\x00" in raw:
        raise ValueError("container path contains NUL")
    if "\\" in raw:
        raise ValueError("container path contains a host separator")
    if not raw.startswith("/") or raw.startswith("//"):
        raise ValueError("container path must have one leading slash")
    if _DRIVE_PREFIX.search(raw) or ":" in raw:
        raise ValueError("container path contains a drive or volume prefix")
    segments = raw.split("/")[1:]
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError("container path is not canonical or traverses")
    rendered = PurePosixPath(raw).as_posix()
    if rendered != raw:
        raise ValueError("container path is not canonical absolute POSIX")
    return rendered


def named_volume_spec(
    volume: str,
    destination: PurePosixPath | str,
    *,
    mode: str | None = None,
) -> str:
    if not _VOLUME_NAME.fullmatch(volume):
        raise ValueError("named volume is not canonical")
    if mode not in {None, "ro", "rw"}:
        raise ValueError("volume mode must be ro, rw, or omitted")
    suffix = "" if mode is None else f":{mode}"
    return f"{volume}:{serialize_container_path(destination)}{suffix}"


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
        raise TypeError("v10 manifest must be an object")
    return value


def _audit_contract(root: Path) -> dict[str, Any]:
    value = read_json(root / RESULT_FREE_AUDIT)
    if not isinstance(value, dict):
        raise TypeError("v10 result-free audit must be an object")
    return value


def _model_registry(root: Path) -> dict[str, Any]:
    value = read_json(root / MODEL_REGISTRY)
    if not isinstance(value, dict):
        raise TypeError("frozen model registry must be an object")
    return value


def _verify_predecessor_hashes(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, str]:
    registry = manifest.get("predecessor_immutable_sha256")
    if not isinstance(registry, dict) or len(registry) != 20:
        raise ValueError("v9 predecessor hash registry must contain 20 files")
    verified: dict[str, str] = {}
    for relative, expected in registry.items():
        path = root / str(relative)
        if not path.is_file():
            raise ValueError(f"v9 predecessor artifact is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"v9 predecessor artifact changed: {relative}")
        verified[str(relative)] = actual
    expected_anchors = {
        V9_RAW_FAILURE_PATH.as_posix(): V9_RAW_FAILURE_SHA256,
        V9_TERMINAL_PATH.as_posix(): V9_TERMINAL_SHA256,
        V9_EVIDENCE_MANIFEST_PATH.as_posix(): V9_EVIDENCE_MANIFEST_SHA256,
    }
    for relative, expected in expected_anchors.items():
        if verified.get(relative) != expected:
            raise ValueError(f"v9 anchor hash is not frozen: {relative}")
    return verified


def _expected_container_paths() -> dict[str, str]:
    return {
        "root": serialize_container_path(CONTAINER_ROOT),
        "predecessor_source": serialize_container_path(CONTAINER_PREDECESSOR_SOURCE),
        "source": serialize_container_path(CONTAINER_SOURCE),
        "model_cache": serialize_container_path(CONTAINER_CACHE),
        "databases": serialize_container_path(CONTAINER_DATABASES),
        "runs": serialize_container_path(CONTAINER_RUNS),
        "archive": serialize_container_path(CONTAINER_ARCHIVE),
        "transport": serialize_container_path(CONTAINER_TRANSPORT),
    }


def validate_prebuild(root: Path = ROOT) -> dict[str, Any]:
    manifest = _manifest(root)
    audit = _audit_contract(root)
    expected_header = {
        "protocol_id": PROTOCOL_ID,
        "phase": "cache-freeze",
        "predecessor_merge_commit": PREDECESSOR_MERGE_COMMIT,
        "accepted_image": {"tag": IMAGE, "id": IMAGE_ID},
        "wslc_version": WSLC_VERSION,
        "cache_freeze_volume": CACHE_FREEZE_VOLUME,
        "future_runtime_volume": FUTURE_RUNTIME_VOLUME,
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            raise ValueError(f"v10 manifest mismatch: {key}")
    if manifest.get("container_paths") != _expected_container_paths():
        raise ValueError("v10 container path registry mismatch")
    if manifest.get("model_registry_sha256") != sha256_file(root / MODEL_REGISTRY):
        raise ValueError("v10 frozen model registry hash mismatch")
    if manifest.get("result_free_audit_sha256") != sha256_file(
        root / RESULT_FREE_AUDIT
    ):
        raise ValueError("v10 result-free audit hash mismatch")
    verified = _verify_predecessor_hashes(root, manifest)
    expected_zero = {
        "development_claim_count": 0,
        "holdout_claim_count": 0,
        "registered_query_execution_count": 0,
        "model_import_count": 0,
        "model_load_count": 0,
        "model_forward_inference_count": 0,
        "observed_result_count": 0,
        "retry_count": 0,
        "accepted_image_rebuild_count": 0,
        "runtime_report_rerun_count": 0,
        "attestation_rerun_count": 0,
    }
    for key, expected in expected_zero.items():
        if audit.get(key) != expected:
            raise ValueError(f"v10 result-free count mismatch: {key}")
    if (
        audit.get("protocol_id") != PROTOCOL_ID
        or audit.get("phase") != "cache-freeze"
        or audit.get("performance") != "not assessed"
        or audit.get("cache_freeze_volume_reusable") is not False
        or audit.get("future_runtime_volume") != FUTURE_RUNTIME_VOLUME
        or audit.get("shared_database_opened") is not False
        or audit.get("model_copy_verifier_run_limit") != 1
    ):
        raise ValueError("v10 result-free boundary mismatch")
    models = _model_registry(root).get("models")
    if not isinstance(models, list) or len(models) != 2:
        raise ValueError("v10 requires exactly two frozen model identities")
    required = [row for model in models for row in model.get("required_files", [])]
    if len(required) != 12:
        raise ValueError("v10 frozen model registry must contain 12 required files")
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "prebuild_contract_valid",
        "predecessor_artifact_count": len(verified),
        "model_count": len(models),
        "required_file_count": len(required),
        "required_file_size": sum(int(row["size"]) for row in required),
        "model_cache_created_by_source_initialization": False,
        "model_copy_verifier_run_limit": 1,
        "registered_query_execution_count": 0,
        "model_import_count": 0,
        "model_load_count": 0,
        "model_forward_inference_count": 0,
        "observed_result_count": 0,
        "performance": "not assessed",
    }


def discover_source_cache(root: Path = ROOT) -> Path:
    candidate = (
        root.resolve().parents[1]
        / "experiments"
        / "github_cross_encoder_precision_v1"
        / "model-cache"
    )
    models = _model_registry(root).get("models", [])
    if not candidate.is_dir():
        raise FileNotFoundError("frozen Windows model cache is unavailable")
    for model in models:
        model_id = str(model["model_id"])
        revision = str(model["revision"])
        snapshot = (
            candidate
            / ("models--" + model_id.replace("/", "--"))
            / "snapshots"
            / revision
        )
        if not snapshot.is_dir():
            raise FileNotFoundError(
                f"frozen Windows model snapshot is unavailable: {model_id}@{revision}"
            )
    return candidate


def _source_cache_identity(root: Path, source_cache: Path) -> dict[str, Any]:
    models = _model_registry(root)["models"]
    resolved = source_cache.resolve()
    return {
        "host_path": str(resolved),
        "host_path_sha256": hashlib.sha256(str(resolved).encode("utf-8")).hexdigest(),
        "container_destination": "/input/models",
        "read_only": True,
        "model_registry_sha256": sha256_file(root / MODEL_REGISTRY),
        "models": [
            {
                "model_id": str(model["model_id"]),
                "revision": str(model["revision"]),
                "required_file_count": len(model["required_files"]),
                "required_file_size": sum(
                    int(row["size"]) for row in model["required_files"]
                ),
            }
            for model in models
        ],
    }


def _command_row(
    command: Sequence[str], completed: subprocess.CompletedProcess[bytes]
) -> dict[str, Any]:
    return {
        "command": list(command),
        "command_sha256": canonical_sha256(list(command)),
        "returncode": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
    }


def _run_logged(
    command: Sequence[str],
    root: Path,
    rows: list[dict[str, Any]],
    *,
    input_bytes: bytes | None = None,
) -> str:
    completed = subprocess.run(
        list(command),
        cwd=root,
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
        text = completed.stderr.decode("utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"command failed ({completed.returncode}): {text}")
    return completed.stdout.decode("utf-8", errors="strict")


def _inspect_absent(volume: str, root: Path, rows: list[dict[str, Any]]) -> bool:
    command = ["wslc", "volume", "inspect", volume]
    completed = subprocess.run(command, cwd=root, check=False, capture_output=True)
    row = _command_row(command, completed)
    row["expected_absence_probe"] = True
    rows.append(row)
    if completed.returncode == 0:
        return False
    text = (completed.stdout + completed.stderr).decode("utf-8", errors="replace")
    if "not found" not in text.lower() and "見つかりません" not in text:
        raise RuntimeError(f"unable to inspect volume {volume}: {text[-2000:]}")
    return True


def _image_identity(root: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    raw = _run_logged(["wslc", "image", "inspect", IMAGE], root, rows)
    value = json.loads(raw)
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError("accepted image inspection shape mismatch")
    image = value[0]
    if image.get("Id") != IMAGE_ID or image.get("Architecture") != "amd64":
        raise ValueError("accepted image identity mismatch")
    if IMAGE not in image.get("RepoTags", []):
        raise ValueError("accepted image tag mismatch")
    return {"tag": IMAGE, "id": IMAGE_ID, "architecture": "amd64"}


def _require_remote_ci_green(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=False,
        capture_output=True,
    ).stdout:
        raise ValueError("working tree must be clean before the one-shot cache freeze")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    remote = subprocess.run(
        ["git", "ls-remote", "origin", f"refs/heads/{branch}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    if not remote or remote[0] != head:
        raise ValueError("one-shot cache freeze requires the pushed remote HEAD")
    raw = _run_logged(
        [
            "gh",
            "api",
            f"repos/Liplus-Project/neuron-graph-rag/commits/{head}/check-runs",
        ],
        root,
        rows,
    )
    checks = json.loads(raw).get("check_runs", [])
    if not checks or any(row.get("status") != "completed" for row in checks):
        raise ValueError("prebuild remote CI is not complete")
    if any(
        row.get("conclusion") not in {"success", "skipped", "neutral"} for row in checks
    ):
        raise ValueError("prebuild remote CI is not green")
    return {
        "commit": head,
        "checks": [
            {
                "name": row["name"],
                "status": row["status"],
                "conclusion": row["conclusion"],
            }
            for row in checks
        ],
    }


def _archive_bytes(commit: str, root: Path, rows: list[dict[str, Any]]) -> bytes:
    command = ["git", "archive", "--format=tar", commit]
    completed = subprocess.run(command, cwd=root, check=False, capture_output=True)
    row = _command_row(command, completed)
    row["binary_export"] = True
    rows.append(row)
    if completed.returncode:
        raise RuntimeError(f"git archive failed for {commit}")
    return completed.stdout


def _predecessor_export_script() -> str:
    target = serialize_container_path(CONTAINER_PREDECESSOR_SOURCE)
    fixture = serialize_container_path(
        CONTAINER_PREDECESSOR_SOURCE
        / "tests/fixtures/github_cross_encoder_precision_v9_observation.manifest.json"
    )
    cache = serialize_container_path(CONTAINER_CACHE)
    return (
        "set -eu; "
        f"test ! -e '{target}'; test ! -e '{cache}'; mkdir '{target}'; "
        f"tar -xf - -C '{target}'; test -f '{fixture}'; test ! -e '{cache}'"
    )


def _source_initialization_script() -> str:
    root = serialize_container_path(CONTAINER_ROOT)
    source = serialize_container_path(CONTAINER_SOURCE)
    cache = serialize_container_path(CONTAINER_CACHE)
    paths = " ".join(
        f"'{serialize_container_path(path)}'"
        for path in (
            CONTAINER_DATABASES,
            CONTAINER_RUNS,
            CONTAINER_ARCHIVE,
            CONTAINER_TRANSPORT,
        )
    )
    fixture = serialize_container_path(
        CONTAINER_SOURCE
        / "tests/fixtures/github_cross_encoder_precision_v10.manifest.json"
    )
    return (
        "set -eu; "
        f"test -d '{root}'; test ! -e '{source}'; test ! -e '{cache}'; "
        f"mkdir {paths}; mkdir '{source}'; tar -xf - -C '{source}'; "
        f"test -f '{fixture}'; test ! -e '{cache}'; "
        "printf '%s\n' 'model_cache_absent_after_source_initialization=true'"
    )


def _archive_import_command(script: str) -> list[str]:
    return [
        "wslc",
        "run",
        "--rm",
        "--interactive",
        "--network",
        "none",
        "--volume",
        named_volume_spec(CACHE_FREEZE_VOLUME, CONTAINER_ROOT),
        "--entrypoint",
        "/bin/sh",
        IMAGE,
        "-c",
        script,
    ]


def model_copy_command(source_cache: Path) -> list[str]:
    return [
        "wslc",
        "run",
        "--rm",
        "--network",
        "none",
        "--volume",
        named_volume_spec(CACHE_FREEZE_VOLUME, CONTAINER_ROOT),
        "--volume",
        host_bind_spec(source_cache, "/input/models", mode="ro"),
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
        "neuron_graph_rag.cross_encoder_precision_v10_observation",
        "model-copy-verify",
        "--source-cache",
        "/input/models",
        "--cache",
        serialize_container_path(CONTAINER_CACHE),
        "--models",
        serialize_container_path(CONTAINER_MODEL_REGISTRY),
        "--output",
        serialize_container_path(CONTAINER_MODEL_REPORT),
    ]


def _file_identity(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    size = path.stat().st_size
    if size != int(expected["size"]):
        raise ValueError(f"frozen model file size mismatch: {path.name}")
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(usedforsecurity=False)
    git_blob.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(chunk)
            git_blob.update(chunk)
    sha256_hex = sha256.hexdigest()
    git_blob_hex = git_blob.hexdigest()
    lfs_sha256 = expected.get("lfs_sha256")
    if lfs_sha256 is not None:
        if sha256_hex != lfs_sha256:
            raise ValueError(f"frozen LFS SHA-256 mismatch: {path.name}")
    elif git_blob_hex != expected.get("git_blob_id"):
        raise ValueError(f"frozen git blob ID mismatch: {path.name}")
    return {
        "path": str(expected["path"]),
        "size": size,
        "sha256": sha256_hex,
        "git_blob_id": git_blob_hex if lfs_sha256 is None else None,
        "registry_git_blob_id": expected.get("git_blob_id"),
        "lfs_sha256": lfs_sha256,
    }


def _snapshot_path(cache: Path, model_id: str, revision: str) -> Path:
    return cache / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision


def model_copy_verify(
    source_cache: Path,
    cache: Path,
    models_path: Path,
    output: Path,
) -> dict[str, Any]:
    if cache.exists():
        raise FileExistsError("dedicated ext4 model cache already exists")
    registry = read_json(models_path)
    models = registry.get("models", [])
    if not isinstance(models, list) or len(models) != 2:
        raise ValueError("frozen model registry identity count mismatch")
    source_rows: list[dict[str, Any]] = []
    for model in models:
        model_id = str(model["model_id"])
        revision = str(model["revision"])
        source = _snapshot_path(source_cache, model_id, revision)
        files = [
            _file_identity(source / str(expected["path"]), expected)
            for expected in model["required_files"]
        ]
        source_rows.append(
            {
                "model_id": model_id,
                "revision": revision,
                "required_files": files,
            }
        )
    cache.mkdir(parents=False, exist_ok=False)
    for model in models:
        destination = _snapshot_path(
            cache, str(model["model_id"]), str(model["revision"])
        )
        source = _snapshot_path(
            source_cache, str(model["model_id"]), str(model["revision"])
        )
        for expected in model["required_files"]:
            relative = str(expected["path"])
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with (
                (source / relative).open("rb") as source_stream,
                target.open("xb") as target_stream,
            ):
                shutil.copyfileobj(source_stream, target_stream, length=1024 * 1024)
    destination_rows: list[dict[str, Any]] = []
    for source_model, model in zip(source_rows, models, strict=True):
        model_id = str(model["model_id"])
        revision = str(model["revision"])
        destination = _snapshot_path(cache, model_id, revision)
        files = [
            _file_identity(destination / str(expected["path"]), expected)
            for expected in model["required_files"]
        ]
        if [row["sha256"] for row in files] != [
            row["sha256"] for row in source_model["required_files"]
        ]:
            raise ValueError("model cache copy byte identity mismatch")
        destination_rows.append(
            {
                "model_id": model_id,
                "revision": revision,
                "required_files": files,
            }
        )
    report = {
        "protocol_id": PROTOCOL_ID,
        "source_cache_path": source_cache.as_posix(),
        "cache_path": cache.as_posix(),
        "source_models": source_rows,
        "models": destination_rows,
        "model_count": len(models),
        "required_file_count": sum(len(model["required_files"]) for model in models),
        "required_file_size": sum(
            int(row["size"]) for model in models for row in model["required_files"]
        ),
        "target_absent_before_exclusive_create": True,
        "target_exclusive_create": True,
        "all_required_files_byte_identical": True,
        "source_cache_read_only": True,
        "model_import_count": 0,
        "model_load_count": 0,
        "model_forward_inference_count": 0,
    }
    _write_json_exclusive(output, report)
    return report


def _container_model_copy_verify(
    source_cache: str, cache: str, models: str, output: str
) -> dict[str, Any]:
    values = [
        serialize_container_path(value)
        for value in (source_cache, cache, models, output)
    ]
    if os.name != "posix":
        raise RuntimeError("container model-copy verifier requires POSIX")
    return model_copy_verify(*(Path(value) for value in values))


def _volume_identity(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw = _run_logged(["wslc", "volume", "inspect", CACHE_FREEZE_VOLUME], root, rows)
    value = json.loads(raw)
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError("cache-freeze volume inspection shape mismatch")
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
    runtime_absent: bool,
    predecessor_unchanged: bool,
) -> dict[str, Any]:
    commands = [row.get("command", []) for row in rows]
    copy_count = sum("model-copy-verify" in command for command in commands)
    volume_create_count = sum(
        command == ["wslc", "volume", "create", CACHE_FREEZE_VOLUME]
        for command in commands
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "cache_freeze_volume_create_count": volume_create_count,
        "model_copy_verifier_run_count": copy_count,
        "model_copy_verifier_retry_count": 0,
        "development_claim_count": 0,
        "holdout_claim_count": 0,
        "registered_query_execution_count": 0,
        "model_import_count": 0,
        "model_load_count": 0,
        "model_forward_inference_count": 0,
        "observed_result_count": 0,
        "retry_count": 0,
        "accepted_image_rebuild_count": 0,
        "runtime_report_rerun_count": 0,
        "attestation_rerun_count": 0,
        "future_runtime_volume_absent": runtime_absent,
        "cache_freeze_volume_reusable": False,
        "predecessor_artifacts_unchanged": predecessor_unchanged,
        "shared_database_opened": False,
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
    source_initialization: Mapping[str, Any] | None,
    model_verification: Mapping[str, Any] | None,
    runtime_absent: bool,
    predecessor_unchanged: bool,
) -> None:
    evidence = root / EVIDENCE
    summary_name = (
        "cache-freeze.pass.json" if status == "pass" else "cache-freeze.error.json"
    )
    _write_json_exclusive(evidence / summary_name, dict(summary))
    _write_json_exclusive(evidence / "cache-freeze-commands.json", {"commands": rows})
    if image is not None:
        _write_json_exclusive(evidence / "accepted-image-inspect.json", dict(image))
    if volume is not None:
        _write_json_exclusive(evidence / "volume-identity.json", dict(volume))
    if source_identity is not None:
        _write_json_exclusive(
            evidence / "source-cache-identity.json", dict(source_identity)
        )
    if source_initialization is not None:
        _write_json_exclusive(
            evidence / "source-initialization.json", dict(source_initialization)
        )
    if model_verification is not None:
        _write_json_exclusive(
            evidence / "model-verification.json", dict(model_verification)
        )
    _write_json_exclusive(
        evidence / "count-audit.json",
        _count_audit(
            status=status,
            rows=rows,
            runtime_absent=runtime_absent,
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


def run_cache_freeze(root: Path = ROOT) -> dict[str, Any]:
    validate_prebuild(root)
    evidence = root / EVIDENCE
    if evidence.exists():
        raise FileExistsError(
            "v10 cache freeze evidence already exists; retry forbidden"
        )
    evidence.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    image: dict[str, Any] | None = None
    volume: dict[str, Any] | None = None
    source_identity: dict[str, Any] | None = None
    source_initialization: dict[str, Any] | None = None
    model_verification: dict[str, Any] | None = None
    runtime_absent = False
    predecessor_unchanged = False
    manifest = _manifest(root)
    try:
        ci = _require_remote_ci_green(root, rows)
        predecessor_before = _verify_predecessor_hashes(root, manifest)
        version = _run_logged(["wslc", "--version"], root, rows).splitlines()[0]
        if version.removeprefix("wslc ").strip() != WSLC_VERSION:
            raise ValueError("WSLC version mismatch")
        if not _inspect_absent(CACHE_FREEZE_VOLUME, root, rows):
            raise FileExistsError("v10 cache-freeze volume already exists")
        if not _inspect_absent(FUTURE_RUNTIME_VOLUME, root, rows):
            raise FileExistsError("v10 future runtime volume already exists")
        image = _image_identity(root, rows)
        source_cache = discover_source_cache(root)
        source_identity = _source_cache_identity(root, source_cache)
        _run_logged(["wslc", "volume", "create", CACHE_FREEZE_VOLUME], root, rows)
        predecessor_archive = _archive_bytes(PREDECESSOR_MERGE_COMMIT, root, rows)
        _run_logged(
            _archive_import_command(_predecessor_export_script()),
            root,
            rows,
            input_bytes=predecessor_archive,
        )
        current_archive = _archive_bytes(ci["commit"], root, rows)
        source_raw = _run_logged(
            _archive_import_command(_source_initialization_script()),
            root,
            rows,
            input_bytes=current_archive,
        )
        if source_raw.strip() != "model_cache_absent_after_source_initialization=true":
            raise ValueError("source initialization target-absence report mismatch")
        source_initialization = {
            "source_commit": ci["commit"],
            "predecessor_commit": PREDECESSOR_MERGE_COMMIT,
            "model_cache_absent_after_source_initialization": True,
            "created_directories": [
                serialize_container_path(path)
                for path in (
                    CONTAINER_SOURCE,
                    CONTAINER_DATABASES,
                    CONTAINER_RUNS,
                    CONTAINER_ARCHIVE,
                    CONTAINER_TRANSPORT,
                )
            ],
            "model_cache_created_by_source_initialization": False,
        }
        model_verification = json.loads(
            _run_logged(model_copy_command(source_cache), root, rows)
        )
        expected_report = {
            "model_count": 2,
            "required_file_count": 12,
            "required_file_size": 3427616927,
            "target_absent_before_exclusive_create": True,
            "target_exclusive_create": True,
            "all_required_files_byte_identical": True,
            "source_cache_read_only": True,
            "model_import_count": 0,
            "model_load_count": 0,
            "model_forward_inference_count": 0,
        }
        for key, expected in expected_report.items():
            if model_verification.get(key) != expected:
                raise ValueError(f"model verification mismatch: {key}")
        volume = _volume_identity(root, rows)
        runtime_absent = _inspect_absent(FUTURE_RUNTIME_VOLUME, root, rows)
        if not runtime_absent:
            raise ValueError("future runtime volume was created during cache freeze")
        predecessor_after = _verify_predecessor_hashes(root, manifest)
        predecessor_unchanged = predecessor_after == predecessor_before
        if not predecessor_unchanged:
            raise ValueError("v9 predecessor artifacts changed during cache freeze")
        counts = _count_audit(
            status="pass",
            rows=rows,
            runtime_absent=True,
            predecessor_unchanged=True,
        )
        if (
            counts["cache_freeze_volume_create_count"] != 1
            or counts["model_copy_verifier_run_count"] != 1
        ):
            raise ValueError("v10 cache freeze exactly-once count mismatch")
        summary = {
            "protocol_id": PROTOCOL_ID,
            "status": "pass",
            "implementation_commit": ci["commit"],
            "remote_ci": ci,
            "wslc_version": WSLC_VERSION,
            "accepted_image": image,
            "volume_identity": volume,
            "source_initialization": source_initialization,
            "model_verification_sha256": canonical_sha256(model_verification),
            "predecessor_sha256_before": predecessor_before,
            "predecessor_sha256_after": predecessor_after,
            "future_runtime_volume_absent": True,
            "development_claim_count": 0,
            "holdout_claim_count": 0,
            "registered_query_execution_count": 0,
            "model_import_count": 0,
            "model_load_count": 0,
            "model_forward_inference_count": 0,
            "observed_result_count": 0,
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
            source_initialization=source_initialization,
            model_verification=model_verification,
            runtime_absent=True,
            predecessor_unchanged=True,
        )
        return summary
    except BaseException as error:
        if not runtime_absent:
            try:
                runtime_absent = _inspect_absent(FUTURE_RUNTIME_VOLUME, root, rows)
            except (OSError, RuntimeError, ValueError):
                runtime_absent = False
        if volume is None:
            try:
                if not _inspect_absent(CACHE_FREEZE_VOLUME, root, rows):
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
            "future_runtime_volume_absent": runtime_absent,
            "development_claim_count": 0,
            "holdout_claim_count": 0,
            "registered_query_execution_count": 0,
            "model_import_count": 0,
            "model_load_count": 0,
            "model_forward_inference_count": 0,
            "observed_result_count": 0,
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
            source_initialization=source_initialization,
            model_verification=model_verification,
            runtime_absent=runtime_absent,
            predecessor_unchanged=predecessor_unchanged,
        )
        raise


def _verify_evidence_manifest(evidence: Path) -> dict[str, Any]:
    value = read_json(evidence / "evidence-manifest.json")
    registry = value.get("files_sha256")
    if not isinstance(registry, dict):
        raise TypeError("v10 evidence hash registry is missing")
    actual_names = {
        path.name
        for path in evidence.iterdir()
        if path.is_file() and path.name != "evidence-manifest.json"
    }
    if set(registry) != actual_names:
        raise ValueError("v10 evidence file set mismatch")
    for name, expected in registry.items():
        if sha256_file(evidence / name) != expected:
            raise ValueError(f"v10 evidence hash mismatch: {name}")
    return value


def audit_evidence(root: Path = ROOT) -> dict[str, Any]:
    prebuild = validate_prebuild(root)
    evidence = root / EVIDENCE
    if not evidence.exists():
        return {
            **prebuild,
            "status": "prebuild_ready_evidence_absent",
            "future_runtime_volume_absent": None,
        }
    manifest = _verify_evidence_manifest(evidence)
    status = str(manifest.get("status"))
    if status not in {"pass", "error"}:
        raise ValueError("v10 evidence status mismatch")
    counts = read_json(evidence / "count-audit.json")
    for key in (
        "development_claim_count",
        "holdout_claim_count",
        "registered_query_execution_count",
        "model_import_count",
        "model_load_count",
        "model_forward_inference_count",
        "observed_result_count",
        "retry_count",
        "model_copy_verifier_retry_count",
    ):
        if counts.get(key) != 0:
            raise ValueError(f"v10 terminal count mismatch: {key}")
    if (
        counts.get("cache_freeze_volume_create_count") not in {0, 1}
        or counts.get("model_copy_verifier_run_count") not in {0, 1}
        or counts.get("cache_freeze_volume_reusable") is not False
        or counts.get("performance") != "not assessed"
        or counts.get("predecessor_artifacts_unchanged") is not True
    ):
        raise ValueError("v10 terminal boundary mismatch")
    if status == "pass" and (
        counts["cache_freeze_volume_create_count"] != 1
        or counts["model_copy_verifier_run_count"] != 1
        or counts.get("future_runtime_volume_absent") is not True
    ):
        raise ValueError("v10 successful cache freeze count mismatch")
    return {
        **prebuild,
        "status": status,
        "cache_freeze_volume_create_count": counts["cache_freeze_volume_create_count"],
        "model_copy_verifier_run_count": counts["model_copy_verifier_run_count"],
        "future_runtime_volume_absent": counts["future_runtime_volume_absent"],
        "predecessor_artifacts_unchanged": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prebuild")
    commands.add_parser("freeze")
    commands.add_parser("audit")
    copy = commands.add_parser("model-copy-verify")
    copy.add_argument("--source-cache", required=True)
    copy.add_argument("--cache", required=True)
    copy.add_argument("--models", required=True)
    copy.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "prebuild":
        result = validate_prebuild()
    elif args.command == "freeze":
        result = run_cache_freeze()
    elif args.command == "audit":
        result = audit_evidence()
    else:
        result = _container_model_copy_verify(
            args.source_cache, args.cache, args.models, args.output
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
