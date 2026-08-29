from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

PROTOCOL_ID = "github-ngr-cross-encoder-precision-v9"
PREDECESSOR_MERGE_COMMIT = "c57e15a0c7cfbec25d6560cf326da7e042db392c"
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path("tests/fixtures/github_cross_encoder_precision_v9.manifest.json")
RESULT_FREE_AUDIT = Path(
    "tests/fixtures/github_cross_encoder_precision_v9.result-free-audit.json"
)
EVIDENCE = Path("tests/evidence/github_cross_encoder_precision_v9")

IMAGE = "ngr-cross-encoder-precision-v8:freeze"
IMAGE_ID = "sha256:136ad9466799109bf32b4b96b611c9db9a099bcc47cf78243f26c7227bc16742"
WSLC_VERSION = "2.9.4.0"
PATH_FREEZE_VOLUME = "github-cross-encoder-precision-v9-path-freeze"
FUTURE_RUNTIME_VOLUME = "github-cross-encoder-precision-v9-runtime"
CONTAINER_ROOT = PurePosixPath("/opt/ngr-v9/path-freeze")
SENTINEL_DIRECTORY = CONTAINER_ROOT / "sentinel"
SENTINEL_FILE = SENTINEL_DIRECTORY / "path-transport-v9.txt"
SENTINEL_VALUE = "ngr-v9-path-transport-freeze"

V8_FAILURE_PATH = Path(
    "tests/evidence/github_cross_encoder_precision_v8/preflight.error.json"
)
V8_FAILURE_SHA256 = "df97b812b052cc421408cdab3b89cbe25529e3167bdc4903c68c892f3c451280"
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value))


def serialize_container_path(value: PurePosixPath | str) -> str:
    """Return one canonical absolute container path or fail closed.

    Host ``Path`` objects are deliberately rejected even when their current
    string happens to look POSIX-like. The type boundary is part of the v9
    protocol, not a platform-dependent normalization convenience.
    """

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
    path = PurePosixPath(raw)
    rendered = path.as_posix()
    if rendered != raw or not path.is_absolute():
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


def _manifest(root: Path) -> dict[str, Any]:
    value = read_json(root / MANIFEST)
    if not isinstance(value, dict):
        raise TypeError("v9 manifest must be an object")
    return value


def _audit_contract(root: Path) -> dict[str, Any]:
    value = read_json(root / RESULT_FREE_AUDIT)
    if not isinstance(value, dict):
        raise TypeError("v9 result-free audit must be an object")
    return value


def _verify_predecessor_hashes(root: Path, manifest: Mapping[str, Any]) -> int:
    registry = manifest.get("predecessor_immutable_sha256")
    if not isinstance(registry, dict) or len(registry) != 29:
        raise ValueError("v8 predecessor hash registry must contain 29 files")
    for relative, expected in registry.items():
        path = root / str(relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"v8 predecessor artifact changed: {relative}")
    if registry.get(V8_FAILURE_PATH.as_posix()) != V8_FAILURE_SHA256:
        raise ValueError("v8 raw failure hash is not frozen")
    return len(registry)


def validate_prebuild(root: Path = ROOT) -> dict[str, Any]:
    manifest = _manifest(root)
    audit = _audit_contract(root)
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("v9 manifest protocol mismatch")
    if manifest.get("predecessor_merge_commit") != PREDECESSOR_MERGE_COMMIT:
        raise ValueError("v9 predecessor commit mismatch")
    if manifest.get("accepted_image") != {"id": IMAGE_ID, "tag": IMAGE}:
        raise ValueError("v9 accepted image registry mismatch")
    paths = manifest.get("path_transport")
    expected_paths = {
        "container_root": serialize_container_path(CONTAINER_ROOT),
        "sentinel_directory": serialize_container_path(SENTINEL_DIRECTORY),
        "sentinel_file": serialize_container_path(SENTINEL_FILE),
        "path_freeze_volume": PATH_FREEZE_VOLUME,
        "future_runtime_volume": FUTURE_RUNTIME_VOLUME,
    }
    if paths != expected_paths:
        raise ValueError("v9 path transport registry mismatch")
    if manifest.get("path_freeze_mount") != named_volume_spec(
        PATH_FREEZE_VOLUME, CONTAINER_ROOT
    ):
        raise ValueError("v9 named volume mount mismatch")
    predecessor_count = _verify_predecessor_hashes(root, manifest)
    if manifest.get("result_free_audit_sha256") != sha256_file(
        root / RESULT_FREE_AUDIT
    ):
        raise ValueError("v9 result-free audit hash mismatch")
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
            raise ValueError(f"v9 result-free count mismatch: {key}")
    if (
        audit.get("protocol_id") != PROTOCOL_ID
        or audit.get("phase") != "path-freeze"
        or audit.get("performance") != "not assessed"
        or audit.get("shared_database_opened") is not False
        or audit.get("predecessor_semantic_content_opened") is not False
        or audit.get("path_freeze_volume_reusable") is not False
    ):
        raise ValueError("v9 result-free boundary mismatch")
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "prebuild_contract_valid",
        "predecessor_artifact_count": predecessor_count,
        "v8_failure_sha256": V8_FAILURE_SHA256,
        "path_freeze_mount": named_volume_spec(PATH_FREEZE_VOLUME, CONTAINER_ROOT),
        "future_runtime_volume": FUTURE_RUNTIME_VOLUME,
        "registered_query_execution_count": 0,
        "model_forward_inference_count": 0,
        "observed_result_count": 0,
        "performance": "not assessed",
    }


def _command_row(command: Sequence[str], completed: subprocess.CompletedProcess[bytes]) -> dict[str, Any]:
    return {
        "command": list(command),
        "command_sha256": canonical_sha256(list(command)),
        "returncode": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
    }


def _run_logged(command: Sequence[str], root: Path, rows: list[dict[str, Any]]) -> str:
    completed = subprocess.run(
        list(command),
        cwd=root,
        check=False,
        capture_output=True,
    )
    row = _command_row(command, completed)
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
        raise ValueError("working tree must be clean before the one-shot smoke")
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
        raise ValueError("one-shot smoke requires the pushed remote HEAD")
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
        row.get("conclusion") not in {"success", "skipped", "neutral"}
        for row in checks
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


def _smoke_script() -> str:
    root = serialize_container_path(CONTAINER_ROOT)
    directory = serialize_container_path(SENTINEL_DIRECTORY)
    sentinel = serialize_container_path(SENTINEL_FILE)
    return (
        "set -eu; "
        f"root='{root}'; directory='{directory}'; sentinel='{sentinel}'; "
        "test -d \"$root\"; test ! -e \"$directory\"; mkdir \"$directory\"; "
        f"printf '%s\\n' '{SENTINEL_VALUE}' > \"$sentinel\"; "
        f"test \"$(cat \"$sentinel\")\" = '{SENTINEL_VALUE}'; "
        "mount_identity=$(awk -v target=\"$root\" '$5 == target "
        "{print $1 \"|\" $3 \"|\" $4 \"|\" $5 \"|\" $9; found=1} "
        "END {if (!found) exit 42}' /proc/self/mountinfo); "
        "sentinel_sha256=$(sha256sum \"$sentinel\" | awk '{print $1}'); "
        "printf 'container_path=%s\\nmount_identity=%s\\nsentinel_path=%s\\n"
        "sentinel_sha256=%s\\n' \"$root\" \"$mount_identity\" \"$sentinel\" "
        "\"$sentinel_sha256\""
    )


def smoke_command() -> list[str]:
    return [
        "wslc",
        "run",
        "--rm",
        "--network",
        "none",
        "--volume",
        named_volume_spec(PATH_FREEZE_VOLUME, CONTAINER_ROOT),
        "--entrypoint",
        "/bin/sh",
        IMAGE,
        "-c",
        _smoke_script(),
    ]


def _parse_smoke_stdout(raw: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in raw.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {
            "container_path",
            "mount_identity",
            "sentinel_path",
            "sentinel_sha256",
        }:
            rows[key] = value
    expected_sha = hashlib.sha256((SENTINEL_VALUE + "\n").encode()).hexdigest()
    if rows != {
        "container_path": serialize_container_path(CONTAINER_ROOT),
        "mount_identity": rows.get("mount_identity"),
        "sentinel_path": serialize_container_path(SENTINEL_FILE),
        "sentinel_sha256": expected_sha,
    } or not rows.get("mount_identity"):
        raise ValueError("container path smoke stdout mismatch")
    return rows


def _volume_identity(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw = _run_logged(
        ["wslc", "volume", "inspect", PATH_FREEZE_VOLUME], root, rows
    )
    value = json.loads(raw)
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError("path-freeze volume inspection shape mismatch")
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
    *, status: str, runtime_absent: bool, smoke_run_count: int
) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "path_smoke_run_count": smoke_run_count,
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
        "path_freeze_volume_reusable": False,
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
    runtime_absent: bool,
) -> None:
    evidence = root / EVIDENCE
    summary_name = "path-smoke.pass.json" if status == "pass" else "path-smoke.error.json"
    _write_json_exclusive(evidence / summary_name, dict(summary))
    _write_json_exclusive(evidence / "path-smoke-commands.json", {"commands": rows})
    if image is not None:
        _write_json_exclusive(evidence / "accepted-image-inspect.json", dict(image))
    if volume is not None:
        _write_json_exclusive(evidence / "volume-identity.json", dict(volume))
    _write_json_exclusive(
        evidence / "count-audit.json",
        _count_audit(
            status=status,
            runtime_absent=runtime_absent,
            smoke_run_count=sum(
                row["command"][:2] == ["wslc", "run"] for row in rows
            ),
        ),
    )
    registry = {
        path.name: sha256_file(path)
        for path in sorted(evidence.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "evidence-manifest.json"
    }
    _write_json_exclusive(
        evidence / "evidence-manifest.json",
        {
            "protocol_id": PROTOCOL_ID,
            "status": status,
            "files_sha256": registry,
        },
    )


def run_result_free_smoke(root: Path = ROOT) -> dict[str, Any]:
    validate_prebuild(root)
    evidence = root / EVIDENCE
    if evidence.exists():
        raise FileExistsError("v9 path smoke evidence already exists; retry forbidden")
    evidence.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    image: dict[str, Any] | None = None
    volume: dict[str, Any] | None = None
    runtime_absent = False
    ci: dict[str, Any] | None = None
    try:
        ci = _require_remote_ci_green(root, rows)
        version = _run_logged(["wslc", "--version"], root, rows).splitlines()[0]
        if version.removeprefix("wslc ").strip() != WSLC_VERSION:
            raise ValueError("WSLC version mismatch")
        if not _inspect_absent(PATH_FREEZE_VOLUME, root, rows):
            raise FileExistsError("v9 path-freeze volume already exists")
        if not _inspect_absent(FUTURE_RUNTIME_VOLUME, root, rows):
            raise FileExistsError("v9 future runtime volume already exists")
        image = _image_identity(root, rows)
        _run_logged(["wslc", "volume", "create", PATH_FREEZE_VOLUME], root, rows)
        raw = _run_logged(smoke_command(), root, rows)
        recognized = _parse_smoke_stdout(raw)
        volume = _volume_identity(root, rows)
        runtime_absent = _inspect_absent(FUTURE_RUNTIME_VOLUME, root, rows)
        if not runtime_absent:
            raise ValueError("future runtime volume was created during path smoke")
        if sum(row["command"][:2] == ["wslc", "run"] for row in rows) != 1:
            raise ValueError("v9 path smoke must run exactly once")
        summary = {
            "protocol_id": PROTOCOL_ID,
            "status": "pass",
            "implementation_commit": ci["commit"],
            "remote_ci": ci,
            "wslc_version": WSLC_VERSION,
            "accepted_image": image,
            "container_recognition": recognized,
            "volume_identity": volume,
            "v8_failure_sha256": V8_FAILURE_SHA256,
            "future_runtime_volume_absent": True,
            "performance": "not assessed",
        }
        _write_evidence(
            root,
            status="pass",
            summary=summary,
            rows=rows,
            image=image,
            volume=volume,
            runtime_absent=True,
        )
        return summary
    except Exception as error:
        if not runtime_absent:
            try:
                runtime_absent = _inspect_absent(FUTURE_RUNTIME_VOLUME, root, rows)
            except (OSError, RuntimeError, ValueError):
                runtime_absent = False
        if volume is None:
            try:
                if not _inspect_absent(PATH_FREEZE_VOLUME, root, rows):
                    volume = _volume_identity(root, rows)
            except (OSError, RuntimeError, ValueError):
                volume = None
        summary = {
            "protocol_id": PROTOCOL_ID,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "implementation_commit": None if ci is None else ci["commit"],
            "v8_failure_sha256": V8_FAILURE_SHA256,
            "future_runtime_volume_absent": runtime_absent,
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
            runtime_absent=runtime_absent,
        )
        raise


def audit_evidence(root: Path = ROOT) -> dict[str, Any]:
    prebuild = validate_prebuild(root)
    evidence = root / EVIDENCE
    if not evidence.exists():
        return {**prebuild, "status": "prebuild_ready_evidence_absent"}
    manifest_path = evidence / "evidence-manifest.json"
    if not manifest_path.is_file():
        raise ValueError("v9 evidence manifest is missing")
    evidence_manifest = read_json(manifest_path)
    if evidence_manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("v9 evidence protocol mismatch")
    registry = evidence_manifest.get("files_sha256")
    if not isinstance(registry, dict):
        raise TypeError("v9 evidence hash registry is missing")
    actual_names = {
        path.name
        for path in evidence.iterdir()
        if path.is_file() and path.name != "evidence-manifest.json"
    }
    if actual_names != set(registry):
        raise ValueError("v9 evidence file set mismatch")
    for name, expected in registry.items():
        if sha256_file(evidence / name) != expected:
            raise ValueError(f"v9 evidence changed: {name}")
    counts = read_json(evidence / "count-audit.json")
    if counts != _count_audit(
        status=evidence_manifest["status"],
        runtime_absent=counts.get("future_runtime_volume_absent") is True,
        smoke_run_count=counts.get("path_smoke_run_count", -1),
    ):
        raise ValueError("v9 count audit mismatch")
    if evidence_manifest["status"] == "pass" and counts["path_smoke_run_count"] != 1:
        raise ValueError("v9 pass evidence must contain exactly one path smoke run")
    if evidence_manifest["status"] == "error" and counts["path_smoke_run_count"] not in {0, 1}:
        raise ValueError("v9 error evidence contains an invalid smoke run count")
    commands = read_json(evidence / "path-smoke-commands.json").get("commands", [])
    if sum(row.get("command", [])[:2] == ["wslc", "run"] for row in commands) != 1:
        raise ValueError("v9 evidence must contain exactly one path smoke run")
    if sha256_file(root / V8_FAILURE_PATH) != V8_FAILURE_SHA256:
        raise ValueError("v8 raw failure changed after v9 smoke")
    return {
        "protocol_id": PROTOCOL_ID,
        "status": evidence_manifest["status"],
        "evidence_file_count": len(registry),
        "v8_failure_sha256": V8_FAILURE_SHA256,
        "future_runtime_volume_absent": counts["future_runtime_volume_absent"],
        "registered_query_execution_count": 0,
        "model_forward_inference_count": 0,
        "observed_result_count": 0,
        "performance": "not assessed",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prebuild", "smoke", "audit"))
    arguments = parser.parse_args(argv)
    if arguments.action == "prebuild":
        value = validate_prebuild()
    elif arguments.action == "smoke":
        value = run_result_free_smoke()
    else:
        value = audit_evidence()
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
