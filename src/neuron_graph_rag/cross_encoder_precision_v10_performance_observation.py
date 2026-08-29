from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from . import cross_encoder_precision_v8_observation as predecessor
from . import cross_encoder_precision_v9_performance_observation as lifecycle
from . import cross_encoder_precision_v10_observation as cache_freeze

PROTOCOL_ID = cache_freeze.PROTOCOL_ID
FREEZE_COMMIT = "e75d1e065441b794ce83b68f62d55747741052e5"
V8_PROTOCOL_COMMIT = predecessor.PROTOCOL_COMMIT
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(
    "tests/fixtures/github_cross_encoder_precision_v10_observation.manifest.json"
)
EVIDENCE = Path("tests/evidence/github_cross_encoder_precision_v10_observation")

IMAGE = cache_freeze.IMAGE
IMAGE_ID = cache_freeze.IMAGE_ID
WSLC_VERSION = cache_freeze.WSLC_VERSION
VOLUME = cache_freeze.FUTURE_RUNTIME_VOLUME
CACHE_FREEZE_VOLUME = cache_freeze.CACHE_FREEZE_VOLUME
CONTAINER_ROOT = PurePosixPath("/opt/ngr-v10/runtime")
CONTAINER_SOURCE = CONTAINER_ROOT / "source"
CONTAINER_CACHE = CONTAINER_ROOT / "model-cache"
CONTAINER_DATABASES = CONTAINER_ROOT / "databases"
CONTAINER_RUNS = CONTAINER_ROOT / "runs"
CONTAINER_ARCHIVE = CONTAINER_ROOT / "archive"
CONTAINER_TRANSPORT = CONTAINER_ROOT / "transport"
CONTAINER_MODEL_REGISTRY = (
    CONTAINER_SOURCE / "tests/fixtures/github_cross_encoder_precision_v8.models.json"
)
BATCH_SIZE = predecessor.BATCH_SIZE
SHARED_DATABASE = predecessor.SHARED_DATABASE
MODEL_CACHE = predecessor.MODEL_CACHE
WORKERS = predecessor.WORKERS

CACHE_FREEZE_PASS_SHA256 = (
    "cca6ee778acb18b7dd921b754657f92bda39af2ccdf6827a86f052634ab41910"
)
CACHE_FREEZE_COUNT_AUDIT_SHA256 = (
    "f55da694e51f8d4e7296cdb03e157452c78410eb40030d55165a30a4c908eaa7"
)
CACHE_FREEZE_EVIDENCE_MANIFEST_SHA256 = (
    "61a46b04b9966ee54663449dff36c48d32a6c079f53a8a231ab12247392fb556"
)
CACHE_FREEZE_MODEL_VERIFICATION_SHA256 = (
    "380ae8c602d1c0049adc495eaea97f31aca25984cbf6ed286594df35bf07e0c9"
)

canonical_sha256 = predecessor.canonical_sha256
sha256_file = predecessor.sha256_file
_write_json_exclusive = predecessor._write_json_exclusive
_git_output = predecessor._git_output
_hash_shared_database = predecessor._hash_shared_database
_command_row = predecessor._command_row
_run_logged = predecessor._run_logged
read_json = lifecycle.read_json


def serialize_container_path(value: PurePosixPath | str) -> str:
    return cache_freeze.serialize_container_path(value)


def named_volume_spec(
    volume: str,
    destination: PurePosixPath | str,
    *,
    mode: str | None = None,
) -> str:
    return cache_freeze.named_volume_spec(volume, destination, mode=mode)


def host_bind_spec(
    source: Path,
    destination: PurePosixPath | str,
    *,
    mode: str,
) -> str:
    return cache_freeze.host_bind_spec(source, destination, mode=mode)


def _manifest(root: Path) -> dict[str, Any]:
    value = read_json(root / MANIFEST)
    if not isinstance(value, dict):
        raise TypeError("v10 observation manifest must be an object")
    return value


def _expected_container_paths() -> dict[str, str]:
    return {
        "root": serialize_container_path(CONTAINER_ROOT),
        "source": serialize_container_path(CONTAINER_SOURCE),
        "model_cache": serialize_container_path(CONTAINER_CACHE),
        "databases": serialize_container_path(CONTAINER_DATABASES),
        "runs": serialize_container_path(CONTAINER_RUNS),
        "archive": serialize_container_path(CONTAINER_ARCHIVE),
        "transport": serialize_container_path(CONTAINER_TRANSPORT),
    }


def _verify_cache_freeze_inputs(root: Path) -> dict[str, Any]:
    manifest = _manifest(root)
    expected_header = {
        "protocol_id": PROTOCOL_ID,
        "phase": "performance-observation",
        "freeze_commit": FREEZE_COMMIT,
        "v8_protocol_commit": V8_PROTOCOL_COMMIT,
        "runtime_volume": VOLUME,
        "cache_freeze_volume": CACHE_FREEZE_VOLUME,
        "cache_freeze_volume_reusable": False,
        "accepted_image": {"tag": IMAGE, "id": IMAGE_ID},
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            raise ValueError(f"v10 observation manifest mismatch: {key}")
    if manifest.get("container_paths") != _expected_container_paths():
        raise ValueError("v10 observation container path registry mismatch")
    registry = manifest.get("cache_freeze_immutable_sha256")
    if not isinstance(registry, dict) or len(registry) != 15:
        raise ValueError("v10 cache-freeze registry must contain exactly 15 files")
    for relative, expected in registry.items():
        path = root / str(relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"v10 cache-freeze artifact changed: {relative}")
    anchors = {
        "tests/evidence/github_cross_encoder_precision_v10/cache-freeze.pass.json": (
            CACHE_FREEZE_PASS_SHA256
        ),
        "tests/evidence/github_cross_encoder_precision_v10/count-audit.json": (
            CACHE_FREEZE_COUNT_AUDIT_SHA256
        ),
        "tests/evidence/github_cross_encoder_precision_v10/evidence-manifest.json": (
            CACHE_FREEZE_EVIDENCE_MANIFEST_SHA256
        ),
        "tests/evidence/github_cross_encoder_precision_v10/model-verification.json": (
            CACHE_FREEZE_MODEL_VERIFICATION_SHA256
        ),
    }
    for relative, expected in anchors.items():
        if registry.get(relative) != expected:
            raise ValueError(f"v10 cache-freeze anchor is not frozen: {relative}")
    prebuild = cache_freeze.validate_prebuild(root)
    terminal = cache_freeze.audit_evidence(root)
    if terminal.get("status") != "pass":
        raise ValueError("successful v10 cache-freeze evidence is required")
    if terminal.get("future_runtime_volume_absent") is not True:
        raise ValueError("v10 cache-freeze did not preserve runtime volume absence")
    return {
        "cache_freeze_artifact_count": len(registry),
        "predecessor_artifact_count": prebuild["predecessor_artifact_count"],
        "cache_freeze_pass_sha256": CACHE_FREEZE_PASS_SHA256,
        "cache_freeze_count_audit_sha256": CACHE_FREEZE_COUNT_AUDIT_SHA256,
        "cache_freeze_evidence_manifest_sha256": (
            CACHE_FREEZE_EVIDENCE_MANIFEST_SHA256
        ),
        "cache_freeze_model_verification_sha256": (
            CACHE_FREEZE_MODEL_VERIFICATION_SHA256
        ),
    }


def _stored_freeze_contract(root: Path) -> dict[str, Any]:
    cache_contract = _verify_cache_freeze_inputs(root)
    image_contract = predecessor._stored_freeze_contract(root)
    return {
        **cache_contract,
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
        "accepted_image_rebuild_count": 0,
        "runtime_content_report_rerun_count": 0,
        "attestation_report_rerun_count": 0,
        "cache_freeze_volume_mounted": False,
        "cache_freeze_volume_read": False,
        "cache_freeze_volume_copied": False,
        "cache_freeze_volume_reused": False,
    }


def _write_lifecycle_json_exclusive(path: Path, value: object) -> None:
    if path.name == "platform-report.json" and isinstance(value, dict):
        value = dict(value)
        value["cache_freeze_volume"] = value.pop("path_freeze_volume")
        value["cache_freeze_volume_mounted"] = value.pop(
            "path_freeze_volume_mounted"
        )
        value["cache_freeze_volume_read"] = value.pop("path_freeze_volume_read")
        value["cache_freeze_volume_reused"] = False
    _write_json_exclusive(path, value)


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
            "neuron_graph_rag.cross_encoder_precision_v10_performance_observation",
            *arguments,
        ]
    )
    return command


def _source_initialization_script() -> str:
    root = serialize_container_path(CONTAINER_ROOT)
    source = serialize_container_path(CONTAINER_SOURCE)
    cache = serialize_container_path(CONTAINER_CACHE)
    paths = " ".join(
        serialize_container_path(path)
        for path in (
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
        f"test -d '{root}'; test ! -e '{source}'; test ! -e '{cache}'; "
        f"mkdir -p {paths}; mkdir '{source}'; "
        f"cp -a /input/source/. '{source}/'; "
        f"rm -rf '{source}/.git'; test -f '{fixture}'; test ! -e '{cache}'"
    )


def _container_model_copy(source: str, cache: str, output: str) -> dict[str, Any]:
    return cache_freeze._container_model_copy_verify(
        source,
        cache,
        serialize_container_path(CONTAINER_MODEL_REGISTRY),
        output,
    )


def _verification_commands(root: Path) -> tuple[list[str], ...]:
    python = root / ".venv" / "Scripts" / "python.exe"
    return (
        [
            "uvx",
            "--offline",
            "ruff",
            "check",
            "src/neuron_graph_rag/cross_encoder_precision_v10_performance_observation.py",
            "tests/test_cross_encoder_precision_v10_performance_observation.py",
        ],
        [
            str(python),
            "-m",
            "unittest",
            "tests.test_cross_encoder_precision_v8",
            "tests.test_cross_encoder_precision_v8_observation",
            "tests.test_cross_encoder_precision_v10",
            "tests.test_cross_encoder_precision_v10_performance_observation",
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
            "neuron_graph_rag.cross_encoder_precision_v10_observation",
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


def _sync_preflight_evidence(
    root: Path, rows: list[dict[str, Any]]
) -> None:
    source = (root / EVIDENCE).resolve()
    destination = serialize_container_path(CONTAINER_SOURCE / EVIDENCE.as_posix())
    lifecycle._run_logged(
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
            f"mkdir -p '{destination}' && cp -a /input/evidence/. '{destination}/'",
        ],
        root,
        rows,
    )


def _export_volume_evidence(
    root: Path, rows: list[dict[str, Any]]
) -> None:
    evidence = (root / EVIDENCE).resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    source = serialize_container_path(CONTAINER_SOURCE / EVIDENCE.as_posix())
    lifecycle._run_logged(
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
    lifecycle._run_logged(_container_command("claim", "--stage", stage), root, rows)
    claim_counts[stage] += 1
    stage_root = CONTAINER_RUNS / stage
    database_root = CONTAINER_DATABASES / stage
    for kind, replay in WORKERS:
        identity = f"ngr-v10-{stage}-{kind}-{replay}"
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
        lifecycle._run_logged(command, root, rows)
    result = json.loads(
        lifecycle._run_logged(
            _container_command("finalize", "--stage", stage), root, rows
        )
    )
    _export_volume_evidence(root, rows)
    return result


def _lifecycle_replacements() -> Mapping[str, Any]:
    replacements: Mapping[str, Any] = {
        "PROTOCOL_ID": PROTOCOL_ID,
        "FREEZE_COMMIT": FREEZE_COMMIT,
        "V8_PROTOCOL_COMMIT": V8_PROTOCOL_COMMIT,
        "ROOT": ROOT,
        "MANIFEST": MANIFEST,
        "EVIDENCE": EVIDENCE,
        "IMAGE": IMAGE,
        "IMAGE_ID": IMAGE_ID,
        "WSLC_VERSION": WSLC_VERSION,
        "VOLUME": VOLUME,
        "PATH_FREEZE_VOLUME": CACHE_FREEZE_VOLUME,
        "CONTAINER_ROOT": CONTAINER_ROOT,
        "CONTAINER_SOURCE": CONTAINER_SOURCE,
        "CONTAINER_CACHE": CONTAINER_CACHE,
        "CONTAINER_DATABASES": CONTAINER_DATABASES,
        "CONTAINER_RUNS": CONTAINER_RUNS,
        "CONTAINER_ARCHIVE": CONTAINER_ARCHIVE,
        "CONTAINER_TRANSPORT": CONTAINER_TRANSPORT,
        "_manifest": _manifest,
        "_write_json_exclusive": _write_lifecycle_json_exclusive,
        "_verify_path_freeze_inputs": _verify_cache_freeze_inputs,
        "_stored_freeze_contract": _stored_freeze_contract,
        "_container_command": _container_command,
        "_source_initialization_script": _source_initialization_script,
        "_container_model_copy": _container_model_copy,
        "_verification_commands": _verification_commands,
        "_sync_preflight_evidence": _sync_preflight_evidence,
        "_export_volume_evidence": _export_volume_evidence,
        "_run_stage_host": _run_stage_host,
    }
    return replacements


@contextmanager
def _lifecycle_scope() -> Any:
    replacements = _lifecycle_replacements()
    original = {name: getattr(lifecycle, name) for name in replacements}
    for name, value in replacements.items():
        setattr(lifecycle, name, value)
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(lifecycle, name, value)


def preflight(
    root: Path = ROOT, model_cache: Path | None = None
) -> dict[str, Any]:
    source_cache = (
        cache_freeze.discover_source_cache(root)
        if model_cache is None
        else model_cache
    )
    with _lifecycle_scope():
        return lifecycle.preflight(root, source_cache)


def verify_preflight(root: Path = ROOT) -> dict[str, Any]:
    with _lifecycle_scope():
        return lifecycle.verify_preflight(root)


def run_once(root: Path = ROOT) -> dict[str, Any]:
    with _lifecycle_scope():
        return lifecycle.run_once(root)


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
        raise FileNotFoundError("v10 raw preflight error evidence is missing")
    if (evidence / "preflight.json").exists():
        raise ValueError("successful preflight cannot be finalized as error")
    if terminal_path.exists() or (
        evidence / "observation-evidence-manifest.json"
    ).exists():
        raise FileExistsError("v10 preflight error is already terminal")
    raw = read_json(raw_path)
    for key in (
        "development_claim_count",
        "holdout_claim_count",
        "registered_query_execution_count",
        "observed_stage_inference_count",
        "result_count",
        "retry_count",
    ):
        if raw.get(key) != 0:
            raise ValueError("v10 raw preflight error result count mismatch")
    if raw.get("preflight_forward_inference_count") not in {0, 2}:
        raise ValueError("v10 raw preflight probe count mismatch")
    before = raw.get("shared_database_sha256_before_preflight")
    after: str | None = None
    post_hash_error: str | None = None
    try:
        after = _hash_shared_database()
    except (OSError, RuntimeError, ValueError) as error:
        post_hash_error = f"{type(error).__name__}: {error}"
    terminal = {
        "protocol_id": PROTOCOL_ID,
        "status": "error",
        "phase": "preflight",
        "implementation_commit": raw.get("implementation_commit"),
        "raw_failure_sha256": sha256_file(raw_path),
        "failure_cause": raw.get("error"),
        "runtime_volume_create_count": raw.get("runtime_volume_create_count"),
        "development_claim_count": 0,
        "holdout_claim_count": 0,
        "registered_query_execution_count": 0,
        "preflight_forward_inference_count": raw.get(
            "preflight_forward_inference_count"
        ),
        "observed_stage_inference_count": 0,
        "result_count": 0,
        "retry_count": 0,
        "same_protocol_retry_allowed": False,
        "accepted_image_rebuild_count": 0,
        "runtime_report_rerun_count": 0,
        "attestation_rerun_count": 0,
        "cache_freeze_volume_mounted": False,
        "cache_freeze_volume_read": False,
        "cache_freeze_volume_reused": False,
        "shared_database_sha256_before_preflight": before,
        "shared_database_sha256_after_error": after,
        "shared_database_post_error_hash_recorded": after is not None,
        "shared_database_post_error_hash_error": post_hash_error,
        "shared_database_unchanged": before is not None and before == after,
        "performance": "not assessed",
    }
    _write_json_exclusive(terminal_path, terminal)
    _write_terminal_manifest(evidence, "preflight-error")
    return terminal


def audit_evidence(root: Path = ROOT) -> dict[str, Any]:
    evidence = root / EVIDENCE
    if (evidence / "preflight.error.json").is_file():
        terminal_path = evidence / "preflight-terminal.json"
        manifest_path = evidence / "observation-evidence-manifest.json"
        if not terminal_path.is_file() or not manifest_path.is_file():
            return {
                "protocol_id": PROTOCOL_ID,
                "status": "preflight-error-unfinalized",
            }
        terminal = read_json(terminal_path)
        with _lifecycle_scope():
            manifest = lifecycle._verify_hash_manifest(
                evidence, "observation-evidence-manifest.json", exact=True
            )
        if (
            terminal.get("raw_failure_sha256")
            != sha256_file(evidence / "preflight.error.json")
            or terminal.get("retry_count") != 0
            or terminal.get("same_protocol_retry_allowed") is not False
            or terminal.get("performance") != "not assessed"
            or manifest.get("status") != "preflight-error"
        ):
            raise ValueError("v10 terminal preflight error evidence mismatch")
        return {
            "protocol_id": PROTOCOL_ID,
            "status": "preflight-error",
            "raw_failure_sha256": terminal["raw_failure_sha256"],
            "shared_database_unchanged": terminal.get(
                "shared_database_unchanged"
            ),
            "retry_count": 0,
            "performance": "not assessed",
        }
    with _lifecycle_scope():
        return lifecycle.audit_evidence(root)


def _read_json_command(path: str) -> dict[str, Any]:
    return lifecycle._read_json_command(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Observe frozen WSLC rank-only benchmark v10 once"
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
        with _lifecycle_scope():
            result = lifecycle._container_model_probe(arguments.cache)
    elif arguments.command == "dependency-report":
        with _lifecycle_scope():
            result = lifecycle._dependency_report()
    elif arguments.command == "read-json":
        result = _read_json_command(arguments.path)
    elif arguments.command == "claim":
        with _lifecycle_scope():
            result = lifecycle._container_claim(arguments.stage)
    elif arguments.command == "worker":
        with _lifecycle_scope():
            result = lifecycle._container_worker(
                arguments.stage,
                arguments.kind,
                arguments.replay,
                arguments.database,
                arguments.output,
            )
    elif arguments.command == "finalize":
        with _lifecycle_scope():
            result = lifecycle._container_finalize(arguments.stage)
    else:
        with _lifecycle_scope():
            result = lifecycle._container_fail_stage(
                arguments.stage, arguments.message
            )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
