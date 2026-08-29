from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from . import cross_encoder_precision_v10_observation as model_freeze
from . import cross_encoder_precision_v10_performance_observation as lifecycle
from . import cross_encoder_precision_v11_observation as root_freeze

PROTOCOL_ID = "github-ngr-cross-encoder-precision-v12"
FREEZE_COMMIT = "39f2cebc6b3b43ac1060a2ce519e8906fa598f57"
V8_PROTOCOL_COMMIT = lifecycle.V8_PROTOCOL_COMMIT
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(
    "tests/fixtures/github_cross_encoder_precision_v12_observation.manifest.json"
)
EVIDENCE = Path("tests/evidence/github_cross_encoder_precision_v12_observation")

IMAGE = lifecycle.IMAGE
IMAGE_ID = lifecycle.IMAGE_ID
WSLC_VERSION = lifecycle.WSLC_VERSION
VOLUME = "github-cross-encoder-precision-v12-runtime"
V11_ROOT_FREEZE_VOLUME = root_freeze.ROOT_FREEZE_VOLUME
V10_RUNTIME_VOLUME = "github-cross-encoder-precision-v10-runtime"
V10_CACHE_FREEZE_VOLUME = "github-cross-encoder-precision-v10-cache-freeze"
CONTAINER_ROOT = PurePosixPath("/opt/ngr-v12/runtime")
CONTAINER_SOURCE = CONTAINER_ROOT / "source"
CONTAINER_CACHE = CONTAINER_ROOT / "model-cache"
CONTAINER_PROTOCOL_SOURCE = CONTAINER_ROOT / "frozen-source"
CONTAINER_DATABASES = CONTAINER_ROOT / "databases"
CONTAINER_RUNS = CONTAINER_ROOT / "runs"
CONTAINER_ARCHIVE = CONTAINER_ROOT / "archive"
CONTAINER_TRANSPORT = CONTAINER_ROOT / "transport"
CONTAINER_MODEL_REGISTRY = (
    CONTAINER_SOURCE / "tests/fixtures/github_cross_encoder_precision_v8.models.json"
)
OLD_V8_ROOT = PurePosixPath("/opt/ngr-v8/runtime")
BATCH_SIZE = lifecycle.BATCH_SIZE
SHARED_DATABASE = lifecycle.SHARED_DATABASE
MODEL_CACHE = lifecycle.MODEL_CACHE
WORKERS = lifecycle.WORKERS

V11_ROOT_BINDING_SHA256 = (
    "af35fa36a1e1be2ed1ef22790dbcc7a3943d351fad892c18e852c947566c8a89"
)
V11_PASS_SHA256 = (
    "ee86431e6603dd1eabd557778366ff05183a843c79d890f0d97fb9bcc9b26387"
)
V11_COUNT_AUDIT_SHA256 = (
    "806c0f0f2c72d57e6e1d0a755086cdb14a8a8812d9e3dc3e0ab723b87f2ddc10"
)
V11_EVIDENCE_MANIFEST_SHA256 = (
    "522e223e57a8855371e18623952404156c593853ba32b91e93039152b2befae6"
)

canonical_sha256 = lifecycle.canonical_sha256
sha256_file = lifecycle.sha256_file
_write_json_exclusive = lifecycle._write_json_exclusive
_hash_shared_database = lifecycle._hash_shared_database
read_json = lifecycle.read_json


def serialize_container_path(value: PurePosixPath | str) -> str:
    return model_freeze.serialize_container_path(value)


def named_volume_spec(
    volume: str,
    destination: PurePosixPath | str,
    *,
    mode: str | None = None,
) -> str:
    return model_freeze.named_volume_spec(volume, destination, mode=mode)


def host_bind_spec(
    source: Path,
    destination: PurePosixPath | str,
    *,
    mode: str,
) -> str:
    return model_freeze.host_bind_spec(source, destination, mode=mode)


def _manifest(root: Path) -> dict[str, Any]:
    value = read_json(root / MANIFEST)
    if not isinstance(value, dict):
        raise TypeError("v12 observation manifest must be an object")
    return value


def _expected_container_paths() -> dict[str, str]:
    return {
        "root": serialize_container_path(CONTAINER_ROOT),
        "source": serialize_container_path(CONTAINER_SOURCE),
        "model_cache": serialize_container_path(CONTAINER_CACHE),
        "protocol_source": serialize_container_path(CONTAINER_PROTOCOL_SOURCE),
        "databases": serialize_container_path(CONTAINER_DATABASES),
        "runs": serialize_container_path(CONTAINER_RUNS),
        "archive": serialize_container_path(CONTAINER_ARCHIVE),
        "transport": serialize_container_path(CONTAINER_TRANSPORT),
        "old_v8_root": serialize_container_path(OLD_V8_ROOT),
    }


def _verify_v11_inputs(root: Path) -> dict[str, Any]:
    manifest = _manifest(root)
    expected_header = {
        "protocol_id": PROTOCOL_ID,
        "phase": "performance-observation",
        "freeze_commit": FREEZE_COMMIT,
        "v8_protocol_commit": V8_PROTOCOL_COMMIT,
        "runtime_volume": VOLUME,
        "v11_root_freeze_volume": V11_ROOT_FREEZE_VOLUME,
        "v11_root_freeze_volume_reusable": False,
        "accepted_image": {"tag": IMAGE, "id": IMAGE_ID},
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            raise ValueError(f"v12 observation manifest mismatch: {key}")
    if manifest.get("container_paths") != _expected_container_paths():
        raise ValueError("v12 observation container path registry mismatch")
    registry = manifest.get("v11_immutable_sha256")
    if not isinstance(registry, dict) or len(registry) != 14:
        raise ValueError("v12 v11 registry must contain exactly 14 files")
    for relative, expected in registry.items():
        path = root / str(relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"v11 artifact changed: {relative}")
    anchors = {
        "tests/evidence/github_cross_encoder_precision_v11/root-binding-verification.json": V11_ROOT_BINDING_SHA256,
        "tests/evidence/github_cross_encoder_precision_v11/root-freeze.pass.json": V11_PASS_SHA256,
        "tests/evidence/github_cross_encoder_precision_v11/count-audit.json": V11_COUNT_AUDIT_SHA256,
        "tests/evidence/github_cross_encoder_precision_v11/evidence-manifest.json": V11_EVIDENCE_MANIFEST_SHA256,
    }
    for relative, expected in anchors.items():
        if registry.get(relative) != expected:
            raise ValueError(f"v11 anchor is not frozen: {relative}")
    prebuild = root_freeze.validate_prebuild(root)
    terminal = root_freeze.audit_evidence(root)
    if terminal.get("status") != "pass":
        raise ValueError("successful v11 root-freeze evidence is required")
    if (
        terminal.get("future_runtime_volume_absent_before") is not True
        or terminal.get("future_runtime_volume_absent_after") is not True
    ):
        raise ValueError("v11 did not preserve v12 runtime volume absence")
    return {
        "v11_artifact_count": len(registry),
        "v11_predecessor_artifact_count": prebuild["predecessor_artifact_count"],
        "v11_root_binding_sha256": V11_ROOT_BINDING_SHA256,
        "v11_pass_sha256": V11_PASS_SHA256,
        "v11_count_audit_sha256": V11_COUNT_AUDIT_SHA256,
        "v11_evidence_manifest_sha256": V11_EVIDENCE_MANIFEST_SHA256,
    }


def _stored_freeze_contract(root: Path) -> dict[str, Any]:
    v11_contract = _verify_v11_inputs(root)
    image_contract = lifecycle.predecessor._stored_freeze_contract(root)
    return {
        **v11_contract,
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
        "v11_root_freeze_volume_mounted": False,
        "v11_root_freeze_volume_read": False,
        "v11_root_freeze_volume_copied": False,
        "v11_root_freeze_volume_reused": False,
        "v10_runtime_volume_mounted": False,
        "v10_runtime_volume_read": False,
        "v10_runtime_volume_reused": False,
        "v10_cache_freeze_volume_mounted": False,
        "v10_cache_freeze_volume_read": False,
        "v10_cache_freeze_volume_reused": False,
        "old_v8_root_created": False,
        "old_v8_root_mounted": False,
        "old_v8_root_read": False,
    }


def _write_lifecycle_json_exclusive(path: Path, value: object) -> None:
    if path.name == "platform-report.json" and isinstance(value, dict):
        value = dict(value)
        value["v11_root_freeze_volume"] = value.pop("path_freeze_volume")
        value["v11_root_freeze_volume_mounted"] = value.pop(
            "path_freeze_volume_mounted"
        )
        value["v11_root_freeze_volume_read"] = value.pop(
            "path_freeze_volume_read"
        )
        value.update(
            {
                "v11_root_freeze_volume_reused": False,
                "v10_runtime_volume_mounted": False,
                "v10_runtime_volume_read": False,
                "v10_runtime_volume_reused": False,
                "v10_cache_freeze_volume_mounted": False,
                "v10_cache_freeze_volume_read": False,
                "v10_cache_freeze_volume_reused": False,
                "old_v8_root_created": False,
                "old_v8_root_mounted": False,
                "old_v8_root_read": False,
            }
        )
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
            "neuron_graph_rag.cross_encoder_precision_v12_performance_observation",
            *arguments,
        ]
    )
    return command


def _source_initialization_script() -> str:
    root = serialize_container_path(CONTAINER_ROOT)
    source = serialize_container_path(CONTAINER_SOURCE)
    cache = serialize_container_path(CONTAINER_CACHE)
    old_root = serialize_container_path(OLD_V8_ROOT)
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
        f"test ! -e '{old_root}'; mkdir -p {paths}; mkdir '{source}'; "
        f"cp -a /input/source/. '{source}/'; "
        f"rm -rf '{source}/.git'; test -f '{fixture}'; "
        f"test ! -e '{cache}'; test ! -e '{old_root}'"
    )


def _container_model_copy(source: str, cache: str, output: str) -> dict[str, Any]:
    return model_freeze._container_model_copy_verify(
        source,
        cache,
        serialize_container_path(CONTAINER_MODEL_REGISTRY),
        output,
    )


def _configure_container_harness() -> None:
    root = Path(serialize_container_path(CONTAINER_ROOT))
    source = Path(serialize_container_path(CONTAINER_SOURCE))
    cache = Path(serialize_container_path(CONTAINER_CACHE))
    protocol_source = Path(serialize_container_path(CONTAINER_PROTOCOL_SOURCE))
    evidence = Path(serialize_container_path(CONTAINER_SOURCE / EVIDENCE.as_posix()))
    old_root = Path(serialize_container_path(OLD_V8_ROOT))
    if old_root.exists():
        raise FileExistsError("old v8 runtime root must remain absent")
    root_freeze.bind_frozen_harness_root(
        lifecycle.predecessor,
        volume=VOLUME,
        root=root,
        source=source,
        cache=cache,
        protocol_source=protocol_source,
        evidence=evidence,
    )


def _verification_commands(root: Path) -> tuple[list[str], ...]:
    python = root / ".venv" / "Scripts" / "python.exe"
    return (
        [
            "uvx",
            "--offline",
            "ruff",
            "check",
            "src/neuron_graph_rag/cross_encoder_precision_v12_performance_observation.py",
            "tests/test_cross_encoder_precision_v12_performance_observation.py",
        ],
        [
            str(python),
            "-m",
            "unittest",
            "tests.test_cross_encoder_precision_v8",
            "tests.test_cross_encoder_precision_v8_observation",
            "tests.test_cross_encoder_precision_v10",
            "tests.test_cross_encoder_precision_v10_performance_observation",
            "tests.test_cross_encoder_precision_v11_observation",
            "tests.test_cross_encoder_precision_v12_performance_observation",
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
            "neuron_graph_rag.cross_encoder_precision_v11_observation",
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


def _run_stage_host(
    stage: str,
    root: Path,
    rows: list[dict[str, Any]],
    claim_counts: dict[str, int],
) -> dict[str, Any]:
    lifecycle.lifecycle._run_logged(
        _container_command("claim", "--stage", stage), root, rows
    )
    claim_counts[stage] += 1
    stage_root = CONTAINER_RUNS / stage
    database_root = CONTAINER_DATABASES / stage
    for kind, replay in WORKERS:
        identity = f"ngr-v12-{stage}-{kind}-{replay}"
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
        lifecycle.lifecycle._run_logged(command, root, rows)
    result = json.loads(
        lifecycle.lifecycle._run_logged(
            _container_command("finalize", "--stage", stage), root, rows
        )
    )
    lifecycle._export_volume_evidence(root, rows)
    return result


def _lifecycle_replacements() -> Mapping[str, Any]:
    return {
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
        "CACHE_FREEZE_VOLUME": V11_ROOT_FREEZE_VOLUME,
        "CONTAINER_ROOT": CONTAINER_ROOT,
        "CONTAINER_SOURCE": CONTAINER_SOURCE,
        "CONTAINER_CACHE": CONTAINER_CACHE,
        "CONTAINER_DATABASES": CONTAINER_DATABASES,
        "CONTAINER_RUNS": CONTAINER_RUNS,
        "CONTAINER_ARCHIVE": CONTAINER_ARCHIVE,
        "CONTAINER_TRANSPORT": CONTAINER_TRANSPORT,
        "CONTAINER_MODEL_REGISTRY": CONTAINER_MODEL_REGISTRY,
        "_manifest": _manifest,
        "_write_lifecycle_json_exclusive": _write_lifecycle_json_exclusive,
        "_verify_cache_freeze_inputs": _verify_v11_inputs,
        "_stored_freeze_contract": _stored_freeze_contract,
        "_container_command": _container_command,
        "_source_initialization_script": _source_initialization_script,
        "_container_model_copy": _container_model_copy,
        "_verification_commands": _verification_commands,
        "_run_stage_host": _run_stage_host,
    }


@contextmanager
def _v12_scope() -> Any:
    replacements = _lifecycle_replacements()
    original = {name: getattr(lifecycle, name) for name in replacements}
    configure = lifecycle.lifecycle._configure_container_harness
    for name, value in replacements.items():
        setattr(lifecycle, name, value)
    lifecycle.lifecycle._configure_container_harness = _configure_container_harness
    try:
        yield
    finally:
        lifecycle.lifecycle._configure_container_harness = configure
        for name, value in original.items():
            setattr(lifecycle, name, value)


def preflight(root: Path = ROOT, model_cache: Path | None = None) -> dict[str, Any]:
    source_cache = (
        model_freeze.discover_source_cache(root)
        if model_cache is None
        else model_cache
    )
    with _v12_scope():
        return lifecycle.preflight(root, source_cache)


def verify_preflight(root: Path = ROOT) -> dict[str, Any]:
    with _v12_scope():
        return lifecycle.verify_preflight(root)


def run_once(root: Path = ROOT) -> dict[str, Any]:
    with _v12_scope():
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
        raise FileNotFoundError("v12 raw preflight error evidence is missing")
    if (evidence / "preflight.json").exists():
        raise ValueError("successful preflight cannot be finalized as error")
    if terminal_path.exists() or (
        evidence / "observation-evidence-manifest.json"
    ).exists():
        raise FileExistsError("v12 preflight error is already terminal")
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
            raise ValueError("v12 raw preflight error result count mismatch")
    if raw.get("preflight_forward_inference_count") not in {0, 2}:
        raise ValueError("v12 raw preflight probe count mismatch")
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
        "v11_root_freeze_volume_mounted": False,
        "v11_root_freeze_volume_read": False,
        "v11_root_freeze_volume_reused": False,
        "v10_runtime_volume_mounted": False,
        "v10_runtime_volume_read": False,
        "v10_runtime_volume_reused": False,
        "v10_cache_freeze_volume_mounted": False,
        "v10_cache_freeze_volume_read": False,
        "v10_cache_freeze_volume_reused": False,
        "old_v8_root_created": False,
        "old_v8_root_mounted": False,
        "old_v8_root_read": False,
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
    with _v12_scope():
        result = lifecycle.audit_evidence(root)
    terminal_path = root / EVIDENCE / "preflight-terminal.json"
    if result.get("status") == "preflight-error" and terminal_path.is_file():
        terminal = read_json(terminal_path)
        for key in (
            "v11_root_freeze_volume_mounted",
            "v11_root_freeze_volume_read",
            "v11_root_freeze_volume_reused",
            "v10_runtime_volume_mounted",
            "v10_runtime_volume_read",
            "v10_runtime_volume_reused",
            "v10_cache_freeze_volume_mounted",
            "v10_cache_freeze_volume_read",
            "v10_cache_freeze_volume_reused",
            "old_v8_root_created",
            "old_v8_root_mounted",
            "old_v8_root_read",
        ):
            if terminal.get(key) is not False:
                raise ValueError(f"v12 terminal boundary mismatch: {key}")
    return result


def _read_json_command(path: str) -> dict[str, Any]:
    return lifecycle._read_json_command(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Observe parameterized WSLC rank benchmark v12 once"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "preflight",
        "verify-preflight",
        "run",
        "audit",
        "finalize-preflight-error",
        "dependency-report",
    ):
        commands.add_parser(name)
    copy = commands.add_parser("model-copy-verify")
    copy.add_argument("--source-cache", required=True)
    copy.add_argument("--cache", required=True)
    copy.add_argument("--output", required=True)
    probe = commands.add_parser("model-probe")
    probe.add_argument("--cache", required=True)
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
    elif arguments.command == "read-json":
        result = _read_json_command(arguments.path)
    else:
        with _v12_scope(), lifecycle._lifecycle_scope():
            if arguments.command == "model-probe":
                result = lifecycle.lifecycle._container_model_probe(arguments.cache)
            elif arguments.command == "dependency-report":
                result = lifecycle.lifecycle._dependency_report()
            elif arguments.command == "claim":
                result = lifecycle.lifecycle._container_claim(arguments.stage)
            elif arguments.command == "worker":
                result = lifecycle.lifecycle._container_worker(
                    arguments.stage,
                    arguments.kind,
                    arguments.replay,
                    arguments.database,
                    arguments.output,
                )
            elif arguments.command == "finalize":
                result = lifecycle.lifecycle._container_finalize(arguments.stage)
            else:
                result = lifecycle.lifecycle._container_fail_stage(
                    arguments.stage, arguments.message
                )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
