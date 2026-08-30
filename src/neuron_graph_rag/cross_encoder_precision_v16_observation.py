from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from . import cross_encoder_precision_v8_observation as frozen_v8
from . import cross_encoder_precision_v15_observation as lifecycle
from . import source_root_propagation

PROTOCOL_ID = "github-ngr-cross-encoder-precision-v16"
PREDECESSOR_MERGE_COMMIT = "452869e8c78e6f93f771864beae3d44a120a2c8a"
FROZEN_PROTOCOL_COMMIT = lifecycle.FROZEN_PROTOCOL_COMMIT
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path("tests/fixtures/github_cross_encoder_precision_v16.manifest.json")
SOURCE_IDENTITY = Path(
    "tests/fixtures/github_cross_encoder_precision_v16.source-identity.json"
)
RESULT_FREE_AUDIT = Path(
    "tests/fixtures/github_cross_encoder_precision_v16.result-free-audit.json"
)
EVIDENCE = Path("tests/evidence/github_cross_encoder_precision_v16")

IMAGE = lifecycle.IMAGE
IMAGE_ID = lifecycle.IMAGE_ID
WSLC_VERSION = lifecycle.WSLC_VERSION
SOURCE_ROOT_PROPAGATION_FREEZE_VOLUME = (
    "github-cross-encoder-precision-v16-source-root-propagation-freeze"
)
FUTURE_RUNTIME_VOLUME = "github-cross-encoder-precision-v16-runtime"
V10_RUNTIME_VOLUME = lifecycle.V10_RUNTIME_VOLUME
V10_CACHE_FREEZE_VOLUME = lifecycle.V10_CACHE_FREEZE_VOLUME
V11_ROOT_FREEZE_VOLUME = lifecycle.V11_ROOT_FREEZE_VOLUME
V12_RUNTIME_VOLUME = lifecycle.V12_RUNTIME_VOLUME
V13_COMMIT_FREEZE_VOLUME = lifecycle.V13_COMMIT_FREEZE_VOLUME
V14_RUNTIME_VOLUME = lifecycle.V14_RUNTIME_VOLUME
V15_ROOT_NORMALIZATION_FREEZE_VOLUME = lifecycle.ROOT_NORMALIZATION_FREEZE_VOLUME

CONTAINER_ROOT = PurePosixPath("/opt/ngr-v16/source-root-propagation-freeze")
CONTAINER_SOURCE = CONTAINER_ROOT / "source"
CONTAINER_CACHE = CONTAINER_ROOT / "model-cache"
CONTAINER_PROTOCOL_SOURCE = CONTAINER_ROOT / "frozen-source"
CONTAINER_REPORT = CONTAINER_ROOT / "source-root-propagation-verification.json"
CONTAINER_SOURCE_IDENTITY = CONTAINER_SOURCE / SOURCE_IDENTITY.as_posix()
OLD_FROZEN_SOURCE = lifecycle.OLD_FROZEN_SOURCE

canonical_sha256 = lifecycle.canonical_sha256
sha256_file = lifecycle.sha256_file
read_json = lifecycle.read_json
_write_json_exclusive = lifecycle._write_json_exclusive
serialize_container_path = lifecycle.serialize_container_path
named_volume_spec = lifecycle.named_volume_spec

SPEC = source_root_propagation.SourceRootFreezeSpec(
    protocol_id=PROTOCOL_ID,
    phase="source-root-propagation-freeze",
    predecessor_merge_commit=PREDECESSOR_MERGE_COMMIT,
    frozen_protocol_commit=FROZEN_PROTOCOL_COMMIT,
    root=ROOT,
    manifest_path=MANIFEST,
    source_identity_path=SOURCE_IDENTITY,
    audit_path=RESULT_FREE_AUDIT,
    evidence_path=EVIDENCE,
    image=IMAGE,
    image_id=IMAGE_ID,
    wslc_version=WSLC_VERSION,
    freeze_volume=SOURCE_ROOT_PROPAGATION_FREEZE_VOLUME,
    future_runtime_volume=FUTURE_RUNTIME_VOLUME,
    container_root=CONTAINER_ROOT,
    container_source=CONTAINER_SOURCE,
    container_cache=CONTAINER_CACHE,
    container_frozen_source=CONTAINER_PROTOCOL_SOURCE,
    container_report=CONTAINER_REPORT,
    container_source_identity=CONTAINER_SOURCE_IDENTITY,
    old_frozen_source=OLD_FROZEN_SOURCE,
    predecessor_artifact_count=14,
    identity_schema="ngr.source-root-propagation/v1",
    evidence_stem="source-root-propagation",
    report_name="source-root-propagation-verification.json",
    forbidden_volumes={
        "v10_runtime_volume": V10_RUNTIME_VOLUME,
        "v10_cache_freeze_volume": V10_CACHE_FREEZE_VOLUME,
        "v11_root_freeze_volume": V11_ROOT_FREEZE_VOLUME,
        "v12_runtime_volume": V12_RUNTIME_VOLUME,
        "v13_commit_freeze_volume": V13_COMMIT_FREEZE_VOLUME,
        "v14_runtime_volume": V14_RUNTIME_VOLUME,
        "v15_root_normalization_freeze_volume": (
            V15_ROOT_NORMALIZATION_FREEZE_VOLUME
        ),
    },
    read_json=read_json,
    sha256_file=sha256_file,
    canonical_sha256=canonical_sha256,
    write_json_exclusive=_write_json_exclusive,
)

_manifest = SPEC.manifest
_source_identity = SPEC.source_identity
_audit_contract = SPEC.audit_contract
_expected_container_paths = SPEC.expected_container_paths
_verify_predecessor_hashes = SPEC.verify_predecessor_hashes
validate_prebuild = SPEC.validate_prebuild
_count_audit = SPEC.count_audit
_write_evidence = SPEC.write_evidence
_verify_evidence_manifest = SPEC.verify_evidence_manifest
audit_evidence = SPEC.audit_evidence


def resolver_aware_verify_protocol_commit(
    protocol_commit: str,
    protocol: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    source: Path | PurePosixPath,
    protocol_source: Path | PurePosixPath,
) -> dict[str, Any]:
    return SPEC.verify_protocol_commit(
        protocol_commit,
        protocol,
        identity=identity,
        source=source,
        protocol_source=protocol_source,
        nested_verifier=lifecycle.git_free.git_free_verify_protocol_commit,
    )


def bind_source_root_propagation_verifier(
    wrapper: Any,
    *,
    volume: str,
    root: Path | PurePosixPath,
    source: Path | PurePosixPath,
    cache: Path | PurePosixPath,
    protocol_source: Path | PurePosixPath,
    evidence: Path,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    return SPEC.bind_verifier(
        wrapper,
        volume=volume,
        root=root,
        source=source,
        cache=cache,
        protocol_source=protocol_source,
        evidence=evidence,
        identity=identity,
        predecessor_binder=lifecycle.git_free.bind_git_free_commit_verifier,
        nested_verifier=lifecycle.git_free.git_free_verify_protocol_commit,
    )


def source_root_propagation_verify(
    root: Path,
    source: Path,
    cache: Path,
    protocol_source: Path,
    identity_path: Path,
    output: Path,
    *,
    wrapper: Any = frozen_v8,
) -> dict[str, Any]:
    return SPEC.verify_execution(
        root,
        source,
        cache,
        protocol_source,
        identity_path,
        output,
        wrapper=wrapper,
        binder=bind_source_root_propagation_verifier,
    )


def _protocol_source_import_script() -> str:
    root = serialize_container_path(CONTAINER_ROOT)
    source = serialize_container_path(CONTAINER_PROTOCOL_SOURCE)
    cache = serialize_container_path(CONTAINER_CACHE)
    old = serialize_container_path(OLD_FROZEN_SOURCE)
    fixture = serialize_container_path(
        CONTAINER_PROTOCOL_SOURCE
        / "tests/fixtures/github_cross_encoder_precision_v8.manifest.json"
    )
    return (
        "set -eu; "
        f"test -d '{root}'; test ! -e '{source}'; test ! -e '{cache}'; "
        f"test ! -e '{old}'; mkdir '{source}'; tar -xf - -C '{source}'; "
        f"test -f '{fixture}'; test ! -e '{cache}'; test ! -e '{old}'"
    )


def _harness_source_import_script() -> str:
    root = serialize_container_path(CONTAINER_ROOT)
    source = serialize_container_path(CONTAINER_SOURCE)
    cache = serialize_container_path(CONTAINER_CACHE)
    old = serialize_container_path(OLD_FROZEN_SOURCE)
    module = serialize_container_path(
        CONTAINER_SOURCE
        / "src/neuron_graph_rag/cross_encoder_precision_v16_observation.py"
    )
    common = serialize_container_path(
        CONTAINER_SOURCE / "src/neuron_graph_rag/source_root_propagation.py"
    )
    return (
        "set -eu; "
        f"test -d '{root}'; test ! -e '{source}'; test ! -e '{cache}'; "
        f"test ! -e '{old}'; mkdir '{source}'; tar -xf - -C '{source}'; "
        f"test -f '{module}'; test -f '{common}'; "
        f"test ! -e '{cache}'; test ! -e '{old}'"
    )


def source_root_propagation_command() -> list[str]:
    return [
        "wslc",
        "run",
        "--rm",
        "--network",
        "none",
        "--volume",
        named_volume_spec(SOURCE_ROOT_PROPAGATION_FREEZE_VOLUME, CONTAINER_ROOT),
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
        "neuron_graph_rag.cross_encoder_precision_v16_observation",
        "source-root-verify",
        "--root",
        serialize_container_path(CONTAINER_ROOT),
        "--source",
        serialize_container_path(CONTAINER_SOURCE),
        "--cache",
        serialize_container_path(CONTAINER_CACHE),
        "--protocol-source",
        serialize_container_path(CONTAINER_PROTOCOL_SOURCE),
        "--identity",
        serialize_container_path(CONTAINER_SOURCE_IDENTITY),
        "--output",
        serialize_container_path(CONTAINER_REPORT),
    ]


@contextmanager
def _v16_scope() -> Any:
    replacements = {
        "PROTOCOL_ID": PROTOCOL_ID,
        "PREDECESSOR_MERGE_COMMIT": PREDECESSOR_MERGE_COMMIT,
        "ROOT": ROOT,
        "MANIFEST": MANIFEST,
        "SOURCE_IDENTITY": SOURCE_IDENTITY,
        "RESULT_FREE_AUDIT": RESULT_FREE_AUDIT,
        "EVIDENCE": EVIDENCE,
        "ROOT_NORMALIZATION_FREEZE_VOLUME": SOURCE_ROOT_PROPAGATION_FREEZE_VOLUME,
        "FUTURE_RUNTIME_VOLUME": FUTURE_RUNTIME_VOLUME,
        "CONTAINER_ROOT": CONTAINER_ROOT,
        "CONTAINER_SOURCE": CONTAINER_SOURCE,
        "CONTAINER_CACHE": CONTAINER_CACHE,
        "CONTAINER_PROTOCOL_SOURCE": CONTAINER_PROTOCOL_SOURCE,
        "CONTAINER_REPORT": CONTAINER_REPORT,
        "CONTAINER_SOURCE_IDENTITY": CONTAINER_SOURCE_IDENTITY,
        "OLD_FROZEN_SOURCE": OLD_FROZEN_SOURCE,
        "_manifest": _manifest,
        "_source_identity": _source_identity,
        "_audit_contract": _audit_contract,
        "_expected_container_paths": _expected_container_paths,
        "_verify_predecessor_hashes": _verify_predecessor_hashes,
        "validate_prebuild": validate_prebuild,
        "_protocol_source_import_script": _protocol_source_import_script,
        "_harness_source_import_script": _harness_source_import_script,
        "root_normalization_command": source_root_propagation_command,
        "_count_audit": _count_audit,
        "_write_evidence": _write_evidence,
    }
    original = {name: getattr(lifecycle, name) for name in replacements}
    for name, value in replacements.items():
        setattr(lifecycle, name, value)
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(lifecycle, name, value)


def run_source_root_propagation_freeze(root: Path = ROOT) -> dict[str, Any]:
    with _v16_scope():
        result = lifecycle.run_root_normalization_freeze(root)
    value = dict(result)
    root_hash = value.pop("claim_source_root_verification_sha256", None)
    if root_hash is not None:
        value["source_root_propagation_verification_sha256"] = root_hash
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prebuild")
    commands.add_parser("freeze")
    commands.add_parser("audit")
    verify = commands.add_parser("source-root-verify")
    verify.add_argument("--root", required=True)
    verify.add_argument("--source", required=True)
    verify.add_argument("--cache", required=True)
    verify.add_argument("--protocol-source", required=True)
    verify.add_argument("--identity", required=True)
    verify.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "prebuild":
        result = validate_prebuild()
    elif arguments.command == "freeze":
        result = run_source_root_propagation_freeze()
    elif arguments.command == "audit":
        result = audit_evidence()
    else:
        if os.name != "posix":
            raise RuntimeError("source root verifier requires the POSIX container")
        result = source_root_propagation_verify(
            Path(arguments.root),
            Path(arguments.source),
            Path(arguments.cache),
            Path(arguments.protocol_source),
            Path(arguments.identity),
            Path(arguments.output),
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
