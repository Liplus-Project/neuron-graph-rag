from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any

from . import cross_encoder_precision_v18_performance_observation as predecessor
from . import rank_observation_stage_contract, source_root_propagation

PROTOCOL_ID = "github-ngr-cross-encoder-precision-v19"
FREEZE_COMMIT = "5106e341522bd6cd9d79a7de48800c607eedc455"
V8_PROTOCOL_COMMIT = predecessor.V8_PROTOCOL_COMMIT
ROOT = Path(__file__).resolve().parents[2]
MODULE = "neuron_graph_rag.cross_encoder_precision_v19_performance_observation"
MANIFEST = Path(
    "tests/fixtures/github_cross_encoder_precision_v19_observation.manifest.json"
)
SOURCE_IDENTITY = Path(
    "tests/fixtures/github_cross_encoder_precision_v19.source-identity.json"
)
OBSERVATION_AUDIT = Path(
    "tests/fixtures/github_cross_encoder_precision_v19.observation-audit.json"
)
EVIDENCE = Path("tests/evidence/github_cross_encoder_precision_v19_observation")

lifecycle = predecessor.lifecycle
source_root_freeze = predecessor.source_root_freeze
IMAGE = predecessor.IMAGE
IMAGE_ID = predecessor.IMAGE_ID
WSLC_VERSION = predecessor.WSLC_VERSION
VOLUME = "github-cross-encoder-precision-v19-runtime"
CONTAINER_ROOT = PurePosixPath("/opt/ngr-v19/runtime")
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

FORBIDDEN_VOLUMES = {
    **predecessor.FORBIDDEN_VOLUMES,
    "v18_runtime_volume": predecessor.VOLUME,
}

PREDECESSOR_ANCHOR_SHA256 = {
    "src/neuron_graph_rag/rank_observation_lifecycle.py": (
        "1f33578f988f365b7daab7202d89309955e9192581f02ae43607cfdcac0c3ff6"
    ),
    "src/neuron_graph_rag/rank_observation_terminal_audit.py": (
        "43420020c0c148d039e8f09b15da362863b2cc0a50e4ac42ee6cabeafe59c5e0"
    ),
    "src/neuron_graph_rag/"
    "cross_encoder_precision_v18_performance_observation.py": (
        "c0eab41a4d44f0f4f5aaecca904368739a26f80c0b66db5b6b67bb6cbbfa1219"
    ),
    "tests/evidence/github_cross_encoder_precision_v18_observation/"
    "terminal-evidence-manifest.json": (
        "ec2c90506d0bec979f64fe33cd052db5726ce4baffb15d356815a9fc0e89816f"
    ),
    "tests/evidence/github_cross_encoder_precision_v18_observation_clarification/"
    "terminal-evidence-manifest.json": (
        "f77ec5c85b4a65f251d82ede69e61c1060ef1c969f6430093bdaf3f5691ec633"
    ),
}


def _verification_commands(root: Path) -> tuple[list[str], ...]:
    python = root / ".venv" / "Scripts" / "python.exe"
    return (
        [
            "uvx",
            "--offline",
            "ruff",
            "check",
            "src/neuron_graph_rag/rank_observation_stage_contract.py",
            (
                "src/neuron_graph_rag/"
                "cross_encoder_precision_v19_performance_observation.py"
            ),
            "tests/test_cross_encoder_precision_v19_performance_observation.py",
        ],
        [
            str(python),
            "-m",
            "unittest",
            "tests.test_cross_encoder_precision_v8",
            "tests.test_cross_encoder_precision_v8_observation",
            "tests.test_cross_encoder_precision_v10",
            "tests.test_cross_encoder_precision_v10_performance_observation",
            "tests.test_cross_encoder_precision_v13_observation",
            "tests.test_cross_encoder_precision_v14_performance_observation",
            "tests.test_cross_encoder_precision_v16_observation",
            "tests.test_cross_encoder_precision_v17_performance_observation",
            "tests.test_cross_encoder_precision_v18_performance_observation",
            "tests.test_cross_encoder_precision_v19_performance_observation",
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
            "neuron_graph_rag.cross_encoder_precision_v13_observation",
            "audit",
        ],
        [
            str(python),
            "-m",
            "neuron_graph_rag.cross_encoder_precision_v16_observation",
            "audit",
        ],
        [
            str(python),
            "-m",
            predecessor.MODULE,
            "audit",
        ],
        [str(python), "-m", MODULE, "audit"],
    )


SOURCE_ROOT_SPEC = source_root_propagation.SourceRootFreezeSpec(
    protocol_id=PROTOCOL_ID,
    phase="performance-observation",
    predecessor_merge_commit=FREEZE_COMMIT,
    frozen_protocol_commit=V8_PROTOCOL_COMMIT,
    root=ROOT,
    manifest_path=MANIFEST,
    source_identity_path=SOURCE_IDENTITY,
    audit_path=OBSERVATION_AUDIT,
    evidence_path=EVIDENCE,
    image=IMAGE,
    image_id=IMAGE_ID,
    wslc_version=WSLC_VERSION,
    freeze_volume=VOLUME,
    future_runtime_volume=VOLUME,
    container_root=CONTAINER_ROOT,
    container_source=CONTAINER_SOURCE,
    container_cache=CONTAINER_CACHE,
    container_frozen_source=CONTAINER_PROTOCOL_SOURCE,
    container_report=CONTAINER_ROOT / "source-root-propagation-verification.json",
    container_source_identity=CONTAINER_SOURCE / SOURCE_IDENTITY.as_posix(),
    old_frozen_source=source_root_freeze.OLD_FROZEN_SOURCE,
    predecessor_artifact_count=26,
    identity_schema="ngr.source-root-propagation/v1",
    evidence_stem="observation",
    report_name="source-root-propagation-verification.json",
    forbidden_volumes=FORBIDDEN_VOLUMES,
    read_json=lifecycle.read_json,
    sha256_file=lifecycle.sha256_file,
    canonical_sha256=lifecycle.canonical_sha256,
    write_json_exclusive=lifecycle._write_json_exclusive,
)

STAGE_CONTRACT = rank_observation_stage_contract.RankObservationStageContract(
    container_databases=CONTAINER_DATABASES,
    container_runs=CONTAINER_RUNS,
    worker_slots_per_stage=len(lifecycle.WORKERS),
)


class V19RankObservationSpec(predecessor.V18RankObservationSpec):
    def run_stage_host(
        self,
        stage: str,
        root: Path,
        rows: list[dict[str, object]],
        claim_counts: dict[str, int],
    ) -> dict[str, object]:
        engine = lifecycle.lifecycle.lifecycle
        initialized = json.loads(
            engine._run_logged(
                self.container_command("stage-init", "--stage", stage), root, rows
            )
        )
        STAGE_CONTRACT.validate_initialization(initialized, stage)
        engine._run_logged(self.container_command("claim", "--stage", stage), root, rows)
        claim_counts[stage] += 1
        for kind, replay in lifecycle.WORKERS:
            identity = f"ngr-v19-{stage}-{kind}-{replay}"
            command = self.container_command(
                "worker",
                "--stage",
                stage,
                "--kind",
                kind,
                "--replay",
                replay,
                "--database",
                str(self.container_databases / stage / f"{kind}-{replay}.sqlite3"),
                "--output",
                str(self.container_runs / stage / f"{kind}-{replay}.json"),
                name=identity,
            )
            insert_at = command.index("--workdir")
            command[insert_at:insert_at] = [
                "--env",
                f"NGR_V8_CONTAINER_IDENTITY={identity}",
            ]
            engine._run_logged(command, root, rows)
        result = json.loads(
            engine._run_logged(
                self.container_command("finalize", "--stage", stage), root, rows
            )
        )
        lifecycle.lifecycle._export_volume_evidence(root, rows)
        return result

    def dispatch_container_command(
        self, command: str, **arguments: str
    ) -> dict[str, Any]:
        if command == "stage-init":
            return STAGE_CONTRACT.initialize_container_stage(arguments["stage"])
        return super().dispatch_container_command(command, **arguments)


SPEC = V19RankObservationSpec(
    protocol_id=PROTOCOL_ID,
    freeze_commit=FREEZE_COMMIT,
    root=ROOT,
    manifest_path=MANIFEST,
    source_identity_path=SOURCE_IDENTITY,
    audit_path=OBSERVATION_AUDIT,
    evidence_path=EVIDENCE,
    module_name=MODULE,
    runtime_volume=VOLUME,
    container_root=CONTAINER_ROOT,
    predecessor_artifact_count=26,
    predecessor_anchor_sha256=PREDECESSOR_ANCHOR_SHA256,
    forbidden_volumes=FORBIDDEN_VOLUMES,
    source_root_spec=SOURCE_ROOT_SPEC,
    verification_commands_factory=_verification_commands,
)
TERMINAL_AUDIT = (
    rank_observation_stage_contract.RankObservationActualCountTerminalAudit(
        SPEC, STAGE_CONTRACT
    )
)

serialize_container_path = lifecycle.serialize_container_path
named_volume_spec = lifecycle.named_volume_spec
host_bind_spec = lifecycle.host_bind_spec
_manifest = SPEC.manifest
_expected_container_paths = SPEC.expected_container_paths
_verify_predecessor_hashes = SPEC.verify_predecessor_hashes
validate_prebuild = SPEC.validate_prebuild
_stored_freeze_contract = SPEC.stored_freeze_contract
_write_lifecycle_json_exclusive = SPEC.write_lifecycle_json_exclusive
_canonical_lifecycle_value = SPEC.canonical_lifecycle_value
_container_command = SPEC.container_command
_source_initialization_script = SPEC.source_initialization_script
_container_model_copy = SPEC.container_model_copy
_configure_container_harness = SPEC.configure_container_harness
_run_stage_host = SPEC.run_stage_host
_v19_scope = SPEC.scope
verify_preflight = SPEC.verify_preflight
audit_evidence = TERMINAL_AUDIT.audit_evidence


def preflight(root: Path = ROOT, model_cache: Path | None = None) -> dict[str, object]:
    try:
        return SPEC.preflight(root, model_cache)
    except BaseException:
        evidence = root / EVIDENCE
        if (evidence / "preflight.error.json").is_file() and not (
            evidence / "preflight-terminal.json"
        ).exists():
            SPEC.finalize_preflight_error(root)
        if (evidence / "preflight-terminal.json").is_file() and not (
            evidence / "terminal-evidence-manifest.json"
        ).exists():
            TERMINAL_AUDIT.fixate_terminal_evidence(root)
        raise


def finalize_preflight_error(root: Path = ROOT) -> dict[str, object]:
    evidence = root / EVIDENCE
    if not (evidence / "preflight-terminal.json").exists():
        result = SPEC.finalize_preflight_error(root)
    else:
        result = lifecycle.read_json(evidence / "preflight-terminal.json")
    if not (evidence / "terminal-evidence-manifest.json").exists():
        TERMINAL_AUDIT.fixate_terminal_evidence(root)
    return result


def run_once(root: Path = ROOT) -> dict[str, object]:
    try:
        result = SPEC.run_once(root)
    except BaseException:
        if (root / EVIDENCE / "observation-evidence-manifest.json").is_file():
            TERMINAL_AUDIT.fixate_terminal_evidence(root)
        raise
    TERMINAL_AUDIT.fixate_terminal_evidence(root)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Observe stage-initialized WSLC rank benchmark v19 once"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "prebuild",
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
    stage_init = commands.add_parser("stage-init")
    stage_init.add_argument("--stage", required=True)
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
    if arguments.command == "prebuild":
        result = validate_prebuild()
    elif arguments.command == "preflight":
        result = preflight()
    elif arguments.command == "verify-preflight":
        result = verify_preflight()
    elif arguments.command == "run":
        result = run_once()
    elif arguments.command == "audit":
        result = audit_evidence()
    elif arguments.command == "finalize-preflight-error":
        result = finalize_preflight_error()
    else:
        values = vars(arguments)
        result = SPEC.dispatch_container_command(
            arguments.command,
            **{
                key: value
                for key, value in values.items()
                if key != "command" and value is not None
            },
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
