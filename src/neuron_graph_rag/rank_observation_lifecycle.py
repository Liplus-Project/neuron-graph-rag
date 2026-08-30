from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from . import cross_encoder_precision_v14_performance_observation as frozen_lifecycle
from . import cross_encoder_precision_v16_observation as frozen_source_root
from .source_root_propagation import SourceRootFreezeSpec


@dataclass(frozen=True)
class RankObservationSpec:
    protocol_id: str
    freeze_commit: str
    root: Path
    manifest_path: Path
    source_identity_path: Path
    audit_path: Path
    evidence_path: Path
    module_name: str
    runtime_volume: str
    container_root: PurePosixPath
    predecessor_artifact_count: int
    predecessor_anchor_sha256: Mapping[str, str]
    forbidden_volumes: Mapping[str, str]
    source_root_spec: SourceRootFreezeSpec
    verification_commands_factory: Callable[[Path], tuple[list[str], ...]]

    @property
    def container_source(self) -> PurePosixPath:
        return self.container_root / "source"

    @property
    def container_cache(self) -> PurePosixPath:
        return self.container_root / "model-cache"

    @property
    def container_protocol_source(self) -> PurePosixPath:
        return self.container_root / "frozen-source"

    @property
    def container_databases(self) -> PurePosixPath:
        return self.container_root / "databases"

    @property
    def container_runs(self) -> PurePosixPath:
        return self.container_root / "runs"

    @property
    def container_archive(self) -> PurePosixPath:
        return self.container_root / "archive"

    @property
    def container_transport(self) -> PurePosixPath:
        return self.container_root / "transport"

    @property
    def container_model_registry(self) -> PurePosixPath:
        return (
            self.container_source
            / "tests/fixtures/github_cross_encoder_precision_v8.models.json"
        )

    @property
    def container_source_identity(self) -> PurePosixPath:
        return self.container_source / self.source_identity_path.as_posix()

    def manifest(self, root: Path) -> dict[str, Any]:
        value = frozen_lifecycle.read_json(root / self.manifest_path)
        if not isinstance(value, dict):
            raise TypeError("rank observation manifest must be an object")
        return value

    def source_identity(self, root: Path) -> dict[str, Any]:
        value = frozen_lifecycle.read_json(root / self.source_identity_path)
        if not isinstance(value, dict):
            raise TypeError("rank observation source identity must be an object")
        return value

    def audit_contract(self, root: Path) -> dict[str, Any]:
        value = frozen_lifecycle.read_json(root / self.audit_path)
        if not isinstance(value, dict):
            raise TypeError("rank observation audit must be an object")
        return value

    def expected_container_paths(self) -> dict[str, str]:
        return {
            "root": str(self.container_root),
            "source": str(self.container_source),
            "model_cache": str(self.container_cache),
            "protocol_source": str(self.container_protocol_source),
            "databases": str(self.container_databases),
            "runs": str(self.container_runs),
            "archive": str(self.container_archive),
            "transport": str(self.container_transport),
            "old_v8_root": str(frozen_source_root.OLD_FROZEN_SOURCE.parent),
        }

    def verify_predecessor_hashes(
        self, root: Path, manifest: Mapping[str, Any]
    ) -> dict[str, str]:
        registry = manifest.get("predecessor_immutable_sha256")
        if (
            not isinstance(registry, dict)
            or len(registry) != self.predecessor_artifact_count
        ):
            raise ValueError("rank observation predecessor cardinality mismatch")
        actual: dict[str, str] = {}
        for relative, expected in registry.items():
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise TypeError("rank observation predecessor entries must be strings")
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(f"rank observation predecessor missing: {relative}")
            observed = frozen_lifecycle.sha256_file(path)
            if observed != expected:
                raise ValueError(f"rank observation predecessor changed: {relative}")
            actual[relative] = observed
        for relative, expected in self.predecessor_anchor_sha256.items():
            if registry.get(relative) != expected:
                raise ValueError(f"rank observation predecessor anchor changed: {relative}")
        return actual

    def validate_prebuild(self, root: Path | None = None) -> dict[str, Any]:
        project_root = self.root if root is None else root
        manifest = self.manifest(project_root)
        expected = {
            "protocol_id": self.protocol_id,
            "phase": "performance-observation",
            "freeze_commit": self.freeze_commit,
            "v8_protocol_commit": frozen_lifecycle.V8_PROTOCOL_COMMIT,
            "runtime_volume": self.runtime_volume,
            "accepted_image": {
                "tag": frozen_lifecycle.IMAGE,
                "id": frozen_lifecycle.IMAGE_ID,
            },
            "accepted_image_rebuild_allowed": False,
            "container_git_executable_allowed": False,
            "preflight_result_free": True,
            "wslc_version": frozen_lifecycle.WSLC_VERSION,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"rank observation manifest mismatch: {key}")
        if manifest.get("container_paths") != self.expected_container_paths():
            raise ValueError("rank observation container path registry mismatch")
        if manifest.get("source_identity_sha256") != frozen_lifecycle.sha256_file(
            project_root / self.source_identity_path
        ):
            raise ValueError("rank observation source identity hash mismatch")
        if manifest.get("observation_audit_sha256") != frozen_lifecycle.sha256_file(
            project_root / self.audit_path
        ):
            raise ValueError("rank observation audit hash mismatch")
        predecessor = self.verify_predecessor_hashes(project_root, manifest)
        identity = self.source_identity(project_root)
        if (
            identity.get("identity_schema") != "ngr.source-root-propagation/v1"
            or identity.get("source_archive_commit") != self.freeze_commit
            or identity.get("configured_claim_source_root")
            != str(self.container_source)
            or identity.get("configured_frozen_source_root")
            != str(self.container_protocol_source)
        ):
            raise ValueError("rank observation source identity mismatch")
        nested = identity.get("git_free_identity")
        if not isinstance(nested, dict) or (
            nested.get("identity_schema") != "ngr.git-free-protocol-identity/v1"
            or nested.get("protocol_artifact_count") != 23
            or nested.get("corpus_document_count") != 24
        ):
            raise ValueError("rank observation git-free identity mismatch")
        audit = self.audit_contract(project_root)
        zero_keys = (
            "accepted_image_rebuild_count",
            "registered_query_execution_count",
            "development_claim_count",
            "holdout_claim_count",
            "observed_result_count",
            "retry_count",
            "shared_database_open_count",
            "container_git_executable_invocation_count",
            "container_subprocess_invocation_count",
        )
        for key in zero_keys:
            if audit.get(key) != 0:
                raise ValueError(f"rank observation preflight count must be zero: {key}")
        if (
            audit.get("protocol_id") != self.protocol_id
            or audit.get("phase") != "preflight"
            or audit.get("preflight_run_limit") != 1
            or audit.get("development_run_limit") != 1
            or audit.get("same_protocol_retry_allowed") is not False
            or audit.get("performance") != "not assessed"
        ):
            raise ValueError("rank observation audit boundary mismatch")
        return {
            "protocol_id": self.protocol_id,
            "status": "prebuild_contract_valid",
            "predecessor_artifact_count": len(predecessor),
            "protocol_artifact_count": 23,
            "corpus_document_count": 24,
            "development_claim_count": 0,
            "holdout_claim_count": 0,
            "observed_result_count": 0,
            "retry_count": 0,
            "performance": "not assessed",
        }

    def stored_freeze_contract(self, root: Path) -> dict[str, Any]:
        prebuild = self.validate_prebuild(root)
        image = frozen_lifecycle.lifecycle.predecessor._stored_freeze_contract(root)
        boundaries: dict[str, Any] = {}
        for field in self.forbidden_volumes:
            boundaries[f"{field}_mounted"] = False
            boundaries[f"{field}_read"] = False
            boundaries[f"{field}_reused"] = False
        return {
            "predecessor_artifact_count": prebuild["predecessor_artifact_count"],
            "source_identity_sha256": frozen_lifecycle.sha256_file(
                root / self.source_identity_path
            ),
            "accepted_image": image["accepted_image"],
            "runtime_content_sha256": image["runtime_content_sha256"],
            "attestation_sha256": image["attestation_sha256"],
            "fingerprint_sha256": image["fingerprint_sha256"],
            "metadata_correspondence_sha256": image[
                "metadata_correspondence_sha256"
            ],
            "expected_distribution_count": image["expected_distribution_count"],
            "accepted_image_rebuild_count": 0,
            "runtime_content_report_rerun_count": 0,
            "attestation_report_rerun_count": 0,
            "predecessor_terminal_evidence_semantic_content_opened": False,
            "predecessor_packet_reused": False,
            "old_v8_root_created": False,
            "old_v8_root_mounted": False,
            "old_v8_root_read": False,
            **boundaries,
        }

    def write_lifecycle_json_exclusive(self, path: Path, value: object) -> None:
        if path.name == "platform-report.json" and isinstance(value, dict):
            value = dict(value)
            value["v16_source_root_propagation_freeze_volume"] = value.pop(
                "path_freeze_volume"
            )
            value["v16_source_root_propagation_freeze_volume_mounted"] = value.pop(
                "path_freeze_volume_mounted"
            )
            value["v16_source_root_propagation_freeze_volume_read"] = value.pop(
                "path_freeze_volume_read"
            )
            for field in self.forbidden_volumes:
                value.setdefault(f"{field}_mounted", False)
                value.setdefault(f"{field}_read", False)
                value.setdefault(f"{field}_reused", False)
            value.update(
                {
                    "predecessor_terminal_evidence_semantic_content_opened": False,
                    "predecessor_packet_reused": False,
                    "old_v8_root_created": False,
                    "old_v8_root_mounted": False,
                    "old_v8_root_read": False,
                }
            )
        frozen_lifecycle._write_json_exclusive(path, value)

    def canonical_lifecycle_value(self, value: object) -> str:
        if isinstance(value, dict) and (
            "v16_source_root_propagation_freeze_volume" in value
        ):
            value = dict(value)
            value["path_freeze_volume"] = value.pop(
                "v16_source_root_propagation_freeze_volume"
            )
            value["path_freeze_volume_mounted"] = value.pop(
                "v16_source_root_propagation_freeze_volume_mounted"
            )
            value["path_freeze_volume_read"] = value.pop(
                "v16_source_root_propagation_freeze_volume_read"
            )
            for field in self.forbidden_volumes:
                if field != "v16_source_root_propagation_freeze_volume":
                    value.pop(f"{field}_mounted", None)
                    value.pop(f"{field}_read", None)
                value.pop(f"{field}_reused", None)
            for key in (
                "predecessor_terminal_evidence_semantic_content_opened",
                "predecessor_packet_reused",
                "old_v8_root_created",
                "old_v8_root_mounted",
                "old_v8_root_read",
            ):
                value.pop(key, None)
        return frozen_lifecycle.canonical_sha256(value)

    def container_command(
        self,
        *arguments: str,
        extra_volumes: Sequence[str] = (),
        name: str | None = None,
    ) -> list[str]:
        command = ["wslc", "run", "--rm", "--network", "none"]
        if name is not None:
            command.extend(["--name", name])
        command.extend(
            [
                "--volume",
                frozen_lifecycle.named_volume_spec(
                    self.runtime_volume, self.container_root
                ),
            ]
        )
        for volume in extra_volumes:
            command.extend(["--volume", volume])
        command.extend(
            [
                "--env",
                f"PYTHONPATH={self.container_source / 'src'}",
                "--env",
                "HF_HUB_OFFLINE=1",
                "--env",
                "TRANSFORMERS_OFFLINE=1",
                "--env",
                f"HF_HOME={self.container_cache}",
                "--env",
                f"HF_HUB_CACHE={self.container_cache}",
                "--env",
                "NO_PROXY=*",
                "--workdir",
                str(self.container_source),
                "--entrypoint",
                "python",
                frozen_lifecycle.IMAGE,
                "-m",
                self.module_name,
                *arguments,
            ]
        )
        return command

    def source_initialization_script(self) -> str:
        paths = " ".join(
            str(path)
            for path in (
                self.container_databases,
                self.container_runs,
                self.container_archive,
                self.container_transport,
            )
        )
        fixture = (
            self.container_source
            / "tests/fixtures/github_cross_encoder_precision_v8.manifest.json"
        )
        old_root = frozen_source_root.OLD_FROZEN_SOURCE.parent
        return (
            "set -eu; "
            f"test -d '{self.container_root}'; "
            f"test ! -e '{self.container_source}'; "
            f"test ! -e '{self.container_cache}'; test ! -e '{old_root}'; "
            f"mkdir -p {paths}; mkdir '{self.container_source}'; "
            f"cp -a /input/source/. '{self.container_source}/'; "
            f"rm -rf '{self.container_source}/.git'; test -f '{fixture}'; "
            f"test ! -e '{self.container_cache}'; test ! -e '{old_root}'"
        )

    def container_model_copy(self, source: str, cache: str, output: str) -> dict[str, Any]:
        return frozen_lifecycle.model_freeze._container_model_copy_verify(
            source, cache, str(self.container_model_registry), output
        )

    def configure_container_harness(self) -> None:
        root = Path(str(self.container_root))
        source = Path(str(self.container_source))
        cache = Path(str(self.container_cache))
        protocol_source = Path(str(self.container_protocol_source))
        evidence = Path(str(self.container_source / self.evidence_path.as_posix()))
        old_root = Path(str(frozen_source_root.OLD_FROZEN_SOURCE.parent))
        if old_root.exists():
            raise FileExistsError("old v8 runtime root must remain absent")
        identity = self.source_identity(source)
        self.source_root_spec.bind_verifier(
            frozen_lifecycle.lifecycle.predecessor,
            volume=self.runtime_volume,
            root=root,
            source=source,
            cache=cache,
            protocol_source=protocol_source,
            evidence=evidence,
            identity=identity,
            predecessor_binder=(
                frozen_source_root.lifecycle.git_free.bind_git_free_commit_verifier
            ),
            nested_verifier=(
                frozen_source_root.lifecycle.git_free.git_free_verify_protocol_commit
            ),
        )

    def run_stage_host(
        self,
        stage: str,
        root: Path,
        rows: list[dict[str, Any]],
        claim_counts: dict[str, int],
    ) -> dict[str, Any]:
        engine = frozen_lifecycle.lifecycle.lifecycle
        engine._run_logged(self.container_command("claim", "--stage", stage), root, rows)
        claim_counts[stage] += 1
        for kind, replay in frozen_lifecycle.WORKERS:
            identity = f"ngr-v17-{stage}-{kind}-{replay}"
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
        frozen_lifecycle.lifecycle._export_volume_evidence(root, rows)
        return result

    @contextmanager
    def scope(self) -> Any:
        replacements = {
            "PROTOCOL_ID": self.protocol_id,
            "FREEZE_COMMIT": self.freeze_commit,
            "ROOT": self.root,
            "MANIFEST": self.manifest_path,
            "EVIDENCE": self.evidence_path,
            "VOLUME": self.runtime_volume,
            "V13_COMMIT_FREEZE_VOLUME": self.forbidden_volumes[
                "v16_source_root_propagation_freeze_volume"
            ],
            "CONTAINER_ROOT": self.container_root,
            "CONTAINER_SOURCE": self.container_source,
            "CONTAINER_CACHE": self.container_cache,
            "CONTAINER_PROTOCOL_SOURCE": self.container_protocol_source,
            "CONTAINER_DATABASES": self.container_databases,
            "CONTAINER_RUNS": self.container_runs,
            "CONTAINER_ARCHIVE": self.container_archive,
            "CONTAINER_TRANSPORT": self.container_transport,
            "CONTAINER_MODEL_REGISTRY": self.container_model_registry,
            "_manifest": self.manifest,
            "_write_lifecycle_json_exclusive": self.write_lifecycle_json_exclusive,
            "_canonical_lifecycle_value": self.canonical_lifecycle_value,
            "_verify_v13_inputs": self.validate_prebuild,
            "_stored_freeze_contract": self.stored_freeze_contract,
            "_container_command": self.container_command,
            "_source_initialization_script": self.source_initialization_script,
            "_container_model_copy": self.container_model_copy,
            "_configure_container_harness": self.configure_container_harness,
            "_verification_commands": self.verification_commands_factory,
            "_run_stage_host": self.run_stage_host,
        }
        original = {name: getattr(frozen_lifecycle, name) for name in replacements}
        for name, value in replacements.items():
            setattr(frozen_lifecycle, name, value)
        try:
            yield
        finally:
            for name, value in original.items():
                setattr(frozen_lifecycle, name, value)

    def preflight(
        self, root: Path | None = None, model_cache: Path | None = None
    ) -> dict[str, Any]:
        project_root = self.root if root is None else root
        with self.scope():
            return frozen_lifecycle.preflight(project_root, model_cache)

    def verify_preflight(self, root: Path | None = None) -> dict[str, Any]:
        project_root = self.root if root is None else root
        with self.scope():
            return frozen_lifecycle.verify_preflight(project_root)

    def run_once(self, root: Path | None = None) -> dict[str, Any]:
        project_root = self.root if root is None else root
        with self.scope():
            return frozen_lifecycle.run_once(project_root)

    def finalize_preflight_error(self, root: Path | None = None) -> dict[str, Any]:
        project_root = self.root if root is None else root
        with self.scope():
            return frozen_lifecycle.finalize_preflight_error(project_root)

    def audit_evidence(self, root: Path | None = None) -> dict[str, Any]:
        project_root = self.root if root is None else root
        prebuild = self.validate_prebuild(project_root)
        evidence = project_root / self.evidence_path
        if not evidence.exists():
            return {**prebuild, "status": "preflight-not-run"}
        if (evidence / "preflight.error.json").is_file():
            terminal = evidence / "preflight-terminal.json"
            manifest = evidence / "observation-evidence-manifest.json"
            if not terminal.is_file() or not manifest.is_file():
                return {**prebuild, "status": "preflight-error-unfinalized"}
            with self.scope(), frozen_lifecycle._v14_scope():
                value = frozen_lifecycle.lifecycle.lifecycle._verify_hash_manifest(
                    evidence, "observation-evidence-manifest.json", exact=True
                )
            report = frozen_lifecycle.read_json(terminal)
            if (
                value.get("status") != "preflight-error"
                or report.get("retry_count") != 0
                or report.get("same_protocol_retry_allowed") is not False
                or report.get("performance") != "not assessed"
            ):
                raise ValueError("rank observation preflight terminal mismatch")
            return {
                **prebuild,
                "status": "preflight-error",
                "retry_count": 0,
                "shared_database_unchanged": report.get("shared_database_unchanged"),
                "performance": "not assessed",
            }
        report = self.verify_preflight(project_root)
        terminal_manifest = evidence / "observation-evidence-manifest.json"
        if not terminal_manifest.exists():
            return {
                **prebuild,
                "status": "preflight-complete",
                "implementation_commit": report["implementation_commit"],
            }
        with self.scope(), frozen_lifecycle._v14_scope():
            manifest = frozen_lifecycle.lifecycle.lifecycle._verify_hash_manifest(
                evidence, "observation-evidence-manifest.json", exact=True
            )
        execution = evidence / "execution.json"
        error = evidence / "execution-error.json"
        if execution.is_file() == error.is_file():
            raise ValueError("rank observation terminal outcome must be exclusive")
        terminal = frozen_lifecycle.read_json(execution if execution.is_file() else error)
        if (
            terminal.get("protocol_id") != self.protocol_id
            or terminal.get("retry_count") != 0
            or terminal.get("development_claim_count") not in {0, 1}
            or terminal.get("holdout_claim_count") not in {0, 1}
            or terminal.get("shared_database_unchanged") is not True
        ):
            raise ValueError("rank observation terminal count mismatch")
        return {
            **prebuild,
            "status": manifest["status"],
            "development_claim_count": terminal["development_claim_count"],
            "holdout_claim_count": terminal["holdout_claim_count"],
            "retry_count": 0,
            "shared_database_unchanged": True,
        }

    def dispatch_container_command(self, command: str, **arguments: str) -> dict[str, Any]:
        if command == "model-copy-verify":
            return self.container_model_copy(
                arguments["source_cache"], arguments["cache"], arguments["output"]
            )
        if command == "read-json":
            with self.scope():
                return frozen_lifecycle._read_json_command(arguments["path"])
        engine = frozen_lifecycle.lifecycle.lifecycle
        with self.scope(), frozen_lifecycle._v14_scope(), frozen_lifecycle.lifecycle._lifecycle_scope():
            if command == "model-probe":
                return engine._container_model_probe(arguments["cache"])
            if command == "dependency-report":
                return engine._dependency_report()
            if command == "claim":
                return engine._container_claim(arguments["stage"])
            if command == "worker":
                return engine._container_worker(
                    arguments["stage"],
                    arguments["kind"],
                    arguments["replay"],
                    arguments["database"],
                    arguments["output"],
                )
            if command == "finalize":
                return engine._container_finalize(arguments["stage"])
            if command == "fail-stage":
                return engine._container_fail_stage(
                    arguments["stage"], arguments["message"]
                )
        raise ValueError(f"unsupported container command: {command}")
