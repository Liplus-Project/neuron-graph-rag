from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath
from typing import Any

Verifier = Callable[[str, Mapping[str, Any]], dict[str, Any]]


def protocol_root_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, PurePath):
        return value.as_posix()
    raise TypeError("protocol root must be a path or string")


def resolve_exact_source_root(
    observed_root: object,
    *,
    configured_source: PurePath | str,
    configured_frozen_source: PurePath | str,
) -> PurePosixPath:
    source = PurePosixPath(str(configured_source))
    frozen_source = PurePosixPath(str(configured_frozen_source))
    if (
        not source.is_absolute()
        or not frozen_source.is_absolute()
        or ".." in source.parts
        or ".." in frozen_source.parts
    ):
        raise ValueError("configured roots must be absolute non-escaping POSIX paths")
    observed = PurePosixPath(protocol_root_text(observed_root))
    if not observed.is_absolute():
        raise ValueError("claim source root must be absolute")
    if ".." in observed.parts:
        raise ValueError("claim source root must not escape its configured root")
    if observed != source:
        raise ValueError("claim source root does not match configured source")
    return frozen_source


def normalize_protocol_root(
    protocol: Mapping[str, Any],
    *,
    configured_source: PurePath | str,
    configured_frozen_source: PurePath | str,
) -> tuple[dict[str, Any], str, PurePosixPath]:
    if "root" not in protocol:
        raise ValueError("protocol root is missing")
    observed = protocol_root_text(protocol["root"])
    resolved = resolve_exact_source_root(
        protocol["root"],
        configured_source=configured_source,
        configured_frozen_source=configured_frozen_source,
    )
    normalized = dict(protocol)
    normalized["root"] = str(resolved)
    return normalized, observed, resolved


def verifier_surfaces(wrapper: Any) -> dict[str, Any]:
    base = getattr(wrapper, "_BASE", None)
    evaluation = getattr(wrapper, "evaluation", None)
    evaluation_base = getattr(evaluation, "_BASE", None)
    nested = getattr(base, "_v4", None)
    nested_base = getattr(nested, "_BASE", None)
    surfaces = {
        "wrapper": wrapper,
        "base": base,
        "evaluation": evaluation,
        "evaluation_base": evaluation_base,
        "nested_protocol_evaluator": nested,
        "nested_protocol_evaluator_base": nested_base,
    }
    if any(module is None for module in surfaces.values()):
        raise TypeError("frozen protocol evaluator object graph is incomplete")
    if len({id(module) for module in surfaces.values()}) != len(surfaces):
        raise ValueError("frozen protocol verifier surfaces must be distinct")
    return surfaces


def bind_verifier_surfaces(wrapper: Any, verifier: Verifier) -> list[str]:
    if not callable(verifier):
        raise TypeError("protocol verifier must be callable")
    surfaces = verifier_surfaces(wrapper)
    for module in surfaces.values():
        module.verify_protocol_commit = verifier
    if not all(
        getattr(module, "verify_protocol_commit", None) is verifier
        for module in surfaces.values()
    ):
        raise ValueError("source root verifier binding diverged")
    return sorted(surfaces)


@dataclass(frozen=True)
class SourceRootFreezeSpec:
    protocol_id: str
    phase: str
    predecessor_merge_commit: str
    frozen_protocol_commit: str
    root: Path
    manifest_path: Path
    source_identity_path: Path
    audit_path: Path
    evidence_path: Path
    image: str
    image_id: str
    wslc_version: str
    freeze_volume: str
    future_runtime_volume: str
    container_root: PurePosixPath
    container_source: PurePosixPath
    container_cache: PurePosixPath
    container_frozen_source: PurePosixPath
    container_report: PurePosixPath
    container_source_identity: PurePosixPath
    old_frozen_source: PurePosixPath
    predecessor_artifact_count: int
    identity_schema: str
    evidence_stem: str
    report_name: str
    forbidden_volumes: Mapping[str, str]
    read_json: Callable[[Path], Any]
    sha256_file: Callable[[Path], str]
    canonical_sha256: Callable[[Any], str]
    write_json_exclusive: Callable[[Path, Any], None]

    def manifest(self, root: Path) -> dict[str, Any]:
        value = self.read_json(root / self.manifest_path)
        if not isinstance(value, dict):
            raise TypeError("source root freeze manifest must be an object")
        return value

    def source_identity(self, root: Path) -> dict[str, Any]:
        value = self.read_json(root / self.source_identity_path)
        if not isinstance(value, dict):
            raise TypeError("source root identity must be an object")
        return value

    def audit_contract(self, root: Path) -> dict[str, Any]:
        value = self.read_json(root / self.audit_path)
        if not isinstance(value, dict):
            raise TypeError("source root result-free audit must be an object")
        return value

    def expected_container_paths(self) -> dict[str, str]:
        return {
            "root": str(self.container_root),
            "source": str(self.container_source),
            "model_cache": str(self.container_cache),
            "protocol_source": str(self.container_frozen_source),
            "source_root_propagation_report": str(self.container_report),
            "source_identity": str(self.container_source_identity),
            "old_frozen_source": str(self.old_frozen_source),
        }

    def expected_evidence(self) -> list[str]:
        return [
            "accepted-image-inspect.json",
            f"{self.evidence_stem}-commands.json",
            f"{self.evidence_stem}.pass.json|{self.evidence_stem}.error.json",
            self.report_name,
            "count-audit.json",
            "evidence-manifest.json",
            "source-identity.json",
            "volume-identity.json",
        ]

    def verify_predecessor_hashes(
        self, root: Path, manifest: Mapping[str, Any]
    ) -> dict[str, str]:
        registry = manifest.get("predecessor_immutable_sha256")
        if (
            not isinstance(registry, dict)
            or len(registry) != self.predecessor_artifact_count
        ):
            raise ValueError("source root predecessor registry cardinality mismatch")
        actual: dict[str, str] = {}
        for relative, expected in registry.items():
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise TypeError("source root predecessor entries must be strings")
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(f"source root predecessor missing: {relative}")
            observed = self.sha256_file(path)
            if observed != expected:
                raise ValueError(f"source root predecessor changed: {relative}")
            actual[relative] = observed
        return actual

    def validate_prebuild(self, root: Path | None = None) -> dict[str, Any]:
        project_root = self.root if root is None else root
        manifest = self.manifest(project_root)
        expected = {
            "protocol_id": self.protocol_id,
            "phase": self.phase,
            "predecessor_merge_commit": self.predecessor_merge_commit,
            "frozen_protocol_commit": self.frozen_protocol_commit,
            "source_root_propagation_freeze_volume": self.freeze_volume,
            "future_runtime_volume": self.future_runtime_volume,
            "accepted_image": {"tag": self.image, "id": self.image_id},
            "accepted_image_rebuild_allowed": False,
            "result_free_only": True,
            "wslc_version": self.wslc_version,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"source root manifest mismatch: {key}")
        if manifest.get("container_paths") != self.expected_container_paths():
            raise ValueError("source root container path registry mismatch")
        if manifest.get("source_identity_sha256") != self.sha256_file(
            project_root / self.source_identity_path
        ):
            raise ValueError("source root identity hash mismatch")
        if manifest.get("result_free_audit_sha256") != self.sha256_file(
            project_root / self.audit_path
        ):
            raise ValueError("source root result-free audit hash mismatch")
        if manifest.get("expected_evidence") != self.expected_evidence():
            raise ValueError("source root expected evidence registry mismatch")
        predecessor_hashes = self.verify_predecessor_hashes(project_root, manifest)
        identity = self.source_identity(project_root)
        if set(identity) != {
            "identity_schema",
            "source_archive_commit",
            "configured_claim_source_root",
            "configured_frozen_source_root",
            "git_free_identity",
        }:
            raise ValueError("source root identity is incomplete")
        if (
            identity["identity_schema"] != self.identity_schema
            or identity["source_archive_commit"] != self.predecessor_merge_commit
            or identity["configured_claim_source_root"]
            != str(self.container_source)
            or identity["configured_frozen_source_root"]
            != str(self.container_frozen_source)
        ):
            raise ValueError("source root identity literal mismatch")
        nested = identity.get("git_free_identity")
        if not isinstance(nested, dict):
            raise TypeError("git-free identity must be an object")
        if (
            nested.get("identity_schema") != "ngr.git-free-protocol-identity/v1"
            or nested.get("frozen_protocol_commit") != self.frozen_protocol_commit
            or nested.get("protocol_artifact_count") != 23
            or nested.get("corpus_document_count") != 24
        ):
            raise ValueError("nested git-free identity mismatch")
        audit = self.audit_contract(project_root)
        zero_keys = (
            "accepted_image_rebuild_count",
            "model_cache_copy_count",
            "model_import_count",
            "model_load_count",
            "model_forward_inference_count",
            "registered_query_execution_count",
            "development_claim_count",
            "holdout_claim_count",
            "worker_process_count",
            "observed_result_count",
            "retry_count",
            "shared_database_open_count",
            "container_git_executable_invocation_count",
            "container_subprocess_invocation_count",
        )
        for key in zero_keys:
            if audit.get(key) != 0:
                raise ValueError(f"source root result-free count must be zero: {key}")
        if (
            audit.get("protocol_id") != self.protocol_id
            or audit.get("phase") != self.phase
            or audit.get("source_root_propagation_verifier_run_limit") != 1
            or audit.get("source_root_propagation_freeze_volume_reusable")
            is not False
            or audit.get("performance") != "not assessed"
        ):
            raise ValueError("source root result-free audit boundary mismatch")
        return {
            "protocol_id": self.protocol_id,
            "status": "prebuild_contract_valid",
            "predecessor_artifact_count": len(predecessor_hashes),
            "protocol_artifact_count": 23,
            "corpus_document_count": 24,
            "source_root_propagation_verifier_run_limit": 1,
            "registered_query_execution_count": 0,
            "model_forward_inference_count": 0,
            "observed_result_count": 0,
            "performance": "not assessed",
        }

    def verify_protocol_commit(
        self,
        protocol_commit: str,
        protocol: Mapping[str, Any],
        *,
        identity: Mapping[str, Any],
        source: Path | PurePosixPath,
        protocol_source: Path | PurePosixPath,
        nested_verifier: Callable[..., dict[str, Any]],
    ) -> dict[str, Any]:
        if identity.get("identity_schema") != self.identity_schema:
            raise ValueError("source root propagation identity schema mismatch")
        if identity.get("source_archive_commit") != self.predecessor_merge_commit:
            raise ValueError("source root propagation archive identity mismatch")
        if identity.get("configured_claim_source_root") != str(source):
            raise ValueError("configured claim source root identity mismatch")
        if identity.get("configured_frozen_source_root") != str(protocol_source):
            raise ValueError("configured frozen source root identity mismatch")
        normalized, observed, resolved = normalize_protocol_root(
            protocol,
            configured_source=source,
            configured_frozen_source=protocol_source,
        )
        nested_identity = identity.get("git_free_identity")
        if not isinstance(nested_identity, dict):
            raise TypeError("git-free identity must be an object")
        verification = nested_verifier(
            protocol_commit,
            normalized,
            identity=nested_identity,
            protocol_source=Path(str(protocol_source)),
        )
        return {
            **verification,
            "observed_claim_source_root": observed,
            "observed_claim_source_root_type": type(protocol["root"]).__name__,
            "configured_claim_source_root": str(source),
            "resolved_frozen_source_root": str(resolved),
            "source_root_propagation_exact": True,
        }

    def bind_verifier(
        self,
        wrapper: Any,
        *,
        volume: str,
        root: Path | PurePosixPath,
        source: Path | PurePosixPath,
        cache: Path | PurePosixPath,
        protocol_source: Path | PurePosixPath,
        evidence: Path,
        identity: Mapping[str, Any],
        predecessor_binder: Callable[..., dict[str, Any]],
        nested_verifier: Callable[..., dict[str, Any]],
    ) -> dict[str, Any]:
        nested_identity = identity.get("git_free_identity")
        if not isinstance(nested_identity, dict):
            raise TypeError("git-free identity must be an object")
        binding = predecessor_binder(
            wrapper,
            volume=volume,
            root=root,
            source=source,
            cache=cache,
            protocol_source=protocol_source,
            evidence=evidence,
            identity=nested_identity,
        )

        def verifier(
            protocol_commit: str, protocol: Mapping[str, Any]
        ) -> dict[str, Any]:
            return self.verify_protocol_commit(
                protocol_commit,
                protocol,
                identity=identity,
                source=source,
                protocol_source=protocol_source,
                nested_verifier=nested_verifier,
            )

        surfaces = bind_verifier_surfaces(wrapper, verifier)
        return {
            **binding,
            "source_root_propagation_verifier_bound": True,
            "configured_claim_source_root": str(source),
            "configured_frozen_source_root": str(protocol_source),
            "verifier_binding_surfaces": surfaces,
        }

    def verify_execution(
        self,
        root: Path,
        source: Path,
        cache: Path,
        protocol_source: Path,
        identity_path: Path,
        output: Path,
        *,
        wrapper: Any,
        binder: Callable[..., dict[str, Any]],
    ) -> dict[str, Any]:
        if output.exists():
            raise FileExistsError("source root propagation report already exists")
        for path, name in (
            (root, "root"),
            (source, "source"),
            (protocol_source, "protocol_source"),
        ):
            if not path.is_dir():
                raise FileNotFoundError(f"parameterized {name} is missing")
        if cache.exists():
            raise FileExistsError("result-free source root freeze created cache")
        old_path = Path(str(self.old_frozen_source))
        if old_path.exists():
            raise FileExistsError("old v8 frozen-source root must remain absent")
        identity = self.read_json(identity_path)
        binding = binder(
            wrapper,
            volume=self.freeze_volume,
            root=root,
            source=source,
            cache=cache,
            protocol_source=protocol_source,
            evidence=self.evidence_path,
            identity=identity,
        )
        protocol = wrapper.evaluation.load_protocol(source)
        verification = wrapper._BASE._v4._BASE.verify_protocol_commit(
            self.frozen_protocol_commit, protocol
        )
        if old_path.exists() or cache.exists():
            raise ValueError("source root propagation crossed a forbidden boundary")
        report = {
            "protocol_id": self.protocol_id,
            "status": "verified",
            "source_root_propagation_verifier_run_count": 1,
            "root_binding_verifier_run_count": 1,
            "binding": binding,
            **verification,
            "old_frozen_source": str(old_path),
            "old_frozen_source_absent_before": True,
            "old_frozen_source_absent_after": True,
            "old_frozen_source_created": False,
            "old_frozen_source_mounted": False,
            "old_frozen_source_read": False,
            "model_cache_absent_before": True,
            "model_cache_absent_after": True,
            "model_cache_copy_count": 0,
            "model_import_count": 0,
            "model_load_count": 0,
            "model_forward_inference_count": 0,
            "registered_query_execution_count": 0,
            "development_claim_count": 0,
            "holdout_claim_count": 0,
            "worker_process_count": 0,
            "observed_result_count": 0,
            "shared_database_open_count": 0,
            "performance": "not assessed",
        }
        self.write_json_exclusive(output, report)
        return report

    def count_audit(
        self,
        *,
        status: str,
        rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
        future_runtime_absent_before: bool,
        future_runtime_absent_after: bool,
        predecessor_unchanged: bool,
    ) -> dict[str, Any]:
        commands = [row.get("command", []) for row in rows]
        create_count = sum(
            command == ["wslc", "volume", "create", self.freeze_volume]
            for command in commands
        )
        verifier_count = sum("source-root-verify" in command for command in commands)
        command_text = "\n".join(
            str(value) for command in commands for value in command
        )
        counts = {
            "protocol_id": self.protocol_id,
            "status": status,
            "source_root_propagation_freeze_volume_create_count": create_count,
            "root_freeze_volume_create_count": create_count,
            "source_root_propagation_verifier_run_count": verifier_count,
            "root_binding_verifier_run_count": verifier_count,
            "source_root_propagation_verifier_retry_count": 0,
            "root_binding_verifier_retry_count": 0,
            "retry_count": 0,
            "accepted_image_rebuild_count": 0,
            "model_cache_copy_count": 0,
            "model_import_count": 0,
            "model_load_count": 0,
            "model_forward_inference_count": 0,
            "registered_query_execution_count": 0,
            "development_claim_count": 0,
            "holdout_claim_count": 0,
            "worker_process_count": 0,
            "observed_result_count": 0,
            "shared_database_open_count": 0,
            "container_git_executable_invocation_count": 0,
            "container_subprocess_invocation_count": 0,
            "old_frozen_source_created": False,
            "old_frozen_source_mounted": False,
            "old_frozen_source_read": False,
            "future_runtime_volume_absent_before": future_runtime_absent_before,
            "future_runtime_volume_absent_after": future_runtime_absent_after,
            "predecessor_artifacts_unchanged": predecessor_unchanged,
            "source_root_propagation_freeze_volume_reusable": False,
            "root_freeze_volume_reusable": False,
            "performance": "not assessed",
        }
        for field, volume in self.forbidden_volumes.items():
            counts[f"{field}_mounted"] = volume in command_text
        return counts

    def write_evidence(
        self,
        root: Path,
        *,
        status: str,
        summary: Mapping[str, Any],
        rows: list[dict[str, Any]],
        image: Mapping[str, Any] | None,
        volume: Mapping[str, Any] | None,
        source_identity: Mapping[str, Any] | None,
        root_binding: Mapping[str, Any] | None,
        future_runtime_absent_before: bool,
        future_runtime_absent_after: bool,
        predecessor_unchanged: bool,
    ) -> None:
        evidence = root / self.evidence_path
        summary_value = dict(summary)
        root_hash = summary_value.pop("root_binding_verification_sha256", None)
        if root_hash is not None:
            summary_value["source_root_propagation_verification_sha256"] = root_hash
        self.write_json_exclusive(
            evidence / f"{self.evidence_stem}.{status}.json", summary_value
        )
        self.write_json_exclusive(
            evidence / f"{self.evidence_stem}-commands.json", {"commands": rows}
        )
        optional = {
            "accepted-image-inspect.json": image,
            "volume-identity.json": volume,
            "source-identity.json": source_identity,
            self.report_name: root_binding,
        }
        for name, value in optional.items():
            if value is not None:
                self.write_json_exclusive(evidence / name, dict(value))
        self.write_json_exclusive(
            evidence / "count-audit.json",
            self.count_audit(
                status=status,
                rows=rows,
                future_runtime_absent_before=future_runtime_absent_before,
                future_runtime_absent_after=future_runtime_absent_after,
                predecessor_unchanged=predecessor_unchanged,
            ),
        )
        registry = {
            path.name: self.sha256_file(path)
            for path in sorted(evidence.iterdir(), key=lambda item: item.name)
            if path.is_file() and path.name != "evidence-manifest.json"
        }
        self.write_json_exclusive(
            evidence / "evidence-manifest.json",
            {
                "protocol_id": self.protocol_id,
                "status": status,
                "files_sha256": registry,
            },
        )

    def verify_evidence_manifest(self, evidence: Path) -> dict[str, Any]:
        value = self.read_json(evidence / "evidence-manifest.json")
        registry = value.get("files_sha256")
        if not isinstance(registry, dict):
            raise TypeError("source root evidence hash registry is missing")
        actual = {
            path.name
            for path in evidence.iterdir()
            if path.is_file() and path.name != "evidence-manifest.json"
        }
        if set(registry) != actual:
            raise ValueError("source root evidence file set mismatch")
        for name, expected in registry.items():
            if self.sha256_file(evidence / name) != expected:
                raise ValueError(f"source root evidence hash mismatch: {name}")
        return value

    def audit_evidence(self, root: Path | None = None) -> dict[str, Any]:
        project_root = self.root if root is None else root
        prebuild = self.validate_prebuild(project_root)
        evidence = project_root / self.evidence_path
        if not evidence.exists():
            return {
                **prebuild,
                "status": "prebuild_ready_evidence_absent",
                "future_runtime_volume_absent_before": None,
                "future_runtime_volume_absent_after": None,
            }
        manifest = self.verify_evidence_manifest(evidence)
        status = str(manifest.get("status"))
        if status not in {"pass", "error"}:
            raise ValueError("source root evidence status mismatch")
        counts = self.read_json(evidence / "count-audit.json")
        zero_keys = (
            "source_root_propagation_verifier_retry_count",
            "retry_count",
            "accepted_image_rebuild_count",
            "model_cache_copy_count",
            "model_import_count",
            "model_load_count",
            "model_forward_inference_count",
            "registered_query_execution_count",
            "development_claim_count",
            "holdout_claim_count",
            "worker_process_count",
            "observed_result_count",
            "shared_database_open_count",
            "container_git_executable_invocation_count",
            "container_subprocess_invocation_count",
        )
        for key in zero_keys:
            if counts.get(key) != 0:
                raise ValueError(f"source root terminal count mismatch: {key}")
        forbidden_mounted = any(
            counts.get(f"{field}_mounted") is not False
            for field in self.forbidden_volumes
        )
        if (
            counts.get("source_root_propagation_freeze_volume_create_count")
            not in {0, 1}
            or counts.get("source_root_propagation_verifier_run_count") not in {0, 1}
            or forbidden_mounted
            or counts.get("old_frozen_source_created") is not False
            or counts.get("old_frozen_source_mounted") is not False
            or counts.get("old_frozen_source_read") is not False
            or counts.get("source_root_propagation_freeze_volume_reusable")
            is not False
            or counts.get("predecessor_artifacts_unchanged") is not True
            or counts.get("performance") != "not assessed"
        ):
            raise ValueError("source root terminal boundary mismatch")
        if status == "pass" and (
            counts.get("source_root_propagation_freeze_volume_create_count") != 1
            or counts.get("source_root_propagation_verifier_run_count") != 1
            or counts.get("future_runtime_volume_absent_before") is not True
            or counts.get("future_runtime_volume_absent_after") is not True
        ):
            raise ValueError("successful source root propagation count mismatch")
        return {
            **prebuild,
            "status": status,
            "source_root_propagation_freeze_volume_create_count": counts[
                "source_root_propagation_freeze_volume_create_count"
            ],
            "source_root_propagation_verifier_run_count": counts[
                "source_root_propagation_verifier_run_count"
            ],
            "future_runtime_volume_absent_before": counts[
                "future_runtime_volume_absent_before"
            ],
            "future_runtime_volume_absent_after": counts[
                "future_runtime_volume_absent_after"
            ],
            "predecessor_artifacts_unchanged": True,
        }
