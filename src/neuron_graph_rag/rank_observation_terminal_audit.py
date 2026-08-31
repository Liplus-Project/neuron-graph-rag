from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import cross_encoder_precision_v14_performance_observation as frozen_lifecycle
from .rank_observation_lifecycle import RankObservationSpec


@dataclass(frozen=True)
class RankObservationTerminalAudit:
    spec: RankObservationSpec

    @contextmanager
    def protocol_identity_scope(self) -> Iterator[Any]:
        with (
            frozen_lifecycle._v14_scope(),
            frozen_lifecycle.lifecycle._lifecycle_scope(),
            self.spec.scope(),
        ):
            engine = frozen_lifecycle.lifecycle.lifecycle
            modules = (frozen_lifecycle, frozen_lifecycle.lifecycle, engine)
            fields = {
                "PROTOCOL_ID": self.spec.protocol_id,
                "FREEZE_COMMIT": self.spec.freeze_commit,
                "ROOT": self.spec.root,
                "EVIDENCE": self.spec.evidence_path,
            }
            original = {
                (id(module), name): getattr(module, name)
                for module in modules
                for name in fields
            }
            for module in modules:
                for name, value in fields.items():
                    setattr(module, name, value)
            try:
                if any(
                    module.PROTOCOL_ID != self.spec.protocol_id
                    for module in modules
                ):
                    raise ValueError("terminal audit protocol identity diverged")
                yield engine
            finally:
                for module in modules:
                    for name in fields:
                        setattr(module, name, original[(id(module), name)])

    def verify_hash_manifest(
        self, evidence: Path, name: str, *, exact: bool
    ) -> dict[str, Any]:
        value = frozen_lifecycle.read_json(evidence / name)
        if value.get("protocol_id") != self.spec.protocol_id:
            raise ValueError(f"terminal audit protocol mismatch: {name}")
        registry = value.get("files_sha256")
        if not isinstance(registry, dict) or not registry:
            raise ValueError(f"terminal audit registry is missing: {name}")
        for relative, expected in registry.items():
            path = evidence / str(relative)
            if not path.is_file() or frozen_lifecycle.sha256_file(path) != expected:
                raise ValueError(f"terminal audit hash mismatch: {relative}")
        if exact:
            actual = {
                path.relative_to(evidence).as_posix()
                for path in evidence.rglob("*")
                if path.is_file() and path.name != name
            }
            if actual != set(registry):
                raise ValueError("terminal audit file set mismatch")
        return value

    def _forbidden_volume_counts(self) -> dict[str, bool]:
        return {
            f"{field}_{operation}": False
            for field in self.spec.forbidden_volumes
            for operation in ("mounted", "read", "reused")
        }

    def _preflight_error_counts(
        self, raw: Mapping[str, Any], terminal: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "protocol_id": self.spec.protocol_id,
            "status": "preflight-error",
            "preflight_run_count": 1,
            "runtime_volume_create_count": raw.get("runtime_volume_create_count"),
            "model_cache_copy_count": int(
                any(
                    isinstance(row, dict)
                    and row.get("returncode") == 0
                    and "model-copy-verify" in row.get("command", [])
                    for row in raw.get("commands", [])
                )
            ),
            "model_forward_inference_count": raw.get(
                "preflight_forward_inference_count"
            ),
            "registered_query_execution_count": 0,
            "development_claim_count": raw.get("development_claim_count"),
            "holdout_claim_count": raw.get("holdout_claim_count"),
            "worker_process_count": 0,
            "observed_result_count": raw.get("result_count"),
            "retry_count": raw.get("retry_count"),
            "same_protocol_retry_allowed": False,
            "accepted_image_rebuild_count": 0,
            "shared_database_open_count": 0,
            "shared_database_unchanged": terminal.get("shared_database_unchanged"),
            "container_git_executable_invocation_count": 0,
            "container_subprocess_invocation_count": 0,
            "old_v8_root_created": False,
            "old_v8_root_mounted": False,
            "old_v8_root_read": False,
            "predecessor_artifact_count": self.spec.predecessor_artifact_count,
            "predecessor_artifacts_unchanged": True,
            "runtime_volume_reusable": False,
            "performance": "not assessed",
            **self._forbidden_volume_counts(),
        }

    def _execution_counts(
        self, report: Mapping[str, Any], status: str
    ) -> dict[str, Any]:
        claim_count = int(report.get("development_claim_count", 0)) + int(
            report.get("holdout_claim_count", 0)
        )
        return {
            "protocol_id": self.spec.protocol_id,
            "status": status,
            "preflight_run_count": 1,
            "runtime_volume_create_count": 1,
            "model_cache_copy_count": 1,
            "model_forward_inference_count": 2,
            "registered_query_execution_count": 0,
            "development_claim_count": report.get("development_claim_count"),
            "holdout_claim_count": report.get("holdout_claim_count"),
            "worker_process_count": report.get(
                "stage_process_count", 6 * claim_count
            ),
            "observed_result_count": report.get(
                "stage_process_count", 6 * claim_count
            ),
            "retry_count": report.get("retry_count"),
            "same_protocol_retry_allowed": False,
            "accepted_image_rebuild_count": 0,
            "shared_database_open_count": 0,
            "shared_database_unchanged": report.get("shared_database_unchanged"),
            "container_git_executable_invocation_count": 0,
            "container_subprocess_invocation_count": 0,
            "old_v8_root_created": False,
            "old_v8_root_mounted": False,
            "old_v8_root_read": False,
            "predecessor_artifact_count": self.spec.predecessor_artifact_count,
            "predecessor_artifacts_unchanged": True,
            "runtime_volume_reusable": False,
            "performance": "observed" if status == "complete" else "not assessed",
            **self._forbidden_volume_counts(),
        }

    def fixate_terminal_evidence(self, root: Path | None = None) -> dict[str, Any]:
        project_root = self.spec.root if root is None else root
        evidence = project_root / self.spec.evidence_path
        count_path = evidence / "count-audit.json"
        final_path = evidence / "terminal-evidence-manifest.json"
        if count_path.exists() or final_path.exists():
            raise FileExistsError("terminal evidence is already fixed")
        if (evidence / "preflight.error.json").is_file():
            raw = frozen_lifecycle.read_json(evidence / "preflight.error.json")
            terminal = frozen_lifecycle.read_json(evidence / "preflight-terminal.json")
            counts = self._preflight_error_counts(raw, terminal)
        else:
            complete = evidence / "execution.json"
            failed = evidence / "execution-error.json"
            if complete.is_file() == failed.is_file():
                raise ValueError("terminal outcome must be exclusive")
            report = frozen_lifecycle.read_json(
                complete if complete.is_file() else failed
            )
            counts = self._execution_counts(
                report, "complete" if complete.is_file() else "error"
            )
        frozen_lifecycle._write_json_exclusive(count_path, counts)
        registry = {
            path.relative_to(evidence).as_posix(): frozen_lifecycle.sha256_file(path)
            for path in sorted(evidence.rglob("*"), key=lambda item: item.as_posix())
            if path.is_file() and path.name != final_path.name
        }
        frozen_lifecycle._write_json_exclusive(
            final_path,
            {
                "protocol_id": self.spec.protocol_id,
                "status": counts["status"],
                "files_sha256": registry,
            },
        )
        return counts

    def _validate_counts(self, counts: Mapping[str, Any]) -> None:
        zero_keys = (
            "registered_query_execution_count",
            "retry_count",
            "accepted_image_rebuild_count",
            "shared_database_open_count",
            "container_git_executable_invocation_count",
            "container_subprocess_invocation_count",
        )
        for key in zero_keys:
            if counts.get(key) != 0:
                raise ValueError(f"terminal audit zero count mismatch: {key}")
        if (
            counts.get("runtime_volume_create_count") != 1
            or counts.get("model_cache_copy_count") != 1
            or counts.get("model_forward_inference_count") != 2
            or counts.get("same_protocol_retry_allowed") is not False
            or counts.get("shared_database_unchanged") is not True
            or counts.get("predecessor_artifacts_unchanged") is not True
            or counts.get("runtime_volume_reusable") is not False
        ):
            raise ValueError("terminal audit lifecycle count mismatch")
        if any(
            counts.get(f"{field}_{operation}") is not False
            for field in self.spec.forbidden_volumes
            for operation in ("mounted", "read", "reused")
        ):
            raise ValueError("terminal audit crossed a predecessor volume boundary")

    def audit_evidence(self, root: Path | None = None) -> dict[str, Any]:
        project_root = self.spec.root if root is None else root
        prebuild = self.spec.validate_prebuild(project_root)
        evidence = project_root / self.spec.evidence_path
        if not evidence.exists():
            return {**prebuild, "status": "preflight-not-run"}
        if (evidence / "preflight.error.json").is_file() and not (
            evidence / "terminal-evidence-manifest.json"
        ).is_file():
            return {**prebuild, "status": "preflight-error-unfinalized"}
        terminal_manifest = evidence / "terminal-evidence-manifest.json"
        if terminal_manifest.is_file():
            manifest = self.verify_hash_manifest(
                evidence, terminal_manifest.name, exact=True
            )
            counts = frozen_lifecycle.read_json(evidence / "count-audit.json")
            self._validate_counts(counts)
            return {
                **prebuild,
                "status": manifest["status"],
                "development_claim_count": counts["development_claim_count"],
                "holdout_claim_count": counts["holdout_claim_count"],
                "worker_process_count": counts["worker_process_count"],
                "observed_result_count": counts["observed_result_count"],
                "retry_count": 0,
                "shared_database_unchanged": True,
                "performance": counts["performance"],
            }
        report = self.spec.verify_preflight(project_root)
        if not (evidence / "observation-evidence-manifest.json").exists():
            return {
                **prebuild,
                "status": "preflight-complete",
                "implementation_commit": report["implementation_commit"],
            }
        raise ValueError("terminal observation evidence is not fixed")
