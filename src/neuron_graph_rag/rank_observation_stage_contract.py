from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from . import cross_encoder_precision_v14_performance_observation as frozen_lifecycle
from .rank_observation_terminal_audit import RankObservationTerminalAudit


@dataclass(frozen=True)
class RankObservationStageContract:
    container_databases: PurePosixPath
    container_runs: PurePosixPath
    worker_slots_per_stage: int = 6

    def stage_paths(self, stage: str) -> tuple[PurePosixPath, PurePosixPath]:
        if stage not in {"development", "holdout"}:
            raise ValueError(f"unsupported rank observation stage: {stage}")
        return self.container_databases / stage, self.container_runs / stage

    def initialize_container_stage(self, stage: str) -> dict[str, Any]:
        database_path, output_path = self.stage_paths(stage)
        paths = tuple(Path(str(path)) for path in (database_path, output_path))
        if any(path.exists() for path in paths):
            raise FileExistsError(f"rank observation stage already initialized: {stage}")
        for path in paths:
            if not path.parent.is_dir():
                raise FileNotFoundError(
                    f"rank observation stage parent is missing: {path.parent}"
                )
            path.mkdir()
        if any(not path.is_dir() for path in paths):
            raise ValueError(f"rank observation stage directory mismatch: {stage}")
        return {
            "protocol_boundary": "fresh-stage-directories",
            "stage": stage,
            "database_directory": str(database_path),
            "output_directory": str(output_path),
            "stage_directory_create_count": 2,
            "exclusive_create": True,
        }

    def validate_initialization(
        self, value: Mapping[str, Any], stage: str
    ) -> dict[str, Any]:
        database_path, output_path = self.stage_paths(stage)
        expected = {
            "protocol_boundary": "fresh-stage-directories",
            "stage": stage,
            "database_directory": str(database_path),
            "output_directory": str(output_path),
            "stage_directory_create_count": 2,
            "exclusive_create": True,
        }
        if dict(value) != expected:
            raise ValueError(f"rank observation stage initialization mismatch: {stage}")
        return expected

    @staticmethod
    def _subcommand(row: Mapping[str, Any]) -> str | None:
        command = row.get("command")
        if not isinstance(command, list) or "-m" not in command:
            return None
        marker = command.index("-m")
        if marker + 2 >= len(command):
            return None
        value = command[marker + 2]
        return value if isinstance(value, str) else None

    def execution_counts(self, report: Mapping[str, Any]) -> dict[str, int]:
        commands = report.get("commands")
        if not isinstance(commands, list):
            raise TypeError("rank observation execution commands must be a list")
        rows = [row for row in commands if isinstance(row, dict)]
        workers = [row for row in rows if self._subcommand(row) == "worker"]
        successful_workers = [row for row in workers if row.get("returncode") == 0]
        finalizers = [
            row
            for row in rows
            if self._subcommand(row) == "finalize" and row.get("returncode") == 0
        ]
        initializers = [
            row
            for row in rows
            if self._subcommand(row) == "stage-init"
            and row.get("returncode") == 0
        ]
        claim_count = int(report.get("development_claim_count", 0)) + int(
            report.get("holdout_claim_count", 0)
        )
        return {
            "planned_worker_slot_count": self.worker_slots_per_stage * claim_count,
            "actual_worker_launch_count": len(workers),
            "actual_successful_worker_count": len(successful_workers),
            "actual_observed_result_count": len(successful_workers),
            "actual_finalize_count": len(finalizers),
            "stage_directory_initialization_count": len(initializers),
        }


@dataclass(frozen=True)
class RankObservationActualCountTerminalAudit(RankObservationTerminalAudit):
    stage_contract: RankObservationStageContract

    def _preflight_error_counts(
        self, raw: Mapping[str, Any], terminal: Mapping[str, Any]
    ) -> dict[str, Any]:
        counts = super()._preflight_error_counts(raw, terminal)
        counts.update(
            {
                "planned_worker_slot_count": 0,
                "actual_worker_launch_count": 0,
                "actual_successful_worker_count": 0,
                "actual_observed_result_count": 0,
                "actual_finalize_count": 0,
                "stage_directory_initialization_count": 0,
            }
        )
        return counts

    def _execution_counts(
        self, report: Mapping[str, Any], status: str
    ) -> dict[str, Any]:
        counts = super()._execution_counts(report, status)
        actual = self.stage_contract.execution_counts(report)
        counts.update(actual)
        counts["worker_process_count"] = actual["actual_worker_launch_count"]
        counts["observed_result_count"] = actual["actual_observed_result_count"]
        return counts

    def _validate_counts(self, counts: Mapping[str, Any]) -> None:
        super()._validate_counts(counts)
        keys = (
            "planned_worker_slot_count",
            "actual_worker_launch_count",
            "actual_successful_worker_count",
            "actual_observed_result_count",
            "actual_finalize_count",
            "stage_directory_initialization_count",
        )
        if any(not isinstance(counts.get(key), int) or counts[key] < 0 for key in keys):
            raise ValueError("rank observation actual count type mismatch")
        claim_count = int(counts.get("development_claim_count", 0)) + int(
            counts.get("holdout_claim_count", 0)
        )
        planned = self.stage_contract.worker_slots_per_stage * claim_count
        launched = counts["actual_worker_launch_count"]
        successful = counts["actual_successful_worker_count"]
        observed = counts["actual_observed_result_count"]
        finalized = counts["actual_finalize_count"]
        initialized = counts["stage_directory_initialization_count"]
        if (
            counts["planned_worker_slot_count"] != planned
            or counts.get("worker_process_count") != launched
            or counts.get("observed_result_count") != observed
            or observed != successful
            or not 0 <= successful <= launched <= planned
            or not 0 <= finalized <= claim_count
            or not claim_count <= initialized <= claim_count + 1
        ):
            raise ValueError("rank observation actual count mismatch")
        if counts.get("status") == "complete" and (
            launched != planned
            or successful != planned
            or finalized != claim_count
            or initialized != claim_count
        ):
            raise ValueError("rank observation complete count mismatch")

    def audit_evidence(self, root: Path | None = None) -> dict[str, Any]:
        result = super().audit_evidence(root)
        project_root = self.spec.root if root is None else root
        count_path = project_root / self.spec.evidence_path / "count-audit.json"
        if not count_path.is_file():
            return result
        counts = frozen_lifecycle.read_json(count_path)
        for key in (
            "planned_worker_slot_count",
            "actual_worker_launch_count",
            "actual_successful_worker_count",
            "actual_observed_result_count",
            "actual_finalize_count",
            "stage_directory_initialization_count",
        ):
            result[key] = counts[key]
        return result
