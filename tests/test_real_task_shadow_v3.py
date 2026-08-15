from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag.config_provenance import (
    effective_config_provenance,
    effective_search_surface,
    search_with_surface,
)
from neuron_graph_rag.evidence_feedback import EngineConfig, NeuronGraphRAG
from neuron_graph_rag.real_task_shadow import (
    canonical_json_bytes,
    read_canonical_json,
    write_json_exclusive,
)
from neuron_graph_rag.real_task_shadow_v3 import (
    PROTOCOL_ID,
    audit_repository_lifecycle,
    bind_capture_fingerprint,
    build_packet_from_search,
    capture_packet,
    create_placeholder_snapshot,
    load_effective_registry,
    probe_placeholder,
    replay_packets,
    verify_result_against_packets,
    write_final_aggregate,
)

try:
    import neuron_graph_rag_mcp.server  # noqa: F401
except ImportError as error:
    module = error.name or ""
    if module != "mcp" and not module.startswith("mcp."):
        raise
    MCP_AVAILABLE = False
else:
    MCP_AVAILABLE = True


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
PLACEHOLDER = FIXTURES / "real_task_shadow_v3.placeholder.json"
MANIFEST = FIXTURES / "real_task_shadow_v3.manifest.json"


class RealTaskShadowV3LifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = read_canonical_json(PLACEHOLDER)

    def _packet(self, root: Path) -> tuple[Path, dict[str, object]]:
        snapshot = root / "placeholder-v3.db"
        create_placeholder_snapshot(self.fixture, snapshot)
        config = EngineConfig(
            relation_feedback_evidence_quorum=3,
            sibling_feedback_normalization=1.0,
        )
        search_surface = effective_search_surface(config)
        with NeuronGraphRAG(snapshot, config=config) as engine:
            trace = search_with_surface(
                engine,
                self.fixture["packet_seed"]["query"],
                limit=self.fixture["packet_seed"]["limit"],
                search_surface=search_surface,
                now=1_755_310_400.0,
            )
            hits = []
            for rank, hit in enumerate(trace.hits, start=1):
                score = (
                    hit.channel_score
                    if hasattr(hit, "channel_score")
                    else hit.final_score
                )
                hits.append(
                    {
                        "node_id": hit.node.node_id,
                        "rank": rank,
                        "text": hit.node.text,
                        "metadata": hit.node.metadata,
                        "confidence": hit.node.confidence,
                        "source_use_stage": "retrieved",
                        "scores": {
                            "sparse": hit.sparse_score,
                            "dense": hit.dense_score,
                            "entry": hit.entry_score,
                            "graph_activation": hit.graph_activation,
                            "final": score,
                        },
                        "paths": [
                            {
                                "seed_id": path.seed_id,
                                "contribution": path.contribution,
                                "steps": [
                                    {
                                        "source_id": step.source_id,
                                        "target_id": step.target_id,
                                        "edge_type": step.edge_type,
                                        "edge_weight": step.edge_weight,
                                        "factuality": step.factuality,
                                    }
                                    for step in path.steps
                                ],
                            }
                            for path in hit.paths
                        ],
                    }
                )
        search = {
            "query": trace.query,
            "created_at": trace.created_at,
            "effective_config_provenance": {
                **effective_config_provenance(config),
                "search_surface": search_surface,
            },
            "hits": hits,
        }
        return snapshot, build_packet_from_search(self.fixture, search, snapshot)

    @staticmethod
    def _rebind(packet: dict[str, object]) -> None:
        packet["capture"]["capture_fingerprint"] = bind_capture_fingerprint(packet)

    def _second_root(self, first: dict[str, object]) -> dict[str, object]:
        packet = copy.deepcopy(first)
        packet["packet_id"] = "placeholder-v3-packet-0002-root"
        packet["slot"] = 2
        packet["task"]["eligible_at"] = "2026-08-16T02:00:01Z"
        packet["task"]["task_url"] = "https://example.invalid/placeholder/task/v3-2"
        packet["capture"]["searched_at"] += 1
        packet["captured_at"] = "2026-08-16T02:00:01Z"
        self._rebind(packet)
        return packet

    def _correction(self, first: dict[str, object]) -> dict[str, object]:
        packet = copy.deepcopy(first)
        packet["packet_id"] = "placeholder-v3-packet-0001-correction"
        packet["supersedes_packet_id"] = first["packet_id"]
        packet["outcome"] = {
            "status": "corrected",
            "summary": "placeholder lifecycle correction",
            "external_ref": "https://example.invalid/placeholder/correction/v3-1",
            "evidence": [
                {
                    "kind": "rollback_or_correction",
                    "node_id": "placeholder-lifecycle-target",
                    "external_ref": "https://example.invalid/placeholder/correction/v3-1",
                    "target_commit": "0000000000000000000000000000000000000000",
                    "details": {"reason": "placeholder lifecycle correction"},
                }
            ],
        }
        return packet

    def test_one_frozen_manifest_accepts_empty_packets_correction_and_final(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            empty = audit_repository_lifecycle(
                MANIFEST, repository_root=ROOT, registered_root=output_root
            )
            self.assertEqual(empty["stage"], "empty")
            snapshot, first = self._packet(output_root)
            registry = output_root / "artifacts" / PROTOCOL_ID / "packets"
            capture_packet(first, registry)
            packet_state = audit_repository_lifecycle(
                MANIFEST, repository_root=ROOT, registered_root=output_root
            )
            self.assertEqual(packet_state["effective_packet_count"], 1)
            second = self._second_root(first)
            correction = self._correction(first)
            capture_packet(second, registry)
            capture_packet(correction, registry)
            corrected_state = audit_repository_lifecycle(
                MANIFEST, repository_root=ROOT, registered_root=output_root
            )
            self.assertEqual(corrected_state["packet_file_count"], 3)
            self.assertEqual(corrected_state["effective_packet_count"], 2)
            effective = load_effective_registry(registry)
            self.assertEqual(
                [packet["packet_id"] for packet in effective],
                [correction["packet_id"], second["packet_id"]],
            )
            result = replay_packets(effective, snapshot)
            aggregate = output_root / "artifacts" / PROTOCOL_ID / "observed" / "final.json"
            write_final_aggregate(result, aggregate)
            final_state = audit_repository_lifecycle(
                MANIFEST, repository_root=ROOT, registered_root=output_root
            )
            self.assertEqual(final_state["stage"], "final_aggregate")
            verify_result_against_packets(result, effective, snapshot)
            with self.assertRaises(FileExistsError):
                write_final_aggregate(result, aggregate)
            write_json_exclusive(aggregate.parent / "second.json", result)
            with self.assertRaisesRegex(ValueError, "more than the one-time"):
                audit_repository_lifecycle(
                    MANIFEST, repository_root=ROOT, registered_root=output_root
                )

    def test_hash_canonical_slot_immutable_and_field_order_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, first = self._packet(root)
            wrong_slot = self._second_root(first)
            with self.assertRaisesRegex(ValueError, "sequential slot 1"):
                capture_packet(wrong_slot, root / "wrong-slot")

            immutable_registry = root / "immutable"
            capture_packet(first, immutable_registry)
            correction = self._correction(first)
            correction["capture"]["searched_at"] += 1
            self._rebind(correction)
            with self.assertRaisesRegex(ValueError, "immutable field: capture"):
                capture_packet(correction, immutable_registry)

            noncanonical_root = root / "noncanonical"
            registry = noncanonical_root / "artifacts" / PROTOCOL_ID / "packets"
            registry.mkdir(parents=True)
            (registry / "0001-noncanonical.json").write_text(
                json.dumps(first, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "canonical JSON"):
                audit_repository_lifecycle(
                    MANIFEST, repository_root=ROOT, registered_root=noncanonical_root
                )

            manifest = read_canonical_json(MANIFEST)
            relative = next(iter(manifest["artifact_sha256"]))
            manifest["artifact_sha256"][relative] = "0" * 64
            forged_manifest = root / "forged.manifest.json"
            forged_manifest.write_bytes(canonical_json_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                audit_repository_lifecycle(
                    forged_manifest, repository_root=ROOT, registered_root=root
                )

            reordered_root = root / "reordered"
            reordered_registry = reordered_root / "artifacts" / PROTOCOL_ID / "packets"
            capture_packet(first, reordered_registry)
            result = replay_packets([first], snapshot)
            aggregate = reordered_root / "artifacts" / PROTOCOL_ID / "observed" / "final.json"
            aggregate.parent.mkdir(parents=True)
            aggregate.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "canonical JSON"):
                audit_repository_lifecycle(
                    MANIFEST, repository_root=ROOT, registered_root=reordered_root
                )

            tampered_root = root / "tampered-result"
            tampered_registry = tampered_root / "artifacts" / PROTOCOL_ID / "packets"
            capture_packet(first, tampered_registry)
            tampered = copy.deepcopy(result)
            tampered["arms"]["used_q3_s1"]["packets"][0]["rank_delta"] += 1
            tampered_aggregate = (
                tampered_root / "artifacts" / PROTOCOL_ID / "observed" / "final.json"
            )
            write_json_exclusive(tampered_aggregate, tampered)
            with self.assertRaisesRegex(ValueError, "rank delta mismatch"):
                audit_repository_lifecycle(
                    MANIFEST, repository_root=ROOT, registered_root=tampered_root
                )

    def test_cli_audits_empty_repository_lifecycle_without_snapshot(self) -> None:
        environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        tool = ROOT / "tools" / "run_real_task_shadow_v3.py"
        with tempfile.TemporaryDirectory() as directory:
            run = subprocess.run(
                [
                    sys.executable,
                    str(tool),
                    "audit-lifecycle",
                    "--manifest",
                    str(MANIFEST),
                    "--repository-root",
                    str(ROOT),
                    "--registered-root",
                    directory,
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('"stage": "empty"', run.stdout)


@unittest.skipUnless(MCP_AVAILABLE, "optional MCP SDK is not installed")
class RealTaskShadowV3MCPTest(unittest.TestCase):
    def test_real_mcp_probe_remains_result_free(self) -> None:
        self.assertEqual(
            probe_placeholder(PLACEHOLDER),
            {
                "protocol_id": PROTOCOL_ID,
                "placeholder_only": True,
                "mcp_search_capture_round_trip": True,
                "capture_time_reused": True,
                "effective_config_provenance_verified": True,
                "replay_round_trip": True,
                "exclusive_writer_verified": True,
            },
        )

        environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        tool = ROOT / "tools" / "run_real_task_shadow_v3.py"
        run = subprocess.run(
            [sys.executable, str(tool), "probe", "--fixture", str(PLACEHOLDER)],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('"placeholder_only": true', run.stdout)


if __name__ == "__main__":
    unittest.main()
