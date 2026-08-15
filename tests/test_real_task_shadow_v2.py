from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag.evidence_feedback import EngineConfig
from neuron_graph_rag.real_task_shadow import read_canonical_json, write_json_exclusive
from neuron_graph_rag.real_task_shadow_v2 import (
    PROTOCOL_ID,
    bind_capture_fingerprint,
    build_packet_from_search,
    capture_packet,
    create_placeholder_snapshot,
    effective_config_provenance,
    load_effective_registry,
    probe_placeholder,
    replay_packets,
    verify_packet_against_snapshot,
    verify_result_against_packets,
)
MCP_AVAILABLE = importlib.util.find_spec("mcp") is not None

if MCP_AVAILABLE:
    from neuron_graph_rag_mcp.server import CONTRACT_VERSION, FeedbackMCPAdapter


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
PLACEHOLDER = FIXTURES / "real_task_shadow_v2.placeholder.json"
MANIFEST = FIXTURES / "real_task_shadow_v2.manifest.json"


@unittest.skipUnless(MCP_AVAILABLE, "optional MCP SDK is not installed")
class RealTaskShadowV2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = read_canonical_json(PLACEHOLDER)

    def _packet(self, root: Path) -> tuple[Path, dict[str, object]]:
        snapshot = root / "placeholder-v2.db"
        create_placeholder_snapshot(self.fixture, snapshot)
        config = EngineConfig(
            relation_feedback_evidence_quorum=3,
            sibling_feedback_normalization=1.0,
        )
        adapter = FeedbackMCPAdapter(snapshot, config=config)
        try:
            search = adapter._search(
                {
                    "contract_version": CONTRACT_VERSION,
                    "query": self.fixture["packet_seed"]["query"],
                    "limit": self.fixture["packet_seed"]["limit"],
                }
            )
        finally:
            adapter.close()
        return snapshot, build_packet_from_search(self.fixture, search, snapshot)

    @staticmethod
    def _rebind(packet: dict[str, object]) -> None:
        capture = packet["capture"]
        config = capture["effective_config"]
        capture["retrieval_config_fingerprint"] = effective_config_provenance(
            EngineConfig(**{**config["retrieval"], **config["feedback"]})
        )["retrieval_config_fingerprint"]
        capture["feedback_config_fingerprint"] = effective_config_provenance(
            EngineConfig(**{**config["retrieval"], **config["feedback"]})
        )["feedback_config_fingerprint"]
        capture["full_config_fingerprint"] = effective_config_provenance(
            EngineConfig(**{**config["retrieval"], **config["feedback"]})
        )["full_config_fingerprint"]
        capture["capture_fingerprint"] = bind_capture_fingerprint(packet)

    def test_runtime_mcp_search_capture_and_exact_replay_round_trip(self) -> None:
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

    def test_capture_uses_serving_retrieval_config_and_arms_only_override_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot, packet = self._packet(Path(directory))
            result = replay_packets([packet], snapshot)
        capture = packet["capture"]
        self.assertEqual(capture["search_surface"], "relation")
        self.assertGreater(capture["searched_at"], 1_700_000_000)
        self.assertEqual(capture["effective_config"]["retrieval"]["sparse_weight"], 0.55)
        self.assertEqual(capture["effective_config"]["retrieval"]["dense_weight"], 0.45)
        retrievals = [result["arms"][arm]["effective_config"]["retrieval"] for arm in result["arms"]]
        self.assertEqual(retrievals[0], retrievals[1])
        self.assertEqual(retrievals[0], capture["effective_config"]["retrieval"])
        used_feedback = result["arms"]["used_q3_s1"]["effective_config"]["feedback"]
        confirmed_feedback = result["arms"]["confirmed_r05_s1"]["effective_config"]["feedback"]
        self.assertEqual(used_feedback["relation_feedback_evidence_quorum"], 3)
        self.assertFalse(used_feedback["confirmed_outcome_reinforcement"])
        self.assertTrue(confirmed_feedback["confirmed_outcome_reinforcement"])
        self.assertEqual(confirmed_feedback["confirmation_decay_ratio"], 0.5)
        self.assertEqual(used_feedback["maximum_edge_weight"], 2.0)
        self.assertEqual(confirmed_feedback["maximum_edge_weight"], 2.0)

    def test_default_combined_search_surface_is_reused_by_both_arms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "placeholder-default.db"
            create_placeholder_snapshot(self.fixture, snapshot)
            adapter = FeedbackMCPAdapter(snapshot)
            try:
                search = adapter._search(
                    {
                        "contract_version": CONTRACT_VERSION,
                        "query": self.fixture["packet_seed"]["query"],
                        "limit": self.fixture["packet_seed"]["limit"],
                    }
                )
            finally:
                adapter.close()
            packet = build_packet_from_search(self.fixture, search, snapshot)
            result = replay_packets([packet], snapshot)
        self.assertEqual(packet["capture"]["search_surface"], "combined")
        self.assertEqual(result["capture_search_surface"], "combined")
        self.assertEqual(
            {arm["search_surface"] for arm in result["arms"].values()},
            {"combined"},
        )

    def test_config_timestamp_candidate_path_and_fingerprint_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot, packet = self._packet(Path(directory))
            cases = []
            config = copy.deepcopy(packet)
            config["capture"]["effective_config"]["retrieval"]["max_hops"] = 0
            cases.append(("config", config))
            timestamp = copy.deepcopy(packet)
            timestamp["capture"]["searched_at"] += 60
            cases.append(("timestamp", timestamp))
            candidate = copy.deepcopy(packet)
            candidate["retrieval"]["candidates"].reverse()
            self._rebind(candidate)
            cases.append(("candidate_order", candidate))
            path = copy.deepcopy(packet)
            path["retrieval"]["credited_path"]["steps"][0]["edge_type"] = "tampered"
            self._rebind(path)
            cases.append(("path", path))
            surface = copy.deepcopy(packet)
            surface["capture"]["search_surface"] = "combined"
            self._rebind(surface)
            cases.append(("search_surface", surface))
            fingerprint = copy.deepcopy(packet)
            fingerprint["capture"]["full_config_fingerprint"] = "sha256:" + "0" * 64
            cases.append(("fingerprint", fingerprint))
            for name, tampered in cases:
                with self.subTest(name=name):
                    with self.assertRaises(ValueError):
                        verify_packet_against_snapshot(tampered, snapshot)

    def test_cumulative_q3_and_confirmed_decay_share_one_capture_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot, first = self._packet(Path(directory))
            packets = []
            for slot in range(1, 4):
                packet = copy.deepcopy(first)
                packet["packet_id"] = f"placeholder-v2-packet-{slot:04d}"
                packet["slot"] = slot
                packet["task"]["task_url"] = f"https://example.invalid/placeholder/task/v2-{slot}"
                packet["capture"]["searched_at"] = first["capture"]["searched_at"] + slot - 1
                packet["outcome"] = {
                    "status": "confirmed",
                    "summary": f"placeholder objective confirmation {slot}",
                    "external_ref": f"https://example.invalid/placeholder/check/{slot}",
                    "evidence": [
                        {
                            "kind": "test_passed",
                            "node_id": "placeholder-runtime-target",
                            "external_ref": f"https://example.invalid/placeholder/check/{slot}",
                            "target_commit": "0000000000000000000000000000000000000000",
                            "details": {"command": f"placeholder-check-{slot}", "exit_code": 0},
                        }
                    ],
                }
                self._rebind(packet)
                packets.append(packet)
            result = replay_packets(packets, snapshot)
        used = result["arms"]["used_q3_s1"]["packets"]
        self.assertEqual([row["source_use"]["feedback"]["evidence"][0]["count"] for row in used], [1, 2, 3])
        confirmed = result["arms"]["confirmed_r05_s1"]["packets"]
        self.assertEqual([row["outcome"]["confirmations"][0]["multiplier"] for row in confirmed], [1.0, 0.5, 0.25])

    def test_batch_config_divergence_registry_and_exact_result_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, first = self._packet(root)
            second = copy.deepcopy(first)
            second["packet_id"] = "placeholder-v2-packet-0002"
            second["slot"] = 2
            second["task"]["task_url"] = "https://example.invalid/placeholder/task/v2-2"
            second["capture"]["searched_at"] += 1
            self._rebind(second)
            registry = root / "registry"
            capture_packet(first, registry)
            capture_packet(second, registry)
            self.assertEqual([item["slot"] for item in load_effective_registry(registry)], [1, 2])
            result = replay_packets([first, second], snapshot)
            stored = root / "result.json"
            write_json_exclusive(stored, result)
            verify_result_against_packets(read_canonical_json(stored), [first, second], snapshot)
            forged = copy.deepcopy(result)
            forged["arms"]["used_q3_s1"]["effective_config"]["retrieval"]["seed_count"] = 1
            with self.assertRaises(ValueError):
                verify_result_against_packets(forged, [first, second], snapshot)
            divergent = copy.deepcopy(second)
            divergent["capture"]["effective_config"]["feedback"]["maximum_edge_weight"] = 2.5
            self._rebind(divergent)
            with self.assertRaisesRegex(ValueError, "share one effective capture config"):
                replay_packets([first, divergent], snapshot)

    def test_cli_probe_and_result_free_manifest(self) -> None:
        environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        tool = ROOT / "tools" / "run_real_task_shadow_v2.py"
        run = subprocess.run(
            [sys.executable, str(tool), "probe", "--fixture", str(PLACEHOLDER)],
            check=False, capture_output=True, text=True, env=environment,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('"placeholder_only": true', run.stdout)
        manifest = read_canonical_json(MANIFEST)
        self.assertTrue(manifest["result_free"])
        self.assertEqual(manifest["observation_status"], "not_started")
        for relative, expected in manifest["artifact_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected, relative)
        for relative in manifest["registered_outputs"].values():
            path = ROOT / relative
            self.assertTrue(not path.exists() or not any(path.iterdir()), relative)


if __name__ == "__main__":
    unittest.main()
