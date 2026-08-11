from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

from neuron_graph_rag.engine import EngineConfig as LegacyEngineConfig
from neuron_graph_rag.engine import NeuronGraphRAG as LegacyNeuronGraphRAG
from neuron_graph_rag.evidence_feedback import EngineConfig, NeuronGraphRAG
from neuron_graph_rag.evidence_gated_evaluation import (
    _atomic_rollback_audit,
    _churn,
    identity_projection,
    validate_observed_outputs,
    validate_protocol,
    write_json_exclusive,
)
from neuron_graph_rag.sibling_normalization_evaluation import (
    EngineConfig as FrozenEngineConfig,
)
from neuron_graph_rag.sibling_normalization_evaluation import (
    NeuronGraphRAG as FrozenNeuronGraphRAG,
)
from neuron_graph_rag.storage import SQLiteStore


class EvidenceGatedProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]

    def test_registered_protocol_and_observed_lifecycle_are_valid(self) -> None:
        validated = validate_protocol(self.repo_root)
        states = validate_observed_outputs(self.repo_root)

        self.assertEqual(validated["manifest"]["protocol"], "evidence-gated-feedback-v1")
        self.assertEqual(set(states), {"development", "holdout"})

    def test_frozen_engine_and_legacy_storage_contract_stay_unchanged(self) -> None:
        self.assertIs(FrozenEngineConfig, LegacyEngineConfig)
        self.assertIs(FrozenNeuronGraphRAG, LegacyNeuronGraphRAG)
        self.assertEqual(LegacyEngineConfig.__module__, "neuron_graph_rag.engine")
        self.assertEqual(EngineConfig.__module__, "neuron_graph_rag.evidence_feedback")
        self.assertEqual(NeuronGraphRAG.__module__, "neuron_graph_rag.evidence_feedback")
        parameters = inspect.signature(SQLiteStore.apply_success_feedback).parameters
        self.assertNotIn("evidence_quorum", parameters)

    def test_identity_projection_skips_forbidden_subtrees(self) -> None:
        projected = identity_projection(
            {
                "node_id": "allowed-node",
                "metrics": {"node_id": "forbidden-node"},
                "nested": {"query": "Allowed Query"},
                "gate_result": {"source_url": "https://forbidden.invalid"},
            },
            allowlist={"node_id", "query", "source_url"},
            forbidden_fragments=("result", "metric", "gate"),
        )

        self.assertEqual(projected["node_ids"], {"allowed-node"})
        self.assertEqual(projected["queries"], {"allowed query"})
        self.assertEqual(projected["source_urls"], set())

    def test_atomicity_probe_uses_unregistered_synthetic_identity(self) -> None:
        self.assertTrue(all(_atomic_rollback_audit("synthetic-probe").values()))

    def test_churn_counts_position_changes_and_entries(self) -> None:
        self.assertEqual(_churn(["a", "b"], ["b", "a", "c"]), 3)
        self.assertEqual(_churn(["a", "b"], ["a", "b"]), 0)

    def test_observed_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observed.json"
            write_json_exclusive(path, {"run_count": 1})
            first = path.read_bytes()

            with self.assertRaises(FileExistsError):
                write_json_exclusive(path, {"run_count": 2})

            self.assertEqual(path.read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
