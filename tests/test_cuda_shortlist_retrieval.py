"""Offline checks for the opt-in CUDA boundary."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from neuron_graph_rag.cpu_shortlist_retrieval import E5_MODEL_ID, E5_REVISION
from neuron_graph_rag.cuda_shortlist_retrieval import (
    CudaShortlistRetriever,
    CudaUnavailableError,
    LocalPinnedCudaV2M3,
    attach_cuda_shortlist_retriever,
)
from neuron_graph_rag.engine import NeuronGraphRAG


class FakeE5:
    model_id = E5_MODEL_ID
    revision = E5_REVISION

    def embed_passages(self, passages):
        return [(1.0, 0.0) for _ in passages]

    def embed_query(self, query):
        return (1.0, 0.0)


class CudaShortlistTests(unittest.TestCase):
    def test_import_and_construction_do_not_load_optional_modules(self):
        script = (
            "from neuron_graph_rag.cuda_shortlist_retrieval import LocalPinnedCudaV2M3; "
            "import sys; backend=LocalPinnedCudaV2M3('missing'); "
            "assert backend._runtime is None and 'torch' not in sys.modules "
            "and 'transformers' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_explicit_cuda_failure_does_not_change_default_search(self):
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"torch": torch}):
            backend = LocalPinnedCudaV2M3(Path(directory) / "missing")
            retriever = CudaShortlistRetriever(Path(directory) / "cache.db", FakeE5(), backend)
            with NeuronGraphRAG(":memory:") as engine:
                original_search = engine.search
                attach_cuda_shortlist_retriever(engine, retriever)
                engine.add_document("a", "target document")
                engine.update_cuda_shortlist_cache()
                self.assertIs(engine.search.__func__, original_search.__func__)
                with self.assertRaisesRegex(CudaUnavailableError, "CUDA device 0"):
                    engine.search_cuda_shortlist("target")
            self.assertIsNone(backend._runtime)

    def test_available_cuda_reports_missing_local_snapshot(self):
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, init=lambda: None)
        transformers = types.ModuleType("transformers")
        transformers.AutoModelForSequenceClassification = object()
        transformers.AutoTokenizer = object()
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
            with self.assertRaisesRegex(FileNotFoundError, "config.json"):
                LocalPinnedCudaV2M3("missing").score_pairs([("q", "p")])


if __name__ == "__main__":
    unittest.main()
