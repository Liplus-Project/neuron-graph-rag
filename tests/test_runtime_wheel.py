"""Exercise the distribution artifact without importing from the checkout."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_MODULES = {
    "__init__", "__main__", "benchmark", "cli", "config_provenance",
    "cpu_shortlist_retrieval", "cuda_shortlist_retrieval", "d1_fixture", "database_home", "dynamics",
    "engine", "evaluation", "evidence_feedback", "exclusion_intent",
    "feedback", "judgments", "models", "ontology", "precision_control",
    "retrieval", "sample", "semantic_retrieval", "storage",
}


class RuntimeWheelTest(unittest.TestCase):
    def test_isolated_wheel_contents_and_entry_points(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            wheels = temporary / "wheels"
            installed = temporary / "installed"
            wheels.mkdir()
            subprocess.run(
                [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(wheels), str(ROOT)],
                check=True, capture_output=True, text=True,
            )
            wheel, = wheels.glob("neuron_graph_rag-*.whl")
            with zipfile.ZipFile(wheel) as archive:
                paths = set(archive.namelist())
                metadata_path = next(path for path in paths if path.endswith(".dist-info/METADATA"))
                metadata = archive.read(metadata_path).decode("utf-8")
                entry_points_path = next(path for path in paths if path.endswith(".dist-info/entry_points.txt"))
                entry_points = archive.read(entry_points_path).decode("utf-8")
            self.assertNotIn("Requires-Dist: torch", metadata)
            self.assertNotIn("Requires-Dist: transformers", metadata)
            modules = {
                Path(path).stem for path in paths
                if path.startswith("neuron_graph_rag/") and path.endswith(".py")
            }
            self.assertEqual(modules, RUNTIME_MODULES)
            self.assertTrue({"neuron_graph_rag_mcp/__init__.py", "neuron_graph_rag_mcp/server.py",
                             "neuron_graph_rag_mcp/__main__.py", "neuron_graph_rag_mcp/http_server.py"} <= paths)
            self.assertIn("neuron-graph-rag-mcp-http", entry_points)
            self.assertNotIn("neuron_graph_rag/cross_encoder_precision_v8_evaluation.py", paths)
            self.assertNotIn("neuron_graph_rag/real_task_shadow_v3.py", paths)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(installed), str(wheel)],
                check=True, capture_output=True, text=True,
            )
            environment = dict(os.environ, PYTHONPATH=str(installed), PYTHONNOUSERSITE="1")

            def run(*args: str) -> str:
                result = subprocess.run(
                    [sys.executable, *args], cwd=temporary, env=environment,
                    check=True, capture_output=True, text=True,
                )
                return result.stdout

            location = run("-c", "import neuron_graph_rag; print(neuron_graph_rag.__file__)").strip()
            self.assertTrue(Path(location).is_relative_to(installed), location)
            run("-c", "from neuron_graph_rag.semantic_retrieval import attach_semantic_retriever; "
                "from neuron_graph_rag.cpu_shortlist_retrieval import attach_cpu_shortlist_retriever; "
                "from neuron_graph_rag.cuda_shortlist_retrieval import attach_cuda_shortlist_retriever")
            run("-c", "import sys; from neuron_graph_rag.cuda_shortlist_retrieval import LocalPinnedCudaV2M3; "
                "assert 'torch' not in sys.modules and 'transformers' not in sys.modules; "
                "assert LocalPinnedCudaV2M3('missing')._runtime is None")
            run("-c", "import importlib.util; assert importlib.util.find_spec("
                "'neuron_graph_rag.real_task_shadow_v3') is None")
            demo = json.loads(run("-m", "neuron_graph_rag", "demo"))
            self.assertEqual(demo["feedback_count"], 1)
            evaluation = json.loads(run("-m", "neuron_graph_rag", "eval"))
            self.assertEqual(evaluation["cases"], 3)
            benchmark = json.loads(run(
                "-m", "neuron_graph_rag", "benchmark", "--fixture",
                str(ROOT / "tests/fixtures/d1_liplus_benchmark.json"), "--gold",
                str(ROOT / "tests/fixtures/d1_liplus_benchmark.gold.json"),
            ))
            self.assertIn("cases", benchmark)
            if importlib.util.find_spec("mcp") is not None:
                self._check_optional_mcp(run)
                run("-c", "from neuron_graph_rag_mcp.http_server import create_http_app; "
                    "assert callable(create_http_app)")

    def _check_optional_mcp(self, run) -> None:
        script = """
import asyncio
from mcp import types
from neuron_graph_rag import NeuronGraphRAG
from neuron_graph_rag_mcp.server import CONTRACT_VERSION, FeedbackMCPAdapter

async def check():
    with NeuronGraphRAG('mcp.db') as engine:
        engine.add_document('source', 'cache invalidation decision')
    adapter = FeedbackMCPAdapter('mcp.db')
    try:
        tools = await adapter.list_tools()
        assert {'search', 'record_source_use', 'record_outcome', 'write_judgment',
                'search_judgments'} <= {tool.name for tool in tools.tools}
        result = await adapter.call_tool(None, types.CallToolRequestParams(
            name='search', arguments={'contract_version': CONTRACT_VERSION,
                                      'query': 'cache invalidation'}))
        assert not result.is_error and result.structured_content['hits']
        trace_id = result.structured_content['trace_id']
        result = await adapter.call_tool(None, types.CallToolRequestParams(
            name='record_source_use', arguments={'contract_version': CONTRACT_VERSION,
              'idempotency_key': 'wheel-source-use', 'trace_id': trace_id,
              'events': [{'node_id': 'source', 'stage': stage}
                         for stage in ('selected', 'validated', 'used')]}))
        assert not result.is_error
        result = await adapter.call_tool(None, types.CallToolRequestParams(
            name='record_outcome', arguments={'contract_version': CONTRACT_VERSION,
              'idempotency_key': 'wheel-outcome', 'trace_id': trace_id,
              'node_ids': ['source'], 'outcome': 'confirmed', 'summary': 'confirmed'}))
        assert not result.is_error
        result = await adapter.call_tool(None, types.CallToolRequestParams(
            name='write_judgment', arguments={'contract_version': CONTRACT_VERSION,
              'action': 'add', 'judgment_id': 'wheel-judgment',
              'statement': 'Use the domain API', 'rationale': 'Wheel API smoke test',
              'provenance': {'source': 'test'}}))
        assert not result.is_error and result.structured_content['judgment']['revision'] == 1
        result = await adapter.call_tool(None, types.CallToolRequestParams(
            name='search_judgments', arguments={'contract_version': CONTRACT_VERSION,
              'query': 'domain API'}))
        assert not result.is_error
        assert 'wheel-judgment' in {
            item['judgment_id'] for item in result.structured_content['judgments']}
    finally:
        adapter.close()

asyncio.run(check())
"""
        run("-c", script)


if __name__ == "__main__":
    unittest.main()
