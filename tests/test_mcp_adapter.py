from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

MCP_AVAILABLE = importlib.util.find_spec("mcp") is not None

if MCP_AVAILABLE:
    from mcp import ClientSession, StdioServerParameters, types
    from mcp.client.stdio import stdio_client

    from neuron_graph_rag_mcp.server import (
        CONTRACT_VERSION,
        OUTCOME_DESCRIPTION,
        SEARCH_DESCRIPTION,
        SOURCE_USE_DESCRIPTION,
        FeedbackMCPAdapter,
    )

from neuron_graph_rag import NeuronGraphRAG
from neuron_graph_rag.evidence_feedback import EngineConfig


@unittest.skipUnless(MCP_AVAILABLE, "optional MCP SDK is not installed")
class MCPAdapterTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary.name) / "ngr.sqlite"
        with NeuronGraphRAG(self.database) as engine:
            engine.add_document("decision", "cache invalidation decision")
            engine.add_document("implementation", "cache implementation")
            engine.add_edge("decision", "implementation", "implemented_by", weight=0.7)
        self.adapter = FeedbackMCPAdapter(self.database)

    def tearDown(self) -> None:
        self.adapter.close()
        self.temporary.cleanup()

    async def test_tools_list_contract_and_structured_search_result(self) -> None:
        listed = await self.adapter.list_tools()
        self.assertEqual([tool.name for tool in listed.tools], ["search", "record_source_use", "record_outcome"])
        self.assertEqual(
            [tool.description for tool in listed.tools],
            [SEARCH_DESCRIPTION, SOURCE_USE_DESCRIPTION, OUTCOME_DESCRIPTION],
        )
        self.assertEqual(
            [tool.annotations.idempotent_hint for tool in listed.tools],
            [False, True, True],
        )
        for tool in listed.tools:
            self.assertFalse(tool.annotations.read_only_hint)
            self.assertFalse(tool.annotations.destructive_hint)
            self.assertFalse(tool.annotations.open_world_hint)
            self.assertFalse(tool.input_schema["additionalProperties"])
            self.assertFalse(tool.output_schema["additionalProperties"])

        result = await self.adapter.call_tool(
            None,
            types.CallToolRequestParams(
                name="search",
                arguments={
                    "contract_version": CONTRACT_VERSION,
                    "query": "cache invalidation",
                },
            ),
        )
        self.assertFalse(result.is_error)
        self.assertEqual(json.loads(result.content[0].text), result.structured_content)
        self.assertIsNone(result.structured_content["trace_expires_at"])
        self.assertTrue(all(hit["source_use_stage"] == "retrieved" for hit in result.structured_content["hits"]))

    async def test_feedback_loop_and_safe_tool_errors(self) -> None:
        search = await self.adapter.call_tool(
            None,
            types.CallToolRequestParams(
                name="search",
                arguments={"contract_version": CONTRACT_VERSION, "query": "cache invalidation", "limit": 2},
            ),
        )
        trace_id = search.structured_content["trace_id"]
        node_id = "implementation"
        source_use_arguments = {
            "contract_version": CONTRACT_VERSION,
            "idempotency_key": "answer-7-source-use-1",
            "trace_id": trace_id,
            "events": [
                {"node_id": node_id, "stage": "selected"},
                {"node_id": node_id, "stage": "validated"},
                {"node_id": node_id, "stage": "used"},
            ],
        }
        source_use = await self.adapter.call_tool(
            None, types.CallToolRequestParams(name="record_source_use", arguments=source_use_arguments)
        )
        replay = await self.adapter.call_tool(
            None, types.CallToolRequestParams(name="record_source_use", arguments=source_use_arguments)
        )
        self.assertFalse(source_use.is_error)
        self.assertEqual(replay.structured_content, source_use.structured_content)
        self.assertEqual(source_use.structured_content["newly_used_node_ids"], [node_id])

        outcome = await self.adapter.call_tool(
            None,
            types.CallToolRequestParams(
                name="record_outcome",
                arguments={
                    "contract_version": CONTRACT_VERSION,
                    "idempotency_key": "outcome-1",
                    "trace_id": trace_id,
                    "node_ids": [node_id],
                    "outcome": "rolled_back",
                    "summary": "the decision was rolled back",
                },
            ),
        )
        self.assertFalse(outcome.is_error)
        self.assertFalse(outcome.structured_content["reinforcement_applied"])

        invalid = await self.adapter.call_tool(
            None,
            types.CallToolRequestParams(
                name="search",
                arguments={"contract_version": "secret query text", "query": "private value"},
            ),
        )
        self.assertTrue(invalid.is_error)
        error = json.loads(invalid.content[0].text)
        self.assertEqual(error["code"], "unsupported_contract_version")
        self.assertNotIn("secret query text", invalid.content[0].text)
        self.assertNotIn("private value", invalid.content[0].text)

    async def test_feedback_receipt_exposes_quorum_evidence_and_replays(self) -> None:
        self.adapter.close()
        self.adapter = FeedbackMCPAdapter(
            self.database,
            config=EngineConfig(relation_feedback_evidence_quorum=3),
        )
        search = await self.adapter.call_tool(
            None,
            types.CallToolRequestParams(
                name="search",
                arguments={
                    "contract_version": CONTRACT_VERSION,
                    "query": "cache invalidation",
                    "limit": 2,
                },
            ),
        )
        arguments = {
            "contract_version": CONTRACT_VERSION,
            "idempotency_key": "evidence-quorum-mcp-1",
            "trace_id": search.structured_content["trace_id"],
            "events": [
                {"node_id": "implementation", "stage": "selected"},
                {"node_id": "implementation", "stage": "validated"},
                {"node_id": "implementation", "stage": "used"},
            ],
        }
        result = await self.adapter.call_tool(
            None,
            types.CallToolRequestParams(
                name="record_source_use", arguments=arguments
            ),
        )
        replay = await self.adapter.call_tool(
            None,
            types.CallToolRequestParams(
                name="record_source_use", arguments=arguments
            ),
        )

        self.assertFalse(result.is_error)
        self.assertEqual(replay.structured_content, result.structured_content)
        feedback = result.structured_content["feedback"]
        self.assertEqual(feedback["reinforced_edges"], [])
        self.assertEqual(
            feedback["evidence"],
            [
                {
                    "source_id": "decision",
                    "target_id": "implementation",
                    "edge_type": "implemented_by",
                    "count": 1,
                    "quorum": 3,
                    "activated": False,
                }
            ],
        )

    async def test_stdio_protocol_smoke(self) -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "neuron_graph_rag_mcp", "--database", str(self.database)],
        )
        async with (
            stdio_client(parameters) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            listed = await session.list_tools()
            self.assertEqual([tool.name for tool in listed.tools], ["search", "record_source_use", "record_outcome"])
            result = await session.call_tool(
                "search",
                {"contract_version": CONTRACT_VERSION, "query": "cache"},
            )
            self.assertFalse(result.is_error)
            self.assertEqual(json.loads(result.content[0].text), result.structured_content)


if __name__ == "__main__":
    unittest.main()
