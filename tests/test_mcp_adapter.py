from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

MCP_AVAILABLE = importlib.util.find_spec("mcp") is not None

if MCP_AVAILABLE:
    from mcp import ClientSession, StdioServerParameters, types
    from mcp.client.stdio import stdio_client

    from neuron_graph_rag_mcp.server import (
        CONFIRMED_OUTCOME_DESCRIPTION,
        CONFIRMED_SOURCE_USE_DESCRIPTION,
        CONTRACT_VERSION,
        DEACTIVATION_OUTCOME_DESCRIPTION,
        OUTCOME_DESCRIPTION,
        SEARCH_DESCRIPTION,
        SOFT_START_OUTCOME_DESCRIPTION,
        SOFT_START_SOURCE_USE_DESCRIPTION,
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
            engine.add_document("implementation", "implementation detail")
            engine.add_document("alternate", "alternate approach")
            engine.add_edge("decision", "implementation", "implemented_by", weight=0.7)
            engine.add_edge("decision", "alternate", "implemented_by", weight=0.6)
        self.adapter = FeedbackMCPAdapter(self.database)

    def tearDown(self) -> None:
        self.adapter.close()
        self.temporary.cleanup()

    def _persistent_state(self) -> dict[str, list[tuple[object, ...]]]:
        connection = self.adapter.engine.store.connection
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        return {
            table: sorted(
                (tuple(row) for row in connection.execute(f'SELECT * FROM "{table}"')),
                key=repr,
            )
            for table in tables
        }

    async def test_tools_list_contract_and_structured_search_result(self) -> None:
        listed = await self.adapter.list_tools()
        self.assertEqual(
            [tool.name for tool in listed.tools],
            [
                "search", "record_source_use", "record_outcome", "write_judgment",
                "search_judgments", "get_judgment", "traverse_judgments",
            ],
        )
        self.assertEqual(
            [tool.description for tool in listed.tools[:3]],
            [SEARCH_DESCRIPTION, SOURCE_USE_DESCRIPTION, OUTCOME_DESCRIPTION],
        )
        self.assertEqual(
            [tool.annotations.idempotent_hint for tool in listed.tools],
            [False, True, True, False, True, True, True],
        )
        for tool in listed.tools[:3]:
            self.assertFalse(tool.annotations.read_only_hint)
            self.assertFalse(tool.annotations.destructive_hint)
            self.assertFalse(tool.annotations.open_world_hint)
            self.assertFalse(tool.input_schema["additionalProperties"])
            self.assertFalse(tool.output_schema["additionalProperties"])
        self.assertTrue(listed.tools[3].annotations.destructive_hint)
        self.assertFalse(listed.tools[3].input_schema["additionalProperties"])
        self.assertFalse(listed.tools[3].output_schema["additionalProperties"])
        for tool in listed.tools[4:]:
            self.assertTrue(tool.annotations.read_only_hint)
            self.assertFalse(tool.annotations.destructive_hint)
            self.assertTrue(tool.annotations.idempotent_hint)
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
        provenance = result.structured_content["effective_config_provenance"]
        self.assertEqual(provenance["search_surface"], "combined")
        self.assertEqual(provenance["effective_config"]["retrieval"]["sparse_weight"], 0.55)
        self.assertEqual(provenance["effective_config"]["feedback"]["maximum_edge_weight"], 2.0)
        self.assertTrue(provenance["retrieval_config_fingerprint"].startswith("sha256:"))
        self.assertTrue(provenance["full_config_fingerprint"].startswith("sha256:"))
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

    async def test_judgment_write_uses_atomic_domain_surface(self) -> None:
        added = await self.adapter.call_tool(
            None,
            types.CallToolRequestParams(
                name="write_judgment",
                arguments={
                    "contract_version": CONTRACT_VERSION,
                    "action": "add",
                    "judgment_id": "mcp-judgment",
                    "statement": "Use the domain API",
                    "rationale": "Raw SQL is outside the model-facing contract",
                    "provenance": {"source": "test"},
                },
            ),
        )
        self.assertFalse(added.is_error)
        self.assertEqual(added.structured_content["judgment"]["revision"], 1)
        stale = await self.adapter.call_tool(
            None,
            types.CallToolRequestParams(
                name="write_judgment",
                arguments={
                    "contract_version": CONTRACT_VERSION,
                    "action": "update",
                    "judgment_id": "mcp-judgment",
                    "statement": "Changed",
                    "rationale": "Stale writes fail closed",
                    "provenance": {"source": "test"},
                    "expected_revision": 2,
                },
            ),
        )
        self.assertTrue(stale.is_error)
        self.assertEqual(
            self.adapter.engine.judgments.get("mcp-judgment")["statement"],
            "Use the domain API",
        )

    async def test_judgment_read_tools_are_filtered_exact_and_persist_nothing(self) -> None:
        graph = self.adapter.engine.judgments
        graph.add(
            "one:root", "Root judgment", "Canonical root",
            {"repository": "Liplus-Project/one", "source": "wiki"},
        )
        graph.add(
            "one:child", "Child judgment", "Supports root",
            {"repository": "Liplus-Project/one", "source": "wiki"},
            relations=[{"target_id": "one:root", "relation_type": "supports"}],
        )
        graph.add(
            "two:legacy", "Legacy judgment", "Archived state",
            {"repository": "Liplus-Project/two", "source": "wiki"},
        )
        graph.archive("two:legacy", expected_revision=1)
        before = self._persistent_state()

        searched = await self.adapter.call_tool(
            None,
            types.CallToolRequestParams(
                name="search_judgments",
                arguments={
                    "contract_version": CONTRACT_VERSION,
                    "query": "judgment",
                    "repository": "one",
                },
            ),
        )
        self.assertFalse(searched.is_error)
        self.assertEqual(
            {item["judgment_id"] for item in searched.structured_content["judgments"]},
            {"one:root", "one:child"},
        )
        self.assertTrue(all("explanation" in item for item in searched.structured_content["judgments"]))

        exact = await self.adapter.call_tool(
            None,
            types.CallToolRequestParams(
                name="get_judgment",
                arguments={
                    "contract_version": CONTRACT_VERSION,
                    "judgment_id": "two:legacy",
                },
            ),
        )
        self.assertFalse(exact.is_error)
        self.assertEqual(exact.structured_content["judgment"]["lifecycle"], "archived")

        traversed = await self.adapter.call_tool(
            None,
            types.CallToolRequestParams(
                name="traverse_judgments",
                arguments={
                    "contract_version": CONTRACT_VERSION,
                    "judgment_id": "one:root",
                    "direction": "incoming",
                    "relation_type": "supports",
                    "max_hops": 2,
                },
            ),
        )
        self.assertFalse(traversed.is_error)
        self.assertEqual(
            [item["judgment"]["judgment_id"] for item in traversed.structured_content["results"]],
            ["one:child"],
        )

        invalid = await self.adapter.call_tool(
            None,
            types.CallToolRequestParams(
                name="search_judgments",
                arguments={
                    "contract_version": CONTRACT_VERSION,
                    "query": "judgment",
                    "unexpected": True,
                },
            ),
        )
        self.assertTrue(invalid.is_error)
        self.assertEqual(json.loads(invalid.content[0].text)["code"], "invalid_argument")
        self.assertEqual(self._persistent_state(), before)

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

    async def test_confirmed_candidate_moves_reinforcement_to_outcome_with_receipt_parity(self) -> None:
        self.adapter.close()
        self.adapter = FeedbackMCPAdapter(
            self.database,
            config=EngineConfig(
                confirmed_outcome_reinforcement=True,
                confirmation_decay_ratio=0.5,
                sibling_feedback_normalization=1.0,
            ),
        )
        listed = await self.adapter.list_tools()
        self.assertEqual(listed.tools[1].description, CONFIRMED_SOURCE_USE_DESCRIPTION)
        self.assertEqual(listed.tools[2].description, CONFIRMED_OUTCOME_DESCRIPTION)
        before = self.adapter.engine.store.edge(
            "decision", "implementation", "implemented_by"
        ).weight
        search = await self.adapter.call_tool(
            None,
            types.CallToolRequestParams(
                name="search",
                arguments={
                    "contract_version": CONTRACT_VERSION,
                    "query": "cache invalidation",
                    "limit": 3,
                },
            ),
        )
        trace_id = search.structured_content["trace_id"]
        source_use = await self.adapter.call_tool(
            None,
            types.CallToolRequestParams(
                name="record_source_use",
                arguments={
                    "contract_version": CONTRACT_VERSION,
                    "idempotency_key": "confirmed-candidate-use",
                    "trace_id": trace_id,
                    "events": [
                        {"node_id": "implementation", "stage": "selected"},
                        {"node_id": "implementation", "stage": "validated"},
                        {"node_id": "implementation", "stage": "used"},
                    ],
                },
            ),
        )
        self.assertIsNone(source_use.structured_content["feedback"])
        self.assertEqual(
            self.adapter.engine.store.edge(
                "decision", "implementation", "implemented_by"
            ).weight,
            before,
        )
        arguments = {
            "contract_version": CONTRACT_VERSION,
            "idempotency_key": "confirmed-candidate-outcome",
            "trace_id": trace_id,
            "node_ids": ["implementation"],
            "outcome": "confirmed",
            "summary": "the implementation was verified",
        }
        outcome = await self.adapter.call_tool(
            None,
            types.CallToolRequestParams(name="record_outcome", arguments=arguments),
        )
        replay = await self.adapter.call_tool(
            None,
            types.CallToolRequestParams(name="record_outcome", arguments=arguments),
        )
        self.assertFalse(outcome.is_error)
        self.assertEqual(replay.structured_content, outcome.structured_content)
        self.assertTrue(outcome.structured_content["reinforcement_applied"])
        confirmation = outcome.structured_content["confirmations"][0]
        self.assertEqual(confirmation["confirmation_count"], 1)
        self.assertEqual(confirmation["multiplier"], 1.0)
        self.assertGreater(confirmation["actual_delta"], 0.0)
        self.assertEqual(
            outcome.structured_content["credited_paths"][0]["steps"][0]["source_id"],
            "decision",
        )

    async def test_soft_start_candidate_receipt_parity_and_provenance(self) -> None:
        self.adapter.close()
        self.adapter = FeedbackMCPAdapter(
            self.database,
            config=EngineConfig(
                soft_start_feedback_reinforcement=True,
                soft_start_feedback_ratio=0.25,
                confirmation_decay_ratio=0.5,
                sibling_feedback_normalization=1.0,
            ),
        )
        listed = await self.adapter.list_tools()
        self.assertEqual(listed.tools[1].description, SOFT_START_SOURCE_USE_DESCRIPTION)
        self.assertEqual(listed.tools[2].description, SOFT_START_OUTCOME_DESCRIPTION)
        before = self.adapter.engine.store.edge(
            "decision", "implementation", "implemented_by"
        ).weight
        sibling_before = self.adapter.engine.store.edge(
            "decision", "alternate", "implemented_by"
        ).weight
        search = await self.adapter.call_tool(
            None,
            types.CallToolRequestParams(
                name="search",
                arguments={
                    "contract_version": CONTRACT_VERSION,
                    "query": "cache invalidation",
                    "limit": 3,
                },
            ),
        )
        provenance = search.structured_content["effective_config_provenance"]
        self.assertEqual(provenance["search_surface"], "relation")
        feedback_config = provenance["effective_config"]["feedback"]
        self.assertTrue(feedback_config["soft_start_feedback_reinforcement"])
        self.assertEqual(feedback_config["soft_start_feedback_ratio"], 0.25)
        trace_id = search.structured_content["trace_id"]
        source_use = await self.adapter.call_tool(
            None,
            types.CallToolRequestParams(
                name="record_source_use",
                arguments={
                    "contract_version": CONTRACT_VERSION,
                    "idempotency_key": "soft-start-use",
                    "trace_id": trace_id,
                    "events": [
                        {"node_id": "implementation", "stage": "selected"},
                        {"node_id": "implementation", "stage": "validated"},
                        {"node_id": "implementation", "stage": "used"},
                    ],
                },
            ),
        )
        feedback = source_use.structured_content["feedback"]
        self.assertIsNotNone(feedback)
        provisional = feedback["reinforced_edges"][0]
        provisional_delta = provisional["new_weight"] - provisional["old_weight"]
        self.assertGreater(provisional_delta, 0.0)
        self.assertEqual(
            self.adapter.engine.store.edge(
                "decision", "alternate", "implemented_by"
            ).weight,
            sibling_before,
        )
        outcome = await self.adapter.call_tool(
            None,
            types.CallToolRequestParams(
                name="record_outcome",
                arguments={
                    "contract_version": CONTRACT_VERSION,
                    "idempotency_key": "soft-start-confirmed",
                    "trace_id": trace_id,
                    "node_ids": ["implementation"],
                    "outcome": "confirmed",
                    "summary": "the implementation was verified",
                },
            ),
        )
        confirmation = outcome.structured_content["confirmations"][0]
        self.assertEqual(confirmation["confirmation_count"], 1)
        self.assertEqual(confirmation["multiplier"], 0.75)
        self.assertGreater(confirmation["actual_delta"], 0.0)
        self.assertAlmostEqual(
            self.adapter.engine.store.edge(
                "decision", "implementation", "implemented_by"
            ).weight
            - before,
            provisional_delta + confirmation["actual_delta"],
        )
        self.assertAlmostEqual(
            sibling_before
            - self.adapter.engine.store.edge(
                "decision", "alternate", "implemented_by"
            ).weight,
            confirmation["actual_delta"],
        )

    async def test_deactivation_candidate_receipt_description_and_provenance(self) -> None:
        self.adapter.close()
        self.adapter = FeedbackMCPAdapter(
            self.database,
            config=EngineConfig(
                soft_start_feedback_reinforcement=True,
                soft_start_feedback_ratio=0.25,
                confirmation_decay_ratio=0.5,
                sibling_feedback_normalization=1.0,
                outcome_driven_feedback_deactivation=True,
            ),
        )
        listed = await self.adapter.list_tools()
        self.assertEqual(listed.tools[2].description, DEACTIVATION_OUTCOME_DESCRIPTION)
        search = await self.adapter.call_tool(
            None,
            types.CallToolRequestParams(
                name="search",
                arguments={
                    "contract_version": CONTRACT_VERSION,
                    "query": "cache invalidation",
                    "limit": 3,
                },
            ),
        )
        self.assertTrue(
            search.structured_content["effective_config_provenance"]
            ["effective_config"]["feedback"]
            ["outcome_driven_feedback_deactivation"]
        )
        trace_id = search.structured_content["trace_id"]
        await self.adapter.call_tool(
            None,
            types.CallToolRequestParams(
                name="record_source_use",
                arguments={
                    "contract_version": CONTRACT_VERSION,
                    "idempotency_key": "deactivation-mcp-use",
                    "trace_id": trace_id,
                    "events": [
                        {"node_id": "implementation", "stage": "selected"},
                        {"node_id": "implementation", "stage": "validated"},
                        {"node_id": "implementation", "stage": "used"},
                    ],
                },
            ),
        )
        await self.adapter.call_tool(
            None,
            types.CallToolRequestParams(
                name="record_outcome",
                arguments={
                    "contract_version": CONTRACT_VERSION,
                    "idempotency_key": "deactivation-mcp-confirmed",
                    "trace_id": trace_id,
                    "node_ids": ["implementation"],
                    "outcome": "confirmed",
                    "summary": "confirmed before correction",
                },
            ),
        )
        corrected = await self.adapter.call_tool(
            None,
            types.CallToolRequestParams(
                name="record_outcome",
                arguments={
                    "contract_version": CONTRACT_VERSION,
                    "idempotency_key": "deactivation-mcp-corrected",
                    "trace_id": trace_id,
                    "node_ids": ["implementation"],
                    "outcome": "corrected",
                    "summary": "credited result was corrected",
                },
            ),
        )
        payload = corrected.structured_content
        self.assertTrue(payload["deactivation_applied"])
        self.assertEqual(len(payload["reversed_contributions"]), 2)
        self.assertEqual(payload["dormancy_changes"], [])
        self.assertEqual(payload["reactivated_edges"], [])
        self.assertEqual(
            {
                mutation["mutation_role"]
                for contribution in payload["reversed_contributions"]
                for mutation in contribution["mutations"]
            },
            {"credited", "sibling"},
        )

    async def test_stdio_protocol_smoke(self) -> None:
        self.adapter.engine.judgments.add(
            "stdio:judgment",
            "Read judgments through stdio",
            "The optional adapter exposes the core read contract",
            {"repository": "Liplus-Project/stdio"},
        )
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
            self.assertEqual(
                [tool.name for tool in listed.tools],
                [
                    "search", "record_source_use", "record_outcome", "write_judgment",
                    "search_judgments", "get_judgment", "traverse_judgments",
                ],
            )
            result = await session.call_tool(
                "search",
                {"contract_version": CONTRACT_VERSION, "query": "cache"},
            )
            self.assertFalse(result.is_error)
            self.assertEqual(json.loads(result.content[0].text), result.structured_content)
            judgment = await session.call_tool(
                "get_judgment",
                {
                    "contract_version": CONTRACT_VERSION,
                    "judgment_id": "stdio:judgment",
                },
            )
            self.assertFalse(judgment.is_error)
            self.assertEqual(
                judgment.structured_content["judgment"]["statement"],
                "Read judgments through stdio",
            )
            source_use = await session.call_tool(
                "record_source_use",
                {
                    "contract_version": CONTRACT_VERSION,
                    "idempotency_key": "stdio-default-source-use",
                    "trace_id": result.structured_content["trace_id"],
                    "events": [
                        {"node_id": "implementation", "stage": "selected"},
                        {"node_id": "implementation", "stage": "validated"},
                        {"node_id": "implementation", "stage": "used"},
                    ],
                },
            )
            self.assertFalse(source_use.is_error)
            evidence = source_use.structured_content["feedback"]["evidence"]
            self.assertEqual(evidence[0]["quorum"], 1)
            self.assertTrue(evidence[0]["activated"])

        with NeuronGraphRAG(self.database) as engine:
            self.assertGreater(
                engine.store.edge(
                    "decision", "implementation", "implemented_by"
                ).weight,
                0.7,
            )
            self.assertEqual(
                engine.store.edge("decision", "alternate", "implemented_by").weight,
                0.6,
            )

    async def test_stdio_q3_s1_delays_then_normalizes_on_third_evidence(self) -> None:
        database = Path(self.temporary.name) / "stdio-q3-s1.sqlite"
        with NeuronGraphRAG(database) as engine:
            engine.add_document("source", "stabilization anchor")
            engine.add_document("target", "credited implementation result")
            engine.add_document("sibling", "uncredited sibling result")
            engine.add_document("other-source", "isolated origin")
            engine.add_document("other-target", "isolated destination")
            engine.add_edge("source", "target", "supports", weight=0.7)
            engine.add_edge("source", "sibling", "supports", weight=0.6)
            engine.add_edge("other-source", "other-target", "isolated", weight=0.8)

        target_weights = []
        sibling_weights = []
        for event_index in range(1, 4):
            parameters = StdioServerParameters(
                command=sys.executable,
                args=[
                    "-m",
                    "neuron_graph_rag_mcp",
                    "--database",
                    str(database),
                    "--relation-feedback-evidence-quorum",
                    "3",
                    "--sibling-feedback-normalization",
                    "1.0",
                ],
            )
            async with (
                stdio_client(parameters) as (read_stream, write_stream),
                ClientSession(read_stream, write_stream) as session,
            ):
                await session.initialize()
                search = await session.call_tool(
                    "search",
                    {
                        "contract_version": CONTRACT_VERSION,
                        "query": "stabilization anchor",
                        "limit": 5,
                    },
                )
                arguments = {
                    "contract_version": CONTRACT_VERSION,
                    "idempotency_key": f"stdio-q3-s1-{event_index}",
                    "trace_id": search.structured_content["trace_id"],
                    "events": [
                        {"node_id": "target", "stage": "selected"},
                        {"node_id": "target", "stage": "validated"},
                        {"node_id": "target", "stage": "used"},
                    ],
                }
                source_use = await session.call_tool(
                    "record_source_use", arguments
                )
                retry = await session.call_tool("record_source_use", arguments)
                self.assertFalse(source_use.is_error)
                self.assertEqual(retry.structured_content, source_use.structured_content)
                feedback = source_use.structured_content["feedback"]
                self.assertEqual(feedback["evidence"][0]["count"], event_index)
                self.assertEqual(feedback["evidence"][0]["quorum"], 3)
                self.assertEqual(
                    feedback["evidence"][0]["activated"], event_index == 3
                )
                self.assertEqual(
                    len(feedback["reinforced_edges"]), 1 if event_index == 3 else 0
                )

            with NeuronGraphRAG(database) as engine:
                target_weights.append(
                    engine.store.edge("source", "target", "supports").weight
                )
                sibling_weights.append(
                    engine.store.edge("source", "sibling", "supports").weight
                )
                self.assertEqual(
                    engine.store.edge(
                        "other-source", "other-target", "isolated"
                    ).weight,
                    0.8,
                )

        self.assertEqual(target_weights[:2], [0.7, 0.7])
        self.assertEqual(sibling_weights[:2], [0.6, 0.6])
        self.assertGreater(target_weights[2], 0.7)
        self.assertLess(sibling_weights[2], 0.6)
        self.assertAlmostEqual(
            target_weights[2] - 0.7,
            0.6 - sibling_weights[2],
        )

    def test_invalid_cli_feedback_settings_do_not_create_database(self) -> None:
        invalid_values = (
            ("--relation-feedback-evidence-quorum", "0"),
            ("--relation-feedback-evidence-quorum", "1.5"),
            ("--sibling-feedback-normalization", "-0.1"),
            ("--sibling-feedback-normalization", "1.1"),
            ("--sibling-feedback-normalization", "nan"),
            ("--confirmation-decay-ratio", "0"),
            ("--confirmation-decay-ratio", "1"),
            ("--soft-start-feedback-ratio", "0"),
            ("--soft-start-feedback-ratio", "1"),
            ("--soft-start-feedback-ratio", "nan"),
            ("--soft-start-feedback-ratio", "inf"),
        )
        for index, (option, value) in enumerate(invalid_values):
            with self.subTest(option=option, value=value):
                database = Path(self.temporary.name) / f"invalid-{index}.sqlite"
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "neuron_graph_rag_mcp",
                        "--database",
                        str(database),
                        option,
                        value,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(option, result.stderr)
                self.assertFalse(database.exists())
                self.assertFalse(Path(f"{database}-wal").exists())
                self.assertFalse(Path(f"{database}-shm").exists())

        missing_pair = Path(self.temporary.name) / "missing-decay.sqlite"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "neuron_graph_rag_mcp",
                "--database",
                str(missing_pair),
                "--confirmed-outcome-reinforcement",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertFalse(missing_pair.exists())

        invalid_combinations = (
            ("--soft-start-feedback-reinforcement", "--confirmation-decay-ratio", "0.5"),
            ("--soft-start-feedback-ratio", "0.25"),
            ("--outcome-driven-feedback-deactivation",),
            (
                "--confirmed-outcome-reinforcement",
                "--soft-start-feedback-reinforcement",
                "--soft-start-feedback-ratio",
                "0.25",
                "--confirmation-decay-ratio",
                "0.5",
            ),
        )
        for index, options in enumerate(invalid_combinations):
            with self.subTest(options=options):
                database = Path(self.temporary.name) / f"invalid-combination-{index}.sqlite"
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "neuron_graph_rag_mcp",
                        "--database",
                        str(database),
                        *options,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertFalse(database.exists())


if __name__ == "__main__":
    unittest.main()
