from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neuron_graph_rag import NeuronGraphRAG
from neuron_graph_rag.user_config import write_config

MCP_AVAILABLE = importlib.util.find_spec("mcp") is not None


@unittest.skipUnless(MCP_AVAILABLE, "optional MCP SDK is not installed")
class SharedHomeMCPTest(unittest.IsolatedAsyncioTestCase):
    TOKEN = "shared_home_mcp_test_token_0123456789abcdef"

    def test_proxy_start_and_stop_resolve_the_same_central_settings(self):
        from neuron_graph_rag_mcp import shared_proxy
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory).resolve()
            database = home / ".ngr/db/knowledge.db"
            write_config({"port": 8912}, home=home)
            seen = []
            def ensure(args, token, db, config):
                seen.append((args.port, db, config))
            async def run(port, token):
                self.assertEqual(port, 8912)
            def stop(port, token, db):
                seen.append((port, db, {}))
            with patch("pathlib.Path.home", return_value=home), patch.dict(os.environ, {"NGR_MCP_HTTP_BEARER_TOKEN": self.TOKEN}, clear=True), patch.object(shared_proxy, "_ensure_service", side_effect=ensure), patch.object(shared_proxy, "_run_proxy", side_effect=run), patch.object(shared_proxy, "_stop_service", side_effect=stop):
                shared_proxy.main([])
                shared_proxy.stop_main([])
            self.assertEqual(seen, [(8912, database, {}), (8912, database, {})])
            self.assertTrue(database.parent.is_dir())

    FEEDBACK = {"soft_start_feedback_reinforcement": True, "soft_start_feedback_ratio": 0.25,
                "confirmation_decay_ratio": 0.5, "outcome_driven_feedback_deactivation": True,
                "sibling_feedback_normalization": 1.0}

    def test_proxy_identity_carries_config_feedback_policy(self):
        from neuron_graph_rag_mcp import shared_proxy
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory).resolve()
            write_config({"port": 8912, **self.FEEDBACK}, home=home)
            seen = []
            def ensure(args, token, db, config):
                seen.append(config)
            async def run(port, token):
                pass
            with patch("pathlib.Path.home", return_value=home), patch.dict(os.environ, {"NGR_MCP_HTTP_BEARER_TOKEN": self.TOKEN}, clear=True), patch.object(shared_proxy, "_ensure_service", side_effect=ensure), patch.object(shared_proxy, "_run_proxy", side_effect=run):
                shared_proxy.main([])
            self.assertEqual(seen, [{"feedback": self.FEEDBACK}])

    async def test_http_service_builds_engine_config_and_identity_from_config(self):
        from mcp import types
        from neuron_graph_rag_mcp import http_server
        from neuron_graph_rag_mcp.server import CONTRACT_VERSION

        for document, identity in (({}, {}), (self.FEEDBACK, {"feedback": self.FEEDBACK})):
            with self.subTest(document=document), tempfile.TemporaryDirectory() as directory:
                home = Path(directory).resolve()
                write_config({"port": 8913, **document}, home=home)
                captured = {}
                def create(database, **kwargs):
                    captured.update(kwargs, database=database)
                    raise SystemExit(0)
                with patch("pathlib.Path.home", return_value=home), patch.dict(os.environ, {"NGR_MCP_HTTP_BEARER_TOKEN": self.TOKEN}, clear=True), patch.object(http_server, "create_http_app", side_effect=create):
                    with self.assertRaises(SystemExit):
                        http_server.main([])
                self.assertEqual(captured["service_config"], identity)
                database = captured.pop("database")
                with NeuronGraphRAG(database) as engine:
                    engine.add_document("decision", "cache invalidation decision")
                    engine.add_document("implementation", "implementation detail")
                    engine.add_edge("decision", "implementation", "implemented_by", weight=0.7)
                app = http_server.create_http_app(database, **captured)
                adapter = app.state.ngr_adapter
                try:
                    result = await adapter.call_tool(None, types.CallToolRequestParams(name="search", arguments={
                        "contract_version": CONTRACT_VERSION, "query": "cache invalidation"}))
                finally:
                    adapter.close()
                self.assertFalse(result.is_error)
                provenance = result.structured_content["effective_config_provenance"]
                feedback = provenance["effective_config"]["feedback"]
                if document:
                    self.assertEqual(provenance["search_surface"], "relation")
                    for name, value in self.FEEDBACK.items():
                        self.assertEqual(feedback[name], value)
                else:
                    self.assertEqual(provenance["search_surface"], "combined")
                    self.assertNotIn("soft_start_feedback_reinforcement", feedback)
                    self.assertNotIn("outcome_driven_feedback_deactivation", feedback)
                    self.assertEqual(feedback["sibling_feedback_normalization"], 0.0)

    async def test_two_clients_use_central_database_from_an_unrelated_working_directory(self):
        import httpx2
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        from neuron_graph_rag_mcp.server import CONTRACT_VERSION

        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory).resolve()
            unrelated = home / "unrelated"
            unrelated.mkdir()
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            write_config({"port": port}, home=home)
            database = home / ".ngr/db/knowledge.db"
            database.parent.mkdir()
            with NeuronGraphRAG(database) as engine:
                engine.add_document("central", "shared configuration migration")
            environment = {key: value for key, value in os.environ.items() if not key.startswith("NGR_")}
            environment.update(HOME=str(home), USERPROFILE=str(home),
                               NGR_MCP_HTTP_BEARER_TOKEN=self.TOKEN,
                               PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
            process = await asyncio.create_subprocess_exec(sys.executable, "-m", "neuron_graph_rag_mcp", "--http",
                cwd=unrelated, env=environment, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try:
                await asyncio.wait_for(process.stdout.readline(), 20)
                url = f"http://127.0.0.1:{port}/mcp/"
                async with httpx2.AsyncClient(headers={"Authorization": f"Bearer {self.TOKEN}"}) as first_http, httpx2.AsyncClient(headers={"Authorization": f"Bearer {self.TOKEN}"}) as second_http:
                    for _ in range(80):
                        try:
                            identity = (await first_http.get(f"http://127.0.0.1:{port}/_ngr/identity")).json()
                            break
                        except httpx2.ConnectError:
                            await asyncio.sleep(0.1)
                    else:
                        self.fail("Central HTTP service did not become ready")
                    self.assertEqual(identity["database"], str(database))
                    self.assertEqual(identity["config"], {})
                    async with streamable_http_client(url, http_client=first_http) as (read1, write1), streamable_http_client(url, http_client=second_http) as (read2, write2):
                        async with ClientSession(read1, write1) as first, ClientSession(read2, write2) as second:
                            await first.initialize()
                            await second.initialize()
                            responses = await asyncio.gather(*[client.call_tool("search", {"contract_version": CONTRACT_VERSION, "query": "shared configuration"}) for client in (first, second)])
                            for response in responses:
                                self.assertFalse(response.is_error)
                                self.assertIn("central", str(response.structured_content))
                stopped = await asyncio.create_subprocess_exec(sys.executable, "-m", "neuron_graph_rag_mcp", "--stop",
                    cwd=unrelated, env=environment, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                await asyncio.wait_for(stopped.communicate(), 30)
                self.assertEqual(stopped.returncode, 0)
                await asyncio.wait_for(process.communicate(), 20)
                self.assertEqual(process.returncode, 0)
            finally:
                if process.returncode is None:
                    process.terminate()
                    await asyncio.wait_for(process.communicate(), 20)


if __name__ == "__main__":
    unittest.main()
