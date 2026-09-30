from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MCP_AVAILABLE = importlib.util.find_spec("mcp") is not None


@unittest.skipUnless(MCP_AVAILABLE, "optional MCP SDK is not installed")
class NativeRegistrationTest(unittest.TestCase):
    def setUp(self):
        from neuron_graph_rag_mcp import native_registration
        self.registration = native_registration
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        home = patch.object(native_registration, "ngr_home", return_value=self.root / ".ngr")
        home.start()
        self.addCleanup(home.stop)

    def test_desktop_resolves_existing_store_and_requires_choice_when_ambiguous(self):
        regular = self.root / "roaming/Claude/claude_desktop_config.json"
        store = self.root / "local/Packages/Claude_example/LocalCache/Roaming/Claude/claude_desktop_config.json"
        store.parent.mkdir(parents=True)
        store.write_text("{}")
        environment = {"APPDATA": str(self.root / "roaming"), "LOCALAPPDATA": str(self.root / "local")}
        candidates = self.registration.desktop_candidates(home=self.root, environ=environment)
        self.assertEqual(candidates, [store])
        regular.parent.mkdir(parents=True)
        regular.write_text("{}")
        candidates = self.registration.desktop_candidates(home=self.root, environ=environment)
        self.assertEqual(candidates, [regular, store])
        with patch.object(self.registration, "desktop_candidates", return_value=candidates), patch("builtins.input", return_value="2"):
            self.assertEqual(self.registration.select_desktop_config(), store)

    def test_json_entry_preserves_other_private_values_and_requires_replacement_choice(self):
        path = self.root / "desktop.json"
        document = {"preferences": {"locale": "ja"}, "mcpServers": {
            "other": {"env": {"PRIVATE": "do not display"}}, "ngr-shared": {"command": "old"}}}
        path.write_text(json.dumps(document), encoding="utf-8")
        original = path.read_bytes()
        output = io.StringIO()
        with patch("builtins.input", return_value="n"), contextlib.redirect_stdout(output):
            self.registration.register_json("desktop", path)
        self.assertEqual(path.read_bytes(), original)
        with patch("builtins.input", return_value="y"), contextlib.redirect_stdout(output):
            self.registration.register_json("desktop", path)
        result = json.loads(path.read_text())
        self.assertEqual(result["preferences"], document["preferences"])
        self.assertEqual(result["mcpServers"]["other"], document["mcpServers"]["other"])
        self.assertEqual(result["mcpServers"]["ngr-shared"], {"command": sys.executable, "args": ["--shared"]})
        self.assertNotIn("do not display", output.getvalue())
        self.assertEqual(next((self.root / ".ngr/backups/clients").iterdir()).read_bytes(), original)

    def test_codex_forwards_environment_names_and_preserves_all_other_settings(self):
        import tomllib
        path = self.root / "config.toml"
        original = 'model = "kept"\n[mcp_servers.other]\ncommand = "private command"\n'
        path.write_text(original)
        calls = []
        def run(command, **kwargs):
            self.assertTrue(kwargs["capture_output"])
            calls.append(command)
            backups = list((self.root / ".ngr/backups/clients").iterdir())
            self.assertEqual(backups[0].read_text(), original)
            path.write_text(original + '\n[mcp_servers.ngr-shared]\ncommand = ' + json.dumps(sys.executable) + '\nargs = ["--shared"]\n')
            return subprocess.CompletedProcess(command, 0, "SECRET OUTPUT", "")
        output = io.StringIO()
        with patch.object(self.registration, "_codex_command", return_value=["codex.exe"]), patch.object(self.registration, "_config_path", return_value=path), patch.object(self.registration.subprocess, "run", side_effect=run), patch("builtins.input", return_value="y"), contextlib.redirect_stdout(output):
            self.registration.register_codex()
        result = tomllib.loads(path.read_text())
        self.assertEqual(result["model"], "kept")
        self.assertEqual(result["mcp_servers"]["other"]["command"], "private command")
        names = result["mcp_servers"]["ngr-shared"]["env_vars"]
        self.assertIn("NGR_MCP_HTTP_BEARER_TOKEN", names)
        self.assertIn("NGR_DATABASE", names)
        self.assertNotIn("SECRET OUTPUT", output.getvalue())
        self.assertEqual(calls[0][-2:], [sys.executable, "--shared"])

    def test_codex_failed_registration_restores_exact_backup(self):
        path = self.root / "config.toml"
        original = b'[mcp_servers.ngr-shared]\ncommand = "old"\n'
        path.write_bytes(original)
        def run(command, **kwargs):
            path.write_bytes(b"partial")
            return subprocess.CompletedProcess(command, 1, "secret", "")
        with patch.object(self.registration, "_codex_command", return_value=["codex.exe"]), patch.object(self.registration, "_config_path", return_value=path), patch.object(self.registration.subprocess, "run", side_effect=run), patch("builtins.input", return_value="y"), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(OSError):
                self.registration.register_codex()
        self.assertEqual(path.read_bytes(), original)

    def test_concurrent_native_configuration_change_refuses_write(self):
        path = self.root / "client.json"
        path.write_bytes(b"{}")
        def choice(*args):
            path.write_bytes(b'{"new":"edit"}')
            return True
        with patch.object(self.registration, "_choice", side_effect=choice):
            with self.assertRaises(RuntimeError):
                self.registration.register_json("desktop", path)
        self.assertEqual(path.read_bytes(), b'{"new":"edit"}')


if __name__ == "__main__":
    unittest.main()
