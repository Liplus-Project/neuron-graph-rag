"""The shared service must relaunch the frozen entry, never Python -m."""

import sys
import unittest
from unittest.mock import patch

from neuron_graph_rag_mcp.shared_proxy import _entry_command


class WindowsPackageEntryTest(unittest.TestCase):
    def test_frozen_children_use_the_executable(self) -> None:
        with patch.object(sys, "frozen", True, create=True):
            self.assertEqual(_entry_command("--http", "--port", "8765"),
                             [sys.executable, "--http", "--port", "8765"])
            self.assertEqual(_entry_command("--tray-controller"),
                             [sys.executable, "--tray-controller"])

    def test_source_children_use_modules(self) -> None:
        with patch.object(sys, "frozen", False, create=True):
            self.assertEqual(_entry_command("--http"),
                             [sys.executable, "-m", "neuron_graph_rag_mcp", "--http"])
            self.assertEqual(_entry_command("--tray-controller"),
                             [sys.executable, "-m", "neuron_graph_rag_mcp.tray_controller"])


if __name__ == "__main__":
    unittest.main()
