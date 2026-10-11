"""Check the dedicated icon's Win32 ownership and packaged ICO directory."""

from __future__ import annotations

import ctypes
import os
import struct
import unittest
from ctypes import wintypes
from pathlib import Path
from unittest.mock import Mock, patch

try:
    from neuron_graph_rag_mcp.tray_controller import _tray_icon
except ModuleNotFoundError:
    _tray_icon = None


@unittest.skipIf(_tray_icon is None, "requires the optional MCP extra")
class TrayIconTest(unittest.TestCase):
    def test_ico_sizes_and_alpha_payloads(self) -> None:
        data = (Path(__file__).resolve().parents[1] / "src" /
                "neuron_graph_rag_mcp/assets/ngr.ico").read_bytes()
        reserved, kind, count = struct.unpack_from("<HHH", data)
        self.assertEqual((reserved, kind, count), (0, 1, 9))
        sizes = []
        for index in range(count):
            width, height, _, _, _, _, length, offset = struct.unpack_from(
                "<BBBBHHII", data, 6 + 16 * index)
            sizes.append((width or 256, height or 256))
            payload = data[offset:offset + length]
            self.assertEqual(len(payload), length)
            self.assertEqual(payload[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(payload[25], 6)  # PNG RGBA color type.
        self.assertEqual(sizes, [(n, n) for n in (16, 20, 24, 32, 40, 48, 64, 128, 256)])

    def test_pointer_width_and_release_on_success_and_exception(self) -> None:
        for failing in (False, True):
            with self.subTest(failing=failing):
                api = Mock()
                handle = 0x123456789 if ctypes.sizeof(ctypes.c_void_p) == 8 else 0x1234567
                api.LoadImageW.return_value = handle
                api.GetSystemMetrics.side_effect = [20, 20]
                with patch("ctypes.WinDLL", return_value=api, create=True):
                    try:
                        with _tray_icon() as loaded:
                            self.assertEqual(loaded, handle)
                            api.DestroyIcon.assert_not_called()
                            if failing:
                                raise RuntimeError("startup failed")
                    except RuntimeError:
                        self.assertTrue(failing)
                self.assertIs(api.LoadImageW.restype, wintypes.HANDLE)
                args = api.LoadImageW.call_args.args
                self.assertTrue(Path(args[1]).is_file())
                self.assertEqual(args[2:], (1, 20, 20, 0x10))
                api.DestroyIcon.assert_called_once_with(handle)

    def test_load_failure_does_not_destroy_null_handle(self) -> None:
        api = Mock()
        api.LoadImageW.return_value = None
        api.GetSystemMetrics.return_value = 16
        with patch("ctypes.WinDLL", return_value=api, create=True), \
                patch("ctypes.get_last_error", return_value=2, create=True), \
                patch("ctypes.WinError", return_value=OSError("missing icon"), create=True):
            with self.assertRaises(OSError):
                with _tray_icon():
                    self.fail("LoadImageW failure must not yield a handle")
        api.DestroyIcon.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "requires Windows icon APIs")
    def test_windows_loads_packaged_icon(self) -> None:
        with _tray_icon() as handle:
            self.assertTrue(handle)


if __name__ == "__main__":
    unittest.main()
