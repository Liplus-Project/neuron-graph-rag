"""Release API and streamed installer integrity tests; no network required."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from neuron_graph_rag_mcp.verified_updates import (
        DownloadCancelled, InstalledBuild, UpdateError, candidate_from_release,
        check_for_update, download_candidate, installed_build,
    )
except ModuleNotFoundError as error:
    if error.name not in {"mcp", "httpx2", "starlette", "uvicorn"}:
        raise
    MCP_AVAILABLE = False
else:
    MCP_AVAILABLE = True


PAYLOAD = b"test installer bytes"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
FILENAME = "NGR-1.2.3-windows-x64-cpu-setup.exe"
URL = f"https://github.com/Liplus-Project/neuron-graph-rag/releases/download/v1.2.3/{FILENAME}"


def release(**overrides):
    result = {
        "tag_name": "v1.2.3", "draft": False, "prerelease": False,
        "published_at": "2026-09-26T00:00:00Z",
        "html_url": "https://github.com/Liplus-Project/neuron-graph-rag/releases/tag/v1.2.3",
        "assets": [{"name": FILENAME, "size": len(PAYLOAD),
                    "digest": f"sha256:{DIGEST}", "browser_download_url": URL}],
    }
    result.update(overrides)
    return result


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, length: str | None = None):
        super().__init__(body)
        self.headers = {"Content-Length": length or str(len(body))}


@unittest.skipUnless(MCP_AVAILABLE, "optional MCP SDK is not installed")
class VerifiedUpdatesTests(unittest.TestCase):
    def setUp(self):
        self.build = InstalledBuild("1.0.0", "cpu")

    def test_installed_manifest_identifies_flavor_without_touching_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "package-manifest.json").write_text(
                json.dumps({"schema": "ngr.windows-package/v1", "version": "1.0.0",
                            "flavor": "cuda"}),
                encoding="utf-8")
            self.assertEqual(installed_build(root / "NGR.exe"), InstalledBuild("1.0.0", "cuda"))

    def test_only_newer_published_stable_compatible_release_is_offered(self):
        self.assertEqual(candidate_from_release(release(), self.build).filename, FILENAME)
        self.assertIsNone(candidate_from_release(release(), InstalledBuild("1.2.3", "cpu")))
        self.assertIsNone(candidate_from_release(release(), InstalledBuild("1.3.0", "cpu")))
        self.assertIsNone(candidate_from_release(release(draft=True), self.build))
        self.assertIsNone(candidate_from_release(release(prerelease=True), self.build))
        self.assertIsNone(candidate_from_release(release(published_at=None), self.build))
        self.assertIsNone(candidate_from_release(release(), InstalledBuild("1.0.0", None)))

    def test_bad_asset_is_rejected(self):
        for changes in (
            {"digest": None}, {"size": 0}, {"size": 16 * 1024**3 + 1},
            {"browser_download_url": "https://example.org/evil.exe"},
        ):
            with self.subTest(changes=changes):
                item = release()
                item["assets"][0].update(changes)
                with self.assertRaises(UpdateError):
                    candidate_from_release(item, self.build)
        with self.assertRaises(UpdateError):
            candidate_from_release(release(assets=[]), self.build)

    def test_api_failure_does_not_block_service(self):
        with patch("neuron_graph_rag_mcp.verified_updates._read_json", return_value=release()):
            self.assertEqual(check_for_update(self.build).version, "1.2.3")
        with patch("neuron_graph_rag_mcp.verified_updates._read_json", side_effect=UpdateError("offline")):
            with self.assertRaisesRegex(UpdateError, "offline"):
                check_for_update(self.build)

    def test_verified_download_and_hash_mismatch(self):
        candidate = candidate_from_release(release(), self.build)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with patch("neuron_graph_rag_mcp.verified_updates.urllib.request.urlopen",
                       return_value=FakeResponse(PAYLOAD)):
                downloaded = download_candidate(candidate, path)
            self.assertEqual(downloaded.read_bytes(), PAYLOAD)
            downloaded.unlink()
            with patch("neuron_graph_rag_mcp.verified_updates.urllib.request.urlopen",
                       return_value=FakeResponse(b"broken", str(len(PAYLOAD)))):
                with self.assertRaises(UpdateError):
                    download_candidate(candidate, path)
            self.assertEqual(list(path.iterdir()), [])

    def test_cancel_removes_partial_file(self):
        candidate = candidate_from_release(release(), self.build)
        cancelled = threading.Event()

        class CancelAfterFirstRead(FakeResponse):
            def read(self, size=-1):
                data = super().read(size)
                if data:
                    cancelled.set()
                return data

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with patch("neuron_graph_rag_mcp.verified_updates.urllib.request.urlopen",
                       return_value=CancelAfterFirstRead(PAYLOAD)):
                with self.assertRaises(DownloadCancelled):
                    download_candidate(candidate, path, cancelled)
            self.assertEqual(list(path.iterdir()), [])

    def test_existing_installer_is_never_replaced(self):
        candidate = candidate_from_release(release(), self.build)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            existing = path / FILENAME
            existing.write_bytes(b"keep existing file")
            with self.assertRaises(UpdateError):
                download_candidate(candidate, path)
            self.assertEqual(existing.read_bytes(), b"keep existing file")


if __name__ == "__main__":
    unittest.main()
