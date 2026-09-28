"""Exercise the manual Windows artifact verification gate with small fixtures."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "packaging" / "windows" / "verify_artifacts.py"


class WindowsPackageArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.version_file = self.root / "release-version.txt"
        self.version_file.write_text("0.2.1\n", encoding="utf-8")
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir()
        for flavor in ("cpu", "cuda"):
            directory = self.artifacts / f"ngr-windows-{flavor}"
            directory.mkdir()
            name = f"NGR-0.2.1-windows-x64-{flavor}-setup.exe"
            content = f"installer-{flavor}".encode()
            (directory / name).write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            (directory / f"{name}.sha256").write_text(f"{digest} *{name}\n", encoding="ascii")
            (directory / "package-manifest.json").write_text(json.dumps({
                "schema": "ngr.windows-package/v1", "version": "0.2.1", "flavor": flavor,
                "setup_file": name, "sha256": digest, "size": len(content),
            }), encoding="utf-8")

    def _verify(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(SCRIPT), str(self.artifacts),
                               str(self.version_file)], capture_output=True, text=True)

    def test_accepts_both_flavors_from_one_run(self) -> None:
        result = self._verify()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("VERIFIED_CPU=", result.stdout)
        self.assertIn("VERIFIED_CUDA=", result.stdout)

    def test_rejects_tampered_installer(self) -> None:
        installer = self.artifacts / "ngr-windows-cuda" / "NGR-0.2.1-windows-x64-cuda-setup.exe"
        installer.write_bytes(b"tampered")
        result = self._verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("installer digest or size mismatch: cuda", result.stderr)

    def test_rejects_missing_flavor(self) -> None:
        directory = self.artifacts / "ngr-windows-cuda"
        for path in directory.iterdir():
            path.unlink()
        directory.rmdir()
        result = self._verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected both package artifacts", result.stderr)


if __name__ == "__main__":
    unittest.main()
