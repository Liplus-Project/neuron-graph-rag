"""Verify both installers downloaded from one Windows package workflow run."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


def verify(root: Path, version_file: Path) -> None:
    version = version_file.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("invalid release version")
    flavors = {"cpu", "cuda"}
    expected_directories = {f"ngr-windows-{flavor}" for flavor in flavors}
    actual_directories = {path.name for path in root.iterdir() if path.is_dir()}
    if actual_directories != expected_directories:
        raise ValueError(f"expected both package artifacts: {sorted(actual_directories)}")

    for flavor in sorted(flavors):
        directory = root / f"ngr-windows-{flavor}"
        manifest = json.loads((directory / "package-manifest.json").read_text(encoding="utf-8-sig"))
        name = f"NGR-{version}-windows-x64-{flavor}-setup.exe"
        if (manifest.get("schema"), manifest.get("version"), manifest.get("flavor"),
                manifest.get("setup_file")) != ("ngr.windows-package/v1", version, flavor, name):
            raise ValueError(f"package manifest mismatch: {flavor}")
        installer = directory / name
        with installer.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if manifest.get("sha256") != digest or manifest.get("size") != installer.stat().st_size:
            raise ValueError(f"installer digest or size mismatch: {flavor}")
        sidecar = (directory / f"{name}.sha256").read_text(encoding="ascii").strip()
        if sidecar != f"{digest} *{name}":
            raise ValueError(f"SHA-256 sidecar mismatch: {flavor}")
        print(f"VERIFIED_{flavor.upper()}={name} sha256:{digest}")


if __name__ == "__main__":
    verify(Path(sys.argv[1]), Path(sys.argv[2]))
