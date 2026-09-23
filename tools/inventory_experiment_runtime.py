"""Write an exclusive installed-distribution inventory for a Python environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path


def inventory() -> dict:
    distributions = []
    for distribution in importlib.metadata.distributions():
        metadata = distribution.read_text("METADATA")
        distributions.append({
            "name": str(distribution.metadata["Name"]).lower(),
            "version": distribution.version,
            "metadata_sha256": hashlib.sha256(metadata.encode("utf-8")).hexdigest() if metadata is not None else None,
        })
    distributions.sort(key=lambda row: (row["name"], row["version"], row["metadata_sha256"] or ""))
    configuration = Path(sys.prefix) / "pyvenv.cfg"
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    dist_info_directories = sorted(path.name for path in site_packages.glob("*.dist-info") if path.is_dir())
    return {
        "schema_version": 1,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "environment_root": sys.prefix,
        "pyvenv_cfg_sha256": hashlib.sha256(configuration.read_bytes()).hexdigest(),
        "distribution_count": len(distributions),
        "distributions": distributions,
        "dist_info_directory_count": len(dist_info_directories),
        "dist_info_directories": dist_info_directories,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
    print(json.dumps({"distribution_count": result["distribution_count"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
