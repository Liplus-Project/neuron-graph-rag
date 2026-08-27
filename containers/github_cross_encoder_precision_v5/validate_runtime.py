from __future__ import annotations

import importlib.metadata
import json
import platform
import socket
from pathlib import Path

import torch

EXPECTED = {
    "certifi": "2026.7.22",
    "charset-normalizer": "3.5.1",
    "filelock": "3.32.4",
    "fsspec": "2026.7.0",
    "hf-xet": "1.6.0",
    "huggingface-hub": "0.36.2",
    "idna": "3.19",
    "jinja2": "3.1.6",
    "markupsafe": "3.0.3",
    "mpmath": "1.3.0",
    "networkx": "3.6.1",
    "numpy": "2.4.6",
    "packaging": "26.3",
    "psutil": "6.0.0",
    "pyyaml": "6.0.3",
    "regex": "2026.7.19",
    "requests": "2.34.2",
    "safetensors": "0.4.5",
    "sentencepiece": "0.2.0",
    "sympy": "1.14.0",
    "tokenizers": "0.19.1",
    "torch": "2.4.1+cpu",
    "tqdm": "4.70.0",
    "transformers": "4.44.2",
    "typing-extensions": "4.16.0",
    "urllib3": "2.7.0",
}
RUN_ROOT = Path("/opt/ngr-v5/runtime")


def _network_is_disabled() -> bool:
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=0.25):
            return False
    except OSError:
        return True


def main() -> int:
    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    if versions != EXPECTED:
        raise RuntimeError(f"dependency version mismatch: {versions!r}")
    if platform.python_version() != "3.11.15":
        raise RuntimeError("Python version mismatch")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Linux amd64 platform mismatch")
    if torch.version.cuda is not None:
        raise RuntimeError("CUDA-enabled torch is forbidden")
    if torch.tensor([-2.0, 3.0], dtype=torch.float32).add(1).tolist() != [-1.0, 4.0]:
        raise RuntimeError("synthetic CPU tensor probe mismatch")

    installed = {
        dist.metadata["Name"].lower() for dist in importlib.metadata.distributions()
    }
    forbidden = sorted(
        name for name in installed if name == "triton" or name.startswith("nvidia-")
    )
    if forbidden:
        raise RuntimeError(f"forbidden accelerator distributions: {forbidden!r}")
    if not _network_is_disabled():
        raise RuntimeError("validation container must run with network disabled")
    if str(RUN_ROOT).startswith("/mnt/"):
        raise RuntimeError("runtime root must not use a Windows bind mount")

    RUN_ROOT.mkdir(parents=True, exist_ok=False)
    probe = RUN_ROOT / "synthetic-probe.txt"
    with probe.open("x", encoding="utf-8") as handle:
        handle.write("synthetic-only\n")
    if probe.read_text(encoding="utf-8") != "synthetic-only\n":
        raise RuntimeError("container filesystem probe mismatch")

    print(
        json.dumps(
            {
                "architecture": "amd64",
                "cuda": None,
                "filesystem_probe": "exclusive-create",
                "model_forward_count": 0,
                "network": "disabled",
                "os": "linux",
                "python_version": platform.python_version(),
                "registered_query_execution_count": 0,
                "versions": versions,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
