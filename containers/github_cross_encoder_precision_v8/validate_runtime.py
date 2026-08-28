from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import sys
from pathlib import Path

import torch

RUN_ROOT = Path("/opt/ngr-v8/runtime")
EXPECTED_REGISTRY = Path("/opt/ngr-v8/expected-distributions.json")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _canonicalize_name(value: str) -> str:
    if not value or value != value.strip():
        raise RuntimeError("distribution Name must be a non-empty trimmed string")
    canonical = re.sub(r"[-_.]+", "-", value).lower()
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", canonical):
        raise RuntimeError("distribution canonical Name is malformed")
    return canonical


def _load_expected() -> tuple[dict[str, str], str]:
    raw = EXPECTED_REGISTRY.read_bytes()
    value = json.loads(raw.decode("utf-8", errors="strict"))
    if raw != (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")):
        raise RuntimeError("expected distribution registry must be canonical JSON")
    if set(value) != {"canonicalization", "distributions", "expected_count", "protocol_id"}:
        raise RuntimeError("expected distribution registry fields mismatch")
    if (
        value["canonicalization"] != "pep503-lowercase-replace-runs"
        or value["expected_count"] != 29
        or value["protocol_id"] != "github-ngr-cross-encoder-precision-v8"
    ):
        raise RuntimeError("expected distribution registry identity mismatch")
    expected: dict[str, str] = {}
    origins: dict[str, str] = {}
    for row in value["distributions"]:
        if set(row) != {"canonical_name", "version", "origin_class"}:
            raise RuntimeError("expected distribution row shape mismatch")
        name = _canonicalize_name(str(row["canonical_name"]))
        version = row["version"]
        origin = row["origin_class"]
        if name != row["canonical_name"] or not isinstance(version, str) or not version:
            raise RuntimeError("expected distribution identity mismatch")
        if origin not in {"ml-runtime-artifact", "image-toolchain"}:
            raise RuntimeError("expected distribution origin class mismatch")
        if name in expected:
            raise RuntimeError("duplicate expected canonical distribution name")
        expected[name] = version
        origins[name] = origin
    if len(expected) != 29 or sum(v == "image-toolchain" for v in origins.values()) != 3:
        raise RuntimeError("expected distribution registry count mismatch")
    return expected, hashlib.sha256(raw).hexdigest()


def _actual_distributions() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    names: set[str] = set()
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.metadata.get("Version")
        if not isinstance(name, str) or not isinstance(version, str) or not version or version != version.strip():
            raise RuntimeError("installed distribution has empty or malformed Name/Version")
        canonical = _canonicalize_name(name)
        if canonical in names:
            raise RuntimeError("duplicate installed canonical distribution name")
        names.add(canonical)
        rows.append({"canonical_name": canonical, "name": name, "version": version})
    rows.sort(key=lambda row: row["canonical_name"])
    return rows


def _network_is_disabled() -> bool:
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=0.25):
            return False
    except OSError:
        return True


def build_attestation() -> dict[str, object]:
    expected, expected_sha256 = _load_expected()
    distributions = _actual_distributions()
    actual = {row["canonical_name"]: row["version"] for row in distributions}
    if actual != expected:
        extra = sorted(set(actual) - set(expected))
        missing = sorted(set(expected) - set(actual))
        mismatched = sorted(
            name for name in set(actual) & set(expected) if actual[name] != expected[name]
        )
        raise RuntimeError(
            "exact installed distribution mismatch: "
            f"extra={extra!r}, missing={missing!r}, version_mismatch={mismatched!r}"
        )
    if platform.python_implementation() != "CPython" or platform.python_version() != "3.11.15":
        raise RuntimeError("CPython version mismatch")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Linux amd64 platform mismatch")
    if torch.version.cuda is not None:
        raise RuntimeError("CUDA-enabled torch is forbidden")
    tensor = torch.tensor([-2.0, 3.0], dtype=torch.float32, device="cpu").add(1)
    if tensor.dtype != torch.float32 or tensor.device.type != "cpu" or tensor.tolist() != [-1.0, 4.0]:
        raise RuntimeError("synthetic CPU float32 tensor probe mismatch")

    installed = set(actual)
    forbidden = sorted(
        name for name in installed if name == "triton" or name.startswith("nvidia-")
    )
    if forbidden:
        raise RuntimeError(f"forbidden accelerator distributions: {forbidden!r}")
    if not _network_is_disabled():
        raise RuntimeError("attestation container must run with outbound network disabled")
    if str(RUN_ROOT).startswith("/mnt/"):
        raise RuntimeError("runtime root must not use a Windows bind mount")

    RUN_ROOT.mkdir(parents=True, exist_ok=False)
    probe = RUN_ROOT / "exclusive-create.txt"
    descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, b"synthetic-only\n")
    finally:
        os.close(descriptor)
    if probe.read_bytes() != b"synthetic-only\n":
        raise RuntimeError("container filesystem probe mismatch")

    return {
        "architecture": "amd64",
        "distributions": distributions,
        "expected_distribution_registry_sha256": expected_sha256,
        "filesystem_probe": "exclusive-create",
        "forbidden_distributions": [],
        "model_forward_inference_count": 0,
        "network": "disabled",
        "observed_result_count": 0,
        "os": "linux",
        "python": {
            "abi": "cp311",
            "implementation": "CPython",
            "version": platform.python_version(),
        },
        "registered_query_count": 0,
        "synthetic_tensor_probe": {
            "device": "cpu",
            "dtype": "float32",
            "output": [-1.0, 4.0],
        },
        "torch_cuda": None,
    }


def main() -> int:
    sys.stdout.buffer.write(canonical_json_bytes(build_attestation()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
