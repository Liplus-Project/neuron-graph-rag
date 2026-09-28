"""Exercise the installed Windows package with no Python on the child PATH."""

from __future__ import annotations

import http.client
import hashlib
import json
import os
import queue
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


def _assert_licenses(install: Path, flavor: str) -> None:
    licenses = install / "licenses"
    readme = (licenses / "README.txt").read_text(encoding="utf-8")
    assert "model weights are not included" in readme, readme
    manifest = json.loads((licenses / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "ngr.windows-licenses/v1"
    assert manifest["flavor"] == flavor and manifest["models_bundled"] is False
    assert any(name.lower().startswith("python3") for name in manifest["python_runtime_files"])
    runtime_prefixes = ("msvcp", "vcruntime", "concrt", "vcomp", "ucrtbase", "api-ms-win-crt-")
    actual_runtime = {path.relative_to(install).as_posix().lower(): path for path in install.rglob("*.dll")
                      if path.name.lower().startswith(runtime_prefixes)}
    vc = manifest["vc_runtime"]
    assert vc["visual_studio_release"] == "2022"
    edition = vc["visual_studio_edition"]
    assert edition == "Community"
    assert vc["edition_terms"].startswith("https://visualstudio.microsoft.com/license-terms/vs2022-ga-")
    assert vc["redist_list"] == f"https://learn.microsoft.com/en-us/visualstudio/releases/{vc['visual_studio_release']}/redistribution"
    expected_runtime = {f"_internal/{name}" for name in
                        ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll")}
    recorded_runtime = {item["path"]: item for item in vc["files"]}
    assert set(actual_runtime) == set(recorded_runtime) == expected_runtime, (
        sorted(actual_runtime), sorted(recorded_runtime), sorted(expected_runtime))
    for path, item in recorded_runtime.items():
        assert item["file_version"] and "VC\\Redist\\MSVC\\" in item["source"]
        assert hashlib.sha256(actual_runtime[path].read_bytes()).hexdigest() == item["sha256"]
    distributions = {item["name"]: item for item in manifest["distributions"]}
    required = {"pyinstaller", "mcp", "httpx2", "uvicorn", "starlette", "numpy",
                "onnxruntime", "tokenizers"}
    if flavor == "cuda":
        required.update(("torch", "transformers", "safetensors"))
    else:
        assert "torch" not in distributions and "transformers" not in distributions
    assert required <= distributions.keys(), sorted(required - distributions.keys())
    assert distributions["numpy"]["version"] == "1.26.4"
    assert all(item["version"] and item["documents"] for item in distributions.values())
    documents = {item["path"]: item for item in manifest["documents"]}
    assert {"NGR-LICENSE.txt", "NGR-NOTICE.txt", "Python-LICENSE.txt",
            "InnoSetup-LICENSE.txt", "Microsoft-VC-Runtime-LICENSE.txt", "Setup-LICENSE.txt"} <= documents.keys()
    setup_terms = (licenses / "Setup-LICENSE.txt").read_text(encoding="utf-8")
    assert "MICROSOFT VISUAL C++ RUNTIME TERMS" in setup_terms and "NGR LICENSE" in setup_terms
    assert "This installer includes four Microsoft Visual C++ runtime DLLs" in setup_terms
    assert vc["edition_terms"] in readme and vc["redist_list"] in readme
    assert any("ONNXRuntime-ThirdPartyNotices.txt" in path for path in documents)
    assert any("Tokenizers-Rust-ThirdPartyNotices.txt" in path for path in documents)
    if flavor == "cuda":
        assert any("PyTorch-NOTICE.txt" in path for path in documents)
    for name, record in documents.items():
        path = licenses / name
        assert path.resolve().is_relative_to(licenses.resolve()), name
        content = path.read_bytes()
        assert content.strip() and b"\x00" not in content, name
        assert hashlib.sha256(content).hexdigest() == record["sha256"], name
    native = {item["path"]: item for item in manifest["native_binaries"]}
    assert len(native) == len(manifest["native_binaries"])
    assert all(item["source_distribution"] == "numpy==1.26.4" for item in native.values())
    numpy_entries = [item for item in native.values() if item["source_distribution"] == "numpy==1.26.4"]
    assert len(numpy_entries) == 1, numpy_entries
    numpy_entry = numpy_entries[0]
    assert numpy_entry["source_record"].startswith("numpy.libs/libopenblas")
    assert (install / numpy_entry["path"]).is_file()
    actual_openblas = [path for path in install.rglob("*.dll") if path.name.lower().startswith("libopenblas")]
    assert actual_openblas == [install / numpy_entry["path"]], actual_openblas
    for item in native.values():
        assert hashlib.sha256((install / item["path"]).read_bytes()).hexdigest() == item["sha256"]
        assert set(item["license_documents"]) <= documents.keys()
    numpy_license = (licenses / numpy_entry["license_documents"][0]).read_text(encoding="utf-8")
    for component in ("Name: OpenBLAS", "Name: LAPACK", "Name: GCC runtime library",
                      "Name: libquadmath", "GCC RUNTIME LIBRARY EXCEPTION"):
        assert component in numpy_license, component
    actual = {path.relative_to(licenses).as_posix() for path in licenses.rglob("*") if path.is_file()}
    assert actual == set(documents) | {"manifest.json", "README.txt"}, sorted(actual ^ (set(documents) | {"manifest.json", "README.txt"}))
    assert not any(path.name.lower() in {"model.onnx", "model.safetensors", "pytorch_model.bin"}
                   for path in install.rglob("*") if path.is_file())
    print(f"LICENSE_DISTRIBUTIONS={len(distributions)} LICENSE_DOCUMENTS={len(documents)}", flush=True)


def _assert_cuda_model_modules(exe: Path) -> None:
    """Check the installed executable against the pinned Transformers source tree."""
    from PyInstaller.archive.readers import pkg_archive_contents
    import transformers.models

    models_root = Path(transformers.models.__file__).resolve().parent
    expected = set()
    for source in models_root.rglob("*.py"):
        parts = source.relative_to(models_root).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        expected.add(".".join(("transformers", "models", *parts)))
    bundled = set(pkg_archive_contents(str(exe)))
    missing = sorted(expected - bundled)
    assert not missing, f"CUDA package omits {len(missing)} Transformers model modules: {missing[:12]}"
    print(f"CUDA_MODEL_MODULES={len(expected)}", flush=True)


def _request(port: int, token: str) -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request("GET", "/_ngr/identity", headers={"Authorization": f"Bearer {token}"})
        response = connection.getresponse()
        if response.status != 200:
            raise AssertionError(f"identity HTTP {response.status}")
        return json.loads(response.read())
    finally:
        connection.close()


def _run(exe: Path, env: dict[str, str], *args: str) -> None:
    result = subprocess.run([str(exe), *args], env=env, capture_output=True,
                            text=True, timeout=90)
    if result.returncode:
        raise AssertionError(f"{args} failed: {result.stderr[-2000:]}")


def main() -> None:
    if os.name != "nt":
        raise SystemExit("Windows only")
    setup, version = Path(sys.argv[1]).resolve(), sys.argv[2]
    release_version = (Path(__file__).resolve().parents[1] / "packaging" / "windows" / "release-version.txt").read_text(encoding="utf-8").strip()
    assert version == release_version, (version, release_version)
    with tempfile.TemporaryDirectory(prefix="ngr-package-") as directory:
        root = Path(directory)
        install = root / "Installed"
        data = root / "user-data"
        data.mkdir()
        database = data / "knowledge.db"
        port = 28800 + secrets.randbelow(1000)
        token = secrets.token_urlsafe(32)
        package_env = dict(os.environ)
        package_env.pop("PYTHONPATH", None)
        package_env.pop("VIRTUAL_ENV", None)
        package_env["PATH"] = ";".join((os.environ["SystemRoot"] + "\\System32",
                                         os.environ["SystemRoot"] + "\\System32\\WindowsPowerShell\\v1.0",
                                         os.environ["SystemRoot"]))
        package_env["USERPROFILE"] = str(data)
        package_env["NGR_MCP_HTTP_BEARER_TOKEN"] = token
        package_env["NGR_DATABASE"] = str(database)
        rejected = subprocess.run([str(setup), "/VERYSILENT", "/SUPPRESSMSGBOXES",
                                   "/NORESTART", f"/DIR={install}"],
                                  capture_output=True, text=True, timeout=30)
        assert rejected.returncode != 0 and not install.exists(), "silent setup bypassed terms acceptance"
        install_started = time.monotonic()
        setup_result = subprocess.run([str(setup), "/VERYSILENT", "/ACCEPTVCRUNTIME=yes", "/SUPPRESSMSGBOXES",
                                       "/NORESTART", f"/DIR={install}"],
                                      capture_output=True, text=True, timeout=180)
        if setup_result.returncode:
            raise AssertionError(f"Setup failed: {setup_result.returncode}")
        print(f"INSTALL_SECONDS={time.monotonic() - install_started:.2f}", flush=True)
        installed_bytes = sum(path.stat().st_size for path in install.rglob("*") if path.is_file())
        print(f"INSTALLED_SIZE_BYTES={installed_bytes}", flush=True)
        exe = install / "NGR.exe"
        try:
            version_result = subprocess.run([str(exe), "--version"], env=package_env,
                                            capture_output=True, text=True, timeout=30)
            assert version_result.returncode == 0 and version_result.stdout.strip() == version
            manifest = json.loads((install / "package-manifest.json").read_text(encoding="utf-8-sig"))
            _assert_licenses(install, manifest["flavor"])
            if manifest["flavor"] == "cuda":
                _assert_cuda_model_modules(exe)
            start_started = time.monotonic()
            proxy = subprocess.Popen([str(exe), "--shared", "--port", str(port)],
                                     env=package_env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, encoding="utf-8")
            try:
                assert proxy.stdin and proxy.stdout
                responses: queue.Queue[str] = queue.Queue()
                threading.Thread(target=lambda: [responses.put(line) for line in proxy.stdout],
                                 daemon=True).start()
                proxy.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1,
                                              "method": "initialize", "params": {
                                                  "protocolVersion": "2025-06-18", "capabilities": {},
                                                  "clientInfo": {"name": "package-smoke", "version": "1"}}}) + "\n")
                proxy.stdin.flush()
                response = json.loads(responses.get(timeout=90))
                assert response["id"] == 1 and "result" in response, response
                assert response["result"]["serverInfo"]["version"] == version, response
                print(f"SHARED_START_SECONDS={time.monotonic() - start_started:.2f}", flush=True)
                proxy.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
                proxy.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}) + "\n")
                proxy.stdin.flush()
                response = json.loads(responses.get(timeout=30))
                assert response["id"] == 2 and response["result"]["tools"], response
                identity = _request(port, token)
                assert identity["database"] == str(database.resolve())
                tray = data / ".ngrdb" / f"shared-local-mcp-{port}.tray.json"
                assert tray.is_file(), "tray process did not register"
                stop_started = time.monotonic()
                _run(exe, package_env, "--tray-action", "stop", "--port", str(port))
                print(f"TRAY_STOP_SECONDS={time.monotonic() - stop_started:.2f}", flush=True)
                try:
                    _request(port, token)
                except (OSError, AssertionError):
                    pass
                else:
                    raise AssertionError("tray stop left the service running")
                resume_started = time.monotonic()
                _run(exe, package_env, "--tray-action", "resume", "--port", str(port))
                print(f"TRAY_RESUME_SECONDS={time.monotonic() - resume_started:.2f}", flush=True)
                assert _request(port, token)["service"] == "neuron-graph-rag-shared-local-mcp/v1"
                _run(exe, package_env, "--tray-action", "exit", "--port", str(port))
            finally:
                proxy.kill()
                proxy.communicate(timeout=10)
            assert database.is_file(), "database not created"
            (data / "model-sentinel").write_text("keep", encoding="utf-8")
            update_result = subprocess.run([str(setup), "/VERYSILENT", "/ACCEPTVCRUNTIME=yes", "/SUPPRESSMSGBOXES",
                                            "/NORESTART", f"/DIR={install}"],
                                           capture_output=True, text=True, timeout=180)
            assert update_result.returncode == 0, f"Update failed: {update_result.returncode}"
            assert database.is_file() and (data / "model-sentinel").is_file()
            uninstaller = install / "unins000.exe"
            _run(uninstaller, package_env, "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART")
            assert database.is_file() and (data / "model-sentinel").is_file()
        finally:
            # A failed assertion should not leave the detached service on the runner.
            if exe.is_file():
                subprocess.run([str(exe), "--tray-action", "exit", "--port", str(port)],
                               env=package_env, capture_output=True, timeout=30)


if __name__ == "__main__":
    main()
