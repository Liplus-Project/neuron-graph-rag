"""Read-only release discovery and verified, user-requested installer download.

This module never starts an installer. The published Windows build is currently
unsigned, so choosing when and whether to run it remains with the user.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from urllib.parse import urlparse


RELEASES_URL = "https://api.github.com/repos/Liplus-Project/neuron-graph-rag/releases/latest"
RELEASE_PAGE_PREFIX = "https://github.com/Liplus-Project/neuron-graph-rag/releases/tag/"
PACKAGE_SCHEMA = "ngr.windows-package/v1"
MAX_INSTALLER_BYTES = 16 * 1024**3
MAX_JSON_BYTES = 1024 * 1024
VERSION_RE = re.compile(r"v?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)\Z")
DIGEST_RE = re.compile(r"sha256:([0-9a-fA-F]{64})\Z")


class UpdateError(Exception):
    """A release cannot safely be presented or downloaded."""


class DownloadCancelled(UpdateError):
    """The user cancelled a download."""


@dataclass(frozen=True)
class InstalledBuild:
    version: str
    flavor: str | None


@dataclass(frozen=True)
class UpdateCandidate:
    version: str
    flavor: str
    filename: str
    size: int
    digest: str
    download_url: str
    release_url: str


def _version(value: str) -> tuple[int, int, int]:
    match = VERSION_RE.fullmatch(value)
    if not match:
        raise UpdateError("invalid stable release version")
    return tuple(int(part) for part in match.groups())


def installed_build(executable: Path | None = None) -> InstalledBuild:
    executable = executable or Path(sys.executable)
    manifest = executable.parent / "package-manifest.json"
    if manifest.is_file():
        try:
            info = json.loads(manifest.read_text(encoding="utf-8"))
            if info["schema"] != PACKAGE_SCHEMA:
                raise ValueError("invalid package schema")
            current = str(info["version"])
            flavor = info["flavor"]
            _version(current)
            if flavor not in ("cpu", "cuda"):
                raise ValueError("invalid flavor")
            return InstalledBuild(current.removeprefix("v"), flavor)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise UpdateError("installed package manifest is invalid") from error
    try:
        current = package_version("neuron-graph-rag")
    except PackageNotFoundError:
        current = "unknown"
    # A source install has no reliable CPU/CUDA distribution identity.
    return InstalledBuild(current, None)


def _read_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json", "User-Agent": "NGR-verified-updates",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            raw = response.read(MAX_JSON_BYTES + 1)
    except (OSError, urllib.error.HTTPError) as error:
        raise UpdateError("release check unavailable") from error
    if len(raw) > MAX_JSON_BYTES:
        raise UpdateError("release response exceeds limit")
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeError) as error:
        raise UpdateError("invalid release response") from error
    if not isinstance(data, dict):
        raise UpdateError("invalid release response")
    return data


def candidate_from_release(release: dict, installed: InstalledBuild) -> UpdateCandidate | None:
    if installed.flavor not in ("cpu", "cuda"):
        return None
    if release.get("draft") or release.get("prerelease") or not release.get("published_at"):
        return None
    try:
        target = str(release["tag_name"])
        if _version(target) <= _version(installed.version):
            return None
        target = target.removeprefix("v")
        filename = f"NGR-{target}-windows-x64-{installed.flavor}-setup.exe"
        page = str(release["html_url"])
        if page != RELEASE_PAGE_PREFIX + str(release["tag_name"]):
            raise UpdateError("release page does not match the official tag")
        matches = [a for a in release["assets"] if isinstance(a, dict) and a.get("name") == filename]
        if len(matches) != 1:
            raise UpdateError("compatible installer is missing or ambiguous")
        asset = matches[0]
        size = asset["size"]
        if type(size) is not int or not 0 < size <= MAX_INSTALLER_BYTES:
            raise UpdateError("installer size is invalid")
        digest = DIGEST_RE.fullmatch(str(asset["digest"]))
        if not digest:
            raise UpdateError("installer SHA-256 digest is missing")
        url = str(asset["browser_download_url"])
        parsed = urlparse(url)
        if (parsed.scheme, parsed.netloc, parsed.path) != (
            "https", "github.com",
            f"/Liplus-Project/neuron-graph-rag/releases/download/{release['tag_name']}/{filename}",
        ) or parsed.query or parsed.fragment:
            raise UpdateError("installer URL is not an official release asset")
        return UpdateCandidate(target, installed.flavor, filename, size,
                               digest.group(1).lower(), url, page)
    except (KeyError, TypeError, ValueError) as error:
        raise UpdateError("release metadata is incomplete") from error


def check_for_update(installed: InstalledBuild | None = None) -> UpdateCandidate | None:
    installed = installed or installed_build()
    if installed.flavor is None:
        return None
    return candidate_from_release(_read_json(RELEASES_URL), installed)


def download_candidate(candidate: UpdateCandidate, directory: Path,
                       cancel: threading.Event | None = None) -> Path:
    """Stream into a private temporary file and publish only after full validation."""
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as error:
        raise UpdateError("update directory is unavailable") from error
    if directory.is_symlink():
        raise UpdateError("update directory must not be a symlink")
    destination = directory / candidate.filename
    if destination.exists():
        raise UpdateError("installer already exists; move it before retrying")
    request = urllib.request.Request(candidate.download_url, headers={
        "Accept": "application/octet-stream", "User-Agent": "NGR-verified-updates",
    })
    temp_path = None
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != candidate.size:
                raise UpdateError("installer transfer size differs from release metadata")
            with tempfile.NamedTemporaryFile(prefix=".ngr-update-", suffix=".part",
                                             dir=directory, delete=False) as output:
                temp_path = Path(output.name)
                digest = hashlib.sha256()
                count = 0
                while True:
                    if cancel is not None and cancel.is_set():
                        raise DownloadCancelled("download cancelled")
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > candidate.size or count > MAX_INSTALLER_BYTES:
                        raise UpdateError("installer exceeds declared size")
                    digest.update(chunk)
                    output.write(chunk)
        if cancel is not None and cancel.is_set():
            raise DownloadCancelled("download cancelled")
        if count != candidate.size or digest.hexdigest() != candidate.digest:
            raise UpdateError("installer size or SHA-256 verification failed")
        os.replace(temp_path, destination)
        return destination
    except (OSError, ValueError, urllib.error.HTTPError) as error:
        raise UpdateError("installer download failed") from error
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
