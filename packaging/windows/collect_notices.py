"""Copy license texts for distributions actually present in a PyInstaller build.

Run after PyInstaller analysis, with the same interpreter that built the bundle.
The Analysis TOC names both archived Python modules and copied native binaries;
installed wheel RECORD files map those source paths back to distributions.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata as metadata
import json
import os
import re
import sys
from pathlib import Path


DOCUMENT_NAME = re.compile(
    r"(?:^|[-_.])(?:licen[cs]e|copying|copyright|notice|third[-_]?party[-_]?notices?)(?:[-_.]|$)",
    re.IGNORECASE,
)
TEXT_SUFFIXES = {"", ".txt", ".md", ".rst", ".text", ".html"}
UPSTREAM_NOTICES = {
    "onnxruntime": ("1.30.0", "ONNXRuntime-ThirdPartyNotices.txt", "https://github.com/microsoft/onnxruntime/blob/v1.30.0/ThirdPartyNotices.txt"),
    "torch": ("2.9.1", "PyTorch-NOTICE.txt", "https://github.com/pytorch/pytorch/blob/v2.9.1/NOTICE"),
    "tokenizers": ("0.22.1", "Tokenizers-Rust-ThirdPartyNotices.txt", "https://github.com/huggingface/tokenizers/tree/v0.22.1/bindings/python"),
}
UPSTREAM_LICENSES = {
    # tokenizers 0.22.1's Windows wheel has no License-File metadata or text.
    "tokenizers": ("0.22.1", "Tokenizers-LICENSE.txt", "https://github.com/huggingface/tokenizers/blob/v0.22.1/LICENSE"),
}
PINNED_DOCUMENT_SHA256 = {
    "InnoSetup-LICENSE.txt": "3df23505b7ec00dc007a1e1e9ba32ee3895e7ce90043bcd1ac1b9b47155921a7",
    "ONNXRuntime-ThirdPartyNotices.txt": "143764b952fdb1a7c69ce653bfba74a7744d6a8a573bfb73e235fba356c83de3",
    "PyTorch-NOTICE.txt": "c2cc7bf0caec7652c2b460a8a470bea1677f241e4ab8e431df34cf17f5a9fec0",
    "Tokenizers-LICENSE.txt": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "Tokenizers-Rust-ThirdPartyNotices.txt": "460b52a6e44ac669b3a4de7b5d7a7aa35847b37e0d9ff7cd04a5b732d5026d71",
}


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def analysis_sources(toc: Path) -> set[str]:
    value = ast.literal_eval(toc.read_text(encoding="utf-8"))
    sources: set[str] = set()

    def visit(item: object) -> None:
        if isinstance(item, str):
            path = Path(item)
            if path.is_absolute() and path.is_file():
                sources.add(path_key(path))
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(key)
                visit(child)
        elif isinstance(item, (tuple, list, set)):
            for child in item:
                visit(child)

    visit(value)
    if not sources:
        raise RuntimeError(f"PyInstaller Analysis contains no source files: {toc}")
    return sources


def _is_document(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES and (
        bool(DOCUMENT_NAME.search(path.name)) or "licenses" in (part.lower() for part in path.parts)
    )


def _copy_document(source: Path, target: Path, manifest_path: str, documents: list[dict], origin: str) -> None:
    data = source.read_bytes()
    if source.name == "InnoSetup-LICENSE.txt":
        data = data.replace(b"\r\n", b"\n")
    if not data.strip() or b"\x00" in data:
        raise RuntimeError(f"License/NOTICE is empty or binary: {source}")
    digest = hashlib.sha256(data).hexdigest()
    if source.name in PINNED_DOCUMENT_SHA256 and digest != PINNED_DOCUMENT_SHA256[source.name]:
        raise RuntimeError(f"Pinned upstream license/NOTICE hash mismatch: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    documents.append({"path": manifest_path, "sha256": digest, "origin": origin})


def collect(toc: Path, bundle: Path, repository: Path, flavor: str) -> dict:
    sources = analysis_sources(toc)
    licenses = bundle / "licenses"
    if licenses.exists():
        raise RuntimeError(f"License output is not clean: {licenses}")
    licenses.mkdir(parents=True)
    documents: list[dict] = []
    for source_name, target_name in (("LICENSE", "NGR-LICENSE.txt"), ("NOTICE", "NGR-NOTICE.txt")):
        _copy_document(repository / source_name, licenses / target_name, target_name, documents, "NGR repository")

    python_root = Path(sys.base_prefix)
    python_license = python_root / "LICENSE.txt"
    _copy_document(python_license, licenses / "Python-LICENSE.txt", "Python-LICENSE.txt", documents, str(python_license))
    inno_license = repository / "packaging" / "windows" / "licenses" / "InnoSetup-LICENSE.txt"
    _copy_document(inno_license, licenses / "InnoSetup-LICENSE.txt", "InnoSetup-LICENSE.txt", documents,
                   "https://github.com/jrsoftware/issrc/blob/is-6_7_3/license.txt")

    installed = list(metadata.distributions())
    owners: dict[str, set[str]] = {}
    by_name: dict[str, metadata.Distribution] = {}
    for dist in installed:
        name = canonical(dist.metadata["Name"])
        by_name[name] = dist
        for entry in dist.files or ():
            key = path_key(Path(dist.locate_file(entry)))
            if key in sources:
                owners.setdefault(key, set()).add(name)

    site_paths = [path_key(Path(path)) + os.sep for path in sys.path if path.lower().endswith("site-packages")]
    unowned = sorted(path for path in sources if any(path.startswith(site) for site in site_paths)
                     and path not in owners)
    if unowned:
        raise RuntimeError("Bundled site-packages files lack installed RECORD ownership: " + repr(unowned[:20]))
    system32 = path_key(Path(os.environ["SystemRoot"]) / "System32") + os.sep
    excluded_names = {"msvcp140.dll", "msvcp140_1.dll"}
    excluded = sorted(path for path in sources if path.startswith(system32)
                      and Path(path).name.lower() in excluded_names)
    for source in excluded:
        if (bundle / "_internal" / Path(source).name).exists():
            raise RuntimeError(f"System32 Visual C++ DLL remains in bundle: {source}")
    allowed_roots = (path_key(python_root) + os.sep, path_key(repository) + os.sep)
    foreign = sorted(path for path in sources if path not in owners and path not in excluded
                     and not any(path.startswith(root) for root in allowed_roots))
    if foreign:
        raise RuntimeError("Bundled source files have no NGR, CPython or wheel ownership: "
                           + repr(foreign[:20]))

    included = sorted({name for names in owners.values() for name in names} | {"pyinstaller"})
    distributions = []
    for name in included:
        if name == "neuron-graph-rag":
            continue
        dist = by_name[name]
        dist_documents = []
        for entry in dist.files or ():
            relative = Path(str(entry))
            if relative.is_absolute() or ".." in relative.parts:
                continue
            source = Path(dist.locate_file(entry))
            if not _is_document(relative) or not source.is_file():
                continue
            # Preserve the wheel-relative name so native-library notices stay identifiable.
            destination = Path("third-party") / f"{name}-{dist.version}" / relative
            record = destination.as_posix()
            _copy_document(source, licenses / destination, record, documents, "installed wheel RECORD")
            dist_documents.append(record)
        if name in UPSTREAM_LICENSES:
            expected_version, filename, url = UPSTREAM_LICENSES[name]
            if dist.version.split("+")[0] != expected_version:
                raise RuntimeError(f"Upstream license version does not match {name}=={dist.version}")
            source = repository / "packaging" / "windows" / "licenses" / filename
            destination = Path("third-party") / f"{name}-{dist.version}" / filename
            record = destination.as_posix()
            _copy_document(source, licenses / destination, record, documents, url)
            dist_documents.append(record)
        if not any("licen" in Path(path).name.lower() or "copying" in Path(path).name.lower()
                   for path in dist_documents):
            raise RuntimeError(f"Bundled distribution has no license text in its installed wheel: {name}=={dist.version}")
        if name in UPSTREAM_NOTICES:
            expected_version, filename, url = UPSTREAM_NOTICES[name]
            if dist.version.split("+")[0] != expected_version:
                raise RuntimeError(f"Upstream notice version does not match {name}=={dist.version}")
            source = repository / "packaging" / "windows" / "licenses" / filename
            destination = Path("third-party") / f"{name}-{dist.version}" / filename
            record = destination.as_posix()
            _copy_document(source, licenses / destination, record, documents, url)
            dist_documents.append(record)
        distributions.append({"name": name, "version": dist.version,
                              "bundled_source_count": sum(name in names for names in owners.values()),
                              "documents": sorted(dist_documents)})

    runtime_sources = [Path(path).name for path in sources if path.startswith(path_key(python_root) + os.sep)]
    vcruntime = [path for path in sources if Path(path).name.lower().startswith("vcruntime")]
    if not vcruntime or any(not path.startswith(path_key(python_root) + os.sep) for path in vcruntime):
        raise RuntimeError("Bundled VCRUNTIME DLLs must come from the licensed CPython distribution: "
                           + repr(vcruntime))
    manifest = {"schema": "ngr.windows-licenses/v1", "flavor": flavor,
                "python_version": sys.version.split()[0],
                "python_runtime_files": sorted(name for name in runtime_sources
                                               if name.lower().startswith(("python3", "vcruntime"))),
                "excluded_system_runtime": sorted(Path(path).name.lower() for path in excluded),
                "distributions": distributions, "documents": sorted(documents, key=lambda item: item["path"]),
                "models_bundled": False}
    (licenses / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = (
        "Neuron Graph RAG - licenses and notices\n\n"
        "NGR-LICENSE.txt and NGR-NOTICE.txt cover the application. Python-LICENSE.txt\n"
        "includes the terms for the bundled Windows Python distribution and its Microsoft\n"
        "Distributable Code. InnoSetup-LICENSE.txt covers the setup program.\n\n"
        "third-party/ holds the license and notice texts from each bundled Python\n"
        "distribution. manifest.json lists the exact names, versions, files and SHA-256\n"
        "digests detected from the PyInstaller Analysis and installed wheel RECORDs.\n\n"
        "The E5 and v2-m3 model weights are not included in either installer. See\n"
        "https://huggingface.co/intfloat/multilingual-e5-small and\n"
        "https://huggingface.co/BAAI/bge-reranker-v2-m3 for their separate terms.\n"
    )
    (licenses / "README.txt").write_text(readme, encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--flavor", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    result = collect(args.analysis, args.bundle, args.repository, args.flavor)
    print(f"LICENSE_DISTRIBUTIONS={len(result['distributions'])}")
    print(f"LICENSE_DOCUMENTS={len(result['documents'])}")


if __name__ == "__main__":
    main()
