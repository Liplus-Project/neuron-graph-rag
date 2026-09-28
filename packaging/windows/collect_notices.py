"""Copy license texts for distributions actually present in a PyInstaller build.

Run after PyInstaller analysis, with the same interpreter that built the bundle.
The Analysis TOC names both archived Python modules and copied native binaries;
installed wheel RECORD files map those source paths back to distributions.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.metadata as metadata
import json
import os
import re
import struct
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
    "Microsoft-VC-Runtime-LICENSE.txt": "a0d066f34af2b1d5c6694902de40cd6fd31b9d471a5594bb22bb58f2e5382dd3",
}
NUMPY_WINDOWS_VERSION = "1.26.4"
MSVC_RUNTIME_PREFIXES = ("msvcp", "vcruntime", "concrt", "vcomp", "ucrtbase", "api-ms-win-crt-")
VC_RUNTIME_NAMES = {"msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll"}


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


def audit_vc_runtime(bundle: Path, provenance_file: Path) -> dict:
    provenance = json.loads(provenance_file.read_text(encoding="utf-8-sig"))
    if provenance.get("schema") != "ngr.vc-redist-source/v1":
        raise RuntimeError("VC runtime provenance schema mismatch")
    installation = Path(provenance["installation"]).resolve(strict=True)
    redist = Path(provenance["redist_directory"]).resolve(strict=True)
    relative = redist.relative_to(installation).parts
    if (len(relative) != 6 or tuple(part.lower() for part in relative[:3]) != ("vc", "redist", "msvc")
            or relative[4].lower() != "x64" or relative[5].lower() != "microsoft.vc143.crt"
            or not re.fullmatch(r"14\.\d+\.\d+", relative[3])
            or any("preview" in part.lower() or "debug_nonredist" in part.lower()
                   for part in redist.parts)):
        raise RuntimeError(f"VC runtime source is not a release x64 VC\\Redist CRT: {redist}")
    release = "2022"
    if release not in installation.parts or "Microsoft Visual Studio" not in installation.parts:
        raise RuntimeError(f"VC runtime source is not the expected Visual Studio {release} installation")
    edition_terms = {"Community": "https://visualstudio.microsoft.com/license-terms/vs2022-ga-community/"}
    edition = provenance["edition"]
    if installation.name != edition or provenance["edition_terms"] != edition_terms.get(edition):
        raise RuntimeError("VC runtime Visual Studio edition or license URL mismatch")
    records = provenance["files"]
    if len(records) != len(VC_RUNTIME_NAMES) or {record["name"].lower() for record in records} != VC_RUNTIME_NAMES:
        raise RuntimeError("VC runtime source must contain exactly the four approved DLLs")
    audited = []
    for record in records:
        name = record["name"].lower()
        source = Path(record["source"]).resolve(strict=True)
        if source.parent != redist or source.name.lower() != name:
            raise RuntimeError(f"VC runtime source path mismatch: {source}")
        target = bundle / "_internal" / name
        data = source.read_bytes()
        pe_offset = struct.unpack_from("<I", data, 0x3c)[0]
        if data[pe_offset:pe_offset + 4] != b"PE\0\0" or struct.unpack_from("<H", data, pe_offset + 4)[0] != 0x8664:
            raise RuntimeError(f"VC runtime is not an x64 PE image: {source}")
        digest = hashlib.sha256(data).hexdigest()
        if digest != record["sha256"] or not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"VC runtime source/bundle SHA-256 mismatch: {name}")
        if not record.get("file_version"):
            raise RuntimeError(f"VC runtime file version missing: {name}")
        audited.append({"path": target.relative_to(bundle).as_posix(), "sha256": digest,
                        "source": str(source), "file_version": record["file_version"]})
    actual = {path.relative_to(bundle).as_posix().lower() for path in bundle.rglob("*.dll")
              if path.name.lower().startswith(MSVC_RUNTIME_PREFIXES)}
    if actual != {entry["path"] for entry in audited}:
        raise RuntimeError(f"Unexpected or missing Visual C++ runtime DLLs in bundle: {sorted(actual)}")
    return {"visual_studio_release": release, "visual_studio_edition": edition,
            "edition_terms": edition_terms[edition], "redist_directory": str(redist),
            "redist_list": f"https://learn.microsoft.com/en-us/visualstudio/releases/{release}/redistribution",
            "files": sorted(audited, key=lambda entry: entry["path"])}


def audit_native_binaries(bundle: Path, sources: set[str],
                          numpy_dist: metadata.Distribution, documents: list[dict]) -> list[dict]:
    """Match NumPy's DLL to wheel RECORD."""
    if numpy_dist.version != NUMPY_WINDOWS_VERSION:
        raise RuntimeError(f"Windows NumPy must be {NUMPY_WINDOWS_VERSION}: {numpy_dist.version}")
    numpy_dlls = [entry for entry in numpy_dist.files or ()
                  if len(entry.parts) == 2 and entry.parts[0].lower() == "numpy.libs"
                  and entry.name.lower().endswith(".dll")]
    if len(numpy_dlls) != 1 or not numpy_dlls[0].name.lower().startswith("libopenblas"):
        raise RuntimeError("NumPy wheel native libraries changed; review their licenses and DLL origins: "
                           + repr([str(entry) for entry in numpy_dlls]))
    entry = numpy_dlls[0]
    if not entry.hash or entry.hash.mode != "sha256":
        raise RuntimeError(f"NumPy native library lacks a SHA-256 wheel RECORD: {entry}")
    source = Path(numpy_dist.locate_file(entry))
    if path_key(source) not in sources:
        raise RuntimeError(f"NumPy native library is absent from PyInstaller Analysis: {source}")
    expected = base64.urlsafe_b64decode(entry.hash.value + "=" * (-len(entry.hash.value) % 4)).hex()
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Installed NumPy native library differs from wheel RECORD: {source}")
    license_path = (Path("third-party") / f"numpy-{numpy_dist.version}" /
                    f"numpy-{numpy_dist.version}.dist-info" / "LICENSE.txt").as_posix()
    if license_path not in {item["path"] for item in documents}:
        raise RuntimeError(f"NumPy wheel license is absent from bundle: {license_path}")
    license_text = (bundle / "licenses" / license_path).read_text(encoding="utf-8")
    for component in ("Name: OpenBLAS", "Name: LAPACK", "Name: GCC runtime library",
                      "Name: libquadmath", "GCC RUNTIME LIBRARY EXCEPTION"):
        if component not in license_text:
            raise RuntimeError(f"NumPy wheel license omits {component}")

    native_binaries = []
    bundled_numpy = []
    for binary in bundle.rglob("*.dll"):
        relative = binary.relative_to(bundle).as_posix()
        name = binary.name.lower()
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        if name == entry.name.lower():
            if digest != expected:
                raise RuntimeError(f"Bundled NumPy native library differs from wheel RECORD: {binary}")
            bundled_numpy.append(relative)
            native_binaries.append({"path": relative, "sha256": digest,
                                    "source_distribution": f"numpy=={numpy_dist.version}",
                                    "source_record": str(entry), "license_documents": [license_path]})
    if len(bundled_numpy) != 1:
        raise RuntimeError(f"Expected one NumPy OpenBLAS DLL in bundle: {bundled_numpy}")
    return sorted(native_binaries, key=lambda item: item["path"])


def collect(toc: Path, bundle: Path, repository: Path, flavor: str, vc_source_manifest: Path) -> dict:
    sources = analysis_sources(toc)
    vc_runtime = audit_vc_runtime(bundle, vc_source_manifest)
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
    microsoft_terms = repository / "packaging" / "windows" / "licenses" / "Microsoft-VC-Runtime-LICENSE.txt"
    _copy_document(microsoft_terms, licenses / "Microsoft-VC-Runtime-LICENSE.txt",
                   "Microsoft-VC-Runtime-LICENSE.txt", documents,
                   "https://visualstudio.microsoft.com/license-terms/vs2022-cruntime/")
    setup_terms = (
        "Neuron Graph RAG installer terms\n\n"
        f"This installer includes four Microsoft Visual C++ runtime DLLs from Visual Studio 2022 {vc_runtime['visual_studio_edition']}.\n"
        "NGR's right to distribute those unmodified DLLs must come from the\n"
        "publisher's own Visual Studio license and Microsoft's REDIST list:\n"
        f"{vc_runtime['edition_terms']}\n"
        f"{vc_runtime['redist_list']}\n"
        "Your acceptance of these installer terms does not establish that the\n"
        "publisher holds that license or meets its distribution requirements.\n\n"
        "As an end user, you must accept the NGR license and the Microsoft Visual\n"
        "C++ runtime terms reproduced below to install this package. The latter\n"
        "restrict your use and any further distribution of the Microsoft runtime;\n"
        "they are not the publisher's Visual Studio redistribution permission.\n"
        "A silent install requires /ACCEPTVCRUNTIME=yes to record this acceptance.\n\n"
        "===== NGR LICENSE =====\n"
        + (licenses / "NGR-LICENSE.txt").read_text(encoding="utf-8")
        + "\n===== MICROSOFT VISUAL C++ RUNTIME TERMS =====\n"
        + (licenses / "Microsoft-VC-Runtime-LICENSE.txt").read_text(encoding="utf-8")
    )
    setup_path = licenses / "Setup-LICENSE.txt"
    setup_path.write_text(setup_terms, encoding="utf-8")
    documents.append({"path": setup_path.name, "sha256": hashlib.sha256(setup_path.read_bytes()).hexdigest(),
                      "origin": "NGR and Microsoft runtime license terms for installer acceptance"})

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
    # Analysis may have collected a CPython, System32 or wheel copy of one of
    # these names. Only the final, replaced bytes from VC\Redist are accepted.
    replaced = {path for path in sources if Path(path).name.lower() in VC_RUNTIME_NAMES}
    allowed_roots = (path_key(python_root) + os.sep, path_key(repository) + os.sep)
    foreign = sorted(path for path in sources if path not in owners and path not in replaced
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
    native_binaries = audit_native_binaries(bundle, sources, by_name["numpy"], documents)
    manifest = {"schema": "ngr.windows-licenses/v1", "flavor": flavor,
                "python_version": sys.version.split()[0],
                "python_runtime_files": sorted(name for name in runtime_sources
                                               if name.lower().startswith("python3")),
                "vc_runtime": vc_runtime,
                "distributions": distributions, "documents": sorted(documents, key=lambda item: item["path"]),
                "native_binaries": native_binaries,
                "models_bundled": False}
    (licenses / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = (
        "Neuron Graph RAG - licenses and notices\n\n"
        "NGR-LICENSE.txt and NGR-NOTICE.txt cover the application. Python-LICENSE.txt\n"
        "includes the terms for the bundled Windows Python distribution and its Microsoft\n"
        "Distributable Code. InnoSetup-LICENSE.txt covers the setup program.\n"
        "Four unmodified x64 Microsoft C++ runtime DLLs are bundled from a\n"
        "Visual Studio VC\\Redist installation. vc_runtime in manifest.json\n"
        "records their exact source, file versions and SHA-256 digests.\n"
        "Microsoft-VC-Runtime-LICENSE.txt is Microsoft's runtime terms. The\n"
        "installer presents them to end users in Setup-LICENSE.txt. Publisher\n"
        "redistribution rights come separately from its Visual Studio license;\n"
        "this bundle does not prove those rights. The bundled CPython binaries\n"
        "also carry their own terms.\n\n"
        "third-party/ holds the license and notice texts from each bundled Python\n"
        "distribution. manifest.json lists the exact names, versions, files and SHA-256\n"
        "digests detected from the PyInstaller Analysis and installed wheel RECORDs.\n"
        "The NumPy license includes OpenBLAS, LAPACK, GCC runtime and libquadmath\n"
        "terms for its bundled OpenBLAS DLL. native_binaries in manifest.json\n"
        "records the actual DLL and its matching wheel RECORD hash.\n\n"
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
    parser.add_argument("--vc-source-manifest", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.analysis, args.bundle, args.repository, args.flavor, args.vc_source_manifest)
    print(f"LICENSE_DISTRIBUTIONS={len(result['distributions'])}")
    print(f"LICENSE_DOCUMENTS={len(result['documents'])}")


if __name__ == "__main__":
    main()
