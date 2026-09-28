# Upstream notice sources

The named upstream texts are copied verbatim from tagged repositories. The
Rust notices aggregate is generated from the tagged tokenizers Python binding's
Cargo.lock and checksum-verified crates.io archives. The build adds these texts
to the license texts found in installed wheels; it does not replace wheel
metadata or claim a source graph proves binary-level linking.

| File | Source | SHA-256 |
| --- | --- | --- |
| `InnoSetup-LICENSE.txt` | https://github.com/jrsoftware/issrc/blob/is-6_7_3/license.txt | `2e5346868c2a18434489824e11d65c3031620f792fefc415d05f19cd441abf5c` |
| `ONNXRuntime-ThirdPartyNotices.txt` | https://github.com/microsoft/onnxruntime/blob/v1.30.0/ThirdPartyNotices.txt | `143764b952fdb1a7c69ce653bfba74a7744d6a8a573bfb73e235fba356c83de3` |
| `PyTorch-NOTICE.txt` | https://github.com/pytorch/pytorch/blob/v2.9.1/NOTICE | `c2cc7bf0caec7652c2b460a8a470bea1677f241e4ab8e431df34cf17f5a9fec0` |
| `Tokenizers-LICENSE.txt` | https://github.com/huggingface/tokenizers/blob/v0.22.1/LICENSE | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| `Tokenizers-Rust-ThirdPartyNotices.txt` | https://github.com/huggingface/tokenizers/blob/v0.22.1/bindings/python/Cargo.lock (SHA-256 `6670f2b198a5016e8c75d704d2808745d30f39657f391055f2a3d81cc955ea68`) | `460b52a6e44ac669b3a4de7b5d7a7aa35847b37e0d9ff7cd04a5b732d5026d71` |

The Rust aggregate covers 125 crates reachable through normal dependency edges
for `x86_64-pc-windows-msvc` and contains 245 license/notice texts. Line endings
and trailing whitespace are normalized; the per-file SHA-256 values in the
aggregate identify the original archive bytes. It includes
the [number_prefix 0.4.0 LICENCE](https://github.com/ogham/rust-number-prefix/blob/v0.4.0/LICENCE)
(`df8d11b64ecce43b1229cd72745c59513b54ba39f05e3ec7c06780617b7b1fcc`),
which that crate excludes from its archive. This conservative dependency graph
may contain crates not linked into the Windows wheel; it does not prove that
every statically linked component is present or that legal obligations are met.

CPython `LICENSE.txt` is copied from the exact Python runtime used by the build.
The other third-party texts are copied from the installed distribution files
listed in their wheel RECORDs. `licenses/manifest.json` in each installer records
the actual distributions, versions, document paths, and hashes.
