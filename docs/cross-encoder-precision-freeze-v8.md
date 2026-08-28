# Cross-encoder precision freeze v8

## 目的

v8は、v7の初回runtime content reportがfail-closedした後継のresult-free freezeである。v7を再試行せず、normalized runtime contentとinstalled-distribution METADATA inventoryの対応意味論、content binding、決定的diagnosticを固定する。registered query、model forward/inference、observed resultは生成しない。

## 固定境界

corpus、development / holdoutのbilingual queryとgold、2 model revision、rank-only candidate、prefilter / CE / RRF、top5、tie-break、selection rule、11 hard gateはv7と同一である。corpus sourceは`c32b3049fd3daaa2190faf5e3e85955a195ee88c`、baseは`python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3`、WSLCは`2.9.4.0`、Pythonは`3.11.15`に固定する。expected registryはv7と同じexact 29 name / version / origin classを持つ。

v1-v7のobserved packet、query / result、model cache / weight、predecessor environmentを設計入力にしない。v7 failure evidenceは対応契約の前提としてのみ扱い、v7 artifactはmanifestのSHA-256 registryでbyte immutabilityを検証する。

## runtime content v3

`ngr.wslc-runtime-content/v3`はv2のnormalized file / symlink entries、base digest、dependency registry、exclusion registry、expected registry bindingを維持し、METADATA correspondenceを次のように定義する。

- installed-distribution pathは`site-packages/<single-component>.dist-info/METADATA`だけである。空component、`.`、`..`、slash、backslash、traversal、duplicate、case collisionを拒否する。
- `site-packages`配下のnested `**/*.dist-info/METADATA`はnormalized fingerprintへ残すが、installed inventoryへ含めない。`nested_metadata_paths`へ分類し、その件数とcanonical JSON SHA-256をreportへ固定する。
- filesystem inventoryの各pathには同じpathのregular-file normalized entryがexactly once必要である。entryの`content_sha256`はinventoryの`metadata_sha256`と一致しなければならない。
- missing normalized path、extra normalized top-level path、digest mismatch、symlink METADATA、malformed Name / Version、duplicate canonical nameをfail-closed拒否する。

diagnostic schemaは`ngr.wslc-metadata-correspondence/v1`である。path集合とdigest mismatchはUTF-8 byte順で決定的にsortし、次を持つ。

- `missing_normalized_metadata_paths`
- `extra_normalized_top_level_metadata_paths`
- `metadata_digest_mismatches`
- `nested_metadata_paths`
- `nested_metadata_count`
- `nested_metadata_paths_sha256`
- expected / actual / filesystem distribution counts

成功reportは空のfailure集合を含み、`metadata_correspondence_sha256`でdiagnostic全体をbindする。correspondence失敗時もruntime toolは同じdiagnostic objectとhashをcanonical JSONでstderrへ出力する。diagnostic処理はquery、model forward、retrieval resultを生成しない。

## one-shot WSLC

prebuild契約、tests、manifest hash registryがgreenになった後、同じpinned Containerfile / inputを`--no-cache`でbuild A `ngr-cross-encoder-precision-v8:freeze`、build B `ngr-cross-encoder-precision-v8:rebuild-check`として各一回だけbuildする。各imageはhost bind mountなし、`--network none`のfresh containerでruntime content reportとattestationを各一回だけ取得する。

local image IDはbuilt artifactの識別だけに用いる。A/Bのnormalized content、correspondence diagnostic、filesystem inventory、actual attestation、expected registryが完全一致した場合だけbuild Aをaccepted imageにする。一項目でも不一致ならdiagnostic evidenceを保存して停止し、追加build、追加report、validator調整、同v8の再試行、successor observationを禁止する。

## lifecycleと安全境界

development / holdoutはv8専用runtime、archive、transport、claim / result / error pathをexclusive-createする。freeze PRでは両stageを実行しない。successful freezeのsquash merge commitだけを別Issueのone-shot observation inputにできる。

共有Windows database、production service、feedback / outcome、NGR default、SQLite schema、default dependency、MCP config、predecessor evidenceを変更しない。performanceは`not assessed`、freeze registered query / model inference / observed resultは`0/0/0`を維持する。

## freeze outcome

one-shot WSLCは成功し、`accepted_exact_installed_distribution_freeze`へ遷移した。build A / Bはそれぞれ一回だけ`--no-cache`で実行し、runtime content reportとattestationも各imageにつき一回だけ取得した。追加buildと追加reportは0回である。

build Aは`sha256:136ad9466799109bf32b4b96b611c9db9a099bcc47cf78243f26c7227bc16742`、build Bは`sha256:67572463b924a21eb039d586a6beb661216ac95db38b497420471192066e1b97`である。local image IDは異なるが、A/Bのnormalized content reportはSHA-256 `7d754e4e1713f90654ae05c749379d08920e31fede89ab25ba075c0b582bcee8`、attestationはSHA-256 `045c813894bed25e3eae29a38fa6366013a0600ce8d0952a40ae29008747a50b`でbyte-for-byte一致した。fingerprintは`8969a259ffdfe822a70ac8bd8ce52dc7223b6e6a2b51ca21c095fe14a388b2bc`、METADATA correspondence diagnosticは`30348d7b8a352e0ab15d98871c71df5240ffbf1547c58564fd951cd451c377c4`で一致し、expected / importlib / filesystemの29 distributionがexact matchした。build Aをaccepted imageとする。

freeze registered query / model inference / observed resultは`0/0/0`、performanceは`not assessed`のままである。このsuccessful freezeのsquash merge commitのみを、別Issueのsuccessor observation inputにできる。
