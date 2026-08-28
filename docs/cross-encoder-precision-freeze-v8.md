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

prebuild段階では`pending_one_shot_wslc`であり、image ID、runtime report、attestation、accepted imageを持たない。one-shot完了後、成功なら`accepted_exact_installed_distribution_freeze`、失敗なら`fail_closed_exact_installed_distribution_freeze`へ一度だけ遷移し、build / report回数とresult-free countをfixtureへ固定する。
