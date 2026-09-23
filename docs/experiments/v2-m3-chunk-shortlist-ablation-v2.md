# v2-m3 chunk shortlist ablation v2

[実験・観測](README.md) / [文書索引](../README.md)

## 目的

v1を再実行せず、append-only claimとworker検証が衝突したallowlist矛盾だけを修正し、同じchunk shortlist pipelineを単一registered one-shotで測定します。既定retrievalとproduction設定は変更しません。

## v1からの唯一の意味差分

protocol IDは `github-retrieval-parity-v5-v2-m3-chunk-shortlist-ablation-v2` です。自protocolのevidence directory全体を禁止対象から外し、実行stageごとに、まだ存在してはならないworker packet、result、error、development-only goldの具体的pathを禁止します。claim後のworker stageではclaimだけを許可します。

v1のpreflight、claim、errorはSHA-256で固定し、v2から変更しません。v1のStage 1 / Stage 2 worker packetとresultは存在せず、v1は再実行しません。

## 変更しない固定値

- Stage 1: pinned multilingual-E5 structural-centroid、全93文書、candidate K=50
- shortlist: E5 query-to-structural-chunk cosine降順、同scoreはchunk index昇順、`m={2,4,8}`
- Stage 2: pinned v2-m3、top 8 union、最大400 pairsを一度だけforward
- 集約: structural normalized log-mean-exp、temperature 1
- 成功条件: 少なくとも1 armがrank 20以内、かつ共通pipeline runtimeが600秒以内
- runtime境界、gold-blind worker、retry 0、観測後の変更禁止

## 登録境界

result-free manifest/schema/model registry/runner/wrapper/docs/testsを先にcommit/pushし、exact source commitへhash-bindします。fresh holdout-absent runtimeでpreflightを実行し、query 0、exact 93文書、model/input/source hash、forbidden path不在、E5/v2-m3 synthetic forwardを確認します。

その後、Stage 1とStage 2を単一one-shotで実行します。完全なgold-blind worker packetsが揃った後だけdevelopment-only goldを追加し、finalizerがcandidate inclusion、各arm rank、quality/practical/combined criteriaを決定します。失敗時は同protocolを再実行しません。

## 登録実行結果

result-free commit `6df8d836a464788ed818d8d408adecdbc3bea7a2` に対するfresh preflightは、query実行0、exact 93文書、gold absent、holdout-bearing input 0、mixed v5 query/gold 0、共有DB open 0でPASSしました。

retry 0の単一one-shotで、Stage 1は正解をrank 39に置き、candidate K=50へ含めました。Stage 2はtop-8 unionの370 pairsを47 batchesで一度だけforwardしました。gold-blind worker完了後にだけdevelopment-only goldを追加し、finalizerが次のrankを確定しました。

| arm | 正解rank | rank 20以内 |
| --- | ---: | :---: |
| m=2 | 19 | PASS |
| m=4 | 22 | FAIL |
| m=8 | 21 | FAIL |

Stage 1 runtimeは67.055秒、Stage 2 runtimeは156.886秒、共通pipeline runtimeは223.941秒でした。m=2がquality primaryを満たし、pipelineは600秒以内だったため、quality、practical、combinedの3条件はすべてPASSです。#242のfull-chunk baseline 717.949秒に対し494.008秒短縮しました。baselineのrank 15に対して最良armはrank 19であり、今回の結論は既知development難問に限定します。

preflight、claim、Stage 1 worker、Stage 2 worker、observed resultをappend-only保存しました。errorはありません。

## 再現入口

```powershell
pwsh tools/run_v2_m3_chunk_shortlist_ablation_v2_wslc.ps1 preflight
pwsh tools/run_v2_m3_chunk_shortlist_ablation_v2_wslc.ps1 run
pwsh tools/run_v2_m3_chunk_shortlist_ablation_v2_wslc.ps1 audit
```
