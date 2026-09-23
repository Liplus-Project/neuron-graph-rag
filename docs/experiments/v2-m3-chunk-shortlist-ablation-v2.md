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

## 再現入口

```powershell
pwsh tools/run_v2_m3_chunk_shortlist_ablation_v2_wslc.ps1 preflight
pwsh tools/run_v2_m3_chunk_shortlist_ablation_v2_wslc.ps1 run
pwsh tools/run_v2_m3_chunk_shortlist_ablation_v2_wslc.ps1 audit
```
