# v2-m3 chunk shortlist ablation v1

[実験・観測](README.md) / [文書索引](../README.md)

## 目的

#234 の既知development難問について、#242のfull-chunk v2-m3 rerankを文書内E5 chunk shortlistへ置き換え、qualityを保ちながらCPU pipelineを600秒以内へ短縮できるか測定します。既定retrievalとproduction設定は変更しません。

## 結果前に固定したprotocol

protocol IDは `github-retrieval-parity-v5-v2-m3-chunk-shortlist-ablation-v1` です。Stage 1は#242と同じpinned `intfloat/multilingual-e5-small` revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`、structural-centroid、全93文書、candidate K=50を使います。

candidate各文書のstructural chunkを、E5 query-to-chunk cosine降順、同scoreはchunk index昇順に並べます。`m={2,4,8}` はこの順位列のprefixです。chunk countがm未満なら全chunkを使います。

Stage 2はpinned `BAAI/bge-reranker-v2-m3` revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` だけを使い、各文書top 8 union、最大400 query-chunk pairsを一度だけscoreします。m=2/4/8の各armはprefix logitsからstructural normalized log-mean-exp（temperature 1）を計算します。文書rankの同scoreはsource ID昇順で解消します。

## 成功条件とruntime境界

- quality primary: 少なくとも1 armが正解をrank 20以内に置く
- practical: 共通pipeline runtimeが600秒以内
- combined: 同一registered one-shotで上記2条件をともに満たす

runtimeは、#242と同じStage 1に加え、v2-m3 loader初期化、top 8選択、最大400 pairのtokenization/CPU forward、3armのNLME集約とrankingを含みます。download、cache copy、container起動、preflight synthetic forward、gold finalization、host export/auditは除外します。

#242のfull-chunk baselineは正解rank 15、pipeline runtime 717.949秒です。これは比較値であり、今回のworker packetや判定値として流用しません。観測後にm、K、式、cutoff、基準を変更しません。

## 登録境界

fresh named volumeへallowlist source、development-only query、exact corpus、pinned E5/v2-m3 model filesだけをコピーします。mixed v5 query/gold、holdout-bearing input、過去evidence、共有DBは登録環境に置かず、networkを無効にします。

preflight後、gold-blind Stage 1 / Stage 2 workersを単一one-shot、retry 0で実行します。worker完了後だけdevelopment-only goldを追加し、finalizerがcandidate inclusion、各arm rank、quality/practical/combined criteriaを決定します。claim、worker packets、resultまたはerrorはappend-onlyです。

## 再現入口

```powershell
pwsh tools/run_v2_m3_chunk_shortlist_ablation_v1_wslc.ps1 preflight
pwsh tools/run_v2_m3_chunk_shortlist_ablation_v1_wslc.ps1 run
pwsh tools/run_v2_m3_chunk_shortlist_ablation_v1_wslc.ps1 audit
```

`run` は既存claim/result/errorがある場合に上書きせず停止し、失敗したregistered protocolを再実行しません。
