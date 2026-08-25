# Cross-encoder precision benchmark freeze v2

## 目的と境界

この文書は、cross-encoder precision v1のone-shot developmentがcandidateの空 `returned_source_paths` をfield欠落と混同して停止したため、そのfield-presence判定だけを修正したresult-free v2 protocolを凍結する。v1 evidenceを変更、再評価、再実行、v2 resultへ変換しない。v1 raw payload、score、rankはcandidate、query、gold、gateの設計入力に使用せず、byte hashだけを確認する。

v2はprotocol ID、fixture stem、runtime/archive path、evaluator、test、docsを専用化する。それ以外はv1の24 corpus identities、development/holdout各8件のbilingual query/gold、2 exact model revisions、dependency lock、passage projection、batch size 8、4 candidate順、threshold、RRF式、11 hard gateを維持する。sourceは `c32b3049fd3daaa2190faf5e3e85955a195ee88c` のgit bytesへ固定し、NGR default、SQLite schema、default dependency、MCP configを変更しない。

## 固定model、passage、candidate

- `BAAI/bge-reranker-base@2cfc18c9415c912f9d8155881c133215df768a70`
- `BAAI/bge-reranker-v2-m3@953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`

freezeではmodel cache、weight、venvを取得、open、推論しない。successor observationだけがclaim前にexact revisionの専用cacheを全hash再検証してofflineへ切り替える。

本文はUnicode code point 480、overlap 80で分割し、入力は `[query, chunk_text]` のみとする。tokenizerは `padding=True`、`truncation=True`、`max_length=512`、実行はCPU / float32 / eval / no-grad、batch size 8である。document scoreはchunk raw logit最大値、tieは最小chunk index、thresholdは `raw_logit >= 0.0` である。

candidate順は `bge-base-rrf-threshold`、`bge-base-ce-threshold`、`bge-v2-m3-rrf-threshold`、`bge-v2-m3-ce-threshold`。RRFは `1/(60+ngr_rank) + 1/(60+ce_rank)` である。11 hard gateと最初の全gate pass candidateを選ぶ規則はv1から変更しない。

## 唯一のevaluator差分

candidate caseに `returned_source_paths` fieldが存在する場合、空配列も有効な0-hit resultとして使用する。fieldが存在しないbaseline caseだけが `ranked_hits[:5]` へfallbackする。空配列とfield欠落をtruthinessで混同しない。

全candidate/all case empty、一部case empty、negative case empty、baseline expected missing + candidate empty、baseline forbidden absent + candidate emptyを、v2 evaluator自身でresult生成からfull verificationまでround-tripする。empty derived fieldまたはgateのtamperは再計算との差として拒否する。この修正以外のcandidate derivation、score、fusion、threshold、rank、gate判定は変更しない。

## Byte immutabilityとcount scope

v2 manifestはv1 freeze artifacts、claim/error/transport/raw manifest、全raw archiveをSHA-256 registryで検証する。v1 evidenceはJSONとして解釈せず、byte hashだけを照合する。

v2 result-free auditのcountは `freeze_registered_query_execution_count`、`freeze_model_inference_count`、`freeze_observed_result_count` と明記し、いずれも0である。これはv2 freeze中の実行countだけを表し、historical v1 observation countを含めない。shared `~/.ngrdb/knowledge.db`、existing experiment DB、production service、feedback/outcomeを開かない。

## One-time lifecycle

v2専用claim/result/error/transportをexclusive-createし、duplicate、overwrite、development再実行、failed/error development後のholdoutを拒否する。同じsynthetic phase verifierでunobserved、development archived pass/fail/error、holdout archivedをround-tripし、実repositoryを永久unobservedへ固定しない。

freeze merge commitだけをsuccessor observationのprotocol inputにできる。successorはv1 packetを再利用せず、fresh process/DBでv2 developmentをexactly once生成する。全hard gate pass時だけholdoutを一度開く。

## 検証

```text
python -m neuron_graph_rag.cross_encoder_precision_v2_evaluation audit
python -m neuron_graph_rag.cross_encoder_precision_v2_evaluation probe
python -m unittest tests.test_cross_encoder_precision_v2
```

audit/probeとsynthetic testsは登録query、model inference、observed resultを生成しない。
