# v5 難問 practical two-stage retrieval v1

[実験・観測](README.md) / [文書索引](../README.md)

## 目的

Issue #234 の既知 development 難問について、CPU で実用可能な二段 retrieval が cutoff 20 を達成できるかを測る探索実験です。対象は既知 development 1件に限り、holdout や production 性能へ一般化しません。既定 retrieval と production 設定は変更しません。

## 結果前に固定した protocol

protocol ID は `github-retrieval-parity-v5-practical-two-stage-retrieval-v1` です。Stage 1 は #240 と同じ `intfloat/multilingual-e5-small` revision `614241f622f53c4eeff9890bdc4f31cfecc418b3` の `structural_centroid` を全93文書へ適用し、`source_id` 昇順で同点を解消して上位50文書を候補にします。

Stage 2 は候補50文書だけを対象に、構造表現と normalized log-mean-exp（temperature 1）を適用します。モデルは次の2つです。

- `cross-encoder/ms-marco-MiniLM-L6-v2` revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`: 新しい CPU 軽量比較軸
- `BAAI/bge-reranker-v2-m3` revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`: #238 から継承した品質軸

query、480/80 codepoint body chunk、構造 prefix literal、tokenizer max length 512、raw logit、`source_id` 昇順 tie-break は既存の固定値を継承します。primary success は両 reranker が正解を rank 20 以内に置くことです。候補漏れは fail closed とし、Stage 2 の結論を作りません。

## MiniLM result-free parity

登録 query を読む前に、同じ offline snapshot から独立にロードした direct `AutoTokenizer` / `AutoModelForSequenceClassification` と Transformers text-classification pipeline を、英語と日本語の synthetic pair で比較しました。token IDs、attention mask、token type IDs は一致し、sigmoid/softmax を通さない raw logit の最大絶対差は `0.0` でした。固定 tolerance は `1e-6` です。

## runtime 境界

Stage 1 は E5 loader 初期化、登録 query embedding、全93文書の structural chunk embedding、centroid、ranking、top-50 選択を含みます。Stage 2 はモデルごとに loader 初期化、候補50文書の tokenization と CPU forward、NLME、ranking を含みます。各 pipeline runtime は共有 Stage 1 と当該 Stage 2 の和で、実用目標はそれぞれ600秒以下です。

モデル download、cache copy、container startup、preflight synthetic forward、gold-only finalization は測定外です。各 worker の peak RSS と source/model/input/output hash を保存します。

## 実行境界

fresh named volume へ allowlist の source、query、corpus、result-free parity と pinned model files だけをコピーします。mixed v5 query/gold、holdout-bearing input、過去 evidence、共有 DB は登録環境に置きません。preflight は gold absent のまま E5 と両 cross-encoder の synthetic forward、model file hash、source hash、offline Linux x86_64 を検証します。

claim 後は Stage 1、MiniLM Stage 2、v2-m3 Stage 2 を retry 0 で一度だけ実行します。worker は gold blind です。両 Stage 2 worker が完了した後だけ development-only gold を追加し、候補包含、rank、primary criteria、runtime criteria を finalizer が計算します。claim、result、error は append-only です。

## 観測結果

source protocol の登録 one-shot は Stage 1、MiniLM Stage 2、v2-m3 Stage 2 の3 workerを完走しました。しかし、finalizer-only goldをnamed volume内のsource treeではなくvolume rootへstreamしたため、finalizer allowlistはdevelopment gold missingとしてfail closedしました。source protocolはresultを作らず、preflight、claim、3 complete worker packets、errorをappend-only evidenceとして保存しています。同protocolは再実行しません。

worker packetのrankはgoldと照合せず、次のSHA-256で固定しました。

- Stage 1: `7a932a6fb6dccbbd2d2a42b7f970ff0595c9c3b4fd9e9191874875a05f76c2ac`
- MiniLM Stage 2: `5f23496625c31ce6be6be64b8fb9fab7ccabb6bf186eda61d5de56653fc19b8a`
- v2-m3 Stage 2: `177e035635f75a961d1e3bf3f120140e6c773ac837cb02399cd8db6d6f036900`

別protocol `github-retrieval-parity-v5-practical-two-stage-finalizer-recovery-v1` は、query 0、model forward 0、retry 0でgold absent claimを完了しました。しかし、manifestに固定したdevelopment gold SHA-256が58桁で、既存ファイルの64桁hashから6文字欠落していたため、rank計算前にfail closedしました。claimとerrorをappend-only evidenceとして保存し、同protocolは再試行しません。

その後の `github-retrieval-parity-v5-practical-two-stage-finalizer-recovery-v2` は、packet verifier、候補50、NLME式、criteria、runtime境界、query 0、model forward 0、retry 0をv1から変えず、development gold SHA-256だけを既存ファイルの正しい64桁 `689028b2a6f827bc915f6d9151c1b6b95164b4dbd59715bb930d573c33c846cf` へ修正した別protocolです。v1 claim/errorをhash-lockし、v1 resultが存在しないことも確認します。fresh gold-absent claimが通った後だけgoldを追加します。回復結果はcompleted gold-blind packetsからの導出であり、source protocol成功、source retry、recovery-v1 retryのいずれでもありません。

recovery-v2 claimはgold absent、holdout-bearing input 0、query 0、model forward 0、retry 0でPASSしました。固定worker packetsから導出した結果は次のとおりです。

| 段階 / model | 正解rank | runtime | peak RSS | cutoff / practical判定 |
|---|---:|---:|---:|---|
| Stage 1 E5 structural centroid | 39 | 62.847秒 | 1,222,049,792 bytes | top 50候補へ包含 |
| MiniLM Stage 2 | 37 | Stage 2 31.524秒 / pipeline 94.371秒 | 814,063,616 bytes | cutoff失敗 / 600秒目標達成 |
| v2-m3 Stage 2 | 15 | Stage 2 655.102秒 / pipeline 717.949秒 | 3,121,111,040 bytes | cutoff達成 / 600秒目標失敗 |

両rerankerがrank 20以内というprimary criteriaは失敗し、両pipelineが600秒以内というpractical criteriaも失敗しました。MiniLMは実用時間内でもrankを改善できず、v2-m3はrankを39から15へ改善しましたが実用時間を超えました。この観測は既知development難問1件だけに対する結果です。recovered result SHA-256は `7166caaac4c1c581140711231de4d373125fad5daeac72b83c199109ea75311e` です。

## 再現入口

```powershell
pwsh tools/run_practical_two_stage_retrieval_v1_wslc.ps1 preflight
pwsh tools/run_practical_two_stage_retrieval_v1_wslc.ps1 run
pwsh tools/run_practical_two_stage_retrieval_v1_wslc.ps1 audit
```

`run` は既存 claim/result/error がある場合に上書きせず停止します。同じ protocol の再試行や置換は行いません。

失敗済みrecovery-v1は再実行しません。finalizer-only recovery-v2は次の段階を別のfresh registered rootで順に実行します。

```powershell
pwsh tools/run_practical_two_stage_retrieval_recovery_v2.ps1 -Phase prepare -RegisteredRoot <fresh-path>
pwsh tools/run_practical_two_stage_retrieval_recovery_v2.ps1 -Phase claim -RegisteredRoot <fresh-path>
pwsh tools/run_practical_two_stage_retrieval_recovery_v2.ps1 -Phase finalize -RegisteredRoot <fresh-path>
pwsh tools/run_practical_two_stage_retrieval_recovery_v2.ps1 -Phase export -RegisteredRoot <fresh-path>
```
