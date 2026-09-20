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

登録 one-shot の完了後に、Stage 1 rank、両 Stage 2 rank、runtime、peak RSS、判定を追記します。

## 再現入口

```powershell
pwsh tools/run_practical_two_stage_retrieval_v1_wslc.ps1 preflight
pwsh tools/run_practical_two_stage_retrieval_v1_wslc.ps1 run
pwsh tools/run_practical_two_stage_retrieval_v1_wslc.ps1 audit
```

`run` は既存 claim/result/error がある場合に上書きせず停止します。同じ protocol の再試行や置換は行いません。
