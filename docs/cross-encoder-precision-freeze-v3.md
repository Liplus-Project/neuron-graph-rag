# Cross-encoder precision benchmark freeze v3

## 目的と境界

この文書は、cross-encoder precision v2 の one-shot development で固定 `raw_logit >= 0.0` filter が positive required documents と relation provenance を落としたため、cross-encoder を除外判定には使わず prefilter hit の順位変更だけに使う result-free v3 protocol を凍結する。v2 evidence を変更、再評価、再実行せず、raw worker payload、raw logit、case/cohort metric、candidate rank を v3 の設計入力に使用しない。v1/v2 evidence は byte hash だけを確認する。

v3 は protocol ID、fixture stem、runtime/archive path、evaluator、test、docs を専用化する。v2 の 24 corpus identities、development/holdout 各8件の bilingual query/gold、2 exact model revisions、dependency lock、passage projection、batch size 8、NGR top 24、model prefilter top 20、4 candidate 順、selection rule、11 hard gate を維持する。source は `c32b3049fd3daaa2190faf5e3e85955a195ee88c` の git bytes へ固定し、NGR default、SQLite schema、default dependency、MCP config を変更しない。

## 固定 model、passage、candidate

- `BAAI/bge-reranker-base@2cfc18c9415c912f9d8155881c133215df768a70`
- `BAAI/bge-reranker-v2-m3@953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`

freeze では model cache、weight、venv を取得、open、推論しない。successor observation だけが claim 前に exact revision の専用 cache を全 hash 再検証して offline へ切り替える。

本文は Unicode code point 480、overlap 80 で分割し、入力は `[query, chunk_text]` のみとする。tokenizer は `padding=True`、`truncation=True`、`max_length=512`、実行は CPU / float32 / eval / no-grad、batch size 8 である。document score は chunk raw logit 最大値、tie は最小 chunk index である。

candidate 順は `bge-base-rrf-rank-only`、`bge-base-ce-rank-only`、`bge-v2-m3-rrf-rank-only`、`bge-v2-m3-ce-rank-only`。CE candidate は全 prefilter hit を `raw_logit` 降順、同値時 `source_path` 昇順で並べ、先頭5件を返す。RRF candidate は全 prefilter hit を `1/(60+ngr_rank) + 1/(60+ce_rank)` 降順、同値時 `source_path` 昇順で並べ、先頭5件を返す。logit の符号や絶対値による filter、sigmoid probability、calibration、query 別 cutoff を導入しない。prefilter hit が5件未満なら全件、0件なら空配列を返す。

既存11 hard gate の意味と順序を維持し、threshold 込み再計算 gate だけを `cross-encoder-fusion-rank-only-recomputation` に更新する。最初の全 gate pass candidate を選ぶ規則は変更しない。

## Result-free verification

全 raw logit が負でも5件以上の prefilter hit から必ず5件を返す。一様な負 shift は CE/RRF の順位と返却集合を変えない。positive/negative/mixed、5件未満、0件、同点を v3 evaluator 自身で result 生成から full verification まで round-trip する。CE/RRF の score、rank、tie-break、returned path、gate derived field の tamper は完全再計算との差として拒否する。

v3 manifest は v1/v2 freeze artifact と全 observed evidence を SHA-256 registry で byte 検証する。evidence は JSON として解釈しない。v3 result-free audit の登録 query、model inference、observed result count はすべて0である。shared `~/.ngrdb/knowledge.db`、existing experiment DB、production service、feedback/outcome を開かない。

## One-time lifecycle

v3 専用 claim/result/error/transport を exclusive-create し、duplicate、overwrite、development 再実行、failed/error development 後の holdout を拒否する。同じ synthetic phase verifier で unobserved、development archived pass/fail/error、holdout archived を round-trip する。

v3 freeze merge commit だけを successor observation の protocol input にできる。successor は v1/v2 packet を移送、変換、再利用せず、fresh process/DB で新しい v3 development packet を exactly once 生成する。全 hard gate pass 時だけ holdout を一度開く。

## 検証

```text
python -m neuron_graph_rag.cross_encoder_precision_v3_evaluation audit
python -m neuron_graph_rag.cross_encoder_precision_v3_evaluation probe
python -m unittest tests.test_cross_encoder_precision_v3
```

audit/probe と synthetic tests は登録 query、model inference、observed result を生成しない。
