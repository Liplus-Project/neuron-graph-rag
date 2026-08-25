# Cross-encoder precision benchmark freeze v1

## 目的と境界

この文書は、NGR の既定検索が返す上位 20 source を cross-encoder の query-passage 関連度で再順位付けする、独立した result-free protocol を凍結する。既定検索、SQLite schema、既定 dependency、MCP 設定は変更しない。凍結中に登録 query、model inference、観測 result は生成せず、既存 DB、production service、feedback / outcome を開かない。

source は freeze 開始時の `c32b3049fd3daaa2190faf5e3e85955a195ee88c` である。precision-control v1 と GitHub parity v1 の source/query/gold identity を除外し、未使用 docs Markdown 18件と未使用 corpus Markdown 6件を path のみで選んだ。選択に過去の score、rank、観測値は用いていない。

## 固定 model と dependency

- `BAAI/bge-reranker-base@2cfc18c9415c912f9d8155881c133215df768a70`（MIT）
- `BAAI/bge-reranker-v2-m3@953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`（Apache-2.0）

`tests/fixtures/github_cross_encoder_precision_v1.models.json` は Hugging Face repository API の exact revision から、必要 file、size、git blob ID、LFS SHA-256、license を固定する。freeze では weight を download しない。後続観測は claim 前に専用 cache へ取得し、全 hash を検証して offline mode に切り替える。`trust_remote_code` は無効のままとする。

実験専用 dependency は `.requirements.in` と hash 付き `.requirements.lock` に隔離し、project の既定 dependency surface へ追加しない。lock は `uv pip compile --generate-hashes --python-version 3.11 --python-platform windows` で生成した。

## Passage、candidate、gate

本文を Unicode code point 480、overlap 80で決定論的に分割し、入力は `[query, chunk_text]` だけとする。tokenizer は固定 artifact、`padding=True`、`truncation=True`、`max_length=512`。CPU / float32 / eval / no-grad、batch size 8で実行する。document score は chunk raw logit の最大値、tie は最小 chunk indexであり、採用境界は `raw_logit >= 0.0` である。

candidate は `bge-base-rrf-threshold`、`bge-base-ce-threshold`、`bge-v2-m3-rrf-threshold`、`bge-v2-m3-ce-threshold` の順である。RRF は `1/(60+ngr_rank) + 1/(60+ce_rank)`。threshold 適用後、score 降順、source identity 昇順で top 5を返し、最初の全 hard gate pass だけを選ぶ。該当なしなら holdout は開かない。

11 hard gate は protocol/source/model/lock integrity、identity separation、CPU offline replay isolation、positive case rank、cohort MRR/Hit@5、negative case non-worsening と stage aggregate strict improvement、positive completeness、relation provenance、CE/fusion/threshold/rank recomputation、NGR/SQLite state immutability、default surface immutabilityを raw rows から再計算する。baseline と candidate の双方で expected missing の case rankは非退行だが completeness は失敗する。双方で forbidden absent は非悪化であり、candidate pass には stage aggregate の厳格な減少が別途必要である。

## One-time lifecycle

runtime と archive path は最初から分離する。claim / result / error / transport は exclusive-create し、duplicate、overwrite、development 再実行、failed/error development 後の holdout を拒否する。失敗 evidence も byte-preserving transport manifest とともに保全する。repository state は永久に unobserved と仮定せず、同じ synthetic test が unobserved、development archived、holdout archived、failed development archived、error archivedを round-tripする。

この freeze が main に squash merge された後、その merge commit だけを後続 one-shot observation の protocol input にできる。pass しても既定採用や parity v2 は本 protocol の範囲外である。

## 検証

```text
python -m neuron_graph_rag.cross_encoder_precision_evaluation audit
python -m neuron_graph_rag.cross_encoder_precision_evaluation probe
python -m unittest tests.test_cross_encoder_precision
```

audit/probe は登録検索も model inference も行わず、count を 0 として報告する。
