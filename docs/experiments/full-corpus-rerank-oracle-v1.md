# v5 semantic rank-62 full-corpus rerank oracle

## 目的

既知の v5 development `v5path-dev-semantic-axis-separation` について、93文書すべてを CPU cross-encoder で直接採点し、正解が既存の実用的な rerank cutoff 20へ入るかを診断する。これは既知 development 1件だけの探索的証拠であり、未見性能、確認的評価、production品質を扱わない。

## 固定契約

- 入力は v5 の保存済み development query / gold と v4由来 exact 93文書 corpusだけで、holdoutを読まない。
- `BAAI/bge-reranker-base` revision `2cfc18c9415c912f9d8155881c133215df768a70` と `BAAI/bge-reranker-v2-m3` revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` を比較する。
- 全文を 480 codepoint、80 overlap で先頭から末尾まで覆い、文書scoreは最大 raw logitとする。両modelで入力、分割、集約、tie-breakを共通化する。
- 両modelの正解rankが20以内なら `candidate_generation_bottleneck`、両方とも20より下なら `semantic_discrimination_bottleneck`、不一致なら `undetermined` とする。
- workerはgoldを読まず全93文書を採点する。finalizeだけがgoldを開いてrankと分類を再計算する。query / document別加点、gold identityによるranking、観測後のcutoff調整を行わない。
- fresh WSLC volume、Linux x86_64、CPU 4 threads、network無効で一度だけ実行する。共有DB、GitHub RAG、既存cross-encoder runtime volumeを接続せず、許可済みのpinned model cacheだけをread-only入力としてfresh volumeへcopyする。

manifestとschemaは `tests/fixtures/full_corpus_rerank_oracle_v1.*.json`、runnerは `neuron_graph_rag.full_corpus_rerank_oracle`、WSLC入口は `tools/run_full_corpus_rerank_oracle_wslc.ps1` である。CIは `audit`、`probe`、unit testだけを実行し、model downloadまたは登録query推論を行わない。

## 観測結果

source commit `d9cbf02f44d69fef7ab5a759de5221d37d5d90f5` を固定し、fresh WSLC volumeでpreflightを通した後にone-shot developmentを実行した。preflightはnetwork無効、93文書、両modelのrevision / file hashを検証し、登録query実行0回、synthetic forward 2回だった。

| model | 正解rank | cutoff 20以内 | runtime | peak RSS | 文書 / chunk |
| --- | ---: | --- | ---: | ---: | ---: |
| `BAAI/bge-reranker-base` | 64 | いいえ | 174.80秒 | 1,976,418,304 bytes | 93 / 2,065 |
| `BAAI/bge-reranker-v2-m3` | 45 | いいえ | 577.96秒 | 3,110,289,408 bytes | 93 / 2,065 |

両modelとも正解が固定cutoff 20の外だったため、契約どおり `semantic_discrimination_bottleneck` と分類する。この1件ではcandidate generationを除いてもreranker単独で正解を実用候補範囲へ上げられず、既存rank-62の主因をsemantic discrimination側に帰属する探索的証拠となった。これは既知development 1件の診断であり、他query、holdout、production品質へ一般化しない。

append-only evidenceは `tests/evidence/full_corpus_rerank_oracle_v1/` にあり、result payload SHA-256は `45588ea2bd7542eefafa1fa98233343320fb1fc359b6d055e265b1456670787a` である。auditは両model各93文書のsource ID、文字数、chunk数、best chunk、score、rank、runtime、peak RSS、dependency / model revision、input / model / output hashを再検証し、holdout、GitHub RAG、共有DBへのアクセスが0だったことを確認する。
