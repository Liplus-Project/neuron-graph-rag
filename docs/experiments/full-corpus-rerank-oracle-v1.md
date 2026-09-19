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

未観測。result-free audit と synthetic probe を通した後、source commitを固定してpreflightとone-shot developmentを実行する。
