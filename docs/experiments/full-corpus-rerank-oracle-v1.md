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

## 無効化された観測

source commit `d9cbf02f44d69fef7ab5a759de5221d37d5d90f5` を固定し、fresh WSLC volumeでpreflightを通した後にone-shot developmentを実行した。preflightはnetwork無効、93文書、両modelのrevision / file hashを検証し、登録query実行0回、synthetic forward 2回だった。

実行後のreviewで、`github_retrieval_parity_v5.queries.json` と `github_retrieval_parity_v5.gold.json` はそれぞれdevelopment 5件とholdout 5件を同居させたmixed-stage fileであり、v1 loaderの `read_json()` はstage選択前にfile全体をparseしていたことが判明した。workerとfinalizeはdevelopment行だけを選んだが、登録run環境がholdout contentを物理的に読んだため、「holdoutを読まない」という固定契約に違反する。

| model | 無効化された報告rank | cutoff 20以内という報告 | runtime | peak RSS | 文書 / chunk |
| --- | ---: | --- | ---: | ---: | ---: |
| `BAAI/bge-reranker-base` | 64 | いいえ | 174.80秒 | 1,976,418,304 bytes | 93 / 2,065 |
| `BAAI/bge-reranker-v2-m3` | 45 | いいえ | 577.96秒 | 3,110,289,408 bytes | 93 / 2,065 |

元resultは `semantic_discrimination_bottleneck` を報告したが、この分類は無効であり、Issue #234のbottleneck診断には使用しない。有効な分類は `null` で、v1の再実行または同じresult pathの置換も行わない。

元claim/resultは失敗履歴としてbyte-for-byte保存し、`development.invalidation.json` がclaim file SHA-256 `217ad55dffcab57095840c1d69bc17af0f291fc262cd45c5500ec87589fbad08`、result file SHA-256 `ea4a64fe75b7f415c25fa26375375f07944078f34637838b66e565b6e88c23ca`、result payload SHA-256 `45588ea2bd7542eefafa1fa98233343320fb1fc359b6d055e265b1456670787a` を拘束する。auditは `observed_invalidated`、holdout-bearing input file 2、そこに含まれたunique holdout record 10、有効分類なしを返す。

## 修正版v2の境界

v2は登録runを始める前に、mixed-stage正本から対象development query 1件とgold 1件を専用JSONへmaterializeし、そのsource hash、抽出条件、出力hashをresult-free manifestへ固定する。登録run用bundleはallowlist copyとし、v2 runner、manifest / schema、development-only query / gold、93文書corpus、model registryだけをfresh volumeへ入れる。元のmixed-stage query / gold pathはvolume内に存在しないことをpreflightでfail closedに検証する。

workerはdevelopment-only queryだけ、finalizeはdevelopment-only goldだけを開く。v2は新しいprotocol ID、schema、evidence path、fresh volumeを使い、v1 runtimeまたはv1 evidenceを入力にしない。この修正protocolがresult-free commitとしてreviewを通るまでone-shotは実行しない。
