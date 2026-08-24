# GitHub RAG / NGR retrieval parity v1 観測結果

## 結論

固定12文書・development 4 case に対する結論は `unsupported` である。10 hard gate のうち `negative-control-non-regression` と `expected-source-top-k-completeness` が不通過だったため、protocol の停止規則に従って holdout は観測していない。この結果は repository 統合、default 変更、production 品質を支持しない。

authority は Issue #131 と freeze merge commit `b3cc03a15b81f0e395ae564387a46fe57d320f31` だけである。観測後に protocol、corpus、query、gold、gate、engine config、relation を変更していない。

## Development

| case | cohort | github-rag-mcp rank | NGR rank | github-rag-mcp forbidden hit | NGR forbidden hit |
|---|---|---:|---:|---:|---:|
| `dev-direct-stored-content` | direct lexical | なし | 2 | false | false |
| `dev-semantic-missed-delivery` | semantic paraphrase | 3 | 2 | false | false |
| `dev-relation-memory-philosophy` | relation linked | 3 | 1 | false | false |
| `dev-negative-vector-id-durability` | negative control | 4 | 2 | false | true |

direct case では github-rag-mcp の top-5 に期待 source `README.md` がなく、`expected-source-top-k-completeness` が不通過になった。negative control では NGR top-5 に禁止 source `docs/installation.md` が含まれ、`negative-control-non-regression` が不通過になった。semantic / relation case の順位と cohort 集約値だけで、この二つの個別 failure を相殺しない。

その他8 gate、すなわち protocol integrity、source provenance integrity、deterministic replay、update following、direct case non-regression、cohort MRR non-regression、cohort Hit@k non-regression、source/path explanation integrity は通過した。

## 一回性と安全境界

- development は frozen order の4 queryを各1回だけ live `github-rag-mcp search` へ渡し、各 keyword result 5件の stored-content を case ごとに1回だけ取得した。
- development capture の exclusive register、stage、verify は各1回だけ実行した。再検索、再登録、再実行はしていない。
- holdout の search、stored-content fetch、capture、stage はすべて0回である。
- NGR replay は fresh temporary SQLite 二つだけを使い、feedback は接続していない。replay database は各 `548864` bytes、elapsed time は `1.2997549999854527` 秒と `1.305595700046979` 秒であり、latency / resource は hard gate ではない。
- shared `~/.ngrdb/knowledge.db` は観測前後とも `2875392` bytes、SHA-256 は `84a3fc590eee990579e3ef8130294129934fe93e25f18ac249eece19813c261e` で不変だった。
- github-rag-mcp の read-only search / stored-content fetch 以外の production 操作、NGR feedback / outcome、共有 DB write は行っていない。

## Evidence

- `tests/fixtures/github_retrieval_parity_v1.development.capture.json`: live response と stored-content response の raw capture
- `tests/fixtures/github_retrieval_parity_v1.development.claim.json`: capture SHA-256 と one-time claim
- `tests/fixtures/github_retrieval_parity_v1.development.observed.json`: case metric、cohort metric、gate、NGR replay resource
- `tests/fixtures/github_retrieval_parity_v1.observation-audit.json`: call count、artifact hash、shared DB 前後 hash、holdout 停止理由

この evidence が扱うのは固定 repository commit の12文書と development 4 case だけである。GitHub issue、PR、review、comment、release、diff の取得 parity、別 corpus、長期運用、物理統合には一般化しない。
