# GitHub RAG / NGR retrieval parity v4

## 目的と観測境界

本protocolは、Issue #220 / PR #221で凍結したretrieval parity v3が、観測前preflightで更新中の`Liplus-Project/liplus-language/main`との索引driftを検出し、登録queryを送らず終了したことを受けたresult-free successorである。v3の固定93 Markdown本文をこのrepositoryの専用pathへbyte-exactにmaterializeし、GitHub RAGの候補母集合をexact `repo + type + path_prefix`で隔離する。

本freeze PRはcorpus、source / query / gold / metric / gate、preflight / capture / claim / result schema、runner、lifecycleだけを固定する。development / holdoutの登録queryは送らず、性能は`not assessed`のままにする。最初の観測はfreeze merge後、github-rag-mcp `v0.11.0` releaseとMCP process再起動を人間が確認した後のsuccessor Issueでexactly once行う。

## 固定corpusとsource identity

専用prefixは`corpora/github-retrieval-parity-v4/`である。末尾`/`を含め64 UTF-8 byte以内とし、prefix以下には固定corpusのMarkdownだけを置く。v3 corpus fixtureの各本文をbyte-exactに複製し、元relative pathをprefix以下に保つ。

`tests/fixtures/github_retrieval_parity_v4.corpus.json`は次を固定する。

- repository: `Liplus-Project/neuron-graph-rag`
- ref: `main`
- path prefixと93 exact paths
- Git blob SHA、UTF-8 content SHA-256、materialized本文
- v3 source repository / commit / original path / commit固定URL
- path、blob SHA、content SHA-256の正規化集合から得た新しいcorpus identity

protocol validationはprefixの実file集合がexact 93 pathsであること、非Markdownや範囲外fileがないこと、各file bytesがv3 fixture本文と一致することを検証する。manifestやschemaなどのmetadataはprefix外に置く。

## Query derivationと非再利用境界

development / holdoutはv3と同じ10 query textを派生利用する。v3 result-free auditが両stageのquery / capture / claim / resultをすべて0件と固定していることを先に検証し、v4 manifestへ同一query text SHA-256集合を明記する。一方、case ID、protocol ID、repository/pathを含むsource identityはv4固有である。

v1 / v2 / v3 / source-grounded relationの観測済みcapture、packet、rank、resultは再利用しない。v3から引き継ぐのは未観測query text、5 cohort、gold semantics、metric、10 hard gateだけである。developmentとholdoutのv4 source identity集合は互いにdisjointである。

## 共通requestとpreflight

GitHub RAG共通requestは次のexact値である。

```text
repo=Liplus-Project/neuron-graph-rag
type=doc
path_prefix=corpora/github-retrieval-parity-v4/
top_k=10
fusion=rrf
rerank=true
graph_expand=true
graph_hops=2
```

各stageは登録query captureより先にqueryless preflightをexclusive-createで登録する。preflightはexact path集合、duplicate / missing / out-of-prefix不在、索引stored content prefix、public source readで得た本文SHA-256を固定corpusと照合する。索引未収束またはdrift時はpreflight登録を拒否し、登録queryへ進めない。`register_capture()`は同じprotocol commitのpassed preflightが存在しない限りfail closedする。

NGR側は同じ93文書、query、`top_k=10`、`graph_hops=2`、固定`EngineConfig()`をfresh temporary SQLiteへ適用する。shared `C:\Users\smile\.ngrdb\knowledge.db`は開かない。

## Metricとhard gate

各caseでexpected rank、MRR、binary relevance nDCG@10、Recall@10、forbidden hit、protected-safe保持、relation path completenessを記録する。5 cohortと10 hard gateの意味・順序はv3を維持する。

- protocol integrity
- source provenance integrity
- fresh isolated DB二回のdeterministic replay
- direct safety
- negative safety
- over-exclusion safety
- cohortごとのnon-regression
- expected-source completeness
- relation path completeness non-regression
- source / path / explanation integrity

latencyとtemporary DB sizeは記録のみでhard gateではない。

## Lifecycle

各stageのregistry順は`preflight -> capture -> claim -> result`で、すべてexclusive-create、上書き・再取得・失敗後retryを拒否する。holdout preflightはdevelopment resultが全hard gateを通るまで開かない。

repository rootはresult-freeで、development / holdoutのpreflight、query、capture、claim、resultはいずれも0件である。whole-module testは登録queryを送らずsynthetic evidenceだけで次を検証する。

- result-free
- development-preflighted / captured / claimed / closed
- holdout-eligible
- preflight drift・prefix逸脱・overwrite拒否
- result metric / provenance tamper拒否

## 実行方法

freeze PRではauditとtestだけを実行する。登録以降はmerge後のsuccessor Issue専用である。

```powershell
$env:PYTHONPATH = "src"
python tools/run_github_retrieval_parity_v4.py --audit
python -m unittest tests.test_github_retrieval_parity_v4 -v

# merge後、release / MCP再起動確認済みのsuccessor Issueだけで実行
python tools/run_github_retrieval_parity_v4.py --register-preflight development --input local-development-preflight.json --protocol-commit <freeze-merge-sha>
python tools/run_github_retrieval_parity_v4.py --register-capture development --input local-development-capture.json --protocol-commit <freeze-merge-sha>
python tools/run_github_retrieval_parity_v4.py --stage development --protocol-commit <freeze-merge-sha>
python tools/run_github_retrieval_parity_v4.py --verify development
```

## 安全境界

- freeze PRでは登録queryを送らない。queryless scanとpublic source readはprefix存在確認にだけ使う。
- shared SQLite、production default、MCP登録、SQLite schema、feedback contract、Li+を変更しない。
- v1 / v2 / v3 artifactとrejected evidenceを変更、削除、上書きしない。
- 本freezeや将来の支持結果だけではdefault adoption、MCP replacement、production品質を主張しない。
