# GitHub RAG / NGR retrieval parity v5

## 目的とfreeze境界

本protocolは、v4 development観測が登録前のpackage importでterminal終了したことを受けた、import-safeなresult-free successorである。v4を同じ方式で再試行せず、repository rootからambient `PYTHONPATH`なしで直接起動できる自己完結runnerへ入口を切り替える。

本freezeではprotocol、runner、schema、fixture、audit testだけを固定する。GitHub RAG request、NGR search、SQLite open、登録artifact作成、development / holdout性能観測は行わない。性能は`not assessed`のまま、観測はfreeze merge後の別successor Issueへ分離する。

## Import-safe共通入口

`tools/run_github_retrieval_parity_v5.py`は、自身のabsolute pathからrepository rootと`src/neuron_graph_rag`を決定論的に解決し、その`src`をprocess-local `sys.path`へ追加してからpackageをimportする。Windows / Linuxとも同じscriptを入口とし、ambient `PYTHONPATH`を成功条件にしない。

`--import-smoke`はpackage import完了後、外部処理を呼ばずに終了する非登録pathである。clean environment testは`PYTHONPATH`を除去し、isolated modeでこのpathを実行して、次のcountがすべて0であることとv5 registry不在を検証する。

- GitHub RAG request
- NGR search
- SQLite open
- artifact create

## Corpusと意味契約の継承

v5は`tests/fixtures/github_retrieval_parity_v4.corpus.json`と`corpora/github-retrieval-parity-v4/`以下の93 Markdownをbyte-exactに参照する。corpus fixtureやMarkdown本文を複製、変更しない。repository、type、path prefixを含む共通requestもv4と同じである。

固定10 query、development / holdout各5 cohort、gold、10 hard gate、GitHub RAG request、NGR `EngineConfig()`、metricの意味と順序はv4から変更しない。case IDとprotocol IDだけをv5固有にし、v4の登録前失敗artifact、capture、rank、resultを入力へ再利用しない。

## Freeze identityと既存証拠

v5 manifestは専用fixture、runner、module、test、文書、production search runtimeをSHA-256で固定する。predecessor registryはv4 protocol inventoryを再帰的に含み、v1からv4までの既存protocol / observation artifactをbyte hashで監査する。特に`tests/evidence/github_retrieval_parity_v4/development.preflight.error.json`について、初回attempt 1、retry 0、登録query 0、shared database open 0のterminal stateを検証する。

v5の出力先は`tests/evidence/github_retrieval_parity_v5/`専用であり、development / holdoutとも`preflight -> capture -> claim -> result`をexclusive-createする。既存stage fileが一つでもあれば同stageを再実行しない。holdoutはdevelopment resultが全hard gateを通るまで開かない。

## Result-free audit

repository rootではdevelopment / holdoutのpreflight、capture、claim、resultがすべて不在で、登録query execution countは0である。whole-module lifecycle testはtemporary rootとsynthetic evidenceだけを使い、result-free、development closed、holdout eligible、drift / overwrite / provenance tamper拒否を検証する。

```powershell
python -I tools/run_github_retrieval_parity_v5.py --import-smoke
python tools/run_github_retrieval_parity_v5.py --audit
python -m unittest tests.test_github_retrieval_parity_v5 -v
```

登録用optionはfreeze後のsuccessor Issue専用である。このIssueでは実行しない。

## 安全境界

- shared `C:\Users\smile\.ngrdb\knowledge.db`を開かず、変更しない。
- v1からv4までのprotocol、corpus、観測証拠を変更、削除、上書きしない。
- production default、MCP登録、SQLite schema、feedback contract、github-rag-mcp production設定、Li+を変更しない。
- result-free freezeや将来の限定観測だけでdefault adoption、MCP replacement、production品質を主張しない。
