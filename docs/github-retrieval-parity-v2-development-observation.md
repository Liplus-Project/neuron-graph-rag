# GitHub RAG / NGR retrieval parity v2 development観測

## 結論

developmentはpre-claim capture validation failureで閉じた。固定5 queryはgithub-rag-mcpへ各1回だけ送信し、keyword result全50件のstored-contentも各1回だけ取得した。しかし、全5 caseのraw searchに固定20-file `docs/*.md` surface外のresultが含まれたため、凍結runnerは候補captureを`ValueError: raw search result identity mismatch`でexclusive registration前に拒否した。

結果を除外・切り詰めず、raw search 5 responseとstored-content 5 responseは`tests/evidence/github_retrieval_parity_v2/development.capture.rejected.json`へ保存した。登録pathへの直接書込やvalidator迂回は行っていない。developmentの登録capture / claim / resultとholdoutの全artifactは不存在である。同じprotocolとqueryを再実行しない。

## 観測前検証

- freeze merge commit: `3b43a36c12fe6141944fafc007ae1d25e6b8279c`
- source: `Liplus-Project/liplus-language` commit `7a2d1a3a9e6fb821c836333900a9f1d145bd1832`の`docs/*.md` 20 files
- result-free audit: source 20 files、development 5 cases、holdout 5 cases、registered observation 0、performance `not assessed`
- target unittest: 8件すべてpass
- artifact / production runtime closure: manifest SHA-256と一致
- freeze commit: v2 manifestを初めて導入し、`origin/main`へmerge済みで、登録artifactを含まない

## 一回性capture

共通requestは凍結値どおり、`repo=Liplus-Project/liplus-language`、`type=doc`、`top_k=10`、`fusion=rrf`、`rerank=true`、`graph_expand=true`、`graph_hops=2`である。

| case | expected sourceのkeyword rank | keyword results | surface外 | graph results | stored-content / not found |
|---|---:|---:|---:|---:|---:|
| `v2-dev-direct-source-format` | 1 | 10 | 9 | 0 | 10 / 0 |
| `v2-dev-semantic-dialogue-driven` | 1 | 10 | 6 | 0 | 10 / 0 |
| `v2-dev-relation-sheepdog` | 取得なし | 10 | 10 | 0 | 10 / 0 |
| `v2-dev-negative-language-config` | 5 | 10 | 5 | 0 | 10 / 0 |
| `v2-dev-overexclusion-operations` | 5 | 10 | 9 | 0 | 10 / 0 |

最初のcaseではrank 1の`docs/K.-Source-File-Format.md`に続き、固定surface外の`docs/K.-Router-Mechanism-Design.md`、`skills/*`が返った。他caseでも`rules/*`、`adapter/*`、`README.md`、`Li+config.md`、`Li+update.md`などが返った。これらを黙って除外すればraw responseを変更し、Issue #218と凍結capture contractのfail-closed境界を破るため、候補capture全体を登録しなかった。

raw evidenceのSHA-256は`ff4efab8118aea2bf7a7af9f0fb03a0421700e8e1843704a79e621bb5bb91a90`である。caseごとの全surface外path / vector ID、登録command、error、registry count、stage closureは`tests/evidence/github_retrieval_parity_v2/terminal-evidence-manifest.json`に記録した。

固定surface内だった11 keyword resultは、対応するstored-contentのrepository / type / path / content / character count / truncation provenanceが固定corpusと一致した。failureはその11件を含むraw response全体にsurface外39件が混在したことによる。

## 評価とstage closure

capture validationはNGR replayとmetric計算より前のhard boundaryで失敗した。このため、NGR fresh temporary DB replayは0回、MRR / nDCG@10 / Recall@10 / forbidden hit / protected-safe保持 / relation path completenessはすべて`not assessed`で、10 hard gateの正式resultも生成していない。protocol-integrity preflightが通った事実と、登録stageのhard-gate resultは区別する。

developmentは`closed-pre-claim-validation-failure`、holdoutは`closed`である。holdoutを開く条件であるdevelopmentの登録`all_hard_gates_pass=true` resultは存在せず、successor Issueも作成しない。production default、MCP登録、github-rag-mcp deployment、GitHub自動同期、SQLite schema、feedback contract、Li+は変更していない。

共有SQLite `~/.ngrdb/knowledge.db`は観測経路から開いていない。byte hash probeは別processがfileを使用中だったため取得不能だったが、NGR replay自体がpre-claimで開始されず、共有DB open countは0である。
