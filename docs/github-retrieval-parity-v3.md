# GitHub RAG / NGR retrieval parity v3

## 目的と観測境界

本protocolは、Issue #216 / PR #217で凍結し、Issue #218 / PR #219でsurface不一致によりpre-claim終了したparity v2を置き換えるresult-free freezeである。現行NGR production retrievalとgithub-rag-mcp hybrid retrievalへ、実際の`type: doc`索引surfaceと一致する同一の固定Markdown corpusを与える。

本PRはsource、query、gold、metric、gate、capture / claim / result schema、runner、lifecycleだけを固定する。登録queryをどちらのretrieverにも実行せず、性能は`not assessed`のままにする。development観測はfreeze merge後の別successor Issueでexactly once実行する。

v1、v2、source-grounded relation v3のquery、gold、capture / packet、rank、resultはhash registryで参照するだけで、v3入力へ再利用しない。特にPR #219のrejected evidenceは変更、削除、上書きしない。

## 固定source surfaceとpreflight

sourceはpublic repository `Liplus-Project/liplus-language` のcommit `51623e200ab6128cf59cf5b5ac7d8115c31268d6`、tree `034101e7ff123f91d089640e0685514a3a890159`にある全`.md` 93 filesである。`tests/fixtures/github_retrieval_parity_v3.corpus.json`はrepository、path、commit、Git blob SHA、UTF-8 content SHA-256、commit固定URL、本文を保持する。`tools/acquire_github_markdown_snapshot_v3.py`はGitHubのpublic commit / tarballだけを読み、共有SQLiteやproduction MCPを開かない。

github-rag-mcpにはpath filterとcommit filterがない。そのため登録query実行前のpreflightは、queryなし、`type: doc`、repository固定scanでcurrent index surfaceを列挙し、固定93 pathとの集合一致を要求する。`tests/fixtures/github_retrieval_parity_v3.surface-audit.json`はrepository作成時点の2026-01-22T12:27:41Zから2026-09-07T07:15:00Zまで、連続7日windowの33 requestで得た93 unique path、duplicate 0、missing 0、out-of-surface 0、error 0を固定する。source content drift、path欠落、範囲外pathのいずれかを検出した場合は、黙って除外せず登録queryを送る前にfail closedにする。

manifestのrelation edgeは固定source本文にあるliteral tokenを根拠にする。

- development: `skills/model-source-check/SKILL.md` -> `skills/model-agentic-search/SKILL.md`
- holdout: `skills/operations-on-pr-creation/SKILL.md` -> `skills/operations-on-ci/SKILL.md`

## Query / gold / request

developmentとholdoutはそれぞれ次の5 cohortを固定順で一件ずつ持つ。

1. `direct_lexical`
2. `semantic_paraphrase`
3. `relation_linked`
4. `negative_control`（positive intent + explicit exclusion）
5. `over_exclusion_control`（明示的な除外句をproduction decompositionへ通し、候補本文の局所否定を安全なsourceとして保護）

両splitのcase ID、query text hash、expected / forbidden / protected-safe / relation-seed source identityの和集合はdisjointである。v1、v2、relation experimentのcase ID、query text hash、gold source identityとも重複しない。goldとcohort labelは取得後の評価だけに使い、production searchへの入力はqueryと共通requestだけである。

candidate-side negationのsource premiseは次の組で固定する。

- development: `excluding self-review record`を除外句として分解する。`skills/task-subagent-delegation/SKILL.md`は`Do not perform the self-review`という局所否定形だけで含むprotected-safe source、`skills/evolution-rule-effect-measurement/SKILL.md`は非否定の`self-review record`を含むunsafe comparison sourceである。
- holdout: `excluding external source`を除外句として分解する。`rules/model/rule-policy.md`は`not an external source`という局所否定形だけで含むprotected-safe source、`skills/model-frame-check/SKILL.md`は非否定の`external source`を含むunsafe comparison sourceである。

protocol validationは登録queryを検索せず、productionの`decompose_exclusion_intent()`と`apply_exclusion_intent()`だけでこのpremiseを検証する。除外句が空、protected-safe候補が局所否定として保持されない、またはunsafe候補が非否定mentionとして除外されない場合はfreezeをfail closedにする。

共通requestは`repo=Liplus-Project/liplus-language`、`type=doc`、`top_k=10`、`fusion=rrf`、`rerank=true`、`graph_expand=true`、`graph_hops=2`である。NGRは同じ固定repository / document type surface、query、候補幅、graph hop条件をfresh temporary DBへ適用し、fixtureに保存した現行`EngineConfig()` defaultだけを使う。local NGRとremote serviceのdeployment差はlatency品質差として扱わない。

manifestはprotocol artifactとは別に、production searchへ影響する`engine.py`、`retrieval.py`、`exclusion_intent.py`、index adapter、graph / dynamics / storage / model依存を`runtime_sha256`で閉包固定する。`freeze_identity_scope=protocol-artifacts-and-production-runtime`は、`artifact_sha256`と`runtime_sha256`の結合をsemantic freeze identityとする。freeze mergeから観測までにこのclosureの1 byteでも変われば、artifact検証とmerge-commit検証は実行前に失敗する。

## Metric

各caseでexpected rank列、MRR、binary relevance nDCG@10、Recall@10、forbidden hit、protected-safe source保持、relation path completenessを記録する。cohort集約はMRR / nDCG@10 / Recall@10 / relation path completenessを別々に保持し、平均だけで個別case failureを隠さない。

strict improvementは要求しない。baselineがceilingのcaseを含め、各non-regression gateで同値または改善の場合だけ通過する。

## Hard gate

10 gateはすべてhardで、順序もfreeze対象である。

- protocol integrity
- source provenance integrity
- fresh isolated DB二回のdeterministic replay
- direct safety
- negative safety（expected non-regressionかつ両retrieverでforbidden不在）
- over-exclusion safety（protected-safe保持、forbidden不在、metric non-regression）
- cohortごとのMRR / nDCG@10 / Recall@10 non-regression
- 両retrieverのexpected-source completeness
- relation path completeness non-regression（NGRはcomplete必須）
- NGR source / path / commit / blob / content hash / explanation integrity

latency、temporary DB bytes、elapsed timeは記録のみでhard gateではない。

## Capture / claim / result

development / holdoutはそれぞれ`capture -> claim -> result`の登録pathを持ち、全writerはexclusive-createで上書きを拒否する。

- capture: frozen merge commit、stage、全caseの共通request、raw `search`、keyword result全件のraw stored-contentを保存する。stored-contentは`content_source=index`、`content_max_chars=8000`、`not_found=[]`、固定path + source本文prefixとの一致を要求する。範囲外repository / type / pathは黙って除外せずfail closedにする。
- claim: protocol / merge commit / stage / capture SHA-256 / `one_time_claim=true`を固定する。claim作成後の再実行はfailureやpartial failureを含め拒否する。
- result: raw capture全体、protocol artifactとproduction runtime closureを合わせたmanifest hash registry、case / cohort metric、deterministic replay、resource、全hard gateを保持する。execution failureもimmutable failure resultとして残す。

holdout capture登録はdevelopment resultが全hard gateを通るまで拒否する。development failureまたはhard gate failure後はholdoutを開かない。

## Lifecycle audit

repository rootは`result-free`で、development / holdoutのcapture、claim、resultは0件である。`tests/test_github_retrieval_parity_v3.py`はprotocol closureだけをtemporary rootへ複製し、登録queryを実行せずsynthetic evidenceから次をwhole-moduleで検証する。

- result-free
- development-captured / development-claimed
- development-closed
- holdout-eligible
- overwrite / retry拒否
- result metric / gate tamper拒否

観測後のappend-only registryを`audit_repository_lifecycle`で検証するため、観測artifactが存在する正常状態とfreeze時の`audit_result_free`を分離する。

## 実行方法

freeze PRではacquisition verify、result-free audit、対象testだけを先に実行する。capture登録以降はmerge後のsuccessor Issue専用である。

```powershell
python tools/acquire_github_markdown_snapshot_v3.py --repo Liplus-Project/liplus-language --ref 51623e200ab6128cf59cf5b5ac7d8115c31268d6 --output tests/fixtures/github_retrieval_parity_v3.corpus.json --verify
$env:PYTHONPATH = "src"
python tools/run_github_retrieval_parity_v3.py --audit
python -m unittest tests.test_github_retrieval_parity_v3 -v

# merge後のsuccessor Issueだけで実行
python tools/run_github_retrieval_parity_v3.py --register-capture development --input local-development-capture.json --protocol-commit <freeze-merge-sha>
python tools/run_github_retrieval_parity_v3.py --stage development --protocol-commit <freeze-merge-sha>
python tools/run_github_retrieval_parity_v3.py --verify development
```

## 安全境界

- shared SQLite `C:\\Users\\smile\\.ngrdb\\knowledge.db`を開かず、検索・fixture作成・検証に使わない。NGR replayはstageごとにfresh temporary SQLiteを二つ作り、終了時に削除する。
- feedback、source-use、success、delayed outcomeを接続しない。
- production default、MCP登録、github-rag-mcp deployment、GitHub自動同期、SQLite schema、feedback contract、既存CI bytes、Li+を変更しない。
- Issue / PR / comment / review / diff ingestion parityは対象外である。
- 本freezeまたは将来の支持結果だけではdefault adoption、MCP replacement、production品質を主張しない。
