# GitHub RAG / NGR retrieval parity v2

## 目的と観測境界

本protocolは、PR #213 / #215を含む現行NGR production retrievalとgithub-rag-mcp hybrid retrievalを、未使用の同一document surfaceで比較するためのresult-free freezeである。本PRはsource、query、gold、metric、gate、capture / claim / result schema、runner、lifecycleだけを固定し、登録queryをどちらのretrieverにも実行しない。性能値は`not assessed`であり、development観測はfreeze merge後の別successor Issueでexactly once実行する。

parity v1とsource-grounded relation v3のquery、gold、capture / packet、rank、resultは参照用のhash registryにのみ置き、v2入力へ再利用しない。既存frozen / observed artifactは変更しない。

## 固定source surface

sourceはpublic repository `Liplus-Project/liplus-language` のcommit `7a2d1a3a9e6fb821c836333900a9f1d145bd1832`直下にある`docs/*.md` 20 files全体である。`tests/fixtures/github_retrieval_parity_v2.corpus.json`はrepository、path、commit、Git blob SHA、UTF-8 content SHA-256、commit固定URL、本文を保持する。取得境界は既存のread-only helper `tools/acquire_github_snapshot.py`であり、共有SQLiteやproduction MCPを開かない。

manifestのrelation edgeはsource本文にあるliteral link tokenを根拠に固定する。

- development: `docs/Home.md` → `docs/G.-Sheepdog-Engineering.md`
- holdout: `docs/F.-Behavior-First.md` → `docs/E.-Li+language.md`

## Query / gold / request

developmentとholdoutはそれぞれ次の5 cohortを固定順で一件ずつ持つ。

1. `direct_lexical`
2. `semantic_paraphrase`
3. `relation_linked`
4. `negative_control`（positive intent + explicit exclusion）
5. `over_exclusion_control`（明示的な除外句をproduction decompositionへ通し、候補本文の局所否定を安全なsourceとして保護）

両splitのcase ID、query、expected / forbidden / protected-safe / relation-seed source identityの和集合はdisjointである。parity v1とrelation v3のcase IDとも重複しない。goldとcohort labelは取得後の評価だけに使い、production searchへの入力はqueryと共通requestだけである。

candidate-side negationのsource premiseは次の組で固定する。

- development: `excluding change`を除外句として分解する。`docs/4.-Operations.md`は`change`を`do not change`という局所否定形だけで含むprotected-safe source、`docs/5.-Notifications.md`は非否定の`changes requested`を含むunsafe comparison sourceである。
- holdout: `excluding export`を除外句として分解する。`docs/C.-Update.md`は`export は不要`という局所否定形で含むprotected-safe source、`docs/6.-Adapter.md`は非否定の`export PATH`を含むunsafe comparison sourceである。

protocol validationは登録queryを検索せず、productionの`decompose_exclusion_intent()`と`apply_exclusion_intent()`だけでこのpremiseを検証する。除外句が空、protected-safe候補が局所否定として保持されない、またはunsafe候補が非否定mentionとして除外されない場合はfreezeをfail closedにする。

共通requestは`repo=Liplus-Project/liplus-language`、`type=doc`、`top_k=10`、`fusion=rrf`、`rerank=true`、`graph_expand=true`、`graph_hops=2`である。NGRは同じ固定repository / document type surface、query、候補幅、graph hop条件をfresh temporary DBへ適用し、fixtureに保存した現行`EngineConfig()` defaultだけを使う。local NGRとremote serviceのdeployment差はlatency品質差として扱わない。

manifestはprotocol artifactとは別に、production searchへ影響する`engine.py`、`retrieval.py`、`exclusion_intent.py`、index adapter、graph / dynamics / storage / model依存を`runtime_sha256`で閉包固定する。lifecycleの`freeze_identity_scope` = `protocol-artifacts-and-production-runtime`は、`artifact_sha256`と`runtime_sha256`の結合をsemantic freeze identityとする。freeze mergeから観測までにこのclosureの1 byteでも変われば、artifact検証とmerge-commit検証は実行前に失敗する。`tools/test_suites.py`はnormal / all分類を維持するが検索・測定結果へ影響しない可変inventoryなので、semantic identityのhash対象には含めない。

## Metric

各caseでexpected rank列、MRR、binary relevance nDCG@10、Recall@10、forbidden hit、protected-safe source保持、relation path completenessを記録する。cohort集約はMRR / nDCG@10 / Recall@10 / relation path completenessを別々に保持し、平均だけで個別case failureを隠さない。

旧v3のHit@5天井を繰り返さないため候補幅は10、順位感度はMRRとnDCG@10で固定した。strict improvementは要求しない。baselineがceilingのcaseも、各non-regression gateで同値または改善の場合だけ通過する。

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

development / holdoutはそれぞれ`capture → claim → result`の登録pathを持ち、全writerはexclusive-createで上書きを拒否する。

- capture: frozen merge commit、stage、全caseの共通request、raw `search`、keyword result全件のraw stored-contentを保存する。stored-contentは`content_source=index`、`content_max_chars=8000`、`not_found=[]`、固定path + source本文prefixとの一致を要求する。範囲外repository / type / pathは黙って除外せずfail closedにする。
- claim: protocol / merge commit / stage / capture SHA-256 / `one_time_claim=true`を固定する。claim作成後の再実行は、failureやpartial failureを含め拒否する。
- result: raw capture全体、protocol artifactとproduction runtime closureを合わせたmanifest hash registry、case / cohort metric、deterministic replay、resource、全hard gateを保持する。execution failureもimmutable failure resultとして残す。

holdout capture登録はdevelopment resultが全hard gateを通るまで拒否する。development failureまたはhard gate failure後はholdoutを開かない。

## Lifecycle audit

repository rootは`result-free`で、development / holdoutのcapture、claim、resultは0件である。`tests/test_github_retrieval_parity_v2.py`はprotocol closureだけをtemporary rootへ複製し、登録queryを実行せずsynthetic evidenceから次をwhole-moduleで検証する。

- result-free
- development-captured / development-claimed
- development-closed
- holdout-eligible
- overwrite / retry拒否
- result metric / gate tamper拒否

観測後のappend-only registryを`audit_repository_lifecycle`で検証するため、観測artifactが存在する正常状態とfreeze時の`audit_result_free`を分離する。

## 実行方法

freeze PRでは先頭2 commandと対象testだけを実行する。capture登録以降はmerge後のsuccessor Issue専用である。

```powershell
$env:PYTHONPATH = "src"
python tools/run_github_retrieval_parity_v2.py --audit
python -m unittest tests.test_github_retrieval_parity_v2 -v

# merge後のsuccessor Issueだけで実行
python tools/run_github_retrieval_parity_v2.py --register-capture development --input local-development-capture.json --protocol-commit <freeze-merge-sha>
python tools/run_github_retrieval_parity_v2.py --stage development --protocol-commit <freeze-merge-sha>
python tools/run_github_retrieval_parity_v2.py --verify development
```

## 安全境界

- shared SQLite `C:\Users\smile\.ngrdb\knowledge.db`を開かず、検索・fixture作成・検証に使わない。NGR replayはstageごとにfresh temporary SQLiteを二つ作り、終了時に削除する。
- feedback、source-use、success、delayed outcomeを接続しない。
- production default、MCP登録、github-rag-mcp deployment、GitHub自動同期、SQLite schema、feedback contract、既存CI bytes、Li+を変更しない。
- Issue / PR / comment / review / diff ingestion parityは対象外である。
- 本freezeまたは将来の支持結果だけではdefault adoption、MCP replacement、production品質を主張しない。
