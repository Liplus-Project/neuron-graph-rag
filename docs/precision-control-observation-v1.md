# Precision-control one-shot observation v1

## 範囲

この文書は、Issue #135 で実施する `github-ngr-precision-control-v1` の一回限りの観測手順と保存結果を記録する。唯一のprotocol inputはfreeze merge commit `c1577cad5753bdafe9abf301bb60b1787a64927f` である。凍結済みのcorpus、query、gold、candidate、threshold、gate、selection rule、evaluatorは変更しない。

観測runnerは次の面だけを使用する。

- source commitの20 Markdown blobをgit objectからbyte取得し、1 pathを1 nodeとして加工せずindexする。
- baselineと固定順5 candidateについて、primaryとexact replayをそれぞれ固有のtemporary SQLite databaseで実行する。
- 登録requestは固定 `now=0.0`、`limit=5` のままとし、同じ実searchが構築した全20 `SearchHit` をpre-filter順位として保存する。
- candidateの全20 decisionは実searchのexplanation diagnosticsから取得する。
- score、ranking、activation、edge、feedback stateをcanonical JSONでSHA-256化する。
- raw rowsから凍結evaluatorがcohort、11 hard gate、selection、statusを一意に再計算する。

shared `~/.ngrdb/knowledge.db`、既存experiment database、github-rag-mcp、feedback、outcome、production serviceは開かない。default searchやprecision-control実装も変更しない。

## 一回性

実行前の `preflight` はprotocol hash、freeze commit導入点、phase state、既存error evidence不在を確認するだけで、登録queryを実行しない。

developmentはruntime claimをexclusive-createした後に一度だけ実行する。resultはexclusive-createして凍結verifierで再検証し、claim/resultをbyte-preservingにarchiveへ移送する。developmentの全hard gateがpassした場合だけholdoutを同じ手順で一度開く。gate failureではholdoutを開かない。exception時はclaimとerrorを専用archiveへ保存し、同stageの再試行を拒否する。専用archiveの存在もrunnerがfail closedに検査するため、凍結manifestのoutput pathを空に戻してもstageを再実行できない。

```text
PYTHONPATH=src python -m neuron_graph_rag.precision_control_observation preflight
PYTHONPATH=src python -m neuron_graph_rag.precision_control_observation run
PYTHONPATH=src python -m neuron_graph_rag.precision_control_evaluation audit
PYTHONPATH=src python -m unittest tests.test_precision_control_observation -v
```

## 観測結果

development claimを一度だけ作成し、固定順のbaselineと5 candidateをprimary / exact replayのfresh databaseで一度だけ実行した。結果は `failed`、`selected_candidate_id` は `null` である。通過candidateは0件だったためholdoutは開いていない。同一protocol versionの再実行、threshold調整、default変更、parity v2実装は行わない。この結果はcross-encoder検討へ停止する条件に該当する。

全candidateで通過したglobal hard gateは次の4件である。

- `protocol-integrity`
- `identity-separation`
- `deterministic-fresh-db`
- `immutable-post-ranking-isolation`

globalで失敗したhard gateは次の7件である。

- `direct-case-non-regression`
- `semantic-case-non-regression`
- `relation-case-non-regression`
- `cohort-mrr-hit-at-k-non-regression`
- `negative-forbidden-strict-improvement`
- `expected-source-top-k-completeness`
- `relation-source-path-provenance`

candidate別では `absolute-floor-025`、`top-ratio-055`、`top-margin-020` が `semantic-case-non-regression`、`negative-forbidden-strict-improvement`、`expected-source-top-k-completeness`、`relation-source-path-provenance` の4件に失敗した。`entry-graph-agreement` と `combined-balanced` はglobalと同じ7件に失敗した。いずれも11 hard gateの全通過には至っていない。

development evidenceのraw SHA-256は次のとおりである。

- claim: `d3528a66849f8a25fcd4e7030bf199e0b3aa74f3a5d72737340f102ce39006af`
- result: `d553830cbc6006170d6b78b5d864a10495b956954e5658075135b0bf93a0e844`
- lifecycle transport: `dfeb0ca23164361d7a4d9560141e8b0a1190576097b1878acb2097c22905c5f2`
- archive transport: `a591e2fd57d8856a1eaf94f517869ed1bfccd07ef21f0453255eba6ec27acac7`

### Phase-boundary archival workaround

凍結protocolにはdefectがある。`precision_control_evaluation.verify_phase_state()` 自体はarchive済みphaseを検証できる一方、同じfreeze commitでhash固定された `test_frozen_protocol_is_complete_disjoint_hashed_and_result_free` はrepository上のphaseを常にdevelopment / holdoutとも `unobserved` と要求する。このため、凍結manifestのarchive pathへ観測evidenceを残すとfull suiteが失敗する。これはfrozen phase-aware lifecycleの成功ではなく、freeze testとobservation output pathの衝突である。凍結artifact、module、test、manifestは変更しない。

development resultの凍結verifier検証と最初のruntime-to-lifecycle archiveが完了した後、検索・claim登録・stage・評価を再実行せず、次のbyte-preserving archivalだけを行った。

- original runtime: `tests/runtime/github_precision_control_v1/development.claim.json` / `development.observed.json`
- original lifecycle archive: `tests/evidence/github_precision_control_v1/development.claim.json` / `development.observed.json` / `development.transport.json`
- final archive: `tests/evidence/github_precision_control_observation_v1/development.claim.json` / `development.observed.json` / `development.lifecycle-transport.json`
- final archive transport: `tests/evidence/github_precision_control_observation_v1/development.archive-transport.json`

archive transportはoriginal runtime path、original lifecycle archive path、final archive path、各SHA-256、`runtime_verified=true`、移送前後 `byte_identity=true` を固定する。claim、result、lifecycle transportのbytesは変更していない。移送後のdevelopment stage再実行は0回、holdout実行は0回である。

custom observation auditは専用archiveからclaim/result相互束縛、凍結evaluatorによるfull recomputation、lifecycle transport hash、archive transport hash、holdout不在を検証し、`registered_stage_execution_count=1`、`post_observation_stage_reexecution_count=0` を返す。同時に `load_protocol()` がfreeze artifact registryとparity v1 immutable registryを検証し、凍結側のbytesが不変であることを証明する。凍結CLI auditはworkaround後のmanifest output pathを `unobserved` と返すため、専用archiveの観測状態を示すauthorityとしては使用しない。

successor protocolでは、freeze PR時点でsyntheticなarchived phaseを、実際にfreeze hashへ登録する同じphase-state testへ通す必要がある。freeze testが永続的なoutput不在を要求せず、unobservedとarchivedの両方を同じrepository lifecycleで検証できることを観測前に確定する。

shared database、既存experiment database、github-rag-mcp、feedback、outcomeは開いていない。default変更やparity v2実装も行っていない。
