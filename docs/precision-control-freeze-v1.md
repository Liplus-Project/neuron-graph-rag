# Precision control freeze v1

## 状態

この文書は GitHub RAG / NGR parity v1 の結果を candidate tuning に使用せず、NGR の既存 score と channel evidence だけを使う post-ranking precision control の result-free protocol を固定する。Issue #133 の PR では登録 query を実行せず、observed result を生成しない。

successor observation の唯一の protocol input は、この protocol manifest を第一親に持たず初めて追加する main 上の squash merge commit である。同じ artifact を含む後続 commit は protocol commit として受理しない。

## 境界

- public `NeuronGraphRAG.search()` の引数と default ranking、score、diagnostics、explanation は変更しない。
- precision control は `EngineConfig.precision_control` に `PrecisionControl` を明示した場合だけ有効になる。
- filter は全候補の final rank を確定した後に適用し、候補を drop / abstain できる。
- filter は final / entry / sparse / dense / graph score、activation、edge、feedback、outcome、SQLite schemaを変更しない。
- query文字列、否定語、case ID、repository固有のpathをrule入力にしない。
- judgment lifecycle、relation type registry、feedback policy、MCP config、shared database、production serviceへ接続しない。

opt-in の最小例は次のとおりである。未指定または `None` は従来どおりの検索になる。

```python
from neuron_graph_rag import EngineConfig, NeuronGraphRAG, PrecisionControl

config = EngineConfig(
    precision_control=PrecisionControl(
        candidate_id="absolute-floor-025",
        minimum_final_score=0.25,
    )
)
engine = NeuronGraphRAG(config=config)
```

## 固定 source surface

公開 source は `Liplus-Project/neuron-graph-rag` の commit `24d948da4ee61b1e5809a5ba89a1c4738512f384` にある20 Markdown documentsである。各pathとgit blob由来のSHA-256は `github_precision_control_v1.corpus.json` に固定する。source identityは `github:{repository}:doc:{path}` とする。

corpusは parity v1 の `Liplus-Project/github-rag-mcp` source identityと交差しない。developmentとholdoutは、expected、forbidden、relation seedを含むgold identityが相互に交差しない。

各stageは次の4 cohortを2件ずつ、計8件持つ。

- `direct_lexical`
- `semantic_paraphrase`
- `relation_linked`
- `negative_control`

全16 query identityはparity v1 query identityと異なる。relation caseは固定source、target、`informs` edgeを持つ。negative controlはexpected sourceを持たず、各caseで一つのdirectionalまたはunrelated forbidden sourceを固定する。

## 固定 candidate set

candidateは次の5件をこの順序で固定する。

1. `absolute-floor-025`: `final_score >= 0.25`
2. `top-ratio-055`: `final_score / top_score >= 0.55`
3. `top-margin-020`: `top_score - final_score <= 0.20`
4. `entry-graph-agreement`: positive entry signalと、1 edge以上を通るpositive graph signalの両方を要求する
5. `combined-balanced`: absolute `0.15`、ratio `0.40`、margin `0.40`、entry / graph agreementをすべて要求する

閾値はscoreのnormalized domainとchannel-presence semanticsから結果観測前に固定した。parity v1のobserved score、rank、forbidden countは閾値、candidate数、順序の入力にしていない。

複数candidateが全development hard gateを通過した場合、上記順序で最初のcandidateを選ぶ。通過candidateが0件の場合はfailed development evidenceを保存し、holdoutを開かずcross-encoder検討へ停止する。

## Explanation contract

opt-in時、各rank済み `SearchHit` は `precision_control` explanationに次を保持する。

- candidate ID、採否、適用rule、threshold、rule別bool
- pre-filter rank / score、top score、top-score ratio / margin
- entry signalとedge-only graph signalの有無
- repository、commit、path、source URL、content hashのうちnode metadataに存在するsource provenance

既存のscore、fusion component、entry / graph rank、raw relation pathはそのまま残る。returned hitは採用候補だけだが、trace diagnosticsにはdropを含む全rank済み候補のdecisionを保存する。したがって採否と順位はscoreを変更せず再計算できる。

## Development gate

次の11 gateはすべてhard gateである。

1. protocol artifact / source blob integrity
2. parity v1 / development / holdout identity separation
3. fresh databaseでのdeterministic replay
4. direct case別rank non-regression
5. semantic case別rank non-regression
6. relation case別rank non-regression
7. direct / semantic / relation cohort MRR・Hit@5 non-regression
8. negative-control forbidden source countとcase別rankのstrict improvementまたは完全排除
9. expected source top-5 completeness
10. relation source / edge-only path provenance
11. score、activation、edge、feedbackを変更しないpost-ranking isolation

baselineはprecision controlを指定しない現行NGR rankingを保存する。各candidateは同一fixtureから作るfresh databaseで比較し、scoreの再学習やfeedbackを行わない。全hard gateが通ったdevelopmentだけがholdoutを一度開ける。holdoutにも同じgateとfresh database isolationを適用する。

### Raw result evaluator

各stageのbaselineと全candidateは、登録順8 caseを省略せず、各caseでcorpus全20件のpre-filter rank、final / entry / normalized graph score、source provenance、edge-only relation path、返却source pathを保存する。candidateはさらに全20 decisionをcaseごとに保存する。case、hit、source、relation path、decisionはunknown fieldを許さないexact schemaである。

cohort rowは4 cohortを固定順で各2 case保持し、MRRとHit@5をraw returned pathとgoldからevaluatorが再計算する。state rowはprimary / replayの異なるfresh database identity、ranking / score / activation digest、edge before / after digest、feedback before / after countをexact schemaで保持する。

result writer / verifierはpayloadの自己申告boolをauthorityにしない。evaluatorがraw case、hit、decision、provenance、stateからcandidate別11 gate、cohort aggregate、candidate summaryを再生成し、最初の全gate通過candidateを選ぶ。top-level gateは選択candidateのgateと一致し、通過candidateがなければcandidate全体のgate別ANDを記録する。derived field、gate、status、selectionのどれかが再計算結果と異なるpayload、または空のcase / cohortをfail closedに拒否する。

## One-time lifecycle と archive

runtime outputとmain保存pathは最初から分離する。

- runtime: `tests/runtime/github_precision_control_v1/`
- archive: `tests/evidence/github_precision_control_v1/`

各stageはexact protocol ID、freeze commit、artifact hash registry、stage、`one_time_claim=true`をexclusive-createしてから実行する。resultはclaim hash、同じfreeze commit、candidate順序、全gateのexact shapeを相互検証し、exclusive-createする。claim / resultのduplicate、overwrite、development再実行を拒否する。

stage完了時はclaim / resultをbyte-preservingにarchiveへ移し、runtime path、archive path、SHA-256、移送後byte identityをstage別transport manifestへ記録する。failed developmentも同じように保存する。freeze phaseではruntime / archive outputがともに不在であり、observation後のCIは完全なarchive pairとtransport manifestを検証するため、artifact不在を永久要求しない。

## Result-free verification

freeze PRで許可するコマンドは次だけである。いずれも登録queryを実行しない。

```text
PYTHONPATH=src python -m neuron_graph_rag.precision_control_evaluation audit
PYTHONPATH=src python -m neuron_graph_rag.precision_control_evaluation probe
PYTHONPATH=src python -m unittest tests.test_precision_control -v
```

`audit`はprotocol hash、source blob、identity、candidate、gate、result-free audit、phase stateを検証する。`probe`はsynthetic claim / resultだけを一時directoryでexclusive-createし、runtimeからarchiveへのbyte-preserving round-tripを検証する。実corpus query、github-rag-mcp、shared `~/.ngrdb/knowledge.db`、既存experiment DB、feedback / outcomeには触れない。

## Parity v1 isolation

manifestはparity v1のmanifest、corpus、queries、gold、gateと保存済みdevelopment capture / claim / result / transportの現行byte hashをimmutable registryとして監査する。このregistryは変更検知だけに使用し、v1 artifactの内容、query、gold、observed valueをsuccessor candidate評価へ入力しない。v1 artifactは移動、変更、再実行しない。

このfreezeだけではprecision改善、production品質、default変更、MCP統合、cross-encoderの要否を支持しない。
