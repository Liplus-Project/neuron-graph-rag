# Local recurrent competition experiment

## 目的

PR #12で観測されたrecurrent competitionのrelation改善とdirect / negative-control退行を分離するため、競合をglobal node集合から同じsourceのsibling neighborへ局所化する。query relevanceとactive path identityを独立ablationとして比較し、未観測holdoutが固定gateを通過した場合だけdefault変更を許可する。

旧developmentはfamilyと`recurrent-balanced` baselineを選んだ探索的根拠に限定する。旧holdoutのgold、rank、metrics、resultは本実験へ読み込まない。

## 固定入力

- manifest: `tests/fixtures/d1_liplus_local_competition_experiment.manifest.json`
- development fixture / gold / provenance: `d1_liplus_local_competition_development.*.json`
- holdout fixture / gold / provenance: `d1_liplus_local_competition_holdout.*.json`
- contamination audit: `d1_liplus_local_competition.contamination.json`
- 両fixtureはproduction D1 `search_docs` / `doc_edges`からread-only取得した9 node / 11 `mention` edgeのweakly-connected subgraph
- developmentはjudgment-learning / self-evolution / retrieval-surface cluster
- holdoutはwiki-sync / Character_Instance cluster

acquisition provenanceはschema fingerprint、取得時刻、coverage、zero-write evidence、known gapを保持する。D1はlossy search snapshotであり、GitHubをbyte-exact sourceの正本とする。

## Contamination boundary

auditは次の重複が空であることを固定する。

- 新developmentと新holdoutのdoc path、node ID、source URL
- 両新splitと旧development fixture
- 両新splitと開封済み旧holdout fixture
- 新split間のnormalized query、expected node、relation endpoint

旧holdoutはfixture identifierだけをdenylist照合する。旧holdout goldとresultはaudit toolもexperiment runnerも読み込まない。

## 固定variant

variantは6件で、追加parameter gridを持たない。全variantは同じentry / graph weight、seed count、hop limit、hop decay、recurrent step、decay、activation budget、inhibition ratioを使う。

| role | variant | structure |
|---|---|---|
| baseline | `current` | current positive additive |
| baseline | `recurrent-balanced` | PR #12のbest prior recurrent |
| candidate | `local-neighbor` | sourceごとのsibling neighbor競合 |
| candidate | `local-neighbor-query` | sibling競合 + query relevance |
| candidate | `local-neighbor-path` | active pathごとのsibling競合 |
| candidate | `local-neighbor-query-path` | active path競合 + query relevance |

path identityはseed IDとruntimeで通過したedge target列から決める。gold path、expected node、case ID、特定query文字列をruntime条件にしない。path stateはnodeごとに最大4件、全伝播は`max_propagation_expansions`で停止する。

## Local update

各sourceまたはactive pathについてoutgoing edge messageを作り、その集合内だけで次を行う。

1. query variantだけtarget text / edge typeとのquery overlapでmessageをgateする。
2. sibling集合の最大messageに固定inhibition ratioを掛け、各messageから引く。
3. 残ったmessage総量を`activation_budget * source_activation`以内へscaleする。
4. path variantだけpath identityを維持してnode集約前に上位pathを決定論的に残す。
5. recurrent decayを前step stateへ適用する。

global maximumを使った抑制はlocal候補へ適用しない。diagnosticsは競合集合ごとのsource、path identity、neighbor数、平均query relevance、message総量の前後を保存する。

## Development gate

新developmentで6 variantsを一度評価する。local候補は次をすべて満たす場合だけ通過する。

1. relation MRRが`current`と`recurrent-balanced`を両方厳密に上回る。
2. direct lookup MRRが`current`から退行しない。
3. negative-control MRRが`current`から退行しない。
4. 全relation caseのendpoint / edge type pathが一致する。
5. credited feedback edgeが1本以上変化する。
6. uncredited edgeと非対象case rankが変化しない。

複数候補のtie-breakはworst-cohort MRR、relation MRR、平均展開数、構造複雑度、variant IDの順とする。候補がなければholdoutを開かずdefaultを維持する。

## Holdout stop rule

development候補がある場合だけ、holdoutで`current`、`recurrent-balanced`、選択候補を一度評価する。result fileが存在する場合は上書きを拒否する。

採用にはdevelopmentと同じgateを要求する。一つでも失敗した場合は`current_positive_additive`を維持し、holdout観測後のparameter、gold、doc path、閾値、選択規則、停止規則変更を禁止する。

## 実行境界

fixture、gold、provenance、audit、manifest、implementation、tests、上記規則をfreeze commitとしてpushするまでdevelopment runnerを実行しない。

freeze後の順序は次に限定する。

1. development resultを新規作成する。
2. 固定gateで候補を一意に決める。
3. 候補がなければ終了する。
4. 候補があればholdout resultを一度だけ新規作成する。
5. 固定gateからdefault維持または候補採用を記録する。

品質数値はCI合格閾値にしない。CIはhash、contamination、determinism、局所性、variant上限、停止規則を検証する。

## 観測結果

freeze commit push後に追記する。追記時に固定入力、variant、gate、停止規則を変更しない。

## 適用限界

9-nodeの固定Li+ wiki subsetとfeature-hashing encoderだけを対象とする。D1 snapshot、別graph topology、learned embedding、一般corpusへ結果を外挿しない。
