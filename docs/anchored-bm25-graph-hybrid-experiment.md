# Anchored BM25 and graph hybrid experiment

## 目的

現行の正方向加算はentry seedのzero-hop値をgraph activationにも含めるため、入口信号とgraph由来信号を分離できない。本実験はentry retrievalを競合外のresidual anchorとして保持し、1 edge以上を通過したmessageだけをlocal graph signalとして統合した場合に、relation retrievalを改善しながらdirect lookupとnegative controlを維持できるか検証する。

既定は`current_positive_additive`のままにする。未観測holdoutで固定gateを通過するまでは変更しない。

## 凍結artifact

- manifest: `tests/fixtures/d1_liplus_anchored_hybrid_experiment.manifest.json`
- development fixture / gold / provenance: `d1_liplus_anchored_hybrid_development.*.json`
- holdout fixture / gold / provenance: `d1_liplus_anchored_hybrid_holdout.*.json`
- contamination audit: `d1_liplus_anchored_hybrid.contamination.json`

developmentはbrake運用clusterの5 nodes / 5 edges、holdoutはDecision Structure clusterの5 nodes / 4 edgesである。両方ともproduction D1へSELECT / WITHだけを実行して取得し、provenanceの全queryで`rows_written=0`、`changes=0`、`changed_db=false`を確認する。

## contamination境界

auditは次の4 prior fixturesからfixture identifierだけを読む。

1. 初回12-node benchmark development
2. neural dynamicsの開封済み9-node holdout
3. local competitionの9-node development
4. local competitionの未開封9-node holdout

合計39 doc pathsをdenylistとし、新development / holdoutとのdoc path、node ID、source URL、relation endpoint重複を拒否する。新しい両split間では、さらにnormalized queryとexpected nodeも分離する。prior goldとprior resultは読み込まない。

## score契約

各traceはBM25とdenseのraw / normalized score、競合前後のentry anchor、graphのraw / normalized activation、final scoreを記録する。pathは`entry_zero_hop`と`graph`を区別する。

`anchored_local_competition`と`anchored_local_query_competition`では、local competitionの初期seedをmessage生成に使うが、最終graph signalから反復decay後のzero-hop seed residualとzero-hop pathを除く。entry anchorは競合処理に渡した値から変更しない。

BM25-onlyでは`use_dense_retrieval=false`と`use_graph_propagation=false`を使い、重みを0にするだけでなくdense encoderとgraph traversal自体を呼ばない。

## 固定variants

最大数と実数を6に固定し、parameter gridを追加しない。

1. `current`: BM25+dense entry + zero-hopを含む現行positive-additive graph
2. `bm25-only`: BM25 entryのみ
3. `bm25-graph-additive`: BM25 entry + 現行positive-additive graph
4. `anchored-local`: BM25+dense anchor + queryなしlocal edge-only graph
5. `anchored-local-query`: BM25+dense anchor + query-conditioned local edge-only graph
6. `bm25-anchored-local`: BM25 anchor + queryなしlocal edge-only graph

## development選択規則

candidateは次の全条件を満たす必要がある。

- relation MRRが`current`より厳密に高い
- direct lookup MRRが`current`から退行しない
- negative-control MRRが`current`から退行しない
- 全relation caseの期待pathが一致する
- success feedbackがcredited path外のedgeと非対象rankを変更しない
- 全caseでentry anchorが競合前後に不変である
- graph signalにzero-hop pathを含まない

複数候補はrelation MRR、worst-cohort MRR、平均展開数、構造複雑度、variant IDの順で一意に選ぶ。候補がなければholdoutを開かず、既定を維持する。

## holdout停止規則

development候補がある場合だけ、holdoutで`current`、`bm25-only`、選択候補を一度評価する。development / holdout resultは既存fileを上書きしない。採用には同じgateを要求し、一つでも失敗すれば`current_positive_additive`を維持する。

fixture、gold、provenance、audit、manifest、variant parameters、threshold、selection rule、stop ruleはfreeze commitをpushするまでrunnerへ渡さない。観測後の変更や再実行は禁止する。

## 実行手順

freeze commitのpush後に次を一度だけ実行する。

```powershell
uv run python tools/run_dynamics_experiment.py development `
  --manifest tests/fixtures/d1_liplus_anchored_hybrid_experiment.manifest.json `
  --output tests/fixtures/d1_liplus_anchored_hybrid_experiment.development.result.json
```

候補がある場合だけholdoutを一度実行し、候補がなければholdout resultを作成しない。

## 観測結果

fixture / gold / provenance / audit / manifest / implementation / tests / 規則をfreeze commit `6400466`としてpushした後、developmentを一度だけ実行した。

| variant | direct MRR | relation MRR | negative MRR | gate |
| --- | ---: | ---: | ---: | --- |
| `current` | 1.0000 | 0.5000 | 1.0000 | baseline |
| `bm25-only` | 1.0000 | 0.2917 | 1.0000 | relation非改善、path / feedbackなし |
| `bm25-graph-additive` | 1.0000 | 0.5000 | 1.0000 | relation非改善、zero-hopを含む |
| `anchored-local` | 0.4167 | 1.0000 | 0.7500 | direct / negative退行 |
| `anchored-local-query` | 0.4167 | 1.0000 | 0.7500 | direct / negative退行 |
| `bm25-anchored-local` | 0.5000 | 0.7500 | 0.7500 | direct / negative退行 |

anchored 3 variantsは全relation path、feedback isolation、entry anchor invariant、edge-only graph signalを満たし、relation MRRを厳密に改善した。しかしdirect lookupとnegative-controlの両方が`current`から退行したため、固定gateを通る候補は0件だった。

selectionは`current`、理由は`no_anchored_variant_passed_frozen_gate`である。停止規則に従ってholdoutは開封せず、holdout resultを作成しない。defaultは`current_positive_additive`のままとする。
