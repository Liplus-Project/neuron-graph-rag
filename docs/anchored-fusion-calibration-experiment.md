# Anchored fusion calibration experiment

## 目的

前回のanchored local experimentはentry anchor invariant、edge-only graph signal、relation path、feedback isolationを満たし、relation MRRを改善した一方でdirect lookupとnegative-controlを退行させた。本実験はactivation dynamicsとBM25/dense entryを変更せず、graph normalizationとfinal fusionだけを切り分ける。

既定は`current_positive_additive`のままとし、未観測holdoutで固定gateを通過した場合だけ変更候補とする。

## 凍結artifact

- manifest: `tests/fixtures/d1_liplus_fusion_calibration_experiment.manifest.json`
- development fixture / gold / provenance: `d1_liplus_fusion_calibration_development.*.json`
- holdout fixture / gold / provenance: `d1_liplus_fusion_calibration_holdout.*.json`
- contamination audit: `d1_liplus_fusion_calibration.contamination.json`

developmentはsubagent parallel width cluster、holdoutはSheepdog Engineering clusterで、各3 nodes / 2 edgesのweakly-connected componentである。production D1へSELECT / WITHだけを実行し、全queryで`rows_written=0`、`changes=0`、`changed_db=false`を検証する。

既存7 fixturesのunionである50 unique doc pathsをdenylistとする。新旧および新split相互のdoc path、node ID、source URL、normalized query、expected node、relation endpoint重複を拒否する。prior goldとprior resultはauditへ読み込まない。

3-node componentは全既存pathを除外したproduction D1で確保できる最大の相互disjoint connected componentである。rank指標が粗くなるため、cohort MRRだけでなく個別case rankもgateに含める。

## normalizationとfusion

Graph normalization:

- `max`: positive graph activationを最大値で割る。最大値0なら全0
- `none`: raw positive graph activation
- `l1_mass`:各activationをpositive activation総和で割る。総和0なら全0

Linear fusionは正規化したweightでentry anchorとgraphを加算する。weight合計が1の本実験では`entry_weight * entry + graph_weight * graph`と同一である。

Weighted RRFはentry全nodeをscore降順・node ID昇順で順位付けし、graphはraw activationが正のnodeだけを同じ規則で順位付けする。graph activationが0のnodeはgraph rankを持たず、graph componentを0とする。

graph-positive node数を`P`として、bottom-centered式を使う。

```text
entry_component = entry_weight / (k + entry_rank)
graph_component = graph_weight * (1 / (k + graph_rank) - 1 / (k + P + 1))
final = entry_component + graph_component
```

`k=60`。bottom subtractionはpositive graph集合の仮想最下位rankを0基準にし、graph channelが欠けるentry seedへ過大なpenaltyを与えないために固定する。

各hitはraw / normalized score、entry / graph rank、両fusion component、final、normalization、fusion strategyを保持する。final降順・node ID昇順をresultから再計算できなければgate不合格とする。

## 固定variants

最大数と実数を6に固定し、parameter gridを追加しない。

1. `current`: current positive-additive、max、linear、0.55 / 0.45
2. `anchored-local-unscaled`: anchored local、none、linear、0.55 / 0.45
3. `anchored-linear-conservative`: anchored local、max、linear、0.80 / 0.20
4. `anchored-linear-mass`: anchored local、l1_mass、linear、0.70 / 0.30
5. `anchored-rrf-conservative`: anchored local、bottom-centered RRF、0.80 / 0.20、k=60
6. `anchored-rrf-balanced`: anchored local、bottom-centered RRF、0.65 / 0.35、k=60

## development選択規則

candidateは次をすべて満たす必要がある。

- relation MRRが`current`より厳密に高い
- relation caseを最低1件個別rank改善する
- direct lookup / negative-control MRRが`current`から退行しない
- 全direct / negative-control caseの個別rankが`current`から退行しない
- 全relation pathが固定endpoint / edge typeと一致する
- success feedbackが最低1 edgeをcreditし、uncredited edgeと非対象rankを変更しない
- entry anchorがcompetition前後で不変である
- graph signalとgraph rankがzero-hop pathを含まない
- traceから全final componentとorderingを再計算できる

複数候補はrelation MRR、worst-cohort MRR、個別改善case数、平均expansions、structural complexity、variant IDの順で一意に選ぶ。候補がなければholdoutを開かない。

## holdout停止規則

development候補がある場合だけ、未観測holdoutで`current`と選択候補を一度評価する。採用には同じgateを要求する。一つでも失敗すればdefaultを維持する。

fixture、gold、provenance、audit、manifest、normalization、weights、RRF k、threshold、selection rule、stop ruleはfreeze commitをpushするまでrunnerへ渡さない。development / holdout resultは既存fileを上書きせず、観測後の変更と再実行を禁止する。

## 実行手順

freeze commitのpush後にdevelopmentを一度だけ実行する。

```powershell
uv run python tools/run_dynamics_experiment.py development `
  --manifest tests/fixtures/d1_liplus_fusion_calibration_experiment.manifest.json `
  --output tests/fixtures/d1_liplus_fusion_calibration_experiment.development.result.json
```

候補がある場合だけholdoutを一度実行する。候補がなければholdout resultを作成しない。

## 観測前状態

fixture / gold / provenance / audit / manifest / implementation / tests / 規則を独立freeze commitとしてpushするまでresultを生成しない。
