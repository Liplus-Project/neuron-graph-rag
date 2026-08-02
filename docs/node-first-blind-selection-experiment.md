# Node-first blind selection experiment

## 目的

blind selection v2はgold-free packetとtrace-bound response validationを固定したが、developmentの3番目のjudgeが4 case中1 caseを欠落させ、capture段階で停止した。retry、replacement、aggregation、accuracy計算、holdout開封は行っていない。

停止後にIssue #23へ記録された非gated診断では、schema-validだった2 judgesは同じ4 node選択を返し、各3/4が固定nodeと一致した。一方、同じnodeが両laneにあるcaseのtrace選択と、relation intentのnode選択は別問題だった。本実験は診断値をv2結果として再採点せず、次の二軸をv3として結果前に分離する。

1. 1 case 1 invocationで複数case responseの欠落状態を構造的に除去する。
2. evidence correctnessをnode IDで採点し、channelは実trace provenanceとして観測する。

既存`search()`、`search_channels()`、default、rank、feedbackは変更しない。learned router、query classifier、cross-lane scalar、LLM invocationをcoreへ追加しない。

## Version境界

v1 / v2のmanifest、prompt、packet、judge artifact、result、source、runner、tests、experiment docsをhash inventoryへ固定する。raw checkout bytesを最初に照合し、raw不一致時はLF / CRLFの完全な相互変換だけを許可する。本文差、mixed newline、bare CRを拒否する。

v2のinvalid responseはretry、補完、再aggregateしない。v2 holdoutはv3 development全gate通過まで次を禁止する。

- holdout stage / case packet生成
- holdout上の`search_channels()`実行
- judge提示
- v3 gold scoring

hash、schema、既存contract testのnon-surfacing readは許可するが、selectionへ使わずholdout open countへ含めない。

## V3 query contract

各splitはv1の4 case順とopaque `case-0001`から`case-0004`を継承する。source direct、target direct、target-anchor directional negativeのqueryはv1のまま保持する。relation queryだけ、anchor sourceではなくedge先targetを求めるtaskとして固定する。

- development: `Which Dialogue-Driven Development method page is linked from the DiDD umbrella naming decision for PR 1468?`
- holdout: `Which agentic search refactor is linked from the skill trigger declaration-in-description decision?`

expected nodeとpathはv1 goldを継承するがpacketへ含めない。query、case順、relation taskはjudge response観測後に変更しない。

## Stage packetとsingle-case packet

stage generatorはsplitごとに一度だけ起動し、各queryへ`search_channels()`を一度実行する。stage packetは4 caseを保持し、各caseからjudge提示専用のsingle-case packetを生成する。

case packetは次を含む。

- opaque case IDとquery
- lexical / relation lane semantics
- laneごとの独立trace ID
- lane内rank、node ID、title、content、source metadata
- relation hitのraw / projected path
- agreement node IDs
- 単一response schema

case packetは`case` object一つだけを持ち、`cases` arrayや他case referenceを持たない。cohort、expected node、intended channel、acceptable rank、gold path、gate、prior result、channel scoreを含めない。

Judge responseもarrayでなく、次のJSON object一つだけである。

```json
{
  "case_id": "case-0001",
  "selected_channel": "lexical",
  "trace_id": "selected lane trace ID",
  "node_id": "selected hit node ID",
  "rationale": "packet-based reason"
}
```

responseのcase IDはpacketの唯一のcaseと一致し、lexical / relation選択ではtraceとnodeが同じlaneに属することを要求する。abstainではtrace / nodeをnullにする。

## Judge isolation

Developmentは4 case x fresh 3 judgesの12 independent invocationsである。

- `fork_turns=none`
- 一つのcase packetと固定promptだけを渡す
- repo、web、tool、gold、他case、他judge、prior responseを渡さない
- agent contextを別caseへ再利用しない
- raw responseをjudgeごとにexclusive writeする
- judge ID、model、agent type、timestamp、raw / parsed responseを保存する

actual LLM invocation、raw capture、aggregateはparent orchestrator境界で行う。implementation subagentはactual responseを読まない。repository coreとCIはLLMを呼ばない。

## Node-first aggregation

caseごとに`node_id`の2/3以上をnode majorityとする。channelをmajority keyへ含めないため、同じnodeへlexical 2票 / relation 1票のように分かれてもnode majorityは成立する。abstainはnode IDなしとして扱い、majorityにしない。

採否の正本はmajority nodeと固定nodeの一致である。channel voteは分布として保存し、gold channelとの一致をgateやmetricにしない。各responseは自身のselected trace / node membershipを満たす必要があり、channel分離はprovenance validationを弱めない。

relation traceを選んだresponseは、selected hitのraw pathを保存したまま`source_id / target_id / edge_type`へ射影し、v1 relation pathと一致してzero-hopでないことを監査する。lexical traceではpathを要求しない。

feedbackは実行しない。選択traceから、lexicalならedge reinforcementなし、relationなら保存済みcredited pathだけが対象になるprovenanceを記録する。検索・capture・scoringのいずれもedgeを変更しない。

## Result-free freeze

次を一つのresult-free commitとしてpushする。

- `src/neuron_graph_rag/node_first_selection.py`
- `tools/run_node_first_selection.py`
- v3 manifestとsingle-case prompt
- synthetic 12-invocation tests
- requirements、README、本文書

freeze commitにはdevelopment / holdout stage packet、case packet、judge response、resultを含めない。manifestは各artifactの固定path template、3 judges x 4 cases、query override、12 gate、stop rule、v1 / v2 hashを保持する。全観測artifactはexclusive createし、上書きを拒否する。

freeze監査:

```powershell
$env:PYTHONPATH='src'
python tools/run_node_first_selection.py audit-freeze `
  --manifest tests/fixtures/d1_liplus_channels_node_first_experiment.manifest.json
```

Development stage生成:

```powershell
$env:PYTHONPATH='src'
python tools/run_node_first_selection.py generate-stage development `
  --manifest tests/fixtures/d1_liplus_channels_node_first_experiment.manifest.json
```

freeze後の順序は固定する。

1. development stageと4 case packetを一度生成
2. 各caseをfresh 3 judgesへ個別提示
3. 12 raw responsesを一度capture
4. development resultを一度aggregate
5. 一つでもgate不合格ならholdoutを生成せず停止
6. 全gate通過時だけholdout stageと4 case packetを一度生成
7. developmentと異なるfresh 12 judgesで一度評価
8. holdout resultを一度aggregateして終了

## 12 hard gates

1. prompt / packetに禁止field、答えを示す値、channel scoreがない
2. 12/12 responseがschema-validでtrace / node membershipを満たす
3. 全4 caseでnode majorityが成立する
4. 全4 caseでmajority nodeが固定nodeと一致する
5. relation taskでmajority nodeがedge targetと一致する
6. relation traceを選んだ全responseのprojected pathが一致しzero-hopでない
7. 各invocationが唯一のcase IDだけへ回答する
8. abstainがcase majorityにならない
9. searchとscoringでedgeが不変である
10. node correctnessとchannel distributionが分離されgold channel gateがない
11. v1 / v2 artifactが固定hashと一致する
12. packet / response / resultの上書きを拒否する

補助metricはjudge node accuracy、majority node accuracy、node unanimity、abstain rate、selected-node MRR、lexical / relation trace usage、correct-node channel split、path-backed correct-node rate、union oracle gapである。

## Holdout gateと主張範囲

Developmentの12 gate全通過時だけ未開封holdoutをfresh 12 judgesで一度評価する。同じgateを全通過した場合だけ`blind node-first LLM selection supported on the frozen minimal holdout`と記録する。

通過してもNGR一般性能、任意modelへの一般化、production router、default変更、`search_channels()`全体のvalidated判定を意味しない。holdout観測後のquery、prompt、judge数、majority単位、gold、threshold、gate、result変更・再実行を拒否する。

## 観測状態

result-free freeze前であり、v3 development stage / case packet、actual judge response、development resultは未生成である。v2 holdoutは未開封である。
