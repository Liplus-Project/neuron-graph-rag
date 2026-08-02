# Blind LLM channel selection experiment

## 目的

independent retrieval channels v1では、relation laneがrelation caseをrank 1へ改善し、trace / feedback分離も成立した。一方、固定direct controlのrank 1と、raw path shapeを直接比較したmatcherが不合格となり、12 gate中10件で停止した。

v2はretrieval実装やlaneを変更しない。正解情報を含まないpacketだけを見たblind LLMが、下流利用に適したlaneとnodeを選べるかを検証する。既存`search()`、`search_channels()`、default、feedback、v1 artifactは変更せず、validated状態も付与しない。

## blind packet契約

各caseは順序から生成したopaque `case-0001`形式のIDだけを持ち、次を含む。

- query
- lexical / relation lane semantics
- laneごとの独立`trace_id`
- lane内rank、node ID、title、content、source metadata
- relation hitのraw pathと、`source_id / target_id / edge_type`だけへ射影したpath
- 両laneに現れるnode ID
- search前後のedge snapshot hashと不変判定
- judge response schema

packetとjudge promptは、cohort、intended channel、期待node / rank / path、gate、過去結果、v1 case ID、答えを示すrole labelを含めない。lane scoreも含めず、rankはlane内だけで解釈し、lane間の数値比較を禁止する。

judge responseは各caseについて次の5 fieldだけを返す。

```json
{
  "case_id": "case-0001",
  "selected_channel": "lexical",
  "trace_id": "selected lane trace ID",
  "node_id": "selected hit node ID",
  "rationale": "packet-based reason"
}
```

`selected_channel`は`lexical / relation / abstain`だけを許可する。abstainではtrace / nodeをnullとし、それ以外ではtraceが選択laneと一致し、nodeがそのtraceのhitに属することを検証する。

## judge分離境界

各splitはfresh Codex agent 3体で判定する。各judgeは`fork_turns=none`、自己完結prompt、packetだけを受け取り、repository、tool、web、正解情報、他judge出力へアクセスしない。actual LLM callはparent orchestratorが実行し、repository core、runner、CIはjudgeを呼ばない。

各judge artifactはjudge ID、model、agent type、実行時刻、packet byte hash、raw response、parse済みresponseを保持する。3 judgeは互いに異なるIDとし、holdoutではdevelopmentと異なるfresh agentを使う。

## 集約とpath照合

各caseは`(selected_channel, node_id)`の2票以上をmajorityとする。2票がない場合とabstainが最多の場合は集約結果をabstainとする。trace / node所属を検証してから集約し、不正responseを補正しない。

relation caseのpath照合ではraw stepをartifactへ残したまま、各stepを次のshapeへ射影して固定pathと比較する。

```json
{
  "source_id": "...",
  "target_id": "...",
  "edge_type": "..."
}
```

空pathは一致とみなさず、zero-hopを拒否する。

## result-free freeze

v1 merge commit `b15e27882f013bd895032e6edd15489eb5206926`を基準に、v1 manifest、development / holdout fixture・gold・provenance、contamination audit、development result、v1 evaluator / runnerの11ファイルをbyte SHA-256で固定する。

v2 developmentが全gateを通るまでは、holdout packet生成、holdout上の`search_channels()`実行、judge提示、v2 evaluatorによるgold照合を禁止する。artifact不変性を証明するhash計算、schema検証、既存contract testによるprocess内readは許可するが、内容を人またはjudgeへ表示せず、selection判断へ使用しない。この非surfacing invariant readはholdout open countへ含めない。

次を一つのresult-free commitとしてpushした後に観測を開始する。

- `src/neuron_graph_rag/blind_selection.py`
- `tools/run_blind_selection.py`
- judge prompt / v2 manifest
- synthetic tests
- requirements / README / 本文書

freeze commitにはdevelopment / holdout packet、judge response、development / holdout resultを含めない。runnerはpacket、response artifact、resultをexclusive createし、既存fileを上書きしない。

## 実行順序と停止規則

1. result-free commitをpushする。
2. development packetを一度だけ生成する。
3. parent orchestratorがfresh blind judge 3体へ同じprompt / packetを渡す。
4. raw responseをimmutable artifactとして一度だけ保存する。
5. development resultを一度だけ生成する。
6. 12 gateの一つでも失敗した場合、holdout packetを生成しない。
7. 全gate通過時だけholdout packetを一度生成する。
8. developmentと異なるfresh blind judge 3体で判定する。
9. holdout resultを一度だけ生成する。

result-free監査:

```powershell
$env:PYTHONPATH='src'
python tools/run_blind_selection.py audit-freeze `
  --manifest tests/fixtures/d1_liplus_channels_blind_experiment.manifest.json
```

development packet生成:

```powershell
$env:PYTHONPATH='src'
python tools/run_blind_selection.py generate-packet development `
  --manifest tests/fixtures/d1_liplus_channels_blind_experiment.manifest.json `
  --output tests/fixtures/d1_liplus_channels_blind.development.packet.json
```

response captureとaggregateはparent orchestrator境界でだけ実行する。runnerは実LLMを呼ばず、渡されたraw responseを検証・保存・集約する。

## 12 hard gates

1. packet / promptが答えを示すfieldを含まない
2. 全judge responseがschema、trace、node所属を満たす
3. 全caseで非abstain majorityが成立する
4. 全caseでmajority nodeが固定nodeと一致する
5. direct 2件とdirectional negativeがlexicalを選ぶ
6. relation caseがrelationを選ぶ
7. relation pathが射影後に一致しzero-hopでない
8. 各judgeが4件中3件以上の固定nodeを選ぶ
9. lane間score比較規則が存在せずpacketにscoreがない
10. search前後でedge weightが不変である
11. v1 11ファイルが固定byte hashと一致する
12. 観測artifactの上書きを拒否する

補助指標はjudge別node accuracy、majority node / channel accuracy、abstention率、全員一致率、selected-node MRR、union oracle coverageとの差である。小さい2-node / 4-case splitのため、holdout通過時も主張は固定split上の限定的支持に留め、一般性能やdefault適格性へ外挿しない。

## Development観測結果

result-free freeze commit `062c1314c1b63ee34b8963b980b2c51eb2c3e9a0`をpushした後、development packetを一度生成し、parent orchestratorがfresh blind judge 3体へ提示した。

packet SHA-256は`9b7e26fc74ce9b84e382e607fad9dcaf34b4def75a234028d7aa6be09fb46c1b`である。3件のraw responseを受領し、先行2件はcapture validationを通過した。3件目は必須4 caseのうちopaque `case-0003`を返さず、3 caseだけだったため、`Judge must answer every packet case exactly once`でcaptureを拒否した。raw responseは上書きせず保存し、retryとreplacement judgeは実行していない。

capture段階で全response validity gateが不合格になったため、majority aggregation、accuracy、channel accuracy、selected-node MRR、path照合は実行していない。これらを0値として解釈せず、`not_evaluated_gates`と`metrics_status=not_computed_invalid_judge_response`で未評価を記録する。

Development gateは不合格であり、`holdout_status=not_opened_invalid_judge_response`で停止した。停止規則の確認ではholdout packet生成がexit 1となり、holdout packet / resultは存在しない。prompt、packet、judge数、response、集約規則、matcher、threshold、gateを観測後に変更せず、既存`search()`とdefault、`search_channels()`の非validated状態を維持する。

## Post-freeze portability correction

PR #22の最初のLinux CI run `30731318651`では、Windowsでfreezeしたv1 text artifactのCRLF checkout byte hashと、GitHub ActionsのLF checkout byte hashが一致せず、`d1_liplus_channels.contamination.json`のfreeze auditで停止した。内容差ではなく、Gitのtext checkout変換だけが原因である。

その後、mainから作成したfresh Windows worktreeでは、v2 judge prompt、development packet、judge artifactもLFの固定hashに対してCRLFでcheckoutされ、同じ内容のままmanifest読込とobserved artifact auditが停止した。verificationをv1だけに限定すると同じfrozen text契約をv2へ一貫して適用できないため、v1 / v2 frozen text-byte verificationを次の順序へ揃えた。

1. checkout raw bytesのSHA-256を最初に照合する。
2. raw不一致時だけ、全改行がLF一種類ならCRLFへ、CRLF一種類ならLFへexact変換する。
3. alternate bytesのSHA-256が固定manifest値と一致した場合だけ許可する。
4. 本文差分、mixed newline、bare CR、その他のbyte差を拒否する。

manifest、prompt、packet、judge artifact、development result、gate、stop rule、観測結果は再生成・変更していない。固定manifest / resultに記録されたLF artifact SHA-256は次の通りである。

| artifact | SHA-256 |
| --- | --- |
| blind manifest | `ae6f639a8a17d91c405df86c240be4cd63264c4dfe3b8bf1855ac612218c47a7` |
| judge prompt | `5d825eea83d2321a11fc69156066c7cebcac56ca7d7e970be3d63d393c05d28c` |
| development packet | `9b7e26fc74ce9b84e382e607fad9dcaf34b4def75a234028d7aa6be09fb46c1b` |
| judge 1 | `b808975131466cc2a98fdcb6d649ab40267f1f4871c80d3b29361e7b3be0a12b` |
| judge 2 | `9efd813d10b549e69ee79d28c9f08dda0ed3cfcb10018f20e72e560d9074c76c` |
| judge 3 raw | `977d2dfcab55a822be9aa6a60eee5b00562400e8fc1c061fbf815d0745db0305` |
| development result | `1f1c25ba3d29aa01acf97a364cfe303178a08defa7fc39e688ffaf1b389b6d88` |

このcorrectionはjudge response validity失敗を再評価せず、retry、aggregation、metric計算、holdout開封を許可しない。
