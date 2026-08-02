# Independent retrieval channels experiment

## 目的

fusion calibrationではrelation改善とdirect / negative-control維持を一つの順位で同時に満たす候補がなかった。本実験は既存`search()`とdefaultを保持したまま、BM25 lexical laneとanchored edge-only relation laneを`search_channels()`で合成せず同時返却する。

NGRはquery classifier、learned router、cross-lane scalar scoreを実装しない。利用するAIが両laneを検査し、実際に判断根拠として使用したlaneの独立trace IDへfeedbackを返す。

## channel契約

Lexical lane:

- BM25 raw score降順・node ID昇順
- dense retrievalとgraph propagationを順位へ使わない
- graph pathを保存しない
- feedbackを記録してもedgeを変更しない

Relation lane:

- BM25+dense entry上位をseedにする
- `anchored_local_competition`を使う
- 1 edge以上を通過したpositive graph nodeだけを返す
- raw graph activation降順・node ID昇順
- entry scoreをrelation順位へ融合しない
- feedbackは保存済みcredited pathだけを強化する

各laneは同一queryと時刻を持つ独立`trace_id`として保存する。同じnodeが両laneに現れても重複を消さず、laneごとのrankと説明を保持する。envelopeは`agreement_node_ids`を持つが、combined hits、combined rank、final score、single winnerを持たない。

## model-facing利用契約

将来のadapterは次の意味をtool description自体へ含める。

```text
Inspect both independent retrieval channels before choosing evidence. The lexical channel contains sources that explicitly match the query by BM25. The relation channel contains sources reached through one or more graph edges from BM25+dense entry seeds. The channels have separate trace IDs and their scores are not comparable across channels. Retrieval alone does not mean selected, validated, or used and never reinforces edges. When a source actually becomes a basis of downstream work, send feedback with the trace ID of the channel that was used and that channel's node ID. Lexical-trace feedback never reinforces graph edges; relation-trace feedback reinforces only its stored credited path.
```

既存`ngr.mcp.feedback/v1`の`hits`をsilent reinterpretationしない。adapter実装時はlane envelopeと独立trace IDを明示する新contract versionを使う。

## 凍結D1 split

production D1 `search_docs` / `doc_edges`へSELECT / WITHだけを実行した。各splitの5 queryすべてで`rows_written=0`、`changes=0`、`changed_db=false`を確認した。

- development: `didd-umbrella-naming` → `DiDD`
- holdout: `skill-trigger-declaration-in-description` → `agentic-search-five-phase-refactor`

各splitは2 nodes / 1 directed `mention` edgeである。既存9 fixturesの58 nonblank `doc_path` / `file_path`値、新split相互についてdoc path、file path、node ID、source URL、normalized query、expected node、relation endpointの重複を拒否する。auditはprior fixture identifiersだけを読み、prior goldとresultを読み込まない。

## 4-case hard gate

各splitは結果観測前に次の4 caseを固定する。

1. source文書のdirect BM25 lookup
2. target文書のdirect BM25 lookup
3. sourceをanchorとしてtargetへ進むrelation lookup
4. targetをanchorとし、outgoing edgeがないrelation laneが空になるdirectional negative

2-node fixtureは一般性能を推定するには小さい。cohort MRR / Hit@1、union coverage、agreement率は補助指標とし、採否は次の個別hard gateを正本とする。

1. lexical hit / rank / raw・normalized scoreがisolated BM25-onlyと一致
2. lexical direct / directional-negativeのrankがBM25-onlyから退行しない
3. relation hit / rank / graph activationがisolated anchored graph-onlyと一致
4. relation MRRがBM25-onlyより厳密に高く、relation caseを個別改善
5. relation pathが固定endpoint / edge typeに一致し、zero-hopを含まない
6. combined score / rank / winnerが存在しない
7. 両laneが異なるtrace IDと同一queryを持つ
8. searchだけではedge不変
9. lexical traceへのsuccessでは全edge不変
10. relation traceへのsuccessではcredited edgeだけが変化
11. cross-lane node / trace誤用をatomicに拒否
12. trace IDを除く全lane結果が決定論的

## result-free freezeと停止規則

凍結artifact:

- `tests/fixtures/d1_liplus_channels_experiment.manifest.json`
- `d1_liplus_channels_development.json` / `.gold.json` / `.provenance.json`
- `d1_liplus_channels_holdout.json` / `.gold.json` / `.provenance.json`
- `d1_liplus_channels.contamination.json`

fixture、gold、provenance、audit、implementation、tests、channel定義、seed規則、rank規則、feedback規則、threshold、gate、stop ruleをfreeze commitとしてpushしてからdevelopmentを一度だけ実行する。

```powershell
uv run python tools/run_channel_experiment.py development `
  --manifest tests/fixtures/d1_liplus_channels_experiment.manifest.json `
  --output tests/fixtures/d1_liplus_channels_experiment.development.result.json
```

全development gateを通過した場合だけholdoutを一度実行する。一つでも失敗した場合はholdout resultを作らず、既存`search()`をdefaultとして維持する。holdout観測後のartifact変更、result上書き、再実行を禁止する。

## 観測結果

result-free freeze後に記録する。
