# Requirements

## 1. Purpose

このプロジェクトは、ハイブリッド検索、型付き知識グラフ、決定論的な活性伝播、成功フィードバックを、一つの観測可能な RAG パイプラインとして提供する。

## 2. Premises

- 公開 API は特定のデータ源に依存しない。
- コアは Python 標準ライブラリだけで動作する。
- MCP 接続は同一 repository 内の任意 adapter とし、コアの必須依存にしない。
- sparse 検索は BM25、dense 検索は決定論的な feature-hashing encoder を既定実装とする。
- dense encoder は差し替え可能とし、既定実装を意味埋め込みモデルとは見なさない。
- グラフは型付き有向エッジであり、重みと事実性を別々に保持する。
- 活性は検索時刻に紐づく動的状態であり、知識の確信度やエッジの事実性とは別軸である。

## 3. Functional requirements

1. 文書ノード、metadata、知識確信度を SQLite に永続化できる。
2. 型付きエッジ、結合重み、事実性を SQLite に永続化できる。
3. BM25 と dense cosine similarity を正規化し、入口スコアへ統合できる。
4. 上位入口ノードから、重み、事実性、hop decay を使って決定論的に活性を伝播できる。
5. 入口スコアとグラフ活性を統合して結果を順位付けできる。
6. 結果ごとに sparse、dense、入口、グラフ、最終スコアと伝播経路を説明できる。
7. 検索 trace と検索結果を、成功フィードバックとは別の記録として保持できる。
8. 成功時に明示された利用ノードへ至る経路だけを強化できる。
9. 検索しただけではエッジ重みを変更しない。
10. 活性値は半減期に従って時間減衰する。
11. 活性減衰はノード確信度とエッジ事実性を変更しない。
12. 同一コーパスで通常のハイブリッド検索とグラフ統合検索を比較できる。
13. 任意 MCP adapter の `search`、`record_source_use`、`record_outcome` 契約を、実装と transport から独立して定義する。
14. source-use を `retrieved`、`selected`、`validated`、`used` に分け、新規 `used` への遷移だけを即時 reinforcement に接続する。
15. `corrected`、`rolled_back` などの delayed outcome を source-use と別に記録し、初期契約では edge weight を自動変更しない。
16. MCP adapter は trace、node、enum、stage 順序、idempotency を境界で検証する。
17. 各 MCP tool の model-facing description 自体が、feedback の呼び分けと reinforcement 条件を consuming AI へ伝える。
18. persistent core の trace は自動 expiry しない。retention を設ける deployment は `search` description と output に期限を明示し、expiry 後の feedback を `unknown_trace` とする。
19. github-rag-mcp の D1 `search_docs` を正本として、repo / type / per-type limit と固定順から決定論的な小 fixture を生成できる。
20. `search_docs.vector_id / content` を node ID / text へ、`doc_edges` の両端と `edge_kind` を typed edge へ変換し、欠損 endpoint は node を捏造せず除外理由を記録できる。
21. D1 取得は単一 SELECT / WITH query のみに制限し、各 query の `rows_written=0`、`changes=0`、`changed_db=false` を検証できる。
22. fixture と分離した provenance report に schema fingerprint、coverage、取得時刻、取得時点で未解消の既知 gap、redaction 件数を記録し、再取得前後の count / commit / 最新時刻を比較できる。未解消の既知 gap がない場合は空配列を記録する。
23. 実 D1 の明示的な wiki doc path 集合から、弱連結で決定論的な評価 fixture を生成できる。gold case と品質結果を見た後に選択集合を変更しない。
24. 12 件以上の gold query を direct lookup、relation、negative control に分け、query、期待 node、許容 rank、source URL、relation 時の期待 endpoint / edge type を保持できる。
25. baseline hybrid と graph-integrated retrieval を同一 corpus、encoder、query で実行し、全体・cohort 別の MRR、Hit@3、rank delta、改善・同値・悪化件数を機械可読に出力できる。
26. relation の説明は score だけでなく、固定した one-hop / two-hop の endpoint と edge type に対して照合する。
27. success feedback 前後で edge weight と全 gold case の rank を比較し、credited path 外の edge 変更と非対象 case の rank 変更を明示する。
28. 活性伝播は共通 interface の下で、現行正方向加算、有限活性 budget、側方抑制、query-conditioned transmission、反復競合を選択できる。
29. 各検索 trace は strategy、伝播 step 数、展開数、活性総量、収束有無、停止理由を決定論的な diagnostics として保持する。
30. neural dynamics experiment は development と doc path が重ならない connected holdout、gold、探索空間、最大 variant 数、選択規則、停止規則を結果観測前に固定する。
31. 候補選択は development result だけで行い、relation MRR と negative-control MRR の Pareto gate、worst-cohort MRR、展開数、構造複雑度、variant ID の順で一意に決める。
32. development gate を通る候補がない場合は holdout を開かず既定を変更しない。候補がある場合だけ holdout を一度評価し、cohort 退行、path 不一致、feedback 汚染のいずれかがあれば採用しない。
33. experiment result は gate 不合格、Pareto 支配、holdout 不採用を含む全 variant を上書きせず versioned artifact として保存する。
34. recurrent activation は global inhibition に加え、同じ source の sibling neighbor だけを競合させる local strategy を選択できる。
35. local recurrent strategy は query relevance と active path identity を独立に有効化でき、競合集合ごとに source、path identity、neighbor 数、query relevance、配分前後の message 総量を記録する。
36. local recurrent experiment は production D1 から read-only 取得した新しい development / holdout を旧 development / 開封済み holdout および相互間で分離し、両 provenance と contamination audit を結果観測前に固定する。
37. 旧 development result は family と baseline の探索的根拠だけに使用し、旧 holdout は fixture identifier の重複拒否以外では読み込まない。
38. local recurrent experiment は `current`、`recurrent-balanced`、neighbor / query / path の4 ablationを合わせた6 variantsに固定し、parameter gridを追加しない。
39. development候補はrelation MRRで両baselineを厳密に上回り、direct / negative-control MRRがcurrentから退行せず、全relation pathとfeedback isolationを満たす場合だけ選択する。
40. development候補がある場合だけ、未観測holdoutで`current`、`recurrent-balanced`、選択候補を一度評価する。resultの再実行と上書きを拒否する。
41. local recurrent strategyは未観測holdoutで同じgateを通過した場合だけ既定候補になり、それ以外では`current_positive_additive`を維持する。
42. entry retrievalはgraph競合の外側にzero-hop anchorとして保持でき、競合前後で同一値であることをtrace diagnosticsで検証できる。
43. anchored graph signalは少なくとも1 edgeを通ったmessageだけで構成し、zero-hop seed residualとzero-hop pathを含めない。
44. dense retrievalとgraph propagationを独立に無効化でき、BM25-only ablationではdense encoderとgraph traversalを実行しない。
45. explanationはBM25 / denseのraw・normalized値、競合前後のentry anchor、graphのraw・normalized値、final score、zero-hop / graph path種別を区別して保持する。
46. anchored hybrid experimentはproduction D1からread-only取得した新development / holdoutを、過去39 doc pathsを含む4 fixturesおよび相互間で分離し、provenanceとcontamination auditを結果観測前に固定する。
47. anchored hybrid experimentは`current`、`bm25-only`、BM25+現行graph、BM25+dense anchorのlocal/query local、BM25 anchorのlocalを合わせた6 variantsだけを比較する。
48. development候補はrelation MRRがcurrentを厳密に上回り、direct / negative-controlが退行せず、relation path、feedback isolation、anchor invariant、edge-only graph signalをすべて満たす場合だけ選択する。
49. 候補がある場合だけ未観測holdoutで`current`、`bm25-only`、選択候補を一度評価し、同じgateを通過した場合だけdefault変更候補とする。
50. graph activationは`max`、`none`、`l1_mass`を一般設定として選択でき、zero totalを決定論的に全0へ変換できる。
51. final fusionは既存linearに加え、entry rankとpositive graph nodeだけのgraph rankを使うbottom-centered weighted RRFを選択できる。
52. 各traceはentry / graph rank、entry / graph fusion component、normalization、fusion strategy、RRF k、positive graph node数を保持し、final orderingを機械的に再計算できる。
53. fusion calibration experimentはproduction D1の新しい3-node development / holdoutを、既存7 fixturesの50 unique doc pathsおよび相互間から分離し、provenance、balanced gold、contamination audit、6 variantsを結果観測前に固定する。
54. development候補はrelation MRRをcurrentから厳密に改善し、少なくとも1 relation caseを個別改善し、direct / negative-controlのcohort MRRと全個別rankを退行させず、path、feedback、anchor、edge-only graph、formula auditを満たす場合だけ選択する。
55. development候補がある場合だけ未観測holdoutでcurrentと選択候補を一度評価し、同じgateを通過した場合だけdefault変更候補とする。
56. `search_channels`は同一queryからBM25 lexical laneとanchored edge-only relation laneを独立trace IDで同時返却し、cross-lane final score、combined rank、single winnerを生成しない。
57. lexical laneはBM25だけで順位付けし、dense retrievalとgraph propagationをlane順位へ使用せず、保存hitへgraph pathを持たせない。
58. relation laneはBM25+dense entryをseed選択だけに使い、`anchored_local_competition`で1 edge以上を通過したpositive graph nodeをraw activation降順・node ID昇順で順位付けする。
59. channel provenanceはcallerのchannel自己申告でなく独立trace IDに保存し、`record_success`はlexical traceでedgeを変更せず、relation traceでは保存済みcredited pathだけを強化する。
60. 同一nodeが両laneに現れる場合も各rankと説明を保持し、片方のtraceに保存されていないnodeへのfeedbackをatomicに拒否する。
61. independent-channel experimentはproduction D1からread-only取得した相互disjointな2-node / 1-edge developmentとholdoutを、既存9 fixturesの全node pathから分離し、各splitの4-case hard gate、provenance、contamination audit、lane規則、feedback規則、停止規則を結果観測前に固定する。
62. developmentでlane parity、relation個別改善、edge-only path、独立trace、edge不変、feedback帰属、cross-lane拒否、決定性の全hard gateを通過した場合だけholdoutを一度開き、同じgateを全通過した場合だけ`search_channels`をvalidatedと記録する。

## 4. Constraints

- GNN 学習、自動正誤判定、分散実行、GitHub 専用 UI は対象外とする。
- reinforcement は成功の申告でのみ発火し、単なる retrieval impression を学習信号にしない。
- MCP SDK、transport、認証、remote deployment はコア要件に含めない。
- 経路は循環を避け、最大 hop 数と結果ごとの最大説明経路数で計算量を制限する。
- edge weight の強化には上限を設ける。ただし、既存 weight が上限を超えている場合も強化処理で現在値を引き下げない。

## 5. Acceptance verification

- `python -m unittest discover -s tests -v` が単体テストと統合テストを通過する。
- `python -m neuron_graph_rag demo` が取り込み、検索、成功フィードバック、再検索を実演する。
- `python -m neuron_graph_rag eval` が baseline hybrid と graph retrieval の比較指標を出力する。
- `python -m neuron_graph_rag benchmark --fixture ... --gold ...` が固定実コーパス上の比較、説明経路、feedback isolation、仮説判定を出力する。
- CI が editable install、test、eval を新規環境で実行する。
- [Optional MCP Feedback Interface](optional-mcp-interface.md) が tool semantics、input、output、failure、core mapping、依存境界、repository 分離条件を定義する。
- `tests/fixtures/d1_liplus_wiki.json` が実 D1 形状から ingest、検索、時系列 metadata、graph activation、success feedback を再現する。
- [D1 corpus fixture](d1-corpus-fixture.md) が read-only 取得、認証境界、provenance、coverage 比較、再取得手順を定義する。
- [Real-corpus benchmark](real-corpus-benchmark.md) が gold freeze、判定規則、観測結果、外挿限界を定義する。
- [Neural dynamics experiment](neural-dynamics-experiment.md) が development / holdout 分離、固定探索空間、候補選択、単一 holdout 開封、停止規則を定義する。
- [Local recurrent competition experiment](neural-dynamics-local-competition-experiment.md) が新規D1 subgraph、contamination audit、query / path ablation、二baseline gateを定義する。
- [Anchored BM25 and graph hybrid experiment](anchored-bm25-graph-hybrid-experiment.md) がzero-hop anchor、edge-only graph signal、BM25 ablation、新規D1 split、単一holdout gateを定義する。
- [Anchored fusion calibration experiment](anchored-fusion-calibration-experiment.md) がgraph normalization、bottom-centered RRF、新規D1 split、個別case gateを定義する。
- [Independent retrieval channels experiment](independent-retrieval-channels-experiment.md) が非融合lane、独立trace provenance、feedback帰属、4-case hard gate、単一holdout開封を定義する。
