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
