# Requirements

## 1. Purpose

このプロジェクトは、ハイブリッド検索、型付き知識グラフ、決定論的な活性伝播、成功フィードバックを、一つの観測可能な RAG パイプラインとして提供する。

## 2. Premises

- 公開 API は特定のデータ源に依存しない。
- コアは Python 標準ライブラリだけで動作する。
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

## 4. Constraints

- GNN 学習、自動正誤判定、分散実行、GitHub 専用 UI は対象外とする。
- reinforcement は成功の申告でのみ発火し、単なる retrieval impression を学習信号にしない。
- 経路は循環を避け、最大 hop 数と結果ごとの最大説明経路数で計算量を制限する。
- edge weight の強化には上限を設ける。ただし、既存 weight が上限を超えている場合も強化処理で現在値を引き下げない。

## 5. Acceptance verification

- `python -m unittest discover -s tests -v` が単体テストと統合テストを通過する。
- `python -m neuron_graph_rag demo` が取り込み、検索、成功フィードバック、再検索を実演する。
- `python -m neuron_graph_rag eval` が baseline hybrid と graph retrieval の比較指標を出力する。
- CI が editable install、test、eval を新規環境で実行する。
