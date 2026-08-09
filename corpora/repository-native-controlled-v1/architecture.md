# Architecture entry

## Controlled benchmark status

この文書は repository-native controlled corpus v1 の public source document です。評価 query、gold、expected path、feedback schedule、runner、result を定義しません。

## Source

- 固定 public source: [README.md at 6bc85d0](https://github.com/Liplus-Project/neuron-graph-rag/blob/6bc85d0/README.md)
- License: Apache-2.0. この repository の [LICENSE](../../LICENSE) を参照します。

## 内容

Neuron Graph RAG は hybrid retrieval を入口にして、typed relation を通じて activation を伝播し、入口 score と graph activation を分けて扱います。検索の記録と成功利用の記録も分離します。この文書は、入口から relation を扱う文書と、出典の境界を扱う文書の両方を参照します。

## 文書間の明示的な参照

- 入口候補と活性伝播の扱いは [Retrieval and activation](retrieval-and-activation.md) を参照します。
- 出典と不変性の扱いは [Provenance and invariants](provenance-and-invariants.md) を参照します。
