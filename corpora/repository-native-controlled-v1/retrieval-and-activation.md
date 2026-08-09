# Retrieval and activation

## Controlled benchmark status

この文書は repository-native controlled corpus v1 の public source document です。評価 query、gold、expected path、feedback schedule、runner、result を定義しません。

## Source

- 固定 public source: [README.md at 6bc85d0](https://github.com/Liplus-Project/neuron-graph-rag/blob/6bc85d0/README.md)
- License: Apache-2.0. この repository の [LICENSE](../../LICENSE) を参照します。

## 内容

Entry score は lexical と dense の入口信号を統合した score です。seed node からの activation は typed relation、factuality、weight、hop decay を用いて伝播します。activation の減衰は confidence や factuality を変更せず、検索だけでは relation weight を変更しません。

## 文書間の明示的な参照

- 成功した利用に対してどの経路だけを強化対象とするかは [Path reinforcement boundary](path-reinforcement-boundary.md) を参照します。
