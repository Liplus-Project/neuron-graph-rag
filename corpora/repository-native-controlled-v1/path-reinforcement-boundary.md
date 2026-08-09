# Path reinforcement boundary

## Controlled benchmark status

この文書は repository-native controlled corpus v1 の public source document です。評価 query、gold、expected path、feedback schedule、runner、result を定義しません。

## Source

- 固定 public source: [README.md at 6bc85d0](https://github.com/Liplus-Project/neuron-graph-rag/blob/6bc85d0/README.md)
- License: Apache-2.0. この repository の [LICENSE](../../LICENSE) を参照します。

## 内容

Reinforcement は、成功した node へ至る上位の説明経路に含まれる relation weight だけを強化する処理です。検索そのものは weight を変えず、成功利用の記録と強化対象の決定を混同しません。

## 文書間の明示的な参照

- 成功利用を明示して記録する境界は [Success use record](success-use-record.md) を参照します。
