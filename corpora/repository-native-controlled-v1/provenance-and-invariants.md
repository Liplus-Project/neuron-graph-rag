# Provenance and invariants

## Controlled benchmark status

この文書は repository-native controlled corpus v1 の public source document です。評価 query、gold、expected path、feedback schedule、runner、result を定義しません。

## Source

- 固定 public source: [docs/d1-corpus-fixture.md at 6bc85d0](https://github.com/Liplus-Project/neuron-graph-rag/blob/6bc85d0/docs/d1-corpus-fixture.md)
- License: Apache-2.0. この repository の [LICENSE](../../LICENSE) を参照します。

## 内容

検索用 snapshot は content truncation や欠損を含み得るため、byte-exact history の正本とは区別します。source の provenance と coverage を記録し、出力へ credential を含めません。この corpus では source URL と commit を固定して、出典を追跡可能にします。

## 文書間の明示的な参照

- source と観測記録を分離する扱いは [Observation record boundary](observation-record-boundary.md) を参照します。
