# Repository-native controlled corpus v1

これは Neuron Graph RAG repository 内で公開する **controlled benchmark** 用の source corpus です。評価 code、evaluation query、gold answer、expected path、feedback schedule、runner、result artifact は含みません。

## 固定 source snapshot

各文書は repository の公開 documentation を出典とし、source URL は corpus 作成前の commit [`6bc85d0`](https://github.com/Liplus-Project/neuron-graph-rag/tree/6bc85d0) に固定します。各 source document の先頭に、対応する public GitHub file URL を記載します。

## Corpus boundary

この corpus の source document は次の 6 ファイルです。ファイル名の列挙は catalog であり、文書間 relation を表しません。

- `architecture.md`
- `retrieval-and-activation.md`
- `path-reinforcement-boundary.md`
- `provenance-and-invariants.md`
- `observation-record-boundary.md`
- `success-use-record.md`

文書間 relation は、各 source document の `## 文書間の明示的な参照` にある同一 directory 内への相対 Markdown link だけから導出します。見出し、ファイル名、catalog、source URL、またはこの README の記述から pseudo edge を作りません。

## License and created scope

この corpus の本文は、この repository の公開 documentation を限定的に再構成したものです。repository と同じ [Apache-2.0](../../LICENSE) で提供します。出典は固定 snapshot で追跡でき、作成時の対象は repository-native controlled benchmark に限られます。外部 corpus、別 repository、一般的な retrieval quality、production adoption への一般化を主張しません。
