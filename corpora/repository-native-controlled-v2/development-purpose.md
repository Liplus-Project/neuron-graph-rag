# Development purpose

## Corpus metadata

- Corpus split: `development`
- Path ordinal: `0`
- Corpus node ID: `v2-dev-0`
- Source URL: https://github.com/Liplus-Project/neuron-graph-rag/blob/03527fd3eab76bbdec652eb74dfdec15b172cb34/docs/requirements.md
- License: Apache-2.0

## Controlled benchmark status

この文書は repository-native controlled corpus v2 の public source document です。評価 query、gold、expected path、feedback schedule、runner、result を定義しません。

## 内容

NGR は hybrid retrieval、型付き有向 edge、決定論的な activation、success feedback を、一つの観測可能な RAG pipeline として扱います。検索 trace と成功 feedback は別の記録であり、検索だけで edge weight を変更しません。

## 文書間の明示的な参照

- trace と feedback を分ける境界は [Development trace boundary](development-trace-boundary.md) を参照します。
