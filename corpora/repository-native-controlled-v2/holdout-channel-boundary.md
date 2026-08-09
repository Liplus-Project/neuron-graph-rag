# Holdout channel boundary

## Corpus metadata

- Corpus split: `holdout`
- Path ordinal: `2`
- Corpus node ID: `v2-holdout-2`
- Source URL: https://github.com/Liplus-Project/neuron-graph-rag/blob/03527fd3eab76bbdec652eb74dfdec15b172cb34/docs/independent-retrieval-channels-experiment.md
- License: Apache-2.0

## Controlled benchmark status

この文書は repository-native controlled corpus v2 の public source document です。評価 query、gold、expected path、feedback schedule、runner、result を定義しません。

## 内容

lexical lane と relation lane は合成せず、同じ query に対する独立 trace として保存します。lexical trace の feedback は graph edge を強化せず、relation trace は保存済み credited path だけを強化対象にします。

## 文書間の明示的な参照

- response の単一観測と holdout 未開封の境界は [Holdout observation boundary](holdout-observation-boundary.md) を参照します。
