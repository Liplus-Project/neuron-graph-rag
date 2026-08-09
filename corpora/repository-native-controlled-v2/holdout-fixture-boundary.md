# Holdout fixture boundary

## Corpus metadata

- Corpus split: `holdout`
- Path ordinal: `1`
- Corpus node ID: `v2-holdout-1`
- Source URL: https://github.com/Liplus-Project/neuron-graph-rag/blob/03527fd3eab76bbdec652eb74dfdec15b172cb34/docs/d1-corpus-fixture.md
- License: Apache-2.0

## Controlled benchmark status

この文書は repository-native controlled corpus v2 の public source document です。評価 query、gold、expected path、feedback schedule、runner、result を定義しません。

## 内容

D1 corpus fixture は read-only acquisition で得る検索用 snapshot です。document の正本、typed edge の endpoint、source metadata を区別し、取得処理が write を行わないことを検証します。

## 文書間の明示的な参照

- independent lane と feedback 帰属の境界は [Holdout channel boundary](holdout-channel-boundary.md) を参照します。
