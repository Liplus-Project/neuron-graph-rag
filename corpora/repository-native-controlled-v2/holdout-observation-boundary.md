# Holdout observation boundary

## Corpus metadata

- Corpus split: `holdout`
- Path ordinal: `3`
- Corpus node ID: `v2-holdout-3`
- Source URL: https://github.com/Liplus-Project/neuron-graph-rag/blob/03527fd3eab76bbdec652eb74dfdec15b172cb34/docs/node-first-blind-selection-experiment.md
- License: Apache-2.0

## Controlled benchmark status

この文書は repository-native controlled corpus v2 の public source document です。評価 query、gold、expected path、feedback schedule、runner、result を定義しません。

## 内容

invalid response は retry、補完、再集約しません。development の全 gate を通過するまで holdout stage、packet、judge 提示、gold scoring を禁止し、観測 artifact の上書きや再生成を拒否します。

## 文書間の明示的な参照

この terminal document には corpus relation を作る相対 Markdown link がありません。
