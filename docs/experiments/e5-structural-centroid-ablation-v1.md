# multilingual-e5-small structural / centroid ablation v1

## 目的

既知v5 developmentの正解がE5 body/maxで62位だった診断を起点に、#238と同じ構造表現とdocument centroid集約を独立に比較する。既定retrievalとproduction設定は変更しない。

## 事前固定protocol

93文書を480 Unicode codepoint、80 overlapでstart-to-end投影する。body chunkは#238と同一で、structural armは#238のliteral prefix生成関数をそのまま適用する。modelは `intfloat/multilingual-e5-small` revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`。`query: ` / `passage: ` prefix、attention-mask mean pooling、L2 normalization、512 token末尾truncationを固定した。

armは `body_max`、`structural_max`、`body_centroid`、`structural_centroid`。centroidは正規化済みchunk embeddingの算術平均を再L2正規化し、query cosineを計算する。cutoff 20、source ID昇順tie-break、primary/directional/length-bias attenuation条件はIssue #240とmanifestに固定した。

## result-free parity gate

登録queryを使わない英語・日本語synthetic 3文で、direct ONNX Runtime + tokenizers実装を既存#230 FastEmbed経路と比較した。事前固定した許容差は最大絶対差 `1e-6` 以下、cosine `0.999999` 以上。実測最大絶対差は `3.5867485e-9` から `5.0746169e-9`、cosineは `0.9999999976` 以上で、parity gateは成功した。

## 観測結果

登録one-shotはまだ実行していない。
