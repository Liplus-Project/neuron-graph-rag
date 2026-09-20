# multilingual-e5-small structural / centroid ablation v1

## 目的

既知v5 developmentの正解がE5 body/maxで62位だった診断を起点に、#238と同じ構造表現とdocument centroid集約を独立に比較する。既定retrievalとproduction設定は変更しない。

## 事前固定protocol

93文書を480 Unicode codepoint、80 overlapでstart-to-end投影する。body chunkは#238と同一で、structural armは#238のliteral prefix生成関数をそのまま適用する。modelは `intfloat/multilingual-e5-small` revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`。`query: ` / `passage: ` prefix、attention-mask mean pooling、L2 normalization、512 token末尾truncationを固定した。

armは `body_max`、`structural_max`、`body_centroid`、`structural_centroid`。centroidは正規化済みchunk embeddingの算術平均を再L2正規化し、query cosineを計算する。cutoff 20、source ID昇順tie-break、primary/directional/length-bias attenuation条件はIssue #240とmanifestに固定した。

## result-free parity gate

登録queryを使わない英語・日本語synthetic 3文で、direct ONNX Runtime + tokenizers実装を既存#230 FastEmbed経路と比較した。事前固定した許容差は最大絶対差 `1e-6` 以下、cosine `0.999999` 以上。実測最大絶対差は `3.5867485e-9` から `5.0746169e-9`、cosineは `0.9999999976` 以上で、parity gateは成功した。

## 観測結果

登録one-shotのgold-blind workerは完了したが、finalizer専用allowlistがdevelopment-only goldを同時にrequiredとforbiddenにしたため、v1 protocolはfinalizerで失敗した。preflight、claim、完全なraw worker packet、errorをappend-only evidenceとして保存した。v1 resultは存在せず、この実行を成功扱いしない。one-shotは再実行しない。

別protocol `github-retrieval-parity-v5-e5-structural-centroid-finalizer-recovery-v1` をgold mount前に固定した。recovery claimは、v1 evidenceのexact hash、93文書・各表現2065 passage、384次元query/centroid、body/structural間のchunk identity、4 armの固定式・tie-break・ranking hash・Spearmanをdevelopment goldなしで再検証する。完全性を証明できた場合だけ、既存development-only goldを追加し、登録query実行0回・model forward 0回で順位を導出する。これは完了済みgold-blind worker packetからの派生結果であり、v1成功またはretryではない。

recovery-v1 claimはpacket完全性を証明したが、manifestへdevelopment-only gold SHA-256を64桁ではなく58桁で転記していたため、finalizerはgold hash gateでrank計算前にfail closedした。claimとerrorを保存し、recovery-v1は再試行しない。別IDのrecovery-v2では、このhashだけを実ファイルの64桁 `689028b2a6f827bc915f6d9151c1b6b95164b4dbd59715bb930d573c33c846cf` へ修正し、packet検証、4 arm、式、criteria、cutoff、query 0回、model forward 0回を不変に固定した。

recovery-v2 claimはdevelopment goldなしでpacket完全性を再証明し、その後のdeterministic finalizerは完了した。これは新しいquery/model推論ではなく、完了済みgold-blind worker packetからの派生である。worker runtimeは `101.52003426600277` 秒、peak RSSは `1229111296` bytesだった。

| arm | expected rank | Spearman(chunk count, document score) | cutoff 20 |
|---|---:|---:|---|
| `body_max` | 78 | 0.5673424621223959 | 未達 |
| `structural_max` | 42 | 0.4791736702875985 | 未達 |
| `body_centroid` | 68 | 0.7438443586611451 | 未達 |
| `structural_centroid` | 39 | 0.6145600057038945 | 未達 |

primary successは未達。`structural_max`、`body_centroid`、`structural_centroid` は同じpacketの `body_max` 78位よりstrictに改善し、directional evidenceを満たした。centroidの絶対Spearmanはbody/structuralの双方でmaxより高く、length-bias attenuationは両方とも未達だった。pre-truncation 512 token超過はbody `0/2065`、structural `0/2065`、bodyが512以下かつstructuralが512超過となったpairは0で、structural表現によるtruncation増加はなかった。
