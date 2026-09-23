# CPU shortlist retrieval v2 実測

## 固定条件

v1 は cache に93文書・2065 chunk を書いたが、結果 artifact がなく、停止理由は確認できなかった。`tests/evidence/cpu_shortlist_benchmark_v1/interrupted.json` に残し、品質・待ち時間は未評価とする。v1 の cache から測定を再開しない。

v2 は `tests/fixtures/cpu_shortlist_benchmark_v2.json` を観測前の正本とする。public Li+ snapshot 93文書、独立 source-grounded 3問、E5 centroid K=50、文書内 top2 chunk、v2-m3 NLME tau=1、最大100 pair、期待 source top5、各 warm query 60秒以内は v1 と同一。登録済み v5 gold / holdout は入力に使用しない。

Windows native、CPU 4 thread、固定した Python package 版、E5 ONNX・tokenizer と v2-m3 weights・tokenizer の SHA-256 を事前検証する。空 cache への cold index、変更なし warm update、1文書変更 update、復元 update、query ごとの E5 / v2-m3 / total runtime と process peak RSS を記録する。既存の cache / output path は拒否する。観測結果は `tests/evidence/cpu_shortlist_benchmark_v2/observed.json` に一度だけ新規作成する。

両 gate が通過するまで利用 guide に設定手順を公開しない。

## 観測結果

result-free 固定時点では未観測。結果を生成した commit で本節を更新する。
