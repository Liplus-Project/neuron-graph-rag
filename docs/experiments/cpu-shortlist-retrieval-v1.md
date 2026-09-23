# CPU shortlist retrieval v1 実測

## 固定条件

`tests/fixtures/cpu_shortlist_benchmark_v1.json` を観測前の正本とする。public Li+ snapshot 93文書へ、既存 v5 query / gold / holdout と独立した3件の source-grounded query を流す。期待 source は各 query 作成時に固定し、top 5 到達を動作確認 gate とする。

実行は CPU 4 thread、E5 centroid K=50、文書内 E5 top2 chunk、v2-m3 NLME tau=1、最大100 forward pair。空の SQLite cache への cold index、変更なしの warm update、1文書変更 update、復元 update、queryごとの E5 / v2-m3 / total runtime と process peak RSS を記録する。warm-cache query 60秒以内と3件の source-grounded gateを両方通過するまで、利用 guide に設定手順を公開しない。

model は固定 revision の local snapshotを明示指定し、runnerは output と cache の既存 path を拒否する。観測結果は `tests/evidence/cpu_shortlist_benchmark_v1/observed.json` に一度だけ新規作成する。

## 観測結果

result-free 固定時点では未観測。結果を生成した commit で本節を更新する。
