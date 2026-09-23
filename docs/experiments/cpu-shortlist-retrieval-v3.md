# CPU shortlist retrieval v3 実測

## 固定条件

v2 は期待 source top5 と各 warm query 60秒以内を通過したが、peak RSS が全段階0 byteで無効だった。v2 の結果を改変・再実行せず、`tests/fixtures/cpu_shortlist_benchmark_v3.json` を結果前の正本とする。v2 と同じ public Li+ snapshot 93文書、独立 source-grounded 3問、CPU 4 thread、モデル・package・model artifact hash、E5 centroid K=50、文書内top2、v2-m3 NLME tau=1、最大100 pair、top5 / 60秒 gate を維持する。

Windows の process handle を明示的な pointer として渡すよう peak RSS 取得を修正した。v3 は cold / warm / 1文書変更 / 復元 update と各 query の peak RSS がすべて正であることを追加 gate とする。空 cache と新規 `tests/evidence/cpu_shortlist_benchmark_v3/observed.json` に一度だけ結果を作成する。登録済み v5 gold / holdout は使わない。全 gate が通過するまで利用 guide の設定手順は公開しない。

## 観測結果

単一 run の正本は `tests/evidence/cpu_shortlist_benchmark_v3/observed.json`。cold index は93文書・2065 chunk で63.641秒、変更なし warm update は記録上0.000秒、1文書変更と復元は各0.094秒。3問の期待 source は1位・3位・1位、warm query は43.343・38.770・40.410秒、forward pair は100・100・99だった。top5 / 60秒 gate はともに通過した。

process peak RSS は cold index 後に2,297,868,288 byte、検索終了までに最大約2.93 GiBで、正の値として記録できた。peak はプロセス開始からの最大値であり、各段階の増分ではない。RSS gate も通過した。これは固定した93文書と3問での opt-in 試作結果であり、未見 corpus 全般の品質や既定検索への採用を示すものではない。

ユーザーから一時停止の指示があったため、利用 guide の設定手順公開と PR 統合は次回へ残す。
