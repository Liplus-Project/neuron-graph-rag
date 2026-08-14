# Feedback policy comparison source corpus v1

これは、後続の独立した比較評価に渡す public corpus-only source です。development と holdout はそれぞれ二つの cluster を持ち、各 cluster は一つの分岐元、二段の有向参照、同じ分岐元から出る別参照で構成します。

relation は各 source document の `## 文書間の明示的な参照` にある同一 directory 内の相対 Markdown link からだけ導出します。catalog、見出し、filename、metadata、manifest 自体から relation を追加しません。

`manifest.json` は split、node、document path、source URL、導出済み edge、raw SHA-256、改行規則、provenance、contamination audit の範囲だけを固定します。後続評価の選択入力や観測物はこの corpus に含めません。

本文は NGR repository の public documentation として Apache-2.0 で提供します。この controlled corpus から外部 corpus、production quality、既定値採用への一般化は行いません。
