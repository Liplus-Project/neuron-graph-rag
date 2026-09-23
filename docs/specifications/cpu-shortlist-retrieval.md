# 任意のCPU shortlist検索

#246 の試作は、#244 で確認した二段検索を独立した opt-in API として提供する。既定の `NeuronGraphRAG.search()` と #230 の `attach_semantic_retriever()` は変更しない。v2-m3 の logit と既存 sparse / dense / graph score は尺度が異なるため合成せず、`search_cpu_shortlist()` は別の順位と診断を返す。

索引は multilingual-e5-small revision `614241f622f53c4eeff9890bdc4f31cfecc418b3` を固定する。文書 path、filename、title、ATX heading chain を本文窓へ付加し、Unicode code point 480文字、overlap 80文字で末尾まで分割する。chunk embedding の正規化平均を document centroid とし、SQLite cache に model identity、構造化規則、node ID、path、本文 SHA-256、chunk 範囲、embedding を保存する。更新は content identity が変わった文書だけを再計算し、削除済み文書は同じ transaction で除く。schema / model / 規則が一致しない cache は明示的な `error` または caller が選んだ全再構築だけを許可する。

query は同じ E5 で全 cached document centroid を cosine 順に並べ、source ID の昇順で同値を解決して上位50件を取る。各候補では chunk cosine 上位2件を選び、最大100 pairを bge-reranker-v2-m3 revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` へ渡す。文書 score は `log(mean(exp(logit / 1)))`、すなわち tau=1 の NLME とし、最終同値も source ID で解決する。cache と現在の corpus が一致しない場合は検索中に補修せず失敗する。

`CpuShortlistRetriever.update_cache()` は索引更新、`attach_cpu_shortlist_retriever()` は engine の専用 slot への接続、`engine.search_cpu_shortlist()` は検索を担う。timeout と cancel は各段階および backend batch 境界で検査する。停止要求や期限超過、backend errorで既定順位へ切り替えない。progress callback は cache、E5 shortlist、v2-m3 の段階と件数を通知する。結果と更新 receipt は E5 / v2-m3 別 runtime、forward pair数、process peak RSS、model revision、cache fingerprint を記録する。

model library と model 自体は import 時にロードしない。既定 backend は caller が明示した local snapshot path だけを開き、ネットワーク取得は行わない。取得は `download_cpu_shortlist_models()` の明示呼び出しに限定する。

利用導線は独立した source-grounded query 群で正しさを確認し、CPU 4 thread、warm cache の query total が固定目標60秒以内の場合だけ公開する。未達時も内部 API と測定結果は残せるが、guide の利用手順を公開しない。#244 の既知 development query は定数・順位計算の一致確認に限り、registered v5 gold / holdout は参照しない。
