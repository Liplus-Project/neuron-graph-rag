# Cross-encoder precision observation v20

## 目的

Issue #191は、v19のdiagnostic evidenceからcandidate-controllable rank gateとbaseline / prefilter由来のprotocol
validityを分離し、否定・除外意図とrelation path preservationを扱うintent-aware rank fusionをresult-freeで固定する。
v20は次の実測protocolを開くものではなく、model execution、registered query、claim、resultを生成しない。

## 前提

v19はmerge commit `2f5f5d7d658681a495bcaab05e8729b567db9fc7`で固定済みである。bge-base CEは
negative forbidden top 5 countを2から1へ減らしたが、semantic completenessとrelation品質を維持しなかった。
bge-v2-m3 CEはdirect / semantic MRRを0.125000 / 0.291666から1.000000 / 0.750000へ上げた一方、relation
MRRを0.750000から0.666666へ下げ、forbidden countは2のままだった。v19 casesと結果はこの設計診断にだけ使い、
v20または将来protocolのunbiased performance evidenceへ再利用しない。

## Gate ownership

`relation-source-edge-only-provenance`を含むbaseline / prefilter validityはcandidate評価前のprotocol validityとする。
candidateのrank操作では修復できないbaseline provenance失敗をcandidate failureへ混在させず、失敗を緩和してpassへ
変換もしない。candidate側はpositive per-case rank non-regression、cohort MRR / hit@5、negative non-worseningかつ
aggregate strict improvement、expected-source completenessを維持する。case regressionをcohort平均で隠さない。

## Intent-aware fusion

共有helper `intent_aware_rank_fusion.py`のranking inputはquery text、NGR prefilter rank / score、cross-encoderの
positive / exclusion clause logits、relation paths、source identityだけである。gold expected path、forbidden path、評価label、
relevance判定をranking inputへ渡すことをschemaで拒否する。

query textはpositive clauseと0個以上のexclusion clauseへdeterministicに分解する。各candidateはpositive clauseのlogitと
exclusion clauseごとのlogitを持ち、exclusion relevanceの最大値をpenaltyとしてfusion scoreから引く。relation intentが
query textにあり、有効なrelation pathがcandidate sourceをtargetにするときだけ固定bonusを加える。relation pathsは
exact schemaで受け、出力では各field/valueを変更せず再構築する。JSONのkey orderやwhitespaceを含むbyte identityは保証しない。
prefilter rank / score、positive logit、exclusion penalty、relation bonus以外のsignalは使わない。

このdecompositionはen / jaの明示markerを対象にしたfrozen contractであり、自然言語理解の完全性を主張しない。次の
実測protocolではfresh development / holdout identitiesを使用し、unknown intentやmarker coverageも独立して評価する。

## Result-free境界

v20ではmodel import / load / forward、query、development / holdout claim、worker、result、shared SQLite open、runtime volume
create、retryをすべて0とする。performanceは`not assessed`であり、v19 candidateの遡及的な再選択、retrieval parity、
production performance、physical integration、NGR default変更を主張しない。実測は別のfresh successor protocolでのみ行う。
