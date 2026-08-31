# Intent-aware observation engine

## 目的

Issue #196は、v21に閉じていたintent-aware workerとfinalizerを、version固有のidentityやpathを持たない
shared engineへ抽出する。抽出後もv20で固定したproduction signal、fusion weights、gate ID、positive
per-case non-regressionを変更せず、既存のobservation lifecycleを二重実装しない。

## Ownership

shared engineは次のpure observation責務だけを持つ。

- protocol、fixture path、stage identity、model identity、gate ID、fusion weightsをspecから受けて検証する。
- worker用にcorpus、relationship、queryを読み、production検索とscorerを注入してcaseを生成する。
- finalizer用にqueryとgoldを読み、quality、protocol validity、candidate gateの順で評価する。
- claim identity、primary/replay一致、model identity、state集約、candidate選択を検証し、上記評価結果のpayloadを組み立てる。

workerはgoldを受け取らない。worker fixture loaderもgold pathを開かず、`gold`、`expected`、`forbidden`、
evaluation labelをworker APIまたはproduction signalへ渡さない。goldを読むのはfinalizerだけであり、protocol
validityが一つでもfailした場合はcandidate gateを評価しない。

stage initialization、planned / actual / successful count、claim count、worker transport、failure transport、
terminal audit、evidence fixateは既存のrank observation lifecycle / stage contractの責務に残す。shared engineは
container command、stage directory、runtime volume、evidence copyを作らず、これらの既存部品を置き換えない。

## v21 parity boundary

v21 source、fixtures、raw evidence、manifestはimmutable predecessorであり、この抽出では変更しない。shared
finalizerへ渡すv21 raw packetは新しいmodel executionではなく、固定済みbytesをbehavior-preserving refactorの
referenceとして読むだけである。

parity比較の対象はcase、quality、protocol / candidate gate、selectionだけとする。v21 raw packetに含まれる
latency、peak RSS、cache bytes、pair countは比較対象にせず、新engineのperformance evidenceへ読み替えない。
payload compatibilityのためmetricsを受け渡しても、新しい測定または性能主張には使用しない。

## v22 result-free composition

`cross_encoder_precision_v22_intent_aware_observation.py`は、contract fixtureをshared specへ変換してengineを
構成する薄いversion moduleである。公開commandは`validate`だけで、model import / load / forward、stage run、
performance observation、result生成を行わない。v22 evidence pathは存在してはならず、statusは
`result-free-engine-scaffold-valid`、performanceは`not assessed`、model execution countとresult countは0に固定する。

v22 contractはv21 registryとterminal evidenceのSHA-256を検証するが、v21結果をv22 resultへ変換しない。
新しいperformance evidenceが必要な場合は、fresh fixtureとfresh stage identityを持つ別protocolで実行する。
